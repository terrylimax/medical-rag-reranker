from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from medical_rag_reranker.retrieval.bm25 import BM25Retriever
from medical_rag_reranker.utils.progress import count_text_lines, progress, timed_stage


@dataclass(frozen=True)
class RerankerExample:
    query: str
    document: str
    label: int
    query_id: str | None = None
    doc_id: str | None = None


class PairDataset(Dataset):
    def __init__(
        self, examples: list[RerankerExample], tokenizer, max_length: int
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex.query,
            ex.document,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(ex.label, dtype=torch.float32)
        return item


class RerankerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        max_length: int = 256,
        batch_size: int = 8,
        num_workers: int = 2,
        negatives_per_query: int = 4,
        prefer_prepared_artifacts: bool = True,
        hard_negative_pool_size: int = 20,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.negatives_per_query = negatives_per_query
        self.prefer_prepared_artifacts = prefer_prepared_artifacts
        self.hard_negative_pool_size = max(
            int(hard_negative_pool_size),
            int(negatives_per_query),
            1,
        )
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.train_ds: PairDataset | None = None
        self.val_ds: PairDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        random.seed(self.seed)

        if self.prefer_prepared_artifacts and self._prepared_artifacts_exist():
            self._setup_from_prepared()
            return

        self._setup_from_raw()

    def _prepared_artifacts_exist(self) -> bool:
        required = (
            self.processed_dir / "qa.jsonl",
            self.processed_dir / "corpus.jsonl",
            self.processed_dir / "splits.json",
        )
        return all(path.exists() for path in required)

    def _setup_from_prepared(self) -> None:
        with timed_stage("Prepare reranker data from processed artifacts"):
            qa_rows = self._read_jsonl(self.processed_dir / "qa.jsonl")
            corpus_rows = self._read_jsonl(self.processed_dir / "corpus.jsonl")
            splits = json.loads((self.processed_dir / "splits.json").read_text("utf-8"))

        qa_by_qid = {
            str(row["question_id"]): row
            for row in qa_rows
            if row.get("question_id") is not None
        }
        docstore = {
            str(row["doc_id"]): row
            for row in corpus_rows
            if row.get("doc_id") is not None
        }
        corpus_by_qid = {
            str(row["question_id"]): row
            for row in corpus_rows
            if row.get("question_id") is not None
        }

        bm25 = BM25Retriever()
        bm25.index(str(self.processed_dir / "corpus.jsonl"))

        train_examples = self._build_examples_from_prepared(
            split_ids=[str(qid) for qid in splits.get("train", [])],
            qa_by_qid=qa_by_qid,
            corpus_by_qid=corpus_by_qid,
            docstore=docstore,
            bm25=bm25,
            rng=random.Random(self.seed),
        )
        val_examples = self._build_examples_from_prepared(
            split_ids=[str(qid) for qid in splits.get("val", [])],
            qa_by_qid=qa_by_qid,
            corpus_by_qid=corpus_by_qid,
            docstore=docstore,
            bm25=bm25,
            rng=random.Random(self.seed + 1),
        )

        self.train_ds = PairDataset(train_examples, self.tokenizer, self.max_length)
        self.val_ds = PairDataset(val_examples, self.tokenizer, self.max_length)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        total = count_text_lines(path)
        with path.open("r", encoding="utf-8") as f:
            iterable = progress(f, desc=f"Reading {path.name}", total=total, unit="row")
            for line in iterable:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _build_examples_from_prepared(
        self,
        *,
        split_ids: list[str],
        qa_by_qid: dict[str, dict[str, Any]],
        corpus_by_qid: dict[str, dict[str, Any]],
        docstore: dict[str, dict[str, Any]],
        bm25: BM25Retriever,
        rng: random.Random,
    ) -> list[RerankerExample]:
        examples: list[RerankerExample] = []
        all_doc_ids = list(docstore.keys())

        iterable = progress(
            split_ids,
            desc="Preparing reranker examples",
            total=len(split_ids),
            unit="query",
        )
        for qid in iterable:
            qa_row = qa_by_qid.get(qid)
            gold_doc = corpus_by_qid.get(qid)
            if qa_row is None or gold_doc is None:
                continue

            query = str(qa_row.get("question") or "").strip()
            gold_doc_id = str(gold_doc.get("doc_id") or "").strip()
            gold_text = str(gold_doc.get("text") or "").strip()
            if not query or not gold_doc_id or not gold_text:
                continue

            examples.append(
                RerankerExample(
                    query=query,
                    document=gold_text,
                    label=1,
                    query_id=qid,
                    doc_id=gold_doc_id,
                )
            )

            negative_doc_ids: list[str] = []
            hits = bm25.retrieve(
                query,
                top_k=int(self.hard_negative_pool_size) + 1,
            )
            for hit in hits:
                if hit.doc_id == gold_doc_id or hit.doc_id in negative_doc_ids:
                    continue
                negative_doc_ids.append(hit.doc_id)
                if len(negative_doc_ids) >= int(self.negatives_per_query):
                    break

            if len(negative_doc_ids) < int(self.negatives_per_query):
                random_pool = [
                    doc_id
                    for doc_id in all_doc_ids
                    if doc_id != gold_doc_id and doc_id not in negative_doc_ids
                ]
                rng.shuffle(random_pool)
                need = int(self.negatives_per_query) - len(negative_doc_ids)
                negative_doc_ids.extend(random_pool[:need])

            for doc_id in negative_doc_ids[: int(self.negatives_per_query)]:
                negative_row = docstore.get(doc_id)
                if negative_row is None:
                    continue
                negative_text = str(negative_row.get("text") or "").strip()
                if not negative_text:
                    continue
                examples.append(
                    RerankerExample(
                        query=query,
                        document=negative_text,
                        label=0,
                        query_id=qid,
                        doc_id=str(doc_id),
                    )
                )

        return examples

    def _setup_from_raw(self) -> None:
        medquad = pd.read_parquet(self.raw_dir / "medquad" / "train.parquet")
        pubmed = pd.read_parquet(self.raw_dir / "pubmed" / "train.parquet")

        # Приводим к ожидаемым полям
        medquad = medquad.rename(columns={"question": "q", "answer": "a"})
        pubmed = pubmed.rename(columns={"title": "title", "abstract": "abstract"})

        medquad = medquad.dropna(subset=["q", "a"]).reset_index(drop=True)
        pubmed = pubmed.dropna(subset=["abstract"]).reset_index(drop=True)

        indices = list(range(len(medquad)))
        random.shuffle(indices)
        cut = int(0.85 * len(indices))
        train_idx, val_idx = indices[:cut], indices[cut:]

        train_examples = self._build_examples(
            medquad.iloc[train_idx],
            medquad,
            pubmed,
            desc="Preparing raw train reranker examples",
        )
        val_examples = self._build_examples(
            medquad.iloc[val_idx],
            medquad,
            pubmed,
            desc="Preparing raw val reranker examples",
        )

        self.train_ds = PairDataset(train_examples, self.tokenizer, self.max_length)
        self.val_ds = PairDataset(val_examples, self.tokenizer, self.max_length)

    def _build_examples(
        self,
        subset: pd.DataFrame,
        all_qa: pd.DataFrame,
        pubmed: pd.DataFrame,
        desc: str,
    ) -> list[RerankerExample]:
        examples: list[RerankerExample] = []
        qa_answers = all_qa["a"].tolist()
        pubmed_docs = pubmed["abstract"].tolist()

        rows = progress(subset.iterrows(), desc=desc, total=len(subset), unit="query")
        for _, row in rows:
            q = str(row["q"])
            a_pos = str(row["a"])
            examples.append(
                RerankerExample(
                    query=q,
                    document=a_pos,
                    label=1,
                    query_id=str(row.get("question_id"))
                    if "question_id" in row
                    else None,
                )
            )

            # негативы: другие ответы + pubmed абстракты
            for _ in range(self.negatives_per_query):
                if random.random() < 0.5:
                    neg = random.choice(qa_answers)
                else:
                    neg = random.choice(pubmed_docs)
                examples.append(
                    RerankerExample(
                        query=q,
                        document=str(neg),
                        label=0,
                        query_id=str(row.get("question_id"))
                        if "question_id" in row
                        else None,
                    )
                )

        return examples

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
