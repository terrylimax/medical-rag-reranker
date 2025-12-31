from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


@dataclass(frozen=True)
class RerankerExample:
    query: str
    document: str
    label: int


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
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        max_length: int = 256,
        batch_size: int = 8,
        num_workers: int = 2,
        negatives_per_query: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.negatives_per_query = negatives_per_query
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.train_ds: PairDataset | None = None
        self.val_ds: PairDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        random.seed(self.seed)

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

        train_examples = self._build_examples(medquad.iloc[train_idx], medquad, pubmed)
        val_examples = self._build_examples(medquad.iloc[val_idx], medquad, pubmed)

        self.train_ds = PairDataset(train_examples, self.tokenizer, self.max_length)
        self.val_ds = PairDataset(val_examples, self.tokenizer, self.max_length)

    def _build_examples(
        self, subset: pd.DataFrame, all_qa: pd.DataFrame, pubmed: pd.DataFrame
    ) -> list[RerankerExample]:
        examples: list[RerankerExample] = []
        qa_answers = all_qa["a"].tolist()
        pubmed_docs = pubmed["abstract"].tolist()

        for _, row in subset.iterrows():
            q = str(row["q"])
            a_pos = str(row["a"])
            examples.append(RerankerExample(query=q, document=a_pos, label=1))

            # негативы: другие ответы + pubmed абстракты
            for _ in range(self.negatives_per_query):
                if random.random() < 0.5:
                    neg = random.choice(qa_answers)
                else:
                    neg = random.choice(pubmed_docs)
                examples.append(RerankerExample(query=q, document=str(neg), label=0))

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
