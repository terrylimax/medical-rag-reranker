from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from medical_rag_reranker.retrieval.bm25 import BM25Retriever


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _doc_text(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def _build_maps(
    qa_rows: list[dict[str, Any]],
    corpus_rows: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    qa_by_qid: dict[str, dict[str, Any]] = {}
    for row in qa_rows:
        qid = row.get("question_id")
        if qid is not None:
            qa_by_qid.setdefault(str(qid), row)

    corpus_by_qid: dict[str, dict[str, Any]] = {}
    docstore: dict[str, dict[str, Any]] = {}
    for row in corpus_rows:
        doc_id = row.get("doc_id")
        if doc_id is not None:
            docstore[str(doc_id)] = row
        qid = row.get("question_id")
        if qid is not None:
            corpus_by_qid.setdefault(str(qid), row)

    return qa_by_qid, corpus_by_qid, docstore


def _write_negative_pool(
    *,
    path: Path,
    doc_ids: list[str],
    docstore: dict[str, dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        row = docstore.get(doc_id)
        if row is None:
            continue
        text = _doc_text(row)
        if not text:
            continue
        rows.append({"doc_id": doc_id, "text": text})
    write_jsonl(path, rows)


def _pool_doc_ids_for_split(
    *,
    split_ids: list[str],
    corpus_by_qid: dict[str, dict[str, Any]],
    docstore: dict[str, dict[str, Any]],
    restrict_to_split: bool,
) -> list[str]:
    if not restrict_to_split:
        return list(docstore.keys())

    doc_ids: list[str] = []
    for qid in split_ids:
        row = corpus_by_qid.get(qid)
        if row is None or row.get("doc_id") is None:
            continue
        doc_ids.append(str(row["doc_id"]))
    return doc_ids


def _select_negatives(
    *,
    query: str,
    gold_doc_id: str,
    bm25: BM25Retriever | None,
    candidate_doc_ids: list[str],
    negatives_per_query: int,
    hard_negative_pool_size: int,
    rng: random.Random,
) -> list[str]:
    negative_doc_ids: list[str] = []

    if bm25 is not None:
        hits = bm25.retrieve(query, top_k=int(hard_negative_pool_size) + 1)
        for hit in hits:
            if hit.doc_id == gold_doc_id or hit.doc_id in negative_doc_ids:
                continue
            negative_doc_ids.append(hit.doc_id)
            if len(negative_doc_ids) >= int(negatives_per_query):
                return negative_doc_ids

    random_pool = [
        doc_id
        for doc_id in candidate_doc_ids
        if doc_id != gold_doc_id and doc_id not in negative_doc_ids
    ]
    rng.shuffle(random_pool)
    need = int(negatives_per_query) - len(negative_doc_ids)
    if need > 0:
        negative_doc_ids.extend(random_pool[:need])

    return negative_doc_ids[: int(negatives_per_query)]


def _build_split_rows(
    *,
    split_name: str,
    split_ids: list[str],
    qa_by_qid: dict[str, dict[str, Any]],
    corpus_by_qid: dict[str, dict[str, Any]],
    docstore: dict[str, dict[str, Any]],
    out_dir: Path,
    restrict_negative_pool_to_split: bool,
    negatives_per_query: int,
    hard_negative_pool_size: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_doc_ids = _pool_doc_ids_for_split(
        split_ids=split_ids,
        corpus_by_qid=corpus_by_qid,
        docstore=docstore,
        restrict_to_split=restrict_negative_pool_to_split,
    )

    pool_path = out_dir / "negative_pools" / f"{split_name}.corpus.jsonl"
    _write_negative_pool(path=pool_path, doc_ids=candidate_doc_ids, docstore=docstore)

    bm25: BM25Retriever | None = None
    if candidate_doc_ids:
        bm25 = BM25Retriever()
        bm25.index(str(pool_path))

    rows: list[dict[str, Any]] = []
    skipped_missing = 0
    skipped_empty = 0
    total_negatives = 0

    for qid in split_ids:
        qa_row = qa_by_qid.get(qid)
        gold_doc = corpus_by_qid.get(qid)
        if qa_row is None or gold_doc is None:
            skipped_missing += 1
            continue

        query = str(qa_row.get("question") or "").strip()
        gold_doc_id = str(gold_doc.get("doc_id") or "").strip()
        positive_text = _doc_text(gold_doc)
        if not query or not gold_doc_id or not positive_text:
            skipped_empty += 1
            continue

        negative_doc_ids = _select_negatives(
            query=query,
            gold_doc_id=gold_doc_id,
            bm25=bm25,
            candidate_doc_ids=candidate_doc_ids,
            negatives_per_query=negatives_per_query,
            hard_negative_pool_size=hard_negative_pool_size,
            rng=rng,
        )
        valid_negative_doc_ids: list[str] = []
        negative_texts: list[str] = []
        for doc_id in negative_doc_ids:
            if doc_id not in docstore:
                continue
            negative_text = _doc_text(docstore[doc_id])
            if not negative_text:
                continue
            valid_negative_doc_ids.append(doc_id)
            negative_texts.append(negative_text)
        negative_doc_ids = valid_negative_doc_ids
        total_negatives += len(negative_doc_ids)

        rows.append(
            {
                "query_id": qid,
                "query": query,
                "positive_doc_id": gold_doc_id,
                "positive_text": positive_text,
                "negative_doc_ids": negative_doc_ids,
                "negative_texts": negative_texts,
                "split": split_name,
            }
        )

    metadata = {
        "split": split_name,
        "num_queries_requested": len(split_ids),
        "num_rows": len(rows),
        "num_negative_pool_docs": len(candidate_doc_ids),
        "num_negatives": total_negatives,
        "avg_negatives_per_query": total_negatives / max(1, len(rows)),
        "skipped_missing": skipped_missing,
        "skipped_empty": skipped_empty,
        "negative_pool_path": str(pool_path),
        "restrict_negative_pool_to_split": bool(restrict_negative_pool_to_split),
    }
    return rows, metadata


def build_retriever_training_data(
    *,
    processed_dir: str,
    out_dir: str,
    train_out: str = "train_retriever.jsonl",
    val_out: str = "val_retriever.jsonl",
    train_split: str = "train",
    val_split: str = "val",
    negatives_per_query: int = 4,
    hard_negative_pool_size: int = 50,
    restrict_train_negatives_to_train: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Build contrastive bi-encoder training rows from prepared RAG artifacts."""
    processed = Path(processed_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    qa_path = processed / "qa.jsonl"
    corpus_path = processed / "corpus.jsonl"
    splits_path = processed / "splits.json"
    missing = [
        str(path) for path in (qa_path, corpus_path, splits_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Prepared retrieval artifacts are missing: " + ", ".join(missing)
        )

    qa_rows = read_jsonl(qa_path)
    corpus_rows = read_jsonl(corpus_path)
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    qa_by_qid, corpus_by_qid, docstore = _build_maps(qa_rows, corpus_rows)

    train_ids = [str(qid) for qid in splits.get(train_split, [])]
    val_ids = [str(qid) for qid in splits.get(val_split, [])]

    train_rows, train_meta = _build_split_rows(
        split_name=train_split,
        split_ids=train_ids,
        qa_by_qid=qa_by_qid,
        corpus_by_qid=corpus_by_qid,
        docstore=docstore,
        out_dir=out_root,
        restrict_negative_pool_to_split=bool(restrict_train_negatives_to_train),
        negatives_per_query=int(negatives_per_query),
        hard_negative_pool_size=int(hard_negative_pool_size),
        rng=random.Random(int(seed)),
    )
    val_rows, val_meta = _build_split_rows(
        split_name=val_split,
        split_ids=val_ids,
        qa_by_qid=qa_by_qid,
        corpus_by_qid=corpus_by_qid,
        docstore=docstore,
        out_dir=out_root,
        restrict_negative_pool_to_split=False,
        negatives_per_query=int(negatives_per_query),
        hard_negative_pool_size=int(hard_negative_pool_size),
        rng=random.Random(int(seed) + 1),
    )

    train_path = out_root / train_out
    val_path = out_root / val_out
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    metadata = {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "processed_dir": str(processed),
        "qa_path": str(qa_path),
        "corpus_path": str(corpus_path),
        "splits_path": str(splits_path),
        "negatives_per_query": int(negatives_per_query),
        "hard_negative_pool_size": int(hard_negative_pool_size),
        "seed": int(seed),
        "train": train_meta,
        "val": val_meta,
    }
    metadata_path = out_root / "retriever_training_data_summary.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def run_from_cfg(cfg) -> dict[str, Any]:
    run_cfg = cfg.run.retriever_training_data
    return build_retriever_training_data(
        processed_dir=str(run_cfg.processed_dir),
        out_dir=str(run_cfg.out_dir),
        train_out=str(run_cfg.train_out),
        val_out=str(run_cfg.val_out),
        train_split=str(run_cfg.train_split),
        val_split=str(run_cfg.val_split),
        negatives_per_query=int(run_cfg.negatives_per_query),
        hard_negative_pool_size=int(run_cfg.hard_negative_pool_size),
        restrict_train_negatives_to_train=bool(
            run_cfg.restrict_train_negatives_to_train
        ),
        seed=int(run_cfg.seed),
    )
