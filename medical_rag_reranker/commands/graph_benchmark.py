from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from medical_rag_reranker.graph.aspects import (
    aspects_from_metadata,
    coerce_str_list,
    format_aspect_list,
)


DEFAULT_COMPOSITES: tuple[tuple[str, ...], ...] = (
    ("symptoms", "causes", "treatments"),
    ("inheritance", "diagnosis"),
    ("symptoms", "causes", "management"),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_split_ids(splits_path: Path, split_name: str) -> set[str] | None:
    if split_name == "all":
        return None
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if split_name not in splits:
        raise ValueError(f"Split `{split_name}` not found in {splits_path}")
    return {str(v) for v in splits[split_name]}


def _object_key(row: dict[str, Any]) -> str | None:
    document_id = str(row.get("document_id") or "").strip()
    if document_id:
        return f"document_id:{document_id}"
    cuis = coerce_str_list(row.get("umls_cui"))
    if cuis:
        return f"cui:{cuis[0]}"
    focus = str(row.get("question_focus") or row.get("topic") or "").strip()
    if focus:
        return f"focus:{focus.lower()}"
    return None


def _object_label(rows: list[dict[str, Any]], object_key: str) -> str:
    for row in rows:
        focus = str(row.get("question_focus") or row.get("topic") or "").strip()
        if focus:
            return focus
    return object_key.split(":", 1)[-1]


def _query_text(label: str, aspects: tuple[str, ...]) -> str:
    aspect_set = set(aspects)
    if {"symptoms", "causes", "treatments"}.issubset(aspect_set):
        return f"What are the symptoms, causes, and treatments of {label}?"
    if {"inheritance", "diagnosis"}.issubset(aspect_set):
        return f"How is {label} inherited and diagnosed?"
    if {"symptoms", "causes", "management"}.issubset(aspect_set):
        return f"Summarize symptoms, genetic causes, and management of {label}."
    return f"Summarize {format_aspect_list(aspects)} of {label}."


def _reference_answer(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows:
        doc_id = row.get("doc_id")
        qtype = row.get("question_type") or "answer"
        text = " ".join(str(row.get("text") or "").split())
        if text:
            parts.append(f"[{doc_id}] {qtype}: {text}")
    return "\n\n".join(parts)


def build_graph_multidoc_benchmark(
    corpus_path: str,
    splits_path: str,
    out_dir: str,
    split_name: str = "test",
    max_queries: int = 300,
    seed: int = 42,
    min_relevant_docs: int = 2,
) -> dict[str, Any]:
    """Build composite graph-RAG queries and multi-document qrels."""
    corpus = _read_jsonl(Path(corpus_path))
    split_ids = _load_split_ids(Path(splits_path), split_name)

    scoped_rows: list[dict[str, Any]] = []
    for row in corpus:
        qid = str(row.get("question_id") or "")
        if split_ids is not None and qid not in split_ids:
            continue
        if not row.get("doc_id"):
            continue
        scoped_rows.append(row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scoped_rows:
        key = _object_key(row)
        if key:
            grouped[key].append(row)

    rng = random.Random(seed)
    object_items = list(grouped.items())
    rng.shuffle(object_items)

    queries: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    qrels_lines: list[str] = []

    for object_key, rows in object_items:
        by_aspect: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            for aspect in aspects_from_metadata(row):
                by_aspect[aspect].append(row)
        if len(by_aspect) < 2:
            continue

        label = _object_label(rows, object_key)
        for composite in DEFAULT_COMPOSITES:
            available = tuple(aspect for aspect in composite if by_aspect.get(aspect))
            if len(available) < 2:
                continue

            relevant_by_doc: dict[str, dict[str, Any]] = {}
            for aspect in available:
                for row in by_aspect[aspect]:
                    relevant_by_doc[str(row["doc_id"])] = row

            relevant_rows = list(relevant_by_doc.values())
            if len(relevant_rows) < int(min_relevant_docs):
                continue

            query_id = f"graph_{len(queries) + 1:05d}"
            gold_doc_ids = [str(row["doc_id"]) for row in relevant_rows]
            cuis = sorted(
                {
                    value
                    for row in relevant_rows
                    for value in coerce_str_list(row.get("umls_cui"))
                }
            )

            queries.append(
                {
                    "query_id": query_id,
                    "question": _query_text(label, available),
                    "question_focus": label,
                    "object_key": object_key,
                    "requested_aspects": list(available),
                    "gold_doc_ids": gold_doc_ids,
                    "anchor_doc_ids": gold_doc_ids[:1],
                    "umls_cui": cuis,
                }
            )
            references.append(
                {
                    "query_id": query_id,
                    "reference_answer": _reference_answer(relevant_rows),
                    "gold_doc_ids": gold_doc_ids,
                    "requested_aspects": list(available),
                    "umls_cui": cuis,
                }
            )
            for doc_id in gold_doc_ids:
                qrels_lines.append(f"{query_id}\t0\t{doc_id}\t1")

            if len(queries) >= int(max_queries):
                break
        if len(queries) >= int(max_queries):
            break

    out_root = Path(out_dir)
    queries_path = out_root / "graph_eval_queries.jsonl"
    qrels_path = out_root / "graph_qrels.tsv"
    references_path = out_root / "graph_references.jsonl"

    _write_jsonl(queries_path, queries)
    _write_jsonl(references_path, references)
    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    qrels_path.write_text("\n".join(qrels_lines) + "\n", encoding="utf-8")

    return {
        "queries_path": str(queries_path),
        "qrels_path": str(qrels_path),
        "references_path": str(references_path),
        "split_name": split_name,
        "num_queries": len(queries),
        "num_qrels": len(qrels_lines),
        "num_scoped_docs": len(scoped_rows),
        "num_objects": len(grouped),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = build_graph_multidoc_benchmark(
        corpus_path=args.corpus,
        splits_path=args.splits,
        out_dir=args.out_dir,
        split_name=args.split_name,
        max_queries=args.max_queries,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
