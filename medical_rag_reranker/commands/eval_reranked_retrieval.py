"""Compare baseline retrieval against the same run after cross-encoder reranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import mlflow
from omegaconf import DictConfig

from medical_rag_reranker.commands.eval_retrieval import (
    evaluate_with_pytrec_eval,
    read_qrels_tsv,
    read_run_trec,
)
from medical_rag_reranker.inference.rerank import (
    CandidateDoc,
    CrossEncoderBatchReranker,
)


def _as_optional_str(value: object) -> str | None:
    """Normalize optional config-like values to either a trimmed string or None."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "null", "None"):
        return None
    return text


def _parse_ks(ks: str) -> list[int]:
    """Parse a comma-separated list of evaluation cutoffs such as `5,10`."""
    values = [int(x.strip()) for x in str(ks).split(",") if x.strip()]
    if not values:
        raise ValueError("ks is empty")
    return values


def _resolve_query_text(row: dict[str, Any]) -> str:
    """Extract the query text from a JSONL row using supported field names."""
    text = row.get("text")
    if text:
        return str(text)

    question = row.get("question")
    if question:
        return str(question)

    raise ValueError("Query row must contain either `text` or `question`.")


def _resolve_query_id(row: dict[str, Any], fallback_idx: int) -> str:
    """Extract a stable query identifier from a JSONL row or synthesize one."""
    qid = row.get("query_id")
    if qid is None:
        qid = row.get("question_id")
    if qid is None:
        qid = f"query-{fallback_idx}"
    return str(qid)


def _load_queries(path: Path) -> dict[str, str]:
    """Load evaluation queries from JSONL into a `query_id -> query_text` mapping."""
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            queries[_resolve_query_id(row, idx)] = _resolve_query_text(row)
    return queries


def _load_docstore(path: Path) -> dict[str, dict[str, Any]]:
    """Load the corpus JSONL into a `doc_id -> document payload` mapping."""
    docstore: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get("doc_id")
            if doc_id:
                docstore[str(doc_id)] = row
    return docstore


def _truncate_text(text: str, max_chars: int = 220) -> str:
    """Collapse whitespace and shorten long text for human-readable reports."""
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def _write_run_trec(
    path: Path,
    results: dict[str, list[tuple[str, float]]],
    run_name: str,
) -> None:
    """Write ranked results to a TREC run file for downstream evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid, docs in results.items():
            for rank, (doc_id, score) in enumerate(docs, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to JSONL, one serialized row per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _serialize_ranked_doc(
    *,
    doc_id: str,
    score: float,
    docstore: dict[str, dict[str, Any]],
    retrieval_score: float | None = None,
    reranker_score: float | None = None,
) -> dict[str, Any]:
    """Build a report-friendly document record from ranked output and corpus data."""
    doc = docstore.get(doc_id, {})
    row: dict[str, Any] = {
        "doc_id": doc_id,
        "score": float(score),
        "text": str(doc.get("text", "")),
        "source": str(doc["source"]) if "source" in doc else None,
    }
    if retrieval_score is not None:
        row["retrieval_score"] = float(retrieval_score)
    if reranker_score is not None:
        row["reranker_score"] = float(reranker_score)
    return row


def _first_relevant_rank(
    ranked_doc_ids: list[str],
    relevant_doc_ids: set[str],
) -> int | None:
    """Return the 1-based rank of the first relevant document, if any."""
    for idx, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return idx
    return None


def _rank_value(rank: int | None) -> int:
    """Convert a possibly missing rank into a sortable numeric sentinel value."""
    return rank if rank is not None else 10**9


def _build_comparison_examples(
    *,
    queries: dict[str, str],
    docstore: dict[str, dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    baseline_run: dict[str, dict[str, float]],
    reranked_run: dict[str, dict[str, float]],
    examples_limit: int,
    top_docs_to_report: int,
) -> list[dict[str, Any]]:
    """Assemble per-query before/after rerank examples for reports and JSONL output."""
    rows: list[dict[str, Any]] = []
    for qid, query_text in queries.items():
        baseline_scores = baseline_run.get(qid)
        reranked_scores = reranked_run.get(qid)
        if not baseline_scores or not reranked_scores:
            continue

        relevant_doc_ids = [
            doc_id for doc_id, rel in qrels.get(qid, {}).items() if int(rel) > 0
        ]
        relevant_set = set(relevant_doc_ids)

        baseline_sorted = sorted(
            baseline_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        reranked_sorted = sorted(
            reranked_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        first_before = _first_relevant_rank(
            [doc_id for doc_id, _score in baseline_sorted],
            relevant_set,
        )
        first_after = _first_relevant_rank(
            [doc_id for doc_id, _score in reranked_sorted],
            relevant_set,
        )

        rows.append(
            {
                "query_id": qid,
                "question": query_text,
                "relevant_doc_ids": relevant_doc_ids,
                "first_relevant_rank_before": first_before,
                "first_relevant_rank_after": first_after,
                "rank_delta": float(
                    _rank_value(first_before) - _rank_value(first_after)
                ),
                "baseline_top_docs": [
                    _serialize_ranked_doc(
                        doc_id=doc_id,
                        score=score,
                        docstore=docstore,
                    )
                    for doc_id, score in baseline_sorted[:top_docs_to_report]
                ],
                "reranked_top_docs": [
                    _serialize_ranked_doc(
                        doc_id=doc_id,
                        score=reranker_score,
                        docstore=docstore,
                        retrieval_score=float(
                            baseline_scores.get(doc_id, reranker_score)
                        ),
                        reranker_score=reranker_score,
                    )
                    for doc_id, reranker_score in reranked_sorted[:top_docs_to_report]
                ],
            }
        )

    rows.sort(
        key=lambda item: (
            float(item["rank_delta"]),
            -float(_rank_value(item["first_relevant_rank_before"])),
            item["query_id"],
        ),
        reverse=True,
    )
    return rows[:examples_limit]


def _write_comparison_report(
    *,
    path: Path,
    retriever_name: str,
    rerank_top_n: int,
    top_docs_to_report: int,
    comparison: dict[str, float],
    examples: list[dict[str, Any]],
) -> None:
    """Render a markdown summary of metric deltas and representative rerank examples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Retrieval Rerank Comparison",
        "",
        f"- retriever: `{retriever_name}`",
        f"- rerank_top_n: `{rerank_top_n}`",
        f"- top_docs_to_report: `{top_docs_to_report}`",
        f"- examples: `{len(examples)}`",
        "",
        "## Metrics",
        "",
    ]

    metric_names = [
        name[len("baseline_") :] for name in comparison if name.startswith("baseline_")
    ]
    for metric_name in sorted(metric_names):
        baseline = comparison.get(f"baseline_{metric_name}")
        reranked = comparison.get(f"reranked_{metric_name}")
        delta = comparison.get(f"delta_{metric_name}")
        if baseline is None or reranked is None:
            continue
        lines.append(
            f"- `{metric_name}`: baseline={baseline:.4f}, "
            f"reranked={reranked:.4f}, delta={delta:.4f}"
        )
    if "rerank_latency_ms_avg" in comparison:
        lines.append(
            f"- `rerank_latency_ms_avg`: {comparison['rerank_latency_ms_avg']:.2f}"
        )
    if "rerank_latency_ms_total" in comparison:
        lines.append(
            f"- `rerank_latency_ms_total`: {comparison['rerank_latency_ms_total']:.2f}"
        )
    lines.append("")

    for idx, row in enumerate(examples, start=1):
        relevant = ", ".join(f"`{doc_id}`" for doc_id in row["relevant_doc_ids"])
        lines.append(f"## Example {idx} (`{row['query_id']}`)")
        lines.append("")
        lines.append(f"**Question**: {row['question']}")
        lines.append("")
        lines.append(f"**Relevant doc_ids**: {relevant if relevant else '_none_'}")
        before_rank = row["first_relevant_rank_before"]
        after_rank = row["first_relevant_rank_after"]
        lines.append(
            "**First relevant rank**: "
            f"baseline={before_rank if before_rank is not None else 'not found'}, "
            f"reranked={after_rank if after_rank is not None else 'not found'}"
        )
        lines.append("")
        lines.append("**Top docs before rerank**:")
        for rank, doc in enumerate(row["baseline_top_docs"], start=1):
            suffix = " [REL]" if doc["doc_id"] in row["relevant_doc_ids"] else ""
            lines.append(
                f"{rank}. `{doc['doc_id']}`{suffix} "
                f"(score={doc['score']:.4f}) "
                f"- {_truncate_text(doc['text'])}"
            )
        lines.append("")
        lines.append("**Top docs after rerank**:")
        for rank, doc in enumerate(row["reranked_top_docs"], start=1):
            suffix = " [REL]" if doc["doc_id"] in row["relevant_doc_ids"] else ""
            retrieval_score = doc.get("retrieval_score", doc["score"])
            reranker_score = doc.get("reranker_score", doc["score"])
            lines.append(
                f"{rank}. `{doc['doc_id']}`{suffix} "
                f"(retrieval_score={retrieval_score:.4f}, "
                f"reranker_score={reranker_score:.4f}) "
                f"- {_truncate_text(doc['text'])}"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_reranked_run(
    *,
    queries: dict[str, str],
    docstore: dict[str, dict[str, Any]],
    baseline_run: dict[str, dict[str, float]],
    reranker: CrossEncoderBatchReranker,
    rerank_top_n: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Rerank each query's top candidates and collect aggregate latency statistics."""
    reranked_run: dict[str, dict[str, float]] = {}
    total_latency_ms = 0.0
    reranked_queries = 0

    for qid, scored_docs in baseline_run.items():
        query_text = queries.get(qid)
        if query_text is None:
            continue

        sorted_candidates = sorted(
            scored_docs.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: int(rerank_top_n)]

        candidates: list[CandidateDoc] = []
        for doc_id, retrieval_score in sorted_candidates:
            doc = docstore.get(doc_id, {})
            text = str(doc.get("text", ""))
            if not text:
                continue
            candidates.append(
                CandidateDoc(
                    doc_id=doc_id,
                    text=text,
                    retrieval_score=float(retrieval_score),
                    source=str(doc["source"]) if "source" in doc else None,
                )
            )

        reranked_docs, latency_ms = reranker.rerank(
            question=query_text,
            candidates=candidates,
            top_k=None,
        )
        reranked_run[qid] = {
            doc.doc_id: float(doc.reranker_score) for doc in reranked_docs
        }
        reranked_queries += 1
        total_latency_ms += float(latency_ms)

    stats = {
        "reranked_queries": float(reranked_queries),
        "rerank_latency_ms_total": float(total_latency_ms),
        "rerank_latency_ms_avg": (
            float(total_latency_ms) / float(reranked_queries)
            if reranked_queries
            else 0.0
        ),
    }
    return reranked_run, stats


def run_from_cfg(cfg: DictConfig) -> Dict[str, float]:
    """Execute the full reranked-retrieval evaluation workflow from Hydra config."""
    run_cfg = cfg.run.eval_reranked_retrieval
    baseline_run_path_text = _as_optional_str(run_cfg.run_path)

    if baseline_run_path_text is None:
        from medical_rag_reranker.commands.retrieval_run import (
            run_from_cfg as retrieval_run_from_cfg,
        )

        baseline_run_path = retrieval_run_from_cfg(cfg)
    else:
        baseline_run_path = Path(baseline_run_path_text)

    reranker_checkpoint_path = _as_optional_str(run_cfg.reranker_checkpoint_path)
    if reranker_checkpoint_path is None:
        raise RuntimeError(
            "`run.eval_reranked_retrieval.reranker_checkpoint_path` must be set."
        )

    ks = _parse_ks(str(run_cfg.ks))
    eval_queries_path = Path(str(run_cfg.eval_queries))
    qrels_path = Path(str(run_cfg.qrels))
    corpus_path = Path(str(run_cfg.corpus_path))
    out_run_path = Path(str(run_cfg.out_run))

    queries = _load_queries(eval_queries_path)
    docstore = _load_docstore(corpus_path)
    qrels = read_qrels_tsv(qrels_path)
    baseline_run = read_run_trec(Path(baseline_run_path))
    baseline_metrics = evaluate_with_pytrec_eval(qrels=qrels, run=baseline_run, ks=ks)

    reranker = CrossEncoderBatchReranker.from_cfg(
        cfg=cfg,
        checkpoint_path=reranker_checkpoint_path,
        batch_size=int(run_cfg.reranker_batch_size),
    )
    reranked_run, rerank_stats = build_reranked_run(
        queries=queries,
        docstore=docstore,
        baseline_run=baseline_run,
        reranker=reranker,
        rerank_top_n=int(run_cfg.rerank_top_n),
    )

    ordered_results = {
        qid: sorted(scored.items(), key=lambda item: item[1], reverse=True)
        for qid, scored in reranked_run.items()
    }
    _write_run_trec(
        path=out_run_path,
        results=ordered_results,
        run_name=str(run_cfg.run_name),
    )

    reranked_metrics = evaluate_with_pytrec_eval(qrels=qrels, run=reranked_run, ks=ks)
    comparison_examples = _build_comparison_examples(
        queries=queries,
        docstore=docstore,
        qrels=qrels,
        baseline_run=baseline_run,
        reranked_run=reranked_run,
        examples_limit=int(run_cfg.examples_limit),
        top_docs_to_report=int(run_cfg.top_docs_to_report),
    )

    comparison: Dict[str, float] = {}
    for key, value in baseline_metrics.items():
        comparison[f"baseline_{key}"] = float(value)
    for key, value in reranked_metrics.items():
        comparison[f"reranked_{key}"] = float(value)
    for key, value in rerank_stats.items():
        comparison[key] = float(value)
    for key, value in reranked_metrics.items():
        baseline_value = baseline_metrics.get(key)
        if baseline_value is not None:
            comparison[f"delta_{key}"] = float(value) - float(baseline_value)

    comparison_path = out_run_path.with_suffix(out_run_path.suffix + ".comparison.json")
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    comparison_report_path = Path(str(run_cfg.comparison_report_path))
    _write_comparison_report(
        path=comparison_report_path,
        retriever_name=_as_optional_str(run_cfg.retriever) or str(cfg.retrieval.name),
        rerank_top_n=int(run_cfg.rerank_top_n),
        top_docs_to_report=int(run_cfg.top_docs_to_report),
        comparison=comparison,
        examples=comparison_examples,
    )
    comparison_jsonl_path = Path(str(run_cfg.comparison_jsonl_path))
    _write_jsonl(comparison_jsonl_path, comparison_examples)

    mlflow_uri = _as_optional_str(run_cfg.mlflow_uri)
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(str(run_cfg.experiment))

    with mlflow.start_run(run_name=str(run_cfg.run_name)):
        run_tag = _as_optional_str(run_cfg.run_tag)
        if run_tag:
            mlflow.set_tag("tag", run_tag)

        params = {
            "retriever": _as_optional_str(run_cfg.retriever) or str(cfg.retrieval.name),
            "eval_queries": str(eval_queries_path),
            "qrels": str(qrels_path),
            "corpus_path": str(corpus_path),
            "baseline_run_path": str(baseline_run_path),
            "reranked_run_path": str(out_run_path),
            "reranker_checkpoint_path": reranker_checkpoint_path,
            "rerank_top_n": int(run_cfg.rerank_top_n),
            "reranker_batch_size": int(run_cfg.reranker_batch_size),
            "model_name": str(cfg.model.model_name),
            "ks": ",".join(map(str, ks)),
        }
        mlflow.log_params(params)

        for metric_name, metric_value in comparison.items():
            mlflow.log_metric(metric_name, float(metric_value))

        mlflow.log_artifact(str(baseline_run_path))
        mlflow.log_artifact(str(out_run_path))
        mlflow.log_artifact(str(comparison_path))
        mlflow.log_artifact(str(comparison_report_path))
        mlflow.log_artifact(str(comparison_jsonl_path))

    return comparison
