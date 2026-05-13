"""Evaluate generated RAG answers with reference-free heuristics."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig

from medical_rag_reranker.evaluation.llm_judge import build_judge_from_cfg
from medical_rag_reranker.evaluation.reference_free import (
    evaluate_generation_result,
    summarize_generation_evaluations,
)
from medical_rag_reranker.inference.generate import (
    generate_from_cfg,
    write_results_jsonl,
)


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "null", "None"):
        return None
    return text


def _write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _truncate_text(text: str, max_chars: int = 220) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def _format_evaluation_scores(evaluation: dict[str, Any]) -> str:
    llm_keys = ("faithfulness", "relevance", "completeness", "safety")
    if all(key in evaluation for key in llm_keys):
        scores = ", ".join(f"{key}={float(evaluation[key]):.1f}" for key in llm_keys)
        verdict = str(evaluation.get("verdict", "")).strip()
        return f"**Scores**: {scores}, verdict={verdict}"

    return (
        f"**Scores**: context_relevance={evaluation['context_relevance']:.3f}, "
        f"groundedness={evaluation['groundedness']:.3f}, "
        f"answer_relevance={evaluation['answer_relevance']:.3f}"
    )


def _write_markdown_report(
    path: Path,
    summary: dict[str, float],
    rows: list[dict[str, Any]],
) -> None:
    """Write a compact markdown report for generation evaluation runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generation Evaluation",
        "",
        "## Summary",
        "",
    ]

    for key in sorted(summary):
        value = summary[key]
        if key == "num_examples":
            lines.append(f"- `{key}`: {int(value)}")
        else:
            lines.append(f"- `{key}`: {value:.4f}")

    lines.append("")
    lines.append("## Examples")
    lines.append("")

    for idx, row in enumerate(rows, start=1):
        qid = row.get("query_id", f"query-{idx}")
        evaluation = row["evaluation"]
        lines.append(f"### Example {idx} (`{qid}`)")
        lines.append("")
        lines.append(f"**Question**: {row['question']}")
        lines.append("")
        lines.append(_format_evaluation_scores(evaluation))
        rationale = str(evaluation.get("rationale", "")).strip()
        if rationale:
            lines.append("")
            lines.append(f"**Judge rationale**: {rationale}")
        lines.append("")
        lines.append("**Top docs**:")
        for rank, doc in enumerate(row.get("retrieved", []), start=1):
            lines.append(
                f"{rank}. `{doc['doc_id']}` "
                f"(score={float(doc['score']):.4f}) "
                f"- {_truncate_text(str(doc.get('text', '')))}"
            )
        lines.append("")
        lines.append("**Answer**:")
        lines.append("")
        lines.append(str(row.get("answer", "")))
        lines.append("")
        supported = ", ".join(
            f"`{item}`" for item in row.get("supported_citations_detected", [])
        )
        unsupported = ", ".join(
            f"`{item}`" for item in row.get("unsupported_citations_detected", [])
        )
        lines.append(f"**Supported citations**: {supported if supported else '_none_'}")
        lines.append(
            f"**Unsupported citations**: {unsupported if unsupported else '_none_'}"
        )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


REFERENCE_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _tokenize_reference_text(text: str) -> list[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in normalized.split() if token]


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    cand_tokens = _tokenize_reference_text(candidate)
    ref_tokens = _tokenize_reference_text(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(cand_tokens, ref_tokens)
    precision = lcs / float(len(cand_tokens))
    recall = lcs / float(len(ref_tokens))
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def lexical_cosine(candidate: str, reference: str) -> float:
    cand = Counter(_tokenize_reference_text(candidate))
    ref = Counter(_tokenize_reference_text(reference))
    if not cand or not ref:
        return 0.0
    shared = set(cand) & set(ref)
    dot = sum(cand[t] * ref[t] for t in shared)
    cand_norm = math.sqrt(sum(v * v for v in cand.values()))
    ref_norm = math.sqrt(sum(v * v for v in ref.values()))
    return dot / float(cand_norm * ref_norm + 1e-12)


def _retrieved_doc_ids(row: dict[str, Any]) -> list[str]:
    return [
        str(item.get("doc_id"))
        for item in row.get("retrieved", [])
        if item.get("doc_id")
    ]


def _gold_doc_ids(row: dict[str, Any]) -> set[str]:
    values = row.get("gold_doc_ids") or []
    if isinstance(values, str):
        return {values}
    return {str(v) for v in values if v}


def _detect_reference_citations(answer: str) -> list[str]:
    found = REFERENCE_CITATION_PATTERN.findall(answer)
    unique: list[str] = []
    seen: set[str] = set()
    for item in found:
        key = item.strip()
        if key and key not in seen:
            unique.append(key)
            seen.add(key)
    return unique


def evaluate_generation_reference(
    results_path: str,
    references_path: str,
    out_path: str | None = None,
) -> dict[str, float]:
    """Evaluate generated answers against graph benchmark references."""
    results = {str(row.get("query_id")): row for row in _read_jsonl(Path(results_path))}
    references = {
        str(row.get("query_id")): row for row in _read_jsonl(Path(references_path))
    }

    rouge_scores: list[float] = []
    semantic_scores: list[float] = []
    citation_precision_scores: list[float] = []
    citation_support_scores: list[float] = []
    context_recall_scores: list[float] = []

    for query_id, reference_row in references.items():
        result = results.get(query_id)
        if not result:
            continue

        answer = str(result.get("answer") or "")
        reference = str(reference_row.get("reference_answer") or "")
        gold_docs = _gold_doc_ids(reference_row)
        retrieved_docs = set(_retrieved_doc_ids(result))
        citations = set(_detect_reference_citations(answer))

        rouge_scores.append(rouge_l_f1(answer, reference))
        semantic_scores.append(lexical_cosine(answer, reference))

        if citations:
            citation_precision_scores.append(
                len(citations & gold_docs) / float(max(1, len(citations)))
            )
            citation_support_scores.append(
                len(citations & retrieved_docs) / float(max(1, len(citations)))
            )
        if gold_docs:
            context_recall_scores.append(
                len(retrieved_docs & gold_docs) / float(max(1, len(gold_docs)))
            )

    def mean_or_zero(values: list[float]) -> float:
        return sum(values) / float(max(1, len(values)))

    metrics = {
        "num_generation_eval": float(len(rouge_scores)),
        "rouge_l_f1": mean_or_zero(rouge_scores),
        "semantic_similarity_lexical": mean_or_zero(semantic_scores),
        "citation_precision": mean_or_zero(citation_precision_scores),
        "citation_support": mean_or_zero(citation_support_scores),
        "context_recall": mean_or_zero(context_recall_scores),
    }

    if out_path:
        _write_summary_json(Path(out_path), metrics)
    return metrics


def run_eval_generation(cfg: DictConfig) -> dict[str, float]:
    """Generate answers for an eval set, score them, and log artifacts to MLflow."""
    run_cfg = cfg.run.eval_generation
    judge_mode = str(run_cfg.judge_mode).strip().lower()
    if judge_mode not in {"heuristic", "llm"}:
        raise ValueError("run.eval_generation.judge_mode must be `heuristic` or `llm`.")

    cfg.generation.mode = "batch"
    cfg.generation.examples_limit = int(run_cfg.examples_limit)
    cfg.generation.report_path = str(run_cfg.generation_report_path)
    cfg.generation.results_jsonl_path = str(run_cfg.generation_results_jsonl_path)

    results = generate_from_cfg(
        cfg=cfg,
        question=None,
        queries_path=_as_optional_str(run_cfg.queries_path),
        output_path=str(run_cfg.generation_report_path),
    )
    if not isinstance(results, list):
        raise RuntimeError("Generation evaluation expects batch output.")

    judge = build_judge_from_cfg(run_cfg) if judge_mode == "llm" else None
    enriched_results: list[dict[str, Any]] = []
    for row in results:
        enriched = dict(row)
        if judge is None:
            enriched["evaluation"] = evaluate_generation_result(row)
        else:
            enriched["evaluation"] = judge.evaluate(row)
        enriched["judge_mode"] = judge_mode
        enriched_results.append(enriched)

    summary = summarize_generation_evaluations(enriched_results)
    summary["generation_remote_concurrency"] = float(
        int(getattr(cfg.generation, "remote_concurrency", 1))
    )

    output_jsonl = Path(str(run_cfg.output_jsonl))
    summary_json = Path(str(run_cfg.summary_json))
    output_report = Path(str(run_cfg.output_report))

    write_results_jsonl(output_jsonl, enriched_results)
    _write_summary_json(summary_json, summary)
    _write_markdown_report(output_report, summary, enriched_results)

    mlflow_uri = _as_optional_str(run_cfg.mlflow_uri)
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(str(run_cfg.experiment))

    with mlflow.start_run(run_name=str(run_cfg.run_name)):
        run_tag = _as_optional_str(run_cfg.run_tag)
        if run_tag:
            mlflow.set_tag("tag", run_tag)

        params = {
            "queries_path": str(run_cfg.queries_path),
            "examples_limit": int(run_cfg.examples_limit),
            "judge_mode": judge_mode,
            "judge_backend": str(getattr(run_cfg, "judge_backend", "")),
            "judge_model": str(getattr(run_cfg, "judge_model", "")),
            "retriever": str(cfg.retrieval.name),
            "top_k": int(cfg.generation.top_k),
            "retrieve_top_k": int(cfg.generation.retrieve_top_k),
            "llm_model_name": str(cfg.generation.llm_model_name),
            "remote_concurrency": int(getattr(cfg.generation, "remote_concurrency", 1)),
            "local_files_only": bool(
                getattr(cfg.generation, "local_files_only", False)
            ),
            "use_reranker": bool(getattr(cfg.generation, "use_reranker", False)),
        }
        mlflow.log_params(params)
        for key, value in summary.items():
            mlflow.log_metric(key, float(value))

        mlflow.log_artifact(str(Path(str(run_cfg.generation_report_path))))
        mlflow.log_artifact(str(Path(str(run_cfg.generation_results_jsonl_path))))
        mlflow.log_artifact(str(output_jsonl))
        mlflow.log_artifact(str(summary_json))
        mlflow.log_artifact(str(output_report))

    return summary
