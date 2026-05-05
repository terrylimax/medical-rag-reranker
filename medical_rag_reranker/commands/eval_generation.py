"""Evaluate generated RAG answers with reference-free heuristics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig

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
        lines.append(
            f"**Scores**: context_relevance={evaluation['context_relevance']:.3f}, "
            f"groundedness={evaluation['groundedness']:.3f}, "
            f"answer_relevance={evaluation['answer_relevance']:.3f}"
        )
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


def run_eval_generation(cfg: DictConfig) -> dict[str, float]:
    """Generate answers for an eval set, score them, and log artifacts to MLflow."""
    run_cfg = cfg.run.eval_generation
    judge_mode = str(run_cfg.judge_mode).strip().lower()
    if judge_mode != "heuristic":
        raise NotImplementedError(
            "Only judge_mode=heuristic is implemented in this repository version."
        )

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

    enriched_results: list[dict[str, Any]] = []
    for row in results:
        enriched = dict(row)
        enriched["evaluation"] = evaluate_generation_result(row)
        enriched["judge_mode"] = judge_mode
        enriched_results.append(enriched)

    summary = summarize_generation_evaluations(enriched_results)

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
            "retriever": str(cfg.retrieval.name),
            "top_k": int(cfg.generation.top_k),
            "retrieve_top_k": int(cfg.generation.retrieve_top_k),
            "llm_model_name": str(cfg.generation.llm_model_name),
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
