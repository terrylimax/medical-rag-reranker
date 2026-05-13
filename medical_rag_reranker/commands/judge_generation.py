from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from medical_rag_reranker.evaluation.llm_judge import build_judge_from_cfg
from medical_rag_reranker.evaluation.reference_free import (
    summarize_generation_evaluations,
)
from medical_rag_reranker.utils.progress import progress


SUMMARY_COLUMNS = (
    "run_name",
    "source_raw_jsonl",
    "output_jsonl",
    "num_examples",
    "pass_rate",
    "fail_rate",
    "avg_faithfulness",
    "avg_relevance",
    "avg_completeness",
    "avg_safety",
)


def _read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})


def _run_name_from_raw_path(path: Path) -> str:
    name = path.name
    if name.endswith(".raw.jsonl"):
        return name[: -len(".raw.jsonl")]
    return path.stem


def _write_markdown_report(
    path: Path,
    *,
    run_name: str,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        f"# LLM-as-a-Judge Sample: {run_name}",
        "",
        f"- examples: {int(summary.get('num_examples', len(rows)))}",
        f"- pass_rate: {summary.get('pass_rate', '')}",
        f"- avg_faithfulness: {summary.get('avg_faithfulness', '')}",
        f"- avg_relevance: {summary.get('avg_relevance', '')}",
        f"- avg_completeness: {summary.get('avg_completeness', '')}",
        f"- avg_safety: {summary.get('avg_safety', '')}",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        evaluation = row.get("evaluation", {})
        lines.extend(
            [
                f"## Example {idx}",
                "",
                f"**Query ID:** `{row.get('query_id', '')}`",
                "",
                "**Question**",
                "",
                str(row.get("question", "")),
                "",
                "**Answer**",
                "",
                str(row.get("answer", "")),
                "",
                "**Judge**",
                "",
                "```json",
                json.dumps(evaluation, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _sample_key(row: dict[str, Any], sample_idx: int | None = None) -> str:
    query_id = row.get("query_id")
    if query_id is not None and str(query_id).strip():
        return f"query_id:{query_id}"
    stored_idx = row.get("sample_idx")
    if stored_idx is not None:
        return f"sample_idx:{stored_idx}"
    return f"sample_idx:{sample_idx}"


def _read_existing_judged_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    existing: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(_read_jsonl(path)):
        if "evaluation" not in row:
            continue
        existing[_sample_key(row, idx)] = row
    return existing


def _enrich_row(
    *,
    row: dict[str, Any],
    sample_idx: int,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(row)
    enriched["sample_idx"] = sample_idx
    enriched["evaluation"] = evaluation
    enriched["judge_mode"] = "llm"
    return enriched


def _judge_rows(
    *,
    judge,
    rows: list[tuple[int, dict[str, Any]]],
    concurrency: int,
    desc: str,
    append_path: Path | None = None,
) -> list[dict[str, Any]]:
    if concurrency <= 1:
        judged: list[dict[str, Any]] = []
        for sample_idx, row in progress(
            rows, desc=desc, total=len(rows), unit="example"
        ):
            enriched = _enrich_row(
                row=row,
                sample_idx=sample_idx,
                evaluation=judge.evaluate(row),
            )
            judged.append(enriched)
            if append_path is not None:
                _append_jsonl(append_path, enriched)
        return judged

    judged_by_idx: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(judge.evaluate, row): (sample_idx, row)
            for sample_idx, row in rows
        }
        for future in progress(
            as_completed(futures),
            desc=desc,
            total=len(futures),
            unit="example",
        ):
            sample_idx, row = futures[future]
            enriched = _enrich_row(
                row=row,
                sample_idx=sample_idx,
                evaluation=future.result(),
            )
            judged_by_idx[sample_idx] = enriched
            if append_path is not None:
                _append_jsonl(append_path, enriched)
    return [judged_by_idx[idx] for idx in sorted(judged_by_idx)]


def run_judge_generation_sample(
    cfg: DictConfig,
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    examples_limit: int = 5,
    pattern: str = "*.raw.jsonl",
    max_files: int | None = None,
    concurrency: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """Run LLM-as-a-Judge on existing generation raw JSONL files."""
    source_dir = Path(input_dir)
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Generation input directory does not exist: {source_dir}"
        )

    raw_paths = sorted(source_dir.glob(pattern))
    if max_files is not None:
        raw_paths = raw_paths[: max(0, int(max_files))]
    if not raw_paths:
        raise FileNotFoundError(
            f"No generation raw JSONL files matched: {source_dir}/{pattern}"
        )

    limit = max(1, int(examples_limit))
    effective_concurrency = max(1, int(concurrency))
    judge = build_judge_from_cfg(cfg.run.eval_generation)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for raw_path in raw_paths:
        run_name = _run_name_from_raw_path(raw_path)
        output_jsonl = target_dir / f"{run_name}.llm_judge_n{limit}.jsonl"
        summary_json = target_dir / f"{run_name}.llm_judge_n{limit}.summary.json"
        output_report = target_dir / f"{run_name}.llm_judge_n{limit}.md"

        if resume and output_jsonl.exists() and summary_json.exists():
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
        else:
            sample_rows = _read_jsonl(raw_path, limit=limit)
            existing_rows = _read_existing_judged_rows(output_jsonl) if resume else {}
            if not resume and output_jsonl.exists():
                output_jsonl.write_text("", encoding="utf-8")
            pending_rows = [
                (idx, row)
                for idx, row in enumerate(sample_rows)
                if _sample_key(row, idx) not in existing_rows
            ]
            judged_rows = _judge_rows(
                judge=judge,
                rows=pending_rows,
                concurrency=effective_concurrency,
                desc=f"LLM judge {run_name}",
                append_path=output_jsonl,
            )
            judged_by_key = {
                **existing_rows,
                **{_sample_key(row): row for row in judged_rows},
            }
            judged_rows = [
                judged_by_key[_sample_key(row, idx)]
                for idx, row in enumerate(sample_rows)
                if _sample_key(row, idx) in judged_by_key
            ]
            summary = summarize_generation_evaluations(judged_rows)
            summary.update(
                {
                    "run_name": run_name,
                    "source_raw_jsonl": str(raw_path),
                    "output_jsonl": str(output_jsonl),
                    "sample_limit": limit,
                }
            )
            _write_jsonl(output_jsonl, judged_rows)
            _write_json(summary_json, summary)
            _write_markdown_report(
                output_report,
                run_name=run_name,
                summary=summary,
                rows=judged_rows,
            )

        summary_rows.append(
            {
                "run_name": run_name,
                "source_raw_jsonl": str(raw_path),
                "output_jsonl": str(output_jsonl),
                **summary,
            }
        )

    aggregate_csv = target_dir / f"llm_judge_sample_n{limit}_summary.csv"
    aggregate_jsonl = target_dir / f"llm_judge_sample_n{limit}_summary.jsonl"
    _write_summary_csv(aggregate_csv, summary_rows)
    _write_jsonl(aggregate_jsonl, summary_rows)
    return {
        "input_dir": str(source_dir),
        "output_dir": str(target_dir),
        "pattern": pattern,
        "examples_limit": limit,
        "num_runs": len(summary_rows),
        "summary_csv": str(aggregate_csv),
        "summary_jsonl": str(aggregate_jsonl),
    }
