from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import fire
from omegaconf import DictConfig

from medical_rag_reranker.data.dvc_data import ensure_data
from medical_rag_reranker.data.scripts.prepare_data import prepare_data
from medical_rag_reranker.utils.hydra_cfg import load_cfg


def _load_cfg(
    config_dir: Optional[str] = None,
    config_name: str = "config",
    overrides: Optional[str | Sequence[str]] = None,
) -> DictConfig:
    """Backward-compatible wrapper around `medical_rag_reranker.utils.hydra_cfg.load_cfg`."""
    return load_cfg(config_dir=config_dir, config_name=config_name, overrides=overrides)


def main() -> None:
    """
    Fire exposes functions as CLI commands:
    - python -m medical_rag_reranker.commands download_data
    - python -m medical_rag_reranker.commands train
    - python -m medical_rag_reranker.commands infer
    - python -m medical_rag_reranker.commands generate
    - python -m medical_rag_reranker.commands prep_data
    - python -m medical_rag_reranker.commands index
    - python -m medical_rag_reranker.commands eval_retrieval
    - python -m medical_rag_reranker.commands rag_demo
    - python -m medical_rag_reranker.commands submit_job
    - python -m medical_rag_reranker.commands job_status
    - python -m medical_rag_reranker.commands job_result
    - python -m medical_rag_reranker.commands migrate_jobs_schema
    - python -m medical_rag_reranker.commands serve_jobs_api

    Examples:
    - python -m medical_rag_reranker.commands download_data
    - python -m medical_rag_reranker.commands train
    - python -m medical_rag_reranker.commands train --overrides '["train.max_epochs=2","train.batch_size=16"]'
    - python -m medical_rag_reranker.commands infer --query "..." --document "..."
    - python -m medical_rag_reranker.commands generate --question "..."
    - python -m medical_rag_reranker.commands prep_data
    - python -m medical_rag_reranker.commands index --overrides "retrieval=hybrid"
    - python -m medical_rag_reranker.commands eval_retrieval
    - python -m medical_rag_reranker.commands rag_demo --question "..."
    - python -m medical_rag_reranker.commands submit_job --question "..."
    - python -m medical_rag_reranker.commands job_status --job_id "..."
    - python -m medical_rag_reranker.commands job_result --job_id "..."
    - python -m medical_rag_reranker.commands migrate_jobs_schema
    - python -m medical_rag_reranker.commands serve_jobs_api
    """
    fire.Fire(
        {
            "download_data": cmd_download_data,
            "train": cmd_train,
            "infer": cmd_infer,
            "retrieval_run": cmd_retrieval_run,
            "generate": cmd_generate,
            "prep_data": cmd_prep_data,
            "index": cmd_index,
            "eval_retrieval": cmd_eval_retrieval,
            "rag_demo": cmd_rag_demo,
            "submit_job": cmd_submit_job,
            "job_status": cmd_job_status,
            "job_result": cmd_job_result,
            "migrate_jobs_schema": cmd_migrate_jobs_schema,
            "serve_jobs_api": cmd_serve_jobs_api,
        }
    )


def cmd_download_data(
    config_dir: Optional[str] = None, overrides: Optional[str] = None
) -> None:
    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)
    ensure_data(cfg)


def cmd_train(
    config_dir: Optional[str] = None, overrides: Optional[str] = None
) -> None:
    from medical_rag_reranker.training.train import train_from_cfg

    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)

    # Ensure data exists (DVC pull -> fallback download)
    ensure_data(cfg)

    # Run training
    train_from_cfg(cfg)


def cmd_infer(
    query: str,
    document: str,
    checkpoint_path: Optional[str] = None,
    config_dir: Optional[str] = None,
    overrides: Optional[str] = None,
) -> float:
    from medical_rag_reranker.inference.infer import infer_from_cfg

    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)

    # Requirement: integrate DVC data pulling into infer as well
    ensure_data(cfg)

    score = infer_from_cfg(
        cfg, query=query, document=document, checkpoint_path=checkpoint_path
    )
    print(score)
    return score


def cmd_retrieval_run(
    config_dir: Optional[str] = None, overrides: Optional[str] = None
) -> None:
    """Run the configured retriever over queries and write a TREC run file."""
    from medical_rag_reranker.commands.retrieval_run import (
        run_from_cfg as retrieval_run_from_cfg,
    )

    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)

    # Ensure data exists (queries/corpus may be managed via DVC)
    ensure_data(cfg)

    retrieval_run_from_cfg(cfg)


def cmd_generate(
    question: Optional[str] = None,
    queries_path: Optional[str] = None,
    output_path: Optional[str] = None,
    config_dir: Optional[str] = None,
    overrides: Optional[str] = None,
) -> dict | list[dict]:
    """Run retrieval + generation baseline (without reranker)."""
    from medical_rag_reranker.inference.generate import generate_from_cfg

    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)

    # Ensure data exists (queries/corpus may be managed via DVC)
    ensure_data(cfg)

    result = generate_from_cfg(
        cfg=cfg,
        question=question,
        queries_path=queries_path,
        output_path=output_path,
    )

    # User-friendly CLI output.
    if isinstance(result, dict):
        print(result["answer"])
        print(f"citations_detected={result.get('citations_detected', [])}")
    else:
        report_path = output_path or str(cfg.generation.report_path)
        print(f"Generated {len(result)} examples")
        print(f"Report: {report_path}")

    return result


def _build_job_runtime(cfg: DictConfig):
    if not bool(getattr(cfg.jobs, "enabled", True)):
        raise RuntimeError("Jobs subsystem is disabled (set jobs.enabled=true).")

    from medical_rag_reranker.jobs.bootstrap import build_job_runtime

    return build_job_runtime(cfg)


def _as_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off", ""):
        return False
    raise ValueError(f"Cannot parse boolean value from: {value!r}")


def cmd_submit_job(
    question: str,
    config_dir: Optional[str] = None,
    overrides: Optional[str] = None,
) -> dict:
    """Create an inference job and dispatch it according to `jobs.dispatch`."""
    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)
    ensure_data(cfg)
    runtime = _build_job_runtime(cfg)

    job = runtime.service.create_job(
        question=question,
        metadata={"source": "cli.submit_job"},
    )

    if _as_bool(getattr(cfg.jobs, "auto_dispatch", True), default=True):
        runtime.dispatcher.dispatch(job.job_id)

    current = runtime.service.get_job(job.job_id)
    if current is None:
        raise RuntimeError(f"Created job was not found in storage: {job.job_id}")

    payload = current.to_dict()
    payload["result"] = runtime.service.get_result(job.job_id)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def cmd_job_status(
    job_id: str,
    config_dir: Optional[str] = None,
    overrides: Optional[str] = None,
) -> dict:
    """Get persisted status for a previously submitted inference job."""
    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)
    runtime = _build_job_runtime(cfg)
    job = runtime.service.get_job(job_id)
    if job is None:
        raise FileNotFoundError(f"Job not found: {job_id}")

    payload = job.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def cmd_job_result(
    job_id: str,
    config_dir: Optional[str] = None,
    overrides: Optional[str] = None,
) -> dict:
    """Get persisted result payload for a previously submitted job."""
    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)
    runtime = _build_job_runtime(cfg)
    result = runtime.service.get_result(job_id)
    if result is None:
        raise FileNotFoundError(
            f"Result for job `{job_id}` was not found or is not ready yet."
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def cmd_migrate_jobs_schema(
    config_dir: Optional[str] = None,
    overrides: Optional[str] = None,
) -> None:
    """Apply Alembic migrations for Postgres job storage."""
    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)
    from medical_rag_reranker.jobs.bootstrap import migrate_jobs_schema

    migrate_jobs_schema(cfg)
    print("Jobs schema migrated to latest revision.")


def cmd_serve_jobs_api(
    host: Optional[str] = None,
    port: Optional[int] = None,
    config_dir: Optional[str] = None,
    overrides: Optional[str] = None,
) -> None:
    """Run HTTP producer API for submit/status/result job flow."""
    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)
    from medical_rag_reranker.jobs.http_api import serve_jobs_api

    serve_jobs_api(cfg=cfg, host=host, port=port)


def cmd_prep_data(
    config_dir: Optional[str] = None,
    overrides: Optional[str] = None,
) -> dict:
    """Build qa/corpus/splits/eval_queries/qrels artifacts for retrieval baseline."""
    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)
    ensure_data(cfg)

    run_cfg = cfg.run.prep_data
    result = prepare_data(
        raw_nih_path=str(run_cfg.raw_nih_path),
        out_dir=str(run_cfg.out_dir),
        eval_size=int(run_cfg.eval_size),
        seed=int(run_cfg.seed),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def cmd_index(
    config_dir: Optional[str] = None,
    overrides: Optional[str] = None,
) -> str | tuple[str, str, str]:
    """Build retrieval index(es) from configured corpus."""
    from medical_rag_reranker.commands.retrieval_index import (
        run_from_cfg as retrieval_index_from_cfg,
    )

    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)
    ensure_data(cfg)
    result = retrieval_index_from_cfg(cfg)
    if isinstance(result, tuple):
        return tuple(str(p) for p in result)
    return str(result)


def cmd_eval_retrieval(
    config_dir: Optional[str] = None,
    overrides: Optional[str] = None,
) -> dict[str, float]:
    """Evaluate retrieval run(s) against qrels and log to MLflow."""
    from medical_rag_reranker.commands.eval_retrieval import (
        run_from_cfg as eval_retrieval_from_cfg,
    )

    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)
    ensure_data(cfg)
    metrics = eval_retrieval_from_cfg(cfg)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def _write_rag_demo_report(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# RAG Demo", ""]
    for i, item in enumerate(results, start=1):
        qid = item.get("query_id", f"demo-{i}")
        lines.append(f"## Example {i} (`{qid}`)")
        lines.append(f"Question: {item['question']}")
        lines.append("")
        lines.append("Answer:")
        lines.append(str(item["answer"]))
        lines.append("")
        citations = ", ".join(f"`{c}`" for c in item.get("citations_detected", []))
        lines.append(f"Citations: {citations if citations else '_none_'}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_rag_demo(
    question: Optional[str] = None,
    queries_path: Optional[str] = None,
    num_questions: Optional[int] = None,
    output_report: Optional[str] = None,
    config_dir: Optional[str] = None,
    overrides: Optional[str] = None,
) -> dict | list[dict]:
    """Run a quick end-to-end retrieval+generation demo (1..5 questions)."""
    from medical_rag_reranker.inference.generate import generate_from_cfg

    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)
    ensure_data(cfg)

    run_cfg = cfg.run.rag_demo
    report_path = Path(output_report or str(run_cfg.output_report))

    if question:
        result = generate_from_cfg(
            cfg=cfg,
            question=question,
            queries_path=None,
            output_path=None,
        )
        _write_rag_demo_report(report_path, [result])
        print(result["answer"])
        print(f"citations_detected={result.get('citations_detected', [])}")
        print(f"report={report_path}")
        return result

    n = int(num_questions if num_questions is not None else run_cfg.num_questions)
    n = max(1, min(5, n))
    cfg.generation.mode = "batch"
    cfg.generation.examples_limit = n
    cfg.generation.report_path = str(report_path)

    result = generate_from_cfg(
        cfg=cfg,
        question=None,
        queries_path=queries_path or str(run_cfg.queries_path),
        output_path=str(report_path),
    )
    assert isinstance(result, list)
    _write_rag_demo_report(report_path, result)
    print(f"Generated {len(result)} demo answers")
    print(f"report={report_path}")
    return result


if __name__ == "__main__":
    main()
