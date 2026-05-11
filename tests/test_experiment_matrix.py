import csv
import json
from pathlib import Path

from medical_rag_reranker.experiments.matrix import (
    PRIMARY_RETRIEVAL_METRICS,
    _build_jobs,
    _build_summary,
    _stage_set,
    run_experiment_matrix,
)
from medical_rag_reranker.utils.hydra_cfg import load_cfg


def test_experiment_matrix_smoke_dry_run_writes_plan(tmp_path: Path) -> None:
    cfg = load_cfg(
        overrides=[
            f"run.experiment_matrix.output_root={tmp_path}",
            "run.experiment_matrix.run_id=test_run",
        ]
    )

    result = run_experiment_matrix(
        cfg,
        profile="smoke",
        stage="all",
        run_id="test_run",
        dry_run=True,
        training_mode="colab_artifacts",
    )

    output_dir = tmp_path / "test_run"
    planned_jobs = output_dir / "planned_jobs.jsonl"
    manifest = json.loads((output_dir / "manifest.json").read_text())

    assert result["dry_run"] is True
    assert planned_jobs.exists()
    assert "P@10" not in manifest["primary_retrieval_metrics"]
    assert "R@10" not in manifest["primary_retrieval_metrics"]
    assert set(PRIMARY_RETRIEVAL_METRICS).issubset(
        set(manifest["primary_retrieval_metrics"])
    )


def test_experiment_matrix_ks_override_is_hydra_safe(tmp_path: Path) -> None:
    cfg = load_cfg(
        overrides=[
            f"run.experiment_matrix.output_root={tmp_path}",
            "run.experiment_matrix.run_id=test_run",
        ]
    )
    jobs = _build_jobs(
        cfg,
        profile="smoke",
        output_dir=tmp_path / "test_run",
        stages=_stage_set("index"),
    )

    index_cfg = load_cfg(overrides=list(jobs[0].overrides))

    assert list(index_cfg.run.eval_retrieval.ks) == [1, 3, 5, 10]


def test_experiment_manifest_masks_secret_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", "secret-value")
    monkeypatch.setenv("VLLM_API_KEY", "secret-value")

    cfg = load_cfg(
        overrides=[
            f"run.experiment_matrix.output_root={tmp_path}",
            "run.experiment_matrix.run_id=secret_safe",
        ]
    )
    run_experiment_matrix(
        cfg,
        profile="smoke",
        stage="summary",
        run_id="secret_safe",
        dry_run=True,
    )

    manifest = json.loads((tmp_path / "secret_safe/manifest.json").read_text())

    assert manifest["env"]["QDRANT_API_KEY"] is True
    assert manifest["env"]["VLLM_API_KEY"] is True
    assert "secret-value" not in json.dumps(manifest)


def test_e2e_summary_uses_primary_metrics_and_generation_latencies(
    tmp_path: Path,
) -> None:
    cfg = load_cfg(
        overrides=[
            f"run.experiment_matrix.output_root={tmp_path}",
            "run.experiment_matrix.run_id=summary_run",
        ]
    )
    output_dir = tmp_path / "summary_run"
    run_path = output_dir / "runs" / "bm25.trec"
    run_path.parent.mkdir(parents=True)
    run_path.write_text("", encoding="utf-8")
    (run_path.with_suffix(run_path.suffix + ".metrics.json")).write_text(
        json.dumps(
            {
                "Hit@1": 0.4,
                "Hit@3": 0.7,
                "Hit@5": 0.8,
                "MRR@10": 0.55,
                "P@10": 0.1,
                "R@10": 0.2,
                "latency_p50_ms": 11.0,
                "latency_p95_ms": 22.0,
            }
        ),
        encoding="utf-8",
    )

    retrieval_status = output_dir / "jobs" / "retrieval" / "retrieval.status.json"
    retrieval_status.parent.mkdir(parents=True)
    retrieval_status.write_text(
        json.dumps(
            {
                "status": "completed",
                "method": "bm25",
                "overrides": [
                    "retrieval=bm25",
                    f"run.eval_retrieval.out_run={run_path}",
                ],
            }
        ),
        encoding="utf-8",
    )

    generation_dir = output_dir / "generation"
    generation_dir.mkdir(parents=True)
    summary_path = generation_dir / "bm25.summary.json"
    output_jsonl = generation_dir / "bm25.jsonl"
    summary_path.write_text(
        json.dumps(
            {
                "pass_rate": 0.9,
                "avg_faithfulness": 4.8,
                "avg_relevance": 5.0,
                "avg_completeness": 4.5,
                "avg_safety": 5.0,
                "avg_supported_citation_rate": 1.0,
                "avg_unsupported_citation_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    output_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "generation_latency_ms": 10.0,
                        "end_to_end_latency_ms": 30.0,
                    }
                ),
                json.dumps(
                    {
                        "generation_latency_ms": 20.0,
                        "end_to_end_latency_ms": 50.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    generation_status = output_dir / "jobs" / "generation" / "generation.status.json"
    generation_status.parent.mkdir(parents=True)
    generation_status.write_text(
        json.dumps(
            {
                "status": "completed",
                "method": "bm25",
                "overrides": [
                    "retrieval=bm25",
                    "generation.top_k=5",
                    "generation.retrieve_top_k=20",
                    "generation.use_reranker=false",
                    f"run.eval_retrieval.out_run={run_path}",
                    f"run.eval_generation.summary_json={summary_path}",
                    f"run.eval_generation.output_jsonl={output_jsonl}",
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = _build_summary(cfg, output_dir=output_dir)

    rows = list(csv.DictReader((output_dir / "e2e_summary.csv").open()))
    assert summary["num_rows"] == 1
    assert rows[0]["hit@1"] == "0.4"
    assert rows[0]["mrr@10"] == "0.55"
    assert rows[0]["answer_pass_rate"] == "0.9"
    assert rows[0]["generation_latency_p50_ms"] == "15.0"
    assert "P@10" not in rows[0]
    assert "R@10" not in rows[0]


def test_practical_remote_keeps_job_count_small_and_outputs_unique(
    tmp_path: Path,
) -> None:
    cfg = load_cfg(
        overrides=[
            f"run.experiment_matrix.output_root={tmp_path}",
            "run.experiment_matrix.run_id=practical_run",
        ]
    )
    output_dir = tmp_path / "practical_run"

    jobs = _build_jobs(
        cfg,
        profile="practical_remote",
        output_dir=output_dir,
        stages=_stage_set("all"),
    )

    assert len(jobs) == 64

    retrieval_paths = [
        next(
            override.split("=", 1)[1]
            for override in job.overrides
            if override.startswith("run.eval_retrieval.out_run=")
        )
        for job in jobs
        if job.stage == "retrieval"
    ]
    generation_outputs = [
        next(
            override.split("=", 1)[1]
            for override in job.overrides
            if override.startswith("run.eval_generation.output_jsonl=")
        )
        for job in jobs
        if job.stage == "generation"
    ]

    assert len(retrieval_paths) == 16
    assert len(set(retrieval_paths)) == 16
    assert len(generation_outputs) == 32
    assert len(set(generation_outputs)) == 32
