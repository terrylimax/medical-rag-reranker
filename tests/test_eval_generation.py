from pathlib import Path

from omegaconf import OmegaConf

from medical_rag_reranker.commands import eval_generation as eval_generation_module


class _DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_run_eval_generation_writes_reports_and_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results = [
        {
            "query_id": "q1",
            "question": "What is metformin used for?",
            "retrieved": [
                {
                    "doc_id": "doc1",
                    "score": 1.0,
                    "text": "Metformin is used for type 2 diabetes.",
                    "source": "TEST",
                }
            ],
            "answer": "Metformin is used for type 2 diabetes [doc1]",
            "citations_detected": ["doc1"],
            "supported_citations_detected": ["doc1"],
            "unsupported_citations_detected": [],
            "reranker_enabled": False,
            "retrieval_latency_ms": 1.0,
            "generation_latency_ms": 2.0,
            "end_to_end_latency_ms": 4.0,
        }
    ]

    monkeypatch.setattr(
        eval_generation_module,
        "generate_from_cfg",
        lambda **kwargs: results,
    )
    monkeypatch.setattr(
        eval_generation_module.mlflow, "set_tracking_uri", lambda *_: None
    )
    monkeypatch.setattr(
        eval_generation_module.mlflow, "set_experiment", lambda *_: None
    )
    monkeypatch.setattr(
        eval_generation_module.mlflow, "start_run", lambda **_: _DummyRun()
    )
    monkeypatch.setattr(eval_generation_module.mlflow, "set_tag", lambda *_: None)
    monkeypatch.setattr(eval_generation_module.mlflow, "log_params", lambda *_: None)
    monkeypatch.setattr(eval_generation_module.mlflow, "log_metric", lambda *_: None)
    monkeypatch.setattr(eval_generation_module.mlflow, "log_artifact", lambda *_: None)

    cfg = OmegaConf.create(
        {
            "retrieval": {"name": "bm25"},
            "generation": {
                "top_k": 5,
                "retrieve_top_k": 20,
                "llm_model_name": "google/flan-t5-small",
                "local_files_only": True,
                "use_reranker": False,
            },
            "run": {
                "eval_generation": {
                    "queries_path": str(tmp_path / "queries.jsonl"),
                    "examples_limit": 1,
                    "judge_mode": "heuristic",
                    "generation_report_path": str(tmp_path / "raw_examples.md"),
                    "generation_results_jsonl_path": str(
                        tmp_path / "raw_examples.jsonl"
                    ),
                    "output_jsonl": str(tmp_path / "eval_generation.jsonl"),
                    "summary_json": str(tmp_path / "eval_generation.summary.json"),
                    "output_report": str(tmp_path / "eval_generation.md"),
                    "experiment": "generation_eval",
                    "run_name": "bm25_generation",
                    "mlflow_uri": None,
                    "run_tag": None,
                }
            },
        }
    )

    summary = eval_generation_module.run_eval_generation(cfg)

    assert summary["num_examples"] == 1.0
    assert (tmp_path / "eval_generation.jsonl").exists()
    assert (tmp_path / "eval_generation.summary.json").exists()
    assert (tmp_path / "eval_generation.md").exists()
