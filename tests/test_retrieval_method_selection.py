from pathlib import Path

from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from medical_rag_reranker.jobs.http_api import create_jobs_api
from medical_rag_reranker.jobs.repositories.file import (
    FileJobRepository,
    FileResultRepository,
)
from medical_rag_reranker.jobs.service import (
    InferenceJobService,
    retrieval_method_options,
    _with_retrieval_method,
)
from medical_rag_reranker.retrieval import loading


def _base_cfg(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    bm25_index = artifacts_dir / "bm25_index.json.gz"
    return OmegaConf.create(
        {
            "paths": {"artifacts_dir": str(artifacts_dir)},
            "retrieval": {"name": "bm25", "index_file": "bm25_index.json.gz"},
            "generation": {"index": str(bm25_index)},
            "retrieval_run": {"index": str(bm25_index)},
            "run": {"retrieval_index": {"out": str(bm25_index)}},
        }
    )


def test_retrieval_method_override_updates_concrete_index_paths(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)

    selected = _with_retrieval_method(cfg, "graph_hybrid")

    expected_index = str(tmp_path / "artifacts" / "graph_hybrid" / "graph_index.json")
    assert selected.retrieval.name == "graph_hybrid"
    assert selected.generation.index == expected_index
    assert selected.retrieval_run.index == expected_index
    assert selected.run.retrieval_index.out == expected_index
    assert cfg.retrieval.name == "bm25"
    assert cfg.generation.index == str(tmp_path / "artifacts" / "bm25_index.json.gz")


def test_inference_job_process_uses_selected_method_for_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _base_cfg(tmp_path)
    jobs_dir = tmp_path / "jobs"
    job_repo = FileJobRepository(jobs_dir)
    result_repo = FileResultRepository(jobs_dir)
    service = InferenceJobService(
        cfg=cfg,
        job_repository=job_repo,
        result_repository=result_repo,
    )
    captured = {}

    from medical_rag_reranker.inference import generate as generate_module

    def fake_generate_from_cfg(*, cfg, question, queries_path, output_path):
        captured["cfg"] = cfg
        captured["question"] = question
        captured["queries_path"] = queries_path
        captured["output_path"] = output_path
        return {"answer": "ok"}

    monkeypatch.setattr(generate_module, "generate_from_cfg", fake_generate_from_cfg)

    job = service.create_job(
        question="What is metformin used for?",
        metadata={"retrieval_method": "hybrid"},
    )
    service.process(job.job_id)

    expected_index = str(tmp_path / "artifacts" / "hybrid" / "hybrid_index.json")
    assert captured["question"] == "What is metformin used for?"
    assert captured["queries_path"] is None
    assert captured["output_path"] is None
    assert captured["cfg"].retrieval.name == "hybrid"
    assert captured["cfg"].generation.index == expected_index
    assert result_repo.get(job.job_id)["retrieval_method"] == "hybrid"


def test_jobs_api_normalizes_retrieval_method_on_submit(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "jobs": {
                "storage": "file",
                "store_dir": str(tmp_path / "store"),
                "dispatch": "sync",
                "auto_dispatch": False,
            }
        }
    )
    client = TestClient(create_jobs_api(cfg))

    response = client.post(
        "/jobs",
        json={"question": "What is metformin used for?", "retrieval_method": "graph"},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["metadata"]["retrieval_method"] == "graph_bm25"
    assert payload["dispatched"] is False


def test_retrieval_method_options_include_index_readiness(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    bm25_index = tmp_path / "artifacts" / "bm25_index.json.gz"
    bm25_index.parent.mkdir(parents=True)
    bm25_index.write_text("{}", encoding="utf-8")

    options = {option["value"]: option for option in retrieval_method_options(cfg)}

    assert options["bm25"]["status"] == "ready"
    assert options["bm25"]["index_ready"] is True
    assert options["bm25"]["index_path"] == str(bm25_index)
    assert options["medcpt"]["status"] == "missing_index"
    assert options["medcpt"]["index_ready"] is False
    assert options["medcpt"]["index_path"] == str(
        tmp_path / "artifacts" / "medcpt_index.pkl"
    )


def test_retrieval_methods_endpoint_reports_index_readiness(tmp_path: Path) -> None:
    cfg = OmegaConf.merge(
        _base_cfg(tmp_path),
        {
            "jobs": {
                "storage": "file",
                "store_dir": str(tmp_path / "store"),
                "dispatch": "sync",
                "auto_dispatch": False,
            }
        },
    )
    bm25_index = tmp_path / "artifacts" / "bm25_index.json.gz"
    bm25_index.parent.mkdir(parents=True)
    bm25_index.write_text("{}", encoding="utf-8")
    client = TestClient(create_jobs_api(cfg))

    response = client.get("/retrieval-methods")

    assert response.status_code == 200
    methods = {method["value"]: method for method in response.json()["methods"]}
    assert methods["bm25"]["label"] == "BM25"
    assert methods["bm25"]["index_ready"] is True
    assert methods["medcpt"]["label"] == "MedCPT"
    assert methods["medcpt"]["status"] == "missing_index"


def test_load_retriever_builds_rag_fusion_from_config(monkeypatch) -> None:
    class FakeBaseRetriever:
        pass

    base = FakeBaseRetriever()
    monkeypatch.setattr(loading.BM25Retriever, "load", lambda index_path: base)

    retriever = loading.load_retriever(
        "rag_fusion_bm25",
        "artifacts/bm25_index.json.gz",
        OmegaConf.create(
            {
                "base_retriever": "bm25",
                "num_queries": 3,
                "cand_k": 25,
                "rrf_k": 42,
                "include_original": False,
            }
        ),
    )

    assert retriever.base is base
    assert retriever.num_queries == 3
    assert retriever.cand_k == 25
    assert retriever.rrf_k == 42
    assert retriever.include_original is False
