from pathlib import Path

from medical_rag_reranker.artifacts.sync import (
    REGISTRY_FORMAT,
    compact_dvc_targets,
    collect_artifact_files,
    pull_artifacts,
    push_artifacts,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_collect_artifact_files_uses_runtime_defaults(tmp_path: Path) -> None:
    _write(tmp_path / "data/processed/corpus.jsonl", "{}\n")
    _write(tmp_path / "artifacts/bm25_index.json.gz", "index")
    _write(tmp_path / "artifacts/hybrid/hybrid_index.json", "{}")
    _write(tmp_path / "artifacts/experiments/run-1/e2e_summary.csv", "method\nbm25\n")
    _write(tmp_path / "reports/eval_generation.md", "report")
    _write(tmp_path / "artifacts/retriever_training/train.jsonl", "{}")

    files = collect_artifact_files(tmp_path)
    rels = {path.relative_to(tmp_path).as_posix() for path in files}

    assert "data/processed/corpus.jsonl" in rels
    assert "artifacts/bm25_index.json.gz" in rels
    assert "artifacts/hybrid/hybrid_index.json" in rels
    assert "artifacts/experiments/run-1/e2e_summary.csv" in rels
    assert "reports/eval_generation.md" not in rels
    assert "artifacts/retriever_training/train.jsonl" not in rels


def test_compact_dvc_targets_groups_runtime_artifacts(tmp_path: Path) -> None:
    _write(tmp_path / "data/processed/corpus.jsonl", '{"doc_id":"d1"}\n')
    _write(tmp_path / "data/processed/eval_queries.jsonl", '{"query_id":"q1"}\n')
    _write(tmp_path / "artifacts/qdrant_index.json", '{"format":"qdrant"}')
    _write(tmp_path / "artifacts/hybrid/hybrid_index.json", "{}")
    _write(
        tmp_path / "artifacts/experiments/run-1/models/reranker/best.ckpt",
        "checkpoint",
    )
    _write(tmp_path / "artifacts/retriever/model/config.json", "{}")

    files = collect_artifact_files(tmp_path)
    targets = compact_dvc_targets(files, local_root=tmp_path)

    assert "data/processed" in targets
    assert "artifacts/qdrant_index.json" in targets
    assert "artifacts/hybrid" in targets
    assert "artifacts/experiments/run-1" in targets
    assert "artifacts/experiments/run-1/models/reranker/best.ckpt" not in targets
    assert "artifacts/retriever" in targets
    assert "artifacts/index_registry.json" not in targets


def test_push_artifacts_dry_run_returns_dvc_commands(tmp_path: Path) -> None:
    _write(tmp_path / "data/processed/corpus.jsonl", '{"doc_id":"d1"}\n')
    _write(tmp_path / "artifacts/qdrant_index.json", '{"format":"qdrant"}')

    result = push_artifacts(
        remote_uri="s3://bucket/medical-rag/medquad-v1",
        local_root=tmp_path,
        dry_run=True,
        region="eu-central-1",
    )

    assert result["format"] == REGISTRY_FORMAT
    assert result["dry_run"] is True
    assert result["dvc_remote"] == "artifact_s3"
    assert [
        "dvc",
        "remote",
        "add",
        "--force",
        "artifact_s3",
        result["remote_uri"],
    ] in result["dvc_commands"]
    assert [
        "dvc",
        "remote",
        "modify",
        "artifact_s3",
        "region",
        "eu-central-1",
    ] in result["dvc_commands"]
    dvc_add = next(cmd for cmd in result["dvc_commands"] if cmd[:2] == ["dvc", "add"])
    assert "artifacts/index_registry.json" not in dvc_add
    assert result["dvc_commands"][-1] == ["dvc", "push", "-r", "artifact_s3"]


def test_pull_artifacts_can_use_existing_dvc_remote_without_remote_uri(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("ARTIFACT_REMOTE_URI", raising=False)

    result = pull_artifacts(
        local_root=tmp_path,
        remote_name="artifact_s3",
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["dvc_remote"] == "artifact_s3"
    assert result["remote_uri"] == ""
    assert result["dvc_commands"] == [["dvc", "pull", "-r", "artifact_s3"]]
