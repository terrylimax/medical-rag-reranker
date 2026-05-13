import csv
import json
from pathlib import Path

from omegaconf import OmegaConf

from medical_rag_reranker.commands import judge_generation as judge_generation_module


class _FakeJudge:
    def __init__(self):
        self.query_ids: list[str] = []

    def evaluate(self, row):
        assert row["query_id"]
        self.query_ids.append(row["query_id"])
        return {
            "faithfulness": 5.0,
            "relevance": 4.0,
            "completeness": 3.0,
            "safety": 5.0,
            "verdict": "pass",
            "rationale": "Supported by the retrieved context.",
        }


def test_run_judge_generation_sample_reads_existing_raw_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "generation"
    input_dir.mkdir()
    for run_name in ("bm25", "hybrid"):
        rows = [
            {
                "query_id": f"q{i}",
                "question": "What is metformin used for?",
                "answer": "Metformin is used for type 2 diabetes [doc1].",
                "retrieved": [
                    {
                        "doc_id": "doc1",
                        "text": "Metformin treats type 2 diabetes.",
                    }
                ],
                "citations_detected": ["doc1"],
            }
            for i in range(2)
        ]
        (input_dir / f"{run_name}.raw.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        judge_generation_module,
        "build_judge_from_cfg",
        lambda run_cfg: _FakeJudge(),
    )
    cfg = OmegaConf.create({"run": {"eval_generation": {}}})

    result = judge_generation_module.run_judge_generation_sample(
        cfg,
        input_dir=input_dir,
        output_dir=tmp_path / "llm_judge_sample",
        examples_limit=1,
    )

    summary_rows = list(csv.DictReader(Path(result["summary_csv"]).open()))
    assert result["num_runs"] == 2
    assert len(summary_rows) == 2
    assert summary_rows[0]["num_examples"] == "1.0"
    assert summary_rows[0]["pass_rate"] == "1.0"
    assert len(list((tmp_path / "llm_judge_sample").glob("*.jsonl"))) == 3


def test_run_judge_generation_sample_resumes_partial_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "generation"
    input_dir.mkdir()
    raw_rows = [
        {
            "query_id": f"q{i}",
            "question": "What is metformin used for?",
            "answer": "Metformin is used for type 2 diabetes [doc1].",
            "retrieved": [{"doc_id": "doc1", "text": "Metformin treats diabetes."}],
        }
        for i in range(2)
    ]
    (input_dir / "bm25.raw.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_rows) + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "llm_judge_sample"
    output_dir.mkdir()
    existing_row = {
        **raw_rows[0],
        "sample_idx": 0,
        "judge_mode": "llm",
        "evaluation": {
            "faithfulness": 4.0,
            "relevance": 4.0,
            "completeness": 4.0,
            "safety": 4.0,
            "verdict": "pass",
            "rationale": "Already judged.",
        },
    }
    (output_dir / "bm25.llm_judge_n2.jsonl").write_text(
        json.dumps(existing_row) + "\n",
        encoding="utf-8",
    )

    fake_judge = _FakeJudge()
    monkeypatch.setattr(
        judge_generation_module,
        "build_judge_from_cfg",
        lambda run_cfg: fake_judge,
    )
    cfg = OmegaConf.create({"run": {"eval_generation": {}}})

    result = judge_generation_module.run_judge_generation_sample(
        cfg,
        input_dir=input_dir,
        output_dir=output_dir,
        examples_limit=2,
        resume=True,
    )

    output_rows = [
        json.loads(line)
        for line in (output_dir / "bm25.llm_judge_n2.jsonl").read_text().splitlines()
    ]
    summary_rows = list(csv.DictReader(Path(result["summary_csv"]).open()))

    assert fake_judge.query_ids == ["q1"]
    assert [row["query_id"] for row in output_rows] == ["q0", "q1"]
    assert summary_rows[0]["num_examples"] == "2.0"
