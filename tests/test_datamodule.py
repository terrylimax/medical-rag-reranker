import json
from pathlib import Path

from medical_rag_reranker.data import datamodule as datamodule_module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_reranker_datamodule_prefers_prepared_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    processed_dir = tmp_path / "processed"
    _write_jsonl(
        processed_dir / "qa.jsonl",
        [
            {
                "question_id": "q1",
                "question": "heart rhythm disease",
                "answer": "arrhythmia answer",
            },
            {
                "question_id": "q2",
                "question": "heart disease treatment",
                "answer": "cardiac treatment answer",
            },
            {
                "question_id": "q3",
                "question": "kidney stone symptoms",
                "answer": "kidney stone answer",
            },
        ],
    )
    _write_jsonl(
        processed_dir / "corpus.jsonl",
        [
            {
                "doc_id": "doc-q1",
                "question_id": "q1",
                "text": "arrhythmia answer",
                "source": "TEST",
            },
            {
                "doc_id": "doc-q2",
                "question_id": "q2",
                "text": "cardiac treatment answer",
                "source": "TEST",
            },
            {
                "doc_id": "doc-q3",
                "question_id": "q3",
                "text": "kidney stone answer",
                "source": "TEST",
            },
        ],
    )
    (processed_dir / "splits.json").write_text(
        json.dumps({"seed": 42, "train": ["q1", "q2"], "val": ["q3"], "test": []}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        datamodule_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    dm = datamodule_module.RerankerDataModule(
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(processed_dir),
        prefer_prepared_artifacts=True,
        negatives_per_query=1,
        hard_negative_pool_size=2,
    )
    dm.setup()

    train_examples = dm.train_ds.examples
    val_examples = dm.val_ds.examples

    assert len(train_examples) == 4
    assert len(val_examples) == 2

    assert {example.query_id for example in train_examples} == {"q1", "q2"}
    assert {example.query_id for example in val_examples} == {"q3"}

    train_positive_doc_ids = {
        example.query_id: example.doc_id
        for example in train_examples
        if example.label == 1
    }
    train_negative_doc_ids = {
        example.query_id: example.doc_id
        for example in train_examples
        if example.label == 0
    }

    assert train_positive_doc_ids == {"q1": "doc-q1", "q2": "doc-q2"}
    assert train_negative_doc_ids["q1"] != "doc-q1"
    assert train_negative_doc_ids["q2"] != "doc-q2"
