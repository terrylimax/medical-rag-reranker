import json
from pathlib import Path

from medical_rag_reranker.commands.retriever_training_data import (
    build_retriever_training_data,
    read_jsonl,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_build_retriever_training_data_uses_split_safe_hard_negatives(
    tmp_path: Path,
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
                "question": "cardiac disease treatment",
                "answer": "cardiac treatment answer",
            },
            {
                "question_id": "q3",
                "question": "kidney stone symptoms",
                "answer": "kidney stone answer",
            },
            {
                "question_id": "q4",
                "question": "skin rash treatment",
                "answer": "rash treatment answer",
            },
        ],
    )
    _write_jsonl(
        processed_dir / "corpus.jsonl",
        [
            {
                "doc_id": "doc-q1",
                "question_id": "q1",
                "text": "arrhythmia heart rhythm answer",
            },
            {
                "doc_id": "doc-q2",
                "question_id": "q2",
                "text": "cardiac disease treatment answer",
            },
            {
                "doc_id": "doc-q3",
                "question_id": "q3",
                "text": "kidney stone symptoms answer",
            },
            {
                "doc_id": "doc-q4",
                "question_id": "q4",
                "text": "skin rash treatment answer",
            },
        ],
    )
    (processed_dir / "splits.json").write_text(
        json.dumps({"train": ["q1", "q2"], "val": ["q3"], "test": ["q4"]}),
        encoding="utf-8",
    )

    result = build_retriever_training_data(
        processed_dir=str(processed_dir),
        out_dir=str(tmp_path / "retriever_training"),
        negatives_per_query=1,
        hard_negative_pool_size=3,
        restrict_train_negatives_to_train=True,
        seed=7,
    )

    train_rows = read_jsonl(Path(result["train_path"]))
    val_rows = read_jsonl(Path(result["val_path"]))

    assert len(train_rows) == 2
    assert len(val_rows) == 1
    assert result["train"]["restrict_negative_pool_to_split"] is True
    assert result["train"]["num_negative_pool_docs"] == 2

    train_doc_ids = {"doc-q1", "doc-q2"}
    for row in train_rows:
        assert row["positive_doc_id"] not in row["negative_doc_ids"]
        assert len(row["negative_doc_ids"]) == 1
        assert set(row["negative_doc_ids"]).issubset(train_doc_ids)
        assert len(row["negative_texts"]) == len(row["negative_doc_ids"])

    assert val_rows[0]["positive_doc_id"] == "doc-q3"
    assert "doc-q3" not in val_rows[0]["negative_doc_ids"]
    assert Path(result["metadata_path"]).exists()
