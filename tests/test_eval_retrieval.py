from pathlib import Path

from medical_rag_reranker.commands.eval_retrieval import (
    _metric_name_for_mlflow,
    read_qrels_tsv,
    read_run_trec,
)


def test_trec_parsers_read_qrels_and_run_files(tmp_path: Path) -> None:
    qrels_path = tmp_path / "qrels.tsv"
    qrels_path.write_text("q1\t0\td1\t1\nq1\t0\td2\t0\n", encoding="utf-8")

    run_path = tmp_path / "run.trec"
    run_path.write_text("q1 Q0 d1 1 2.0 bm25\nq1 Q0 d2 2 1.0 bm25\n", encoding="utf-8")

    qrels = read_qrels_tsv(qrels_path)
    run = read_run_trec(run_path)

    assert qrels == {"q1": {"d1": 1, "d2": 0}}
    assert run == {"q1": {"d1": 2.0, "d2": 1.0}}


def test_metric_name_for_mlflow_rewrites_at_sign() -> None:
    assert _metric_name_for_mlflow("P@5") == "P_at_5"
    assert _metric_name_for_mlflow("NDCG@10") == "NDCG_at_10"
