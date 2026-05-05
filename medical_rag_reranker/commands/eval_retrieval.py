"""CLI for offline evaluation of retrieval run files against qrels."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import mlflow
from omegaconf import DictConfig

from medical_rag_reranker.graph.aspects import aspects_from_metadata, coerce_str_list
from medical_rag_reranker.utils.progress import path_size_bytes, progress


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "null", "None"):
        return None
    return text


def _parse_ks(ks: str | List[int]) -> List[int]:
    if isinstance(ks, list):
        values = [int(v) for v in ks]
    else:
        values = [int(x.strip()) for x in str(ks).split(",") if x.strip()]
    if not values:
        raise ValueError("ks is empty")
    if any(v <= 0 for v in values):
        raise ValueError("ks must contain positive integers")
    return values


def _metric_name_for_mlflow(name: str) -> str:
    """Convert human-readable retrieval metric names to MLflow-safe keys."""
    return str(name).replace("@", "_at_")


def read_qrels_tsv(path: Path) -> Dict[str, Dict[str, int]]:
    """Parse qrels in TREC format: qid <iter> docid rel."""
    qrels: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Bad qrels line (expected 4 columns): {line}")
            qid, _iter, docid, rel = parts
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def read_run_trec(path: Path) -> Dict[str, Dict[str, float]]:
    """Parse run in TREC format: qid Q0 docid rank score run_name."""
    run: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Bad run line (expected >=6 columns): {line}")
            qid, _q0, docid, _rank, score, _runname = parts[:6]
            run.setdefault(qid, {})[docid] = float(score)
    return run


def read_queries_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    queries: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("query_id") or row.get("question_id") or f"query-{idx}"
            queries[str(qid)] = row
    return queries


def read_corpus_jsonl(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    docs: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get("doc_id")
            if doc_id:
                docs[str(doc_id)] = row
    return docs


def _anchors_from_query(row: Dict[str, Any]) -> list[str]:
    for key in ("anchor_doc_ids", "gold_doc_ids"):
        values = coerce_str_list(row.get(key))
        if values:
            return values
    return coerce_str_list(row.get("gold_doc_id"))


def _requested_aspects(query: Dict[str, Any]) -> set[str]:
    requested = set()
    for value in coerce_str_list(query.get("requested_aspects")):
        requested.add(value.strip().lower().replace(" ", "_"))
    if not requested:
        requested.update(aspects_from_metadata(query))
    return requested


def _expected_cuis(
    query: Dict[str, Any],
    qrels_for_query: Dict[str, int],
    docstore: Dict[str, Dict[str, Any]],
) -> set[str]:
    explicit = set(coerce_str_list(query.get("umls_cui") or query.get("expected_cuis")))
    if explicit:
        return explicit
    values: set[str] = set()
    for doc_id, rel in qrels_for_query.items():
        if int(rel) <= 0:
            continue
        values.update(coerce_str_list(docstore.get(doc_id, {}).get("umls_cui")))
    return values


def evaluate_graph_metadata_metrics(
    eval_queries: Path,
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    ks: List[int],
    corpus_path: Path | None,
) -> Dict[str, float]:
    queries = read_queries_jsonl(eval_queries)
    docstore = read_corpus_jsonl(corpus_path)
    metrics: Dict[str, float] = {}

    ranked_by_query = {qid: _ranked_doc_ids(scores) for qid, scores in run.items()}

    for k in ks:
        type_scores: list[float] = []
        cui_scores: list[float] = []
        retention_scores: list[float] = []

        for qid, query in queries.items():
            hits = ranked_by_query.get(qid, [])[:k]

            requested = _requested_aspects(query)
            if requested and docstore:
                covered: set[str] = set()
                for doc_id in hits:
                    covered.update(aspects_from_metadata(docstore.get(doc_id, {})))
                type_scores.append(
                    len(covered & requested) / float(max(1, len(requested)))
                )

            expected_cuis = _expected_cuis(query, qrels.get(qid, {}), docstore)
            if expected_cuis and docstore:
                seen_cuis: set[str] = set()
                for doc_id in hits:
                    seen_cuis.update(
                        coerce_str_list(docstore.get(doc_id, {}).get("umls_cui"))
                    )
                cui_scores.append(
                    len(seen_cuis & expected_cuis) / float(max(1, len(expected_cuis)))
                )

            anchors = _anchors_from_query(query)
            if anchors:
                retention_scores.append(
                    len(set(hits) & set(anchors)) / float(max(1, len(anchors)))
                )

        if type_scores:
            metrics[f"question_type_coverage@{k}"] = sum(type_scores) / len(type_scores)
        if cui_scores:
            metrics[f"cui_coverage@{k}"] = sum(cui_scores) / len(cui_scores)
        if retention_scores:
            metrics[f"gold_retention@{k}"] = sum(retention_scores) / len(
                retention_scores
            )

    return metrics


def _ranked_doc_ids(scores: Dict[str, float]) -> list[str]:
    return [
        doc_id
        for doc_id, _score in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def _dcg(relevances: list[int]) -> float:
    return sum(
        ((2.0 ** float(rel) - 1.0) / math.log2(idx + 2.0))
        for idx, rel in enumerate(relevances)
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * float(q)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _load_latency_metrics(run_path: Path) -> dict[str, float]:
    latency_path = run_path.with_suffix(run_path.suffix + ".latency.json")
    if not latency_path.exists():
        return {}
    payload = json.loads(latency_path.read_text(encoding="utf-8"))
    latencies = [float(v) for v in payload.get("latencies_ms", [])]
    if not latencies:
        return {}
    return {
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "latency_mean_ms": sum(latencies) / len(latencies),
        "num_latency_queries": float(len(latencies)),
    }


def _index_size_mb(index_path: Path | None) -> float | None:
    if index_path is None or not index_path.exists():
        return None

    paths = [index_path]
    if index_path.is_file() and index_path.suffix == ".json":
        try:
            manifest = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        if manifest.get("format") == "medical-rag-reranker.hybrid-index":
            for key in ("bm25_index", "dense_index"):
                value = manifest.get(key)
                if not value:
                    continue
                component = Path(str(value))
                if not component.is_absolute():
                    component = index_path.parent / component
                paths.append(component)

    total_bytes = sum(path_size_bytes(path) for path in paths)
    return total_bytes / (1024.0 * 1024.0)


def evaluate_with_pytrec_eval(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    ks: List[int],
) -> Dict[str, float]:
    """Compute mean P/R/NDCG/Hit/MRR over qrels queries.

    The implementation is kept local so product metrics do not depend on the
    exact measure names supported by a pytrec_eval build.
    """
    if not qrels:
        raise RuntimeError("No queries were evaluated (empty qrels).")

    sums: Dict[str, float] = {}
    n = 0
    items = progress(
        qrels.items(),
        desc="Computing retrieval metrics",
        total=len(qrels),
        unit="query",
    )
    for qid, rels in items:
        relevant = {doc_id: int(rel) for doc_id, rel in rels.items() if int(rel) > 0}
        ranked = _ranked_doc_ids(run.get(qid, {}))
        ideal_rels = sorted(relevant.values(), reverse=True)
        n += 1

        for k in ks:
            top_docs = ranked[:k]
            hit_count = sum(1 for doc_id in top_docs if doc_id in relevant)
            precision = hit_count / float(k)
            recall = hit_count / float(max(1, len(relevant)))
            hit = 1.0 if hit_count > 0 else 0.0

            first_rank = 0
            for rank, doc_id in enumerate(top_docs, start=1):
                if doc_id in relevant:
                    first_rank = rank
                    break
            mrr = 0.0 if first_rank == 0 else 1.0 / float(first_rank)

            observed_rels = [relevant.get(doc_id, 0) for doc_id in top_docs]
            ideal_dcg = _dcg(ideal_rels[:k])
            ndcg = 0.0 if ideal_dcg <= 0.0 else _dcg(observed_rels) / ideal_dcg

            sums[f"P@{k}"] = sums.get(f"P@{k}", 0.0) + precision
            sums[f"R@{k}"] = sums.get(f"R@{k}", 0.0) + recall
            sums[f"NDCG@{k}"] = sums.get(f"NDCG@{k}", 0.0) + ndcg
            sums[f"Hit@{k}"] = sums.get(f"Hit@{k}", 0.0) + hit
            sums[f"MRR@{k}"] = sums.get(f"MRR@{k}", 0.0) + mrr

    out: Dict[str, float] = {key: value / float(n) for key, value in sums.items()}
    out["num_queries_eval"] = float(n)
    return out


def call_retriever(cmd: List[str]) -> None:
    """Call external retriever command expected to write run file."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Retriever command failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )


def run_eval(
    eval_queries: Path,
    qrels: Path,
    run_path: Path | None,
    retrieve_cmd: str | None,
    out_run: Path,
    ks: str | List[int],
    run_name: str,
    experiment: str,
    mlflow_uri: str | None,
    run_tag: str | None,
    retriever: str | None,
    top_k: int | None,
    alpha: float | None,
    embedding_model: str | None,
    index_path: Path | None = None,
    corpus_path: Path | None = None,
) -> Dict[str, float]:
    """Run retrieval evaluation and log metrics/artifacts to MLflow."""
    ks_values = _parse_ks(ks)

    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)

    effective_run_path = run_path or out_run
    if run_path is None:
        if not retrieve_cmd:
            raise ValueError(
                "Either run_path must be provided or retrieve_cmd must be set."
            )
        cmd_str = retrieve_cmd.format(queries=str(eval_queries), out_run=str(out_run))
        cmd = shlex.split(cmd_str)
        call_retriever(cmd)
        if not out_run.exists():
            raise RuntimeError(f"Retriever did not create run file: {out_run}")

    qrels_map = read_qrels_tsv(qrels)
    run_map = read_run_trec(effective_run_path)
    metrics = evaluate_with_pytrec_eval(qrels=qrels_map, run=run_map, ks=ks_values)
    metrics.update(
        evaluate_graph_metadata_metrics(
            eval_queries=eval_queries,
            qrels=qrels_map,
            run=run_map,
            ks=ks_values,
            corpus_path=corpus_path,
        )
    )
    metrics.update(_load_latency_metrics(effective_run_path))
    index_size_mb = _index_size_mb(index_path)
    if index_size_mb is not None:
        metrics["index_size_mb"] = float(index_size_mb)

    out_metrics = effective_run_path.with_suffix(
        effective_run_path.suffix + ".metrics.json"
    )
    out_metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with mlflow.start_run(run_name=run_name):
        if run_tag:
            mlflow.set_tag("tag", run_tag)

        params = {
            "retriever": retriever,
            "top_k": top_k,
            "alpha": alpha,
            "embedding_model": embedding_model,
            "index_path": None if index_path is None else str(index_path),
            "corpus_path": None if corpus_path is None else str(corpus_path),
            "ks": ",".join(map(str, ks_values)),
            "eval_queries": str(eval_queries),
            "qrels": str(qrels),
            "run_path": str(effective_run_path),
        }
        mlflow.log_params({k: v for k, v in params.items() if v is not None})

        for k, v in metrics.items():
            mlflow.log_metric(_metric_name_for_mlflow(k), float(v))

        mlflow.log_artifact(str(effective_run_path))
        mlflow.log_artifact(str(out_metrics))

    return metrics


def run_from_cfg(cfg: DictConfig) -> Dict[str, float]:
    """Hydra-config entrypoint used by `medical_rag_reranker.commands`."""
    run_cfg = cfg.run.eval_retrieval
    run_path_text = _as_optional_str(run_cfg.run_path)
    retrieve_cmd = _as_optional_str(run_cfg.retrieve_cmd)

    run_path: Path | None = Path(run_path_text) if run_path_text else None
    if run_path is None and retrieve_cmd is None:
        from medical_rag_reranker.commands.retrieval_run import (
            run_from_cfg as retrieval_run_from_cfg,
        )

        run_path = retrieval_run_from_cfg(cfg)

    alpha_val = _as_optional_str(run_cfg.alpha)
    top_k_val = _as_optional_str(run_cfg.top_k)
    index_path: Path | None = None
    if "retrieval_run" in cfg and "index" in cfg.retrieval_run:
        index_path = Path(str(cfg.retrieval_run.index))

    return run_eval(
        eval_queries=Path(str(run_cfg.eval_queries)),
        qrels=Path(str(run_cfg.qrels)),
        run_path=run_path,
        retrieve_cmd=retrieve_cmd,
        out_run=Path(str(run_cfg.out_run)),
        ks=str(run_cfg.ks),
        run_name=str(run_cfg.run_name),
        experiment=str(run_cfg.experiment),
        mlflow_uri=_as_optional_str(run_cfg.mlflow_uri),
        run_tag=_as_optional_str(run_cfg.run_tag),
        retriever=_as_optional_str(run_cfg.retriever),
        top_k=int(top_k_val) if top_k_val else None,
        alpha=float(alpha_val) if alpha_val else None,
        embedding_model=_as_optional_str(run_cfg.embedding_model),
        index_path=index_path,
        corpus_path=(
            Path(str(run_cfg.corpus))
            if _as_optional_str(getattr(run_cfg, "corpus", None))
            else None
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_queries", type=Path, required=True)
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--run_path", type=Path, default=None)
    p.add_argument(
        "--retrieve_cmd",
        type=str,
        default=None,
        help="Command template. Use placeholders {queries} and {out_run}.",
    )
    p.add_argument("--out_run", type=Path, default=Path("runs/retrieval.run.trec"))
    p.add_argument("--ks", type=str, default="5,10")
    p.add_argument("--run_name", type=str, default="baseline")
    p.add_argument("--mlflow_uri", type=str, default=None)
    p.add_argument("--experiment", type=str, default="retrieval_eval")
    p.add_argument("--run_tag", type=str, default=None)
    p.add_argument("--retriever", type=str, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--embedding_model", type=str, default=None)
    p.add_argument("--index_path", type=Path, default=None)
    p.add_argument("--corpus", type=Path, default=None)
    args = p.parse_args()

    metrics = run_eval(
        eval_queries=args.eval_queries,
        qrels=args.qrels,
        run_path=args.run_path,
        retrieve_cmd=args.retrieve_cmd,
        out_run=args.out_run,
        ks=args.ks,
        run_name=args.run_name,
        experiment=args.experiment,
        mlflow_uri=args.mlflow_uri,
        run_tag=args.run_tag,
        retriever=args.retriever,
        top_k=args.top_k,
        alpha=args.alpha,
        embedding_model=args.embedding_model,
        index_path=args.index_path,
        corpus_path=args.corpus,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
