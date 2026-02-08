"""CLI for offline evaluation of retrieval run files against qrels."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List

import mlflow
from omegaconf import DictConfig


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
    return values


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


def evaluate_with_pytrec_eval(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    ks: List[int],
) -> Dict[str, float]:
    """Compute mean P@k / R@k / NDCG@k over evaluated queries."""
    try:
        import pytrec_eval
    except ImportError as e:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "pytrec_eval is not installed. Add it to your env/poetry deps."
        ) from e

    measures = set()
    for k in ks:
        measures.add(f"P_{k}")
        measures.add(f"recall_{k}")
        measures.add(f"ndcg_cut_{k}")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query = evaluator.evaluate(run)
    if not per_query:
        raise RuntimeError(
            "No queries were evaluated (empty run or mismatch with qrels)."
        )

    avg: Dict[str, float] = {m: 0.0 for m in measures}
    n = 0
    for scores in per_query.values():
        n += 1
        for m in measures:
            avg[m] += float(scores.get(m, 0.0))
    for m in avg:
        avg[m] /= float(max(n, 1))

    out: Dict[str, float] = {}
    for k in ks:
        out[f"P@{k}"] = avg[f"P_{k}"]
        out[f"R@{k}"] = avg[f"recall_{k}"]
        out[f"NDCG@{k}"] = avg[f"ndcg_cut_{k}"]
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
            "ks": ",".join(map(str, ks_values)),
            "eval_queries": str(eval_queries),
            "qrels": str(qrels),
            "run_path": str(effective_run_path),
        }
        mlflow.log_params({k: v for k, v in params.items() if v is not None})

        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

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
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
