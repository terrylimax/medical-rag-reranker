"""Generate a retrieval run file (TREC format).

This CLI loads a retriever (BM25, Dense, or Hybrid), runs it over a list of
queries, and writes a run file suitable for standard IR evaluation.

Inputs
------
- `--queries`: JSONL with objects containing:
    - `query_id`: identifier (string/int)
    - `text` or `question`: query text
- `--index`:
    - `bm25`: path to saved BM25 retriever file (e.g. `.json`/`.json.gz`)
    - `dense`: path to saved Dense index file (currently `.pkl`)
    - `hybrid`: path to `hybrid_index.json` manifest, or a directory containing it

Output
------
- `--out`: TREC run file, one line per hit:
    `query_id\tQ0\tdoc_id\trank\tscore\trun_name`

Examples
--------

BM25:
    python -m medical_rag_reranker.commands.retrieval_run \
        --retriever bm25 \
        --index artifacts/bm25_index.json.gz \
        --queries data/eval_queries.jsonl \
        --out runs/bm25.trec \
        --top_k 10 \
        --run_name bm25

Dense:
    python -m medical_rag_reranker.commands.retrieval_run \
        --retriever dense \
        --index artifacts/dense_index.pkl \
        --queries data/eval_queries.jsonl \
        --out runs/dense.trec

Hybrid (manifest path):
    python -m medical_rag_reranker.commands.retrieval_run \
        --retriever hybrid \
        --index artifacts/hybrid/hybrid_index.json \
        --queries data/eval_queries.jsonl \
        --out runs/hybrid.trec

Hybrid (directory shortcut):
    python -m medical_rag_reranker.commands.retrieval_run \
        --retriever hybrid \
        --index artifacts/hybrid/ \
        --queries data/eval_queries.jsonl \
        --out runs/hybrid.trec
"""

import argparse
import json
import time
from pathlib import Path

from omegaconf import DictConfig

from medical_rag_reranker.retrieval.loading import load_retriever
from medical_rag_reranker.utils.progress import count_text_lines, progress


def _resolve_query_text(query_obj: dict) -> str:
    """Return query text from either `text` (preferred) or `question`."""
    text = query_obj.get("text")
    if text:
        return str(text)

    question = query_obj.get("question")
    if question:
        return str(question)

    raise ValueError("Query row must contain either `text` or `question`.")


def _resolve_query_id(query_obj: dict) -> str:
    """Return query id from `query_id` or fallback `question_id`."""
    qid = query_obj.get("query_id")
    if qid is None:
        qid = query_obj.get("question_id")
    if qid is None:
        raise ValueError("Query row must contain `query_id` (or `question_id`).")
    return str(qid)


def _load_retriever(
    retriever_name: str,
    index_path: str,
    retrieval_cfg: DictConfig | None = None,
):
    return load_retriever(
        retriever_name=retriever_name,
        index_path=index_path,
        retrieval_cfg=retrieval_cfg,
    )


def run_retrieval(
    retriever_name: str,
    index_path: str,
    queries_path: str,
    out_path: str,
    top_k: int = 10,
    run_name: str = "baseline",
    retrieval_cfg: DictConfig | None = None,
) -> Path:
    """Run retrieval over a queries JSONL and write a TREC run file."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    retriever = _load_retriever(
        retriever_name=retriever_name,
        index_path=index_path,
        retrieval_cfg=retrieval_cfg,
    )
    total_queries = count_text_lines(queries_path)
    latencies_ms: list[float] = []
    fusion_query_rows: list[dict] = []

    # TREC format: query_id Q0 doc_id rank score run_name
    with (
        open(queries_path, "r", encoding="utf-8") as fq,
        open(out, "w", encoding="utf-8") as fo,
    ):
        rows = progress(
            fq,
            desc=f"Running {retriever_name} retrieval",
            total=total_queries,
            unit="query",
        )
        for line in rows:
            if not line.strip():
                continue
            q = json.loads(line)
            qid = _resolve_query_id(q)
            text = _resolve_query_text(q)

            started = time.perf_counter()
            hits = retriever.retrieve(text, top_k=int(top_k))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            expanded_queries = getattr(retriever, "last_queries", None)
            if expanded_queries:
                fusion_query_rows.append(
                    {
                        "query_id": qid,
                        "question": text,
                        "queries": list(expanded_queries),
                    }
                )
            if hasattr(rows, "set_postfix"):
                rows.set_postfix(latency_ms=f"{elapsed_ms:.1f}")
            for rank, h in enumerate(hits, start=1):
                fo.write(f"{qid}\tQ0\t{h.doc_id}\t{rank}\t{h.score:.6f}\t{run_name}\n")

    latency_path = out.with_suffix(out.suffix + ".latency.json")
    latency_payload = {
        "run_path": str(out),
        "num_queries": len(latencies_ms),
        "latencies_ms": latencies_ms,
    }
    latency_path.write_text(
        json.dumps(latency_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved retrieval run to: {out}")
    print(f"Saved retrieval latency profile to: {latency_path}")
    if fusion_query_rows:
        fusion_queries_path = out.with_suffix(out.suffix + ".queries.jsonl")
        with fusion_queries_path.open("w", encoding="utf-8") as f:
            for row in fusion_query_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved RAG Fusion query variants to: {fusion_queries_path}")
    return out


def run_from_cfg(cfg: DictConfig) -> Path:
    """Hydra-config entrypoint used by `medical_rag_reranker.commands`."""
    return run_retrieval(
        retriever_name=str(cfg.retrieval.name),
        index_path=str(cfg.retrieval_run.index),
        queries_path=str(cfg.retrieval_run.queries),
        out_path=str(cfg.retrieval_run.out),
        top_k=int(cfg.retrieval_run.top_k),
        run_name=str(cfg.retrieval_run.run_name),
        retrieval_cfg=cfg.retrieval,
    )


def main():
    """Run a retrieval baseline and write results in TREC run format.

    This entrypoint:
    - Loads a retriever from `--index` (bm25/dense index file, or a hybrid manifest/directory).
    - Reads queries from `--queries` (JSONL with keys: `query_id`, `text`/`question`).
    - For each query, retrieves `--top_k` documents and writes a run file to `--out`.

    Output file format (one line per hit):
        query_id  Q0  doc_id  rank  score  run_name
    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run a retriever over queries.jsonl and write a TREC run file.",
        epilog=(
            "Examples:\n"
            "  python -m medical_rag_reranker.commands.retrieval_run \\\n+  --retriever bm25 \\\n+  --index artifacts/bm25_index.json.gz \\\n+  --queries data/eval_queries.jsonl \\\n+  --out runs/bm25.trec\n\n"
            "  python -m medical_rag_reranker.commands.retrieval_run \\\n+  --retriever dense \\\n+  --index artifacts/dense_index.pkl \\\n+  --queries data/eval_queries.jsonl \\\n+  --out runs/dense.trec\n\n"
            "  python -m medical_rag_reranker.commands.retrieval_run \\\n+  --retriever hybrid \\\n+  --index artifacts/hybrid/hybrid_index.json \\\n+  --queries data/eval_queries.jsonl \\\n+  --out runs/hybrid.trec\n"
        ),
    )
    p.add_argument(
        "--retriever",
        choices=[
            "bm25",
            "dense",
            "bi_encoder",
            "hybrid",
            "rag_fusion",
            "rag_fusion_bm25",
            "rag_fusion_dense",
            "rag_fusion_medcpt_pilot",
            "graph_bm25",
            "graph_hybrid",
            "graph_hybrid_medcpt",
        ],
        required=True,
    )
    p.add_argument(
        "--index",
        required=True,
        help=(
            "path to saved index. For bm25/dense: path to the index file. "
            "For hybrid: path to hybrid manifest JSON (or a directory containing hybrid_index.json)."
        ),
    )
    p.add_argument("--queries", required=True, help="path to queries.jsonl")
    p.add_argument("--out", required=True, help="path to run.trec")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--run_name", default="baseline")
    args = p.parse_args()

    out = run_retrieval(
        retriever_name=args.retriever,
        index_path=args.index,
        queries_path=args.queries,
        out_path=args.out,
        top_k=args.top_k,
        run_name=args.run_name,
    )

    print(f"Wrote run file: {out}")


if __name__ == "__main__":
    main()
