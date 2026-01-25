"""Generate a retrieval run file (TREC format).

This CLI loads a retriever (BM25, Dense, or Hybrid), runs it over a list of
queries, and writes a run file suitable for standard IR evaluation.

Inputs
------
- `--queries`: JSONL with objects containing:
    - `query_id`: identifier (string/int)
    - `text`: query text
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
import os
from pathlib import Path

from medical_rag_reranker.retrieval.bm25 import BM25Retriever
from medical_rag_reranker.retrieval.dense import DenseRetriever
from medical_rag_reranker.retrieval.hybrid import HybridRetriever


def _resolve_manifest_path(index_arg: str) -> Path:
    """Resolve `--index` into a hybrid manifest path.

    If `index_arg` is a directory, expects `hybrid_index.json` inside it; otherwise
    treats `index_arg` as a direct path to the manifest.
    """
    p = Path(index_arg)
    if p.exists() and p.is_dir():
        candidate = p / "hybrid_index.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Expected hybrid_index.json inside directory: {p}")
        return candidate
    return p


def _load_hybrid_from_manifest(manifest_path: Path) -> HybridRetriever:
    """Load a `HybridRetriever` from a hybrid manifest JSON.

    The manifest is expected to reference saved bm25/dense indices (paths may be
    relative to the manifest location) and store hybrid parameters like `alpha`
    and `cand_k`.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    if m.get("format") != "medical-rag-reranker.hybrid-index":
        raise ValueError("Unsupported hybrid manifest format.")

    base_dir = manifest_path.parent
    bm25_index = Path(m["bm25_index"])
    dense_index = Path(m["dense_index"])

    # Allow relative paths in the manifest.
    if not bm25_index.is_absolute():
        bm25_index = base_dir / bm25_index
    if not dense_index.is_absolute():
        dense_index = base_dir / dense_index

    bm25 = BM25Retriever.load(str(bm25_index))
    dense = DenseRetriever.load(str(dense_index))

    return HybridRetriever(
        bm25=bm25,
        dense=dense,
        alpha=float(m.get("alpha", 0.5)),
        cand_k=int(m.get("cand_k", 50)),
    )


def main():
    """Run a retrieval baseline and write results in TREC run format.

    This entrypoint:
    - Loads a retriever from `--index` (bm25/dense index file, or a hybrid manifest/directory).
    - Reads queries from `--queries` (JSONL with keys: `query_id`, `text`).
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
    p.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], required=True)
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.retriever == "bm25":
        r = BM25Retriever.load(args.index)
    elif args.retriever == "dense":
        r = DenseRetriever.load(args.index)
    else:
        manifest_path = _resolve_manifest_path(args.index)
        r = _load_hybrid_from_manifest(manifest_path)

    # TREC format: query_id Q0 doc_id rank score run_name
    with open(args.queries, "r", encoding="utf-8") as fq, open(out, "w", encoding="utf-8") as fo:
        for line in fq:
            q = json.loads(line)
            qid = str(q["query_id"])
            text = q["text"]

            hits = r.retrieve(text, top_k=args.top_k)
            for rank, h in enumerate(hits, start=1):
                fo.write(f"{qid}\tQ0\t{h.doc_id}\t{rank}\t{h.score:.6f}\t{args.run_name}\n")

    print(f"Wrote run file: {out}")


if __name__ == "__main__":
    main()