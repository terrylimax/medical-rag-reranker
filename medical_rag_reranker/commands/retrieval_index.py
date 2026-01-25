"""Build and persist retrieval indices.

This CLI builds an index from a JSONL corpus and saves it to disk.

Supported retrievers:
- bm25: builds a BM25 index (stored as JSON/JSON.GZ metadata for safe reload/rebuild).
- dense: builds a dense vector index (currently persisted as a pickle file in DenseRetriever).
- hybrid: builds both bm25 + dense indices and writes a small JSON manifest that describes
    how to combine them (alpha/cand_k) and where the two index files live.

Inputs:
- --corpus: path to a corpus.jsonl file

Output behavior (--out):
- If you pass a directory path (existing dir or a path ending with '/'), the command will
    place default filenames inside it.
- If you pass a file path without an extension, the command will append a sensible extension.
- For hybrid, --out controls where the manifest goes and how the two component index files are named.

Examples
--------

1) Build BM25 index (write into a directory):

     python -m medical_rag_reranker.commands.retrieval_index \
         --retriever bm25 \
         --corpus data/corpus.jsonl \
         --out artifacts/

2) Build BM25 index (explicit file path):

     python -m medical_rag_reranker.commands.retrieval_index \
         --retriever bm25 \
         --corpus data/corpus.jsonl \
         --out artifacts/my_bm25.json.gz

3) Build Dense index (auto extension):

     python -m medical_rag_reranker.commands.retrieval_index \
         --retriever dense \
         --corpus data/corpus.jsonl \
         --out artifacts/my_dense \
         --model sentence-transformers/all-MiniLM-L6-v2

4) Build Hybrid (write 3 files into a directory):

     python -m medical_rag_reranker.commands.retrieval_index \
         --retriever hybrid \
         --corpus data/corpus.jsonl \
         --out artifacts/hybrid/ \
         --model sentence-transformers/all-MiniLM-L6-v2 \
         --alpha 0.5 \
         --cand-k 50

     This will create:
     - artifacts/hybrid/hybrid_index.json        (manifest)
     - artifacts/hybrid/bm25_index.json.gz       (bm25)
     - artifacts/hybrid/dense_index.pkl          (dense)

5) Build Hybrid with a prefix (creates manifest + suffixed component files):

     python -m medical_rag_reranker.commands.retrieval_index \
         --retriever hybrid \
         --corpus data/corpus.jsonl \
         --out artifacts/my_hybrid \
         --model sentence-transformers/all-MiniLM-L6-v2

     This will create:
     - artifacts/my_hybrid.json
     - artifacts/my_hybrid_bm25.json.gz
     - artifacts/my_hybrid_dense.pkl
"""

import argparse
import json
import os
from pathlib import Path

from medical_rag_reranker.retrieval.bm25 import BM25Retriever
from medical_rag_reranker.retrieval.dense import DenseRetriever


def _default_index_filename(retriever: str) -> str:
    if retriever == "bm25":
        return "bm25_index.json.gz"
    if retriever == "dense":
        return "dense_index.pkl"
    if retriever == "hybrid":
        return "hybrid_index.json"
    raise ValueError(f"Unknown retriever: {retriever}")


def _resolve_out_path(retriever: str, out_arg: str) -> Path:
    """Resolve output path.

    - If `out_arg` points to a directory (existing, or ends with '/'), place a
      default filename inside it.
    - If `out_arg` has no suffix, append a sensible suffix based on retriever.
    - If `out_arg` already has a suffix, keep it as-is.
    """
    out = Path(out_arg)

    is_dir_hint = out_arg.endswith("/") or out_arg.endswith("\\")
    if is_dir_hint or (out.exists() and out.is_dir()):
        return out / _default_index_filename(retriever)

    if out.suffix:
        return out

    default_name = _default_index_filename(retriever)
    # default_name has either one suffix (.pkl) or two (.json.gz)
    return out.with_suffix("").with_name(out.name + "".join(Path(default_name).suffixes))


def _relpath(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path, start)
    except Exception:
        return str(path)


def _resolve_hybrid_out_paths(out_arg: str) -> tuple[Path, Path, Path]:
    """Return (manifest, bm25_index_path, dense_index_path)."""
    out = Path(out_arg)

    is_dir_hint = out_arg.endswith("/") or out_arg.endswith("\\")
    if is_dir_hint or (out.exists() and out.is_dir()):
        base_dir = out
        manifest = base_dir / "hybrid_index.json"
        bm25_path = base_dir / "bm25_index.json.gz"
        dense_path = base_dir / "dense_index.pkl"
        return manifest, bm25_path, dense_path

    # Treat `out` as a manifest file path if it has a suffix.
    if out.suffix:
        manifest = out
        stem = out.name
        for suf in out.suffixes:
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
        base = out.with_name(stem)
    else:
        # Treat `out` as a prefix; write manifest next to it.
        manifest = out.with_suffix(".json")
        base = out

    bm25_path = base.parent / f"{base.name}_bm25.json.gz"
    dense_path = base.parent / f"{base.name}_dense.pkl"
    return manifest, bm25_path, dense_path


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build and persist retrieval indices from a corpus.jsonl file.",
        epilog=(
            "Examples:\n"
            "  python -m medical_rag_reranker.commands.retrieval_index \\\n+  --retriever bm25 \\\n+  --corpus data/corpus.jsonl \\\n+  --out artifacts/\n\n"
            "  python -m medical_rag_reranker.commands.retrieval_index \\\n+  --retriever dense \\\n+  --corpus data/corpus.jsonl \\\n+  --out artifacts/my_dense \\\n+  --model sentence-transformers/all-MiniLM-L6-v2\n\n"
            "  python -m medical_rag_reranker.commands.retrieval_index \\\n+  --retriever hybrid \\\n+  --corpus data/corpus.jsonl \\\n+  --out artifacts/hybrid/ \\\n+  --alpha 0.5 \\\n+  --cand-k 50 \\\n+  --model sentence-transformers/all-MiniLM-L6-v2\n"
        ),
    )
    p.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], required=True)
    p.add_argument("--corpus", required=True, help="path to corpus.jsonl")
    p.add_argument(
        "--out",
        required=True,
        help=(
            "output path (file or directory). If you pass a directory or a path "
            "without extension, a sensible filename/extension will be chosen automatically. "
            "For hybrid, this will write a manifest JSON plus separate bm25/dense index files."
        ),
    )
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="only for dense")
    p.add_argument("--alpha", type=float, default=0.5, help="only for hybrid")
    p.add_argument("--cand-k", type=int, default=50, help="only for hybrid")
    args = p.parse_args()

    if args.retriever == "hybrid":
        manifest_path, bm25_path, dense_path = _resolve_hybrid_out_paths(args.out)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        dense_path.parent.mkdir(parents=True, exist_ok=True)

        bm25 = BM25Retriever()
        dense = DenseRetriever(model_name=args.model)

        bm25.index(args.corpus)
        bm25.save(str(bm25_path))

        dense.index(args.corpus)
        dense.save(str(dense_path))

        manifest = {
            "format": "medical-rag-reranker.hybrid-index",
            "version": 1,
            "retriever": "hybrid",
            "alpha": float(args.alpha),
            "cand_k": int(args.cand_k),
            "corpus": str(args.corpus),
            "dense_model": str(args.model),
            "bm25_index": _relpath(bm25_path, manifest_path.parent),
            "dense_index": _relpath(dense_path, manifest_path.parent),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"Saved hybrid manifest to: {manifest_path}")
        print(f"Saved bm25 index to: {bm25_path}")
        print(f"Saved dense index to: {dense_path}")
        return

    out = _resolve_out_path(args.retriever, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.retriever == "bm25":
        r = BM25Retriever()
    else:
        r = DenseRetriever(model_name=args.model)

    r.index(args.corpus)
    r.save(str(out))
    print(f"Saved index to: {out}")


if __name__ == "__main__":
    main()