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

from omegaconf import DictConfig

from medical_rag_reranker.graph.builder import build_medquad_graph
from medical_rag_reranker.retrieval.bm25 import BM25Retriever
from medical_rag_reranker.retrieval.graph_expanded import DEFAULT_RELATION_WEIGHTS
from medical_rag_reranker.utils.progress import timed_stage


def _default_index_filename(retriever: str) -> str:
    if retriever == "bm25":
        return "bm25_index.json.gz"
    if retriever in ("dense", "bi_encoder"):
        return "dense_index.pkl"
    if retriever == "hybrid":
        return "hybrid_index.json"
    if retriever.startswith("graph"):
        return "graph_index.json"
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
    return out.with_suffix("").with_name(
        out.name + "".join(Path(default_name).suffixes)
    )


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


def _resolve_graph_out_paths(out_arg: str) -> tuple[Path, Path]:
    """Return (graph_manifest_path, base_index_out)."""
    out = Path(out_arg)

    is_dir_hint = out_arg.endswith("/") or out_arg.endswith("\\")
    if is_dir_hint or (out.exists() and out.is_dir()):
        return out / "graph_index.json", out / "base"

    if out.suffix:
        manifest = out
        stem = out.name
        for suf in out.suffixes:
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
        return manifest, out.with_name(f"{stem}_base")

    return out.with_suffix(".json"), out.with_name(f"{out.name}_base")


def _infer_graph_base_retriever(retriever: str, base_retriever: str | None) -> str:
    if base_retriever:
        return base_retriever
    if "hybrid" in retriever:
        return "hybrid"
    if "dense" in retriever:
        return "dense"
    return "bm25"


def _relation_weights_from_config(value: object | None) -> dict[str, float]:
    weights = dict(DEFAULT_RELATION_WEIGHTS)
    if value:
        for key, weight in dict(value).items():
            weights[str(key)] = float(weight)
    return weights


def run_index(
    retriever: str,
    corpus: str,
    out: str,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    query_model: str = "ncbi/MedCPT-Query-Encoder",
    doc_model: str = "ncbi/MedCPT-Article-Encoder",
    dense_backend: str = "dense",
    pooling: str = "cls",
    normalize: bool = True,
    query_max_length: int = 64,
    doc_max_length: int = 256,
    encode_batch_size: int = 32,
    max_seq_length: int | None = None,
    local_files_only: bool = False,
    fusion: str = "score",
    alpha: float = 0.5,
    cand_k: int = 50,
    rrf_k: int = 60,
    graph_path: str | None = None,
    base_retriever: str | None = None,
    seed_k: int = 20,
    expand_k: int = 50,
    max_hops: int = 2,
    base_weight: float = 0.7,
    graph_weight: float = 0.3,
    hop_decay: float = 0.65,
    relation_weights: dict[str, float] | None = None,
) -> Path | tuple[Path, Path, Path]:
    """Build index(es) for a retriever and persist them to disk."""
    if retriever.startswith("graph"):
        graph_manifest, base_out = _resolve_graph_out_paths(out)
        graph_manifest.parent.mkdir(parents=True, exist_ok=True)
        base_out.parent.mkdir(parents=True, exist_ok=True)

        effective_graph_path = (
            Path(graph_path)
            if graph_path
            else graph_manifest.parent / "medquad_graph.json"
        )
        if not effective_graph_path.exists():
            with timed_stage("Build MedQuAD metadata graph"):
                build_medquad_graph(corpus, effective_graph_path)

        effective_base = _infer_graph_base_retriever(retriever, base_retriever)
        base_index_result = run_index(
            retriever=effective_base,
            corpus=corpus,
            out=str(base_out),
            model=model,
            query_model=query_model,
            doc_model=doc_model,
            dense_backend=dense_backend,
            pooling=pooling,
            normalize=normalize,
            query_max_length=query_max_length,
            doc_max_length=doc_max_length,
            encode_batch_size=encode_batch_size,
            max_seq_length=max_seq_length,
            local_files_only=local_files_only,
            fusion=fusion,
            alpha=alpha,
            cand_k=cand_k,
            rrf_k=rrf_k,
        )
        base_index_path = (
            base_index_result[0]
            if isinstance(base_index_result, tuple)
            else base_index_result
        )

        manifest = {
            "format": "medical-rag-reranker.graph-expanded-index",
            "version": 1,
            "retriever": retriever,
            "base_retriever": effective_base,
            "base_index": _relpath(Path(base_index_path), graph_manifest.parent),
            "graph_path": _relpath(effective_graph_path, graph_manifest.parent),
            "seed_k": int(seed_k),
            "expand_k": int(expand_k),
            "max_hops": int(max_hops),
            "base_weight": float(base_weight),
            "graph_weight": float(graph_weight),
            "hop_decay": float(hop_decay),
            "relation_weights": _relation_weights_from_config(relation_weights),
        }
        with graph_manifest.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"Saved graph retriever manifest to: {graph_manifest}")
        print(f"Saved graph artifact to: {effective_graph_path}")
        return graph_manifest

    if retriever == "hybrid":
        manifest_path, bm25_path, dense_path = _resolve_hybrid_out_paths(out)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        dense_path.parent.mkdir(parents=True, exist_ok=True)

        bm25 = BM25Retriever()
        dense_backend = str(dense_backend).strip()
        if dense_backend == "bi_encoder":
            from medical_rag_reranker.retrieval.bi_encoder import BiEncoderRetriever

            dense = BiEncoderRetriever(
                query_model_name=str(query_model),
                doc_model_name=str(doc_model),
                pooling=str(pooling),
                normalize=bool(normalize),
                query_max_length=int(query_max_length),
                doc_max_length=int(doc_max_length),
                batch_size=int(encode_batch_size),
                local_files_only=bool(local_files_only),
            )
        elif dense_backend == "dense":
            try:
                from medical_rag_reranker.retrieval.dense import DenseRetriever
            except Exception as e:
                raise RuntimeError(
                    "Dense/hybrid index requires `sentence-transformers`. "
                    "Install dependencies or switch to retrieval=bm25."
                ) from e

            dense = DenseRetriever(
                model_name=model,
                batch_size=int(encode_batch_size),
                max_seq_length=max_seq_length,
            )
        else:
            raise ValueError(
                f"Unsupported hybrid dense_backend: {dense_backend!r}. "
                "Expected `dense` or `bi_encoder`."
            )

        with timed_stage("Build hybrid BM25 index"):
            bm25.index(corpus)
            bm25.save(str(bm25_path))

        with timed_stage("Build hybrid dense index"):
            dense.index(corpus)
            dense.save(str(dense_path))

        manifest = {
            "format": "medical-rag-reranker.hybrid-index",
            "version": 1,
            "retriever": "hybrid",
            "fusion": str(fusion),
            "alpha": float(alpha),
            "cand_k": int(cand_k),
            "rrf_k": int(rrf_k),
            "dense_backend": dense_backend,
            "corpus": str(corpus),
            "dense_model": str(model),
            "query_model": str(query_model),
            "doc_model": str(doc_model),
            "pooling": str(pooling),
            "normalize": bool(normalize),
            "query_max_length": int(query_max_length),
            "doc_max_length": int(doc_max_length),
            "bm25_index": _relpath(bm25_path, manifest_path.parent),
            "dense_index": _relpath(dense_path, manifest_path.parent),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"Saved hybrid manifest to: {manifest_path}")
        print(f"Saved bm25 index to: {bm25_path}")
        print(f"Saved dense index to: {dense_path}")
        return manifest_path, bm25_path, dense_path

    resolved_out = _resolve_out_path(retriever, out)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)

    if retriever == "bm25":
        retriever_impl = BM25Retriever()
    elif retriever == "dense":
        try:
            from medical_rag_reranker.retrieval.dense import DenseRetriever
        except Exception as e:
            raise RuntimeError(
                "Dense index requires `sentence-transformers`. "
                "Install dependencies or switch to retrieval=bm25."
            ) from e
        retriever_impl = DenseRetriever(
            model_name=model,
            batch_size=int(encode_batch_size),
            max_seq_length=max_seq_length,
        )
    elif retriever == "bi_encoder":
        from medical_rag_reranker.retrieval.bi_encoder import BiEncoderRetriever

        retriever_impl = BiEncoderRetriever(
            query_model_name=str(query_model),
            doc_model_name=str(doc_model),
            pooling=str(pooling),
            normalize=bool(normalize),
            query_max_length=int(query_max_length),
            doc_max_length=int(doc_max_length),
            batch_size=int(encode_batch_size),
            local_files_only=bool(local_files_only),
        )
    else:
        raise ValueError(f"Unknown retriever: {retriever}")

    with timed_stage(f"Build {retriever} index"):
        retriever_impl.index(corpus)
        retriever_impl.save(str(resolved_out))
    print(f"Saved index to: {resolved_out}")
    return resolved_out


def run_from_cfg(cfg: DictConfig) -> Path | tuple[Path, Path, Path]:
    """Hydra-config entrypoint used by `medical_rag_reranker.commands`."""
    run_cfg = cfg.run.retrieval_index
    retrieval_cfg = cfg.retrieval
    return run_index(
        retriever=str(retrieval_cfg.name),
        corpus=str(run_cfg.corpus),
        out=str(run_cfg.out),
        model=str(retrieval_cfg.get("model_name", run_cfg.model)),
        query_model=str(retrieval_cfg.get("query_model_name", run_cfg.query_model)),
        doc_model=str(retrieval_cfg.get("doc_model_name", run_cfg.doc_model)),
        dense_backend=str(retrieval_cfg.get("dense_backend", run_cfg.dense_backend)),
        pooling=str(retrieval_cfg.get("pooling", run_cfg.pooling)),
        normalize=bool(retrieval_cfg.get("normalize", run_cfg.normalize)),
        query_max_length=int(
            retrieval_cfg.get("query_max_length", run_cfg.query_max_length)
        ),
        doc_max_length=int(retrieval_cfg.get("doc_max_length", run_cfg.doc_max_length)),
        encode_batch_size=int(retrieval_cfg.get("batch_size", run_cfg.batch_size)),
        max_seq_length=(
            int(retrieval_cfg.max_seq_length)
            if retrieval_cfg.get("max_seq_length") is not None
            else None
        ),
        local_files_only=bool(
            retrieval_cfg.get("local_files_only", run_cfg.local_files_only)
        ),
        fusion=str(run_cfg.fusion),
        alpha=float(run_cfg.alpha),
        cand_k=int(run_cfg.cand_k),
        rrf_k=int(run_cfg.rrf_k),
        graph_path=str(retrieval_cfg.get("graph_path", "") or ""),
        base_retriever=str(retrieval_cfg.get("base_retriever", "") or ""),
        seed_k=int(retrieval_cfg.get("seed_k", 20)),
        expand_k=int(retrieval_cfg.get("expand_k", 50)),
        max_hops=int(retrieval_cfg.get("max_hops", 2)),
        base_weight=float(retrieval_cfg.get("base_weight", 0.7)),
        graph_weight=float(retrieval_cfg.get("graph_weight", 0.3)),
        hop_decay=float(retrieval_cfg.get("hop_decay", 0.65)),
        relation_weights=retrieval_cfg.get("relation_weights"),
    )


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
    p.add_argument(
        "--retriever",
        choices=[
            "bm25",
            "dense",
            "bi_encoder",
            "hybrid",
            "graph_bm25",
            "graph_hybrid",
            "graph_hybrid_medcpt",
        ],
        required=True,
    )
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
    p.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="only for dense",
    )
    p.add_argument("--query-model", default="ncbi/MedCPT-Query-Encoder")
    p.add_argument("--doc-model", default="ncbi/MedCPT-Article-Encoder")
    p.add_argument(
        "--dense-backend",
        choices=["dense", "bi_encoder"],
        default="dense",
        help="only for hybrid",
    )
    p.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--query-max-length", type=int, default=64)
    p.add_argument("--doc-max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument(
        "--fusion",
        choices=["score", "rrf"],
        default="score",
        help="only for hybrid",
    )
    p.add_argument("--alpha", type=float, default=0.5, help="only for hybrid")
    p.add_argument("--cand-k", type=int, default=50, help="only for hybrid")
    p.add_argument("--rrf-k", type=int, default=60, help="only for hybrid")
    p.add_argument("--graph-path", default=None, help="only for graph retrievers")
    p.add_argument("--base-retriever", default=None, help="only for graph retrievers")
    p.add_argument("--seed-k", type=int, default=20, help="only for graph retrievers")
    p.add_argument("--expand-k", type=int, default=50, help="only for graph retrievers")
    p.add_argument("--max-hops", type=int, default=2, help="only for graph retrievers")
    p.add_argument(
        "--base-weight", type=float, default=0.7, help="only for graph retrievers"
    )
    p.add_argument(
        "--graph-weight", type=float, default=0.3, help="only for graph retrievers"
    )
    p.add_argument(
        "--hop-decay", type=float, default=0.65, help="only for graph retrievers"
    )
    args = p.parse_args()

    run_index(
        retriever=args.retriever,
        corpus=args.corpus,
        out=args.out,
        model=args.model,
        query_model=args.query_model,
        doc_model=args.doc_model,
        dense_backend=args.dense_backend,
        pooling=args.pooling,
        normalize=args.normalize,
        query_max_length=args.query_max_length,
        doc_max_length=args.doc_max_length,
        encode_batch_size=args.batch_size,
        local_files_only=args.local_files_only,
        fusion=args.fusion,
        alpha=args.alpha,
        cand_k=args.cand_k,
        rrf_k=args.rrf_k,
        graph_path=args.graph_path,
        base_retriever=args.base_retriever,
        seed_k=args.seed_k,
        expand_k=args.expand_k,
        max_hops=args.max_hops,
        base_weight=args.base_weight,
        graph_weight=args.graph_weight,
        hop_decay=args.hop_decay,
    )


if __name__ == "__main__":
    main()
