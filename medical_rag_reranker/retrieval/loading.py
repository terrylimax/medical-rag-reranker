from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medical_rag_reranker.graph.builder import load_medquad_graph
from medical_rag_reranker.retrieval.bm25 import BM25Retriever
from medical_rag_reranker.retrieval.graph_expanded import (
    DEFAULT_RELATION_WEIGHTS,
    GraphExpandedRetriever,
)


HYBRID_FORMAT = "medical-rag-reranker.hybrid-index"
GRAPH_RETRIEVER_FORMAT = "medical-rag-reranker.graph-expanded-index"


def _resolve_manifest_path(index_arg: str | Path, expected_name: str) -> Path:
    p = Path(index_arg)
    if p.exists() and p.is_dir():
        candidate = p / expected_name
        if not candidate.exists():
            raise FileNotFoundError(f"Expected {expected_name} inside directory: {p}")
        return candidate
    return p


def _resolve_relative(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _load_hybrid_from_manifest(manifest_path: Path):
    try:
        from medical_rag_reranker.retrieval.hybrid import HybridRetriever
    except Exception as e:
        raise RuntimeError(
            "Hybrid retriever dependencies are missing. "
            "Install retrieval dependencies or switch to retrieval=bm25."
        ) from e

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("format") != HYBRID_FORMAT:
        raise ValueError("Unsupported hybrid manifest format.")

    base_dir = manifest_path.parent
    bm25_index = _resolve_relative(str(manifest["bm25_index"]), base_dir)
    dense_index = _resolve_relative(str(manifest["dense_index"]), base_dir)

    bm25 = BM25Retriever.load(str(bm25_index))
    dense_backend = str(manifest.get("dense_backend", "dense"))
    if dense_backend == "bi_encoder":
        from medical_rag_reranker.retrieval.bi_encoder import BiEncoderRetriever

        dense = BiEncoderRetriever.load(str(dense_index))
    elif dense_backend == "dense":
        from medical_rag_reranker.retrieval.dense import DenseRetriever

        dense = DenseRetriever.load(str(dense_index))
    else:
        raise ValueError(f"Unsupported hybrid dense_backend: {dense_backend!r}")

    return HybridRetriever(
        bm25=bm25,
        dense=dense,
        fusion=str(manifest.get("fusion", "score")),
        alpha=float(manifest.get("alpha", 0.5)),
        cand_k=int(manifest.get("cand_k", 50)),
        rrf_k=int(manifest.get("rrf_k", 60)),
    )


def _load_graph_from_manifest(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest: dict[str, Any] = json.load(f)

    if manifest.get("format") != GRAPH_RETRIEVER_FORMAT:
        raise ValueError("Unsupported graph retriever manifest format.")

    base_dir = manifest_path.parent
    base_retriever = str(manifest.get("base_retriever") or "bm25")
    base_index = _resolve_relative(str(manifest["base_index"]), base_dir)
    graph_path = _resolve_relative(str(manifest["graph_path"]), base_dir)

    base = load_retriever(base_retriever, str(base_index))
    graph = load_medquad_graph(graph_path)
    weights = dict(DEFAULT_RELATION_WEIGHTS)
    weights.update(manifest.get("relation_weights") or {})

    return GraphExpandedRetriever(
        base=base,
        graph=graph,
        seed_k=int(manifest.get("seed_k", 20)),
        expand_k=int(manifest.get("expand_k", 50)),
        max_hops=int(manifest.get("max_hops", 2)),
        base_weight=float(manifest.get("base_weight", 0.7)),
        graph_weight=float(manifest.get("graph_weight", 0.3)),
        hop_decay=float(manifest.get("hop_decay", 0.65)),
        relation_weights={str(k): float(v) for k, v in weights.items()},
    )


def load_retriever(retriever_name: str, index_path: str):
    if retriever_name == "bm25":
        return BM25Retriever.load(index_path)
    if retriever_name == "dense":
        try:
            from medical_rag_reranker.retrieval.dense import DenseRetriever
        except Exception as e:
            raise RuntimeError(
                "Dense retriever requires `sentence-transformers`. "
                "Install dependencies or switch to retrieval=bm25."
            ) from e
        return DenseRetriever.load(index_path)
    if retriever_name == "bi_encoder":
        from medical_rag_reranker.retrieval.bi_encoder import BiEncoderRetriever

        return BiEncoderRetriever.load(index_path)
    if retriever_name == "hybrid":
        manifest_path = _resolve_manifest_path(index_path, "hybrid_index.json")
        return _load_hybrid_from_manifest(manifest_path)
    if retriever_name.startswith("graph"):
        manifest_path = _resolve_manifest_path(index_path, "graph_index.json")
        return _load_graph_from_manifest(manifest_path)
    raise ValueError(f"Unsupported retriever: {retriever_name}")
