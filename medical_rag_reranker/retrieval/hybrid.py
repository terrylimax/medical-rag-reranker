from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

from . import Retriever, ScoredDoc

def minmax(scores: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Min-max normalize scores to [0, 1].

    If `mask` is provided, statistics are computed only on `scores[mask]`.
    Elements where mask is False will be set to 0.0 in the output.
    mask — булев массив той же длины, что и scores. 
    True означает “этот элемент реально пришёл из данного ретривера”, False — “это заглушка (missing)”.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if mask is None:
        mn, mx = float(scores.min()), float(scores.max())
        if mx - mn < 1e-9:
            return np.zeros_like(scores, dtype=np.float32)
        return (scores - mn) / (mx - mn)

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != scores.shape:
        raise ValueError("mask must have the same shape as scores")
    if not mask.any(): # если нет ни одного True
        return np.zeros_like(scores, dtype=np.float32)

    masked_scores = scores[mask]
    mn, mx = float(masked_scores.min()), float(masked_scores.max())
    if mx - mn < 1e-9:
        out = np.zeros_like(scores, dtype=np.float32)
        out[mask] = 0.0
        return out

    out = np.zeros_like(scores, dtype=np.float32)
    out[mask] = (scores[mask] - mn) / (mx - mn)
    return out

@dataclass
class HybridRetriever(Retriever):
    bm25: Retriever
    dense: Retriever
    alpha: float = 0.5
    cand_k: int = 50  # candidates per retriever

    def retrieve(self, query: str, top_k: int) -> List[ScoredDoc]:
        a = self.bm25.retrieve(query, self.cand_k)
        b = self.dense.retrieve(query, self.cand_k)

        # merge by doc_id
        all_ids = list({d.doc_id for d in a} | {d.doc_id for d in b})
        bm = {d.doc_id: d.score for d in a}
        de = {d.doc_id: d.score for d in b}

        bm_scores = np.array([bm.get(i, 0.0) for i in all_ids], dtype=np.float32)
        de_scores = np.array([de.get(i, 0.0) for i in all_ids], dtype=np.float32)

        bm_present = np.array([i in bm for i in all_ids], dtype=bool)
        de_present = np.array([i in de for i in all_ids], dtype=bool)

        bm_n = minmax(bm_scores, mask=bm_present)
        de_n = minmax(de_scores, mask=de_present)

        s = self.alpha * bm_n + (1.0 - self.alpha) * de_n
        idx = np.argsort(-s)[:top_k]
        return [ScoredDoc(all_ids[i], float(s[i])) for i in idx]