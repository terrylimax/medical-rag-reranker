import json
import pickle
import numpy as np
from dataclasses import dataclass
from typing import List
from sentence_transformers import SentenceTransformer

from . import Retriever, ScoredDoc

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

@dataclass
class DenseRetriever(Retriever):
    model_name: str
    model: SentenceTransformer = None
    doc_ids: List[str] = None
    emb: np.ndarray = None  # (N, D) normalized

    def __post_init__(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def index(self, corpus_path: str) -> None:
        texts, doc_ids = [], []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                doc_ids.append(r["doc_id"])
                texts.append(r["text"])
        E = self.model.encode(texts, batch_size=64, show_progress_bar=True)
        E = np.asarray(E, dtype=np.float32)
        self.emb = l2_normalize(E)
        self.doc_ids = doc_ids

    def retrieve(self, query: str, top_k: int) -> List[ScoredDoc]:
        q = self.model.encode([query])
        q = np.asarray(q, dtype=np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        scores = (self.emb @ q.T).squeeze(1)  #(N, 1) --> (N,)
        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [ScoredDoc(self.doc_ids[i], float(scores[i])) for i in idx]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"model_name": self.model_name, "doc_ids": self.doc_ids, "emb": self.emb},
                f,
            )

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        inst = cls(model_name=obj["model_name"])
        inst.doc_ids = obj["doc_ids"]
        inst.emb = obj["emb"]
        return inst