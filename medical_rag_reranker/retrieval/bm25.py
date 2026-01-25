import json
import gzip
from dataclasses import dataclass
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np

from . import Retriever, ScoredDoc

def simple_tokenize(text: str) -> List[str]:
    return text.lower().split()

@dataclass
class BM25Retriever(Retriever):
    bm25: BM25Okapi = None
    doc_ids: List[str] = None
    corpus_path: str = None

    def index(self, corpus_path: str) -> None:
        docs = []
        doc_ids = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                doc_ids.append(r["doc_id"])
                docs.append(simple_tokenize(r["text"]))
        self.doc_ids = doc_ids
        self.corpus_path = corpus_path
        self.bm25 = BM25Okapi(docs)

    def retrieve(self, query: str, top_k: int) -> List[ScoredDoc]:
        """Retrieve top-k documents for a query using BM25.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return.

        Purpose:
            Scores every document in the indexed corpus with BM25 and returns the best matches.
            If the retriever was loaded from disk in a compact form, the index may be rebuilt
            lazily on the first call (using the stored `corpus_path`).

        Returns:
            A list of `ScoredDoc(doc_id, score)` sorted by descending score. If `top_k <= 0`,
            returns an empty list.

        Raises:
            ValueError: If the retriever is not indexed and has no `corpus_path` to rebuild from,
                or if internal index state is missing/corrupted.
        """
        if self.bm25 is None:
            if not self.corpus_path:
                raise ValueError(
                    "BM25Retriever is not indexed yet. Call index(corpus_path) or load() from a saved retriever."
                )
            self.index(self.corpus_path)

        q = simple_tokenize(query)
        scores = self.bm25.get_scores(q)  # np.array
        # top_k indices
        if self.doc_ids is None:
            raise ValueError("BM25Retriever has no doc_ids. Rebuild the index by calling index().")

        k = min(int(top_k), len(self.doc_ids))
        if k <= 0:
            return []

        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [ScoredDoc(self.doc_ids[i], float(scores[i])) for i in idx]

    def save(self, path: str) -> None:
        if not self.corpus_path:
            raise ValueError(
                "BM25Retriever has no corpus_path. Call index(corpus_path) first so the retriever can be rebuilt."
            )

        payload = {
            "format": "bm25-retriever",
            "version": 2,
            "corpus_path": self.corpus_path,
            "tokenizer": "simple_tokenize",
            "doc_count": None if self.doc_ids is None else len(self.doc_ids),
        }

        if path.endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                obj = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)

        if obj.get("format") != "bm25-retriever":
            raise ValueError("Unsupported retriever format.")

        if int(obj.get("version", 0)) < 2:
            raise ValueError(
                "Unsupported BM25Retriever version. Re-save the retriever using the current code."
            )

        corpus_path = obj.get("corpus_path")
        if not corpus_path:
            raise ValueError("Saved retriever is missing corpus_path.")

        inst = cls()
        inst.corpus_path = str(corpus_path)
        # Index is rebuilt lazily on first retrieve() to keep load fast and the saved file small.
        inst.bm25 = None
        inst.doc_ids = None
        return inst