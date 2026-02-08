from dataclasses import dataclass
from typing import List


@dataclass
class ScoredDoc:
    doc_id: str
    score: float


class Retriever:
    def index(self, corpus_path: str) -> None: ...
    def retrieve(self, query: str, top_k: int) -> List[ScoredDoc]: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str): ...
