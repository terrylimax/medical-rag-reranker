from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from medical_rag_reranker.utils.progress import count_text_lines, progress

from . import Retriever, ScoredDoc


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off", ""):
        return False
    raise ValueError(f"Cannot parse boolean value from: {value!r}")


def _pool_embeddings(
    model_output,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    mode = str(pooling).strip().lower()
    if mode == "cls":
        return model_output.last_hidden_state[:, 0, :]
    if mode == "mean":
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.shape).float()
        return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(
            mask.sum(dim=1), min=1e-9
        )
    raise ValueError(
        f"Unsupported pooling mode: {pooling!r}. Expected `cls` or `mean`."
    )


@dataclass
class BiEncoderRetriever(Retriever):
    """Transformer bi-encoder retriever with separate query/document encoders.

    This supports biomedical asymmetric encoders such as MedCPT:
    - query encoder for questions/search queries
    - document/article encoder for indexed chunks
    """

    query_model_name: str
    doc_model_name: str
    pooling: str = "cls"
    normalize: bool = True
    query_max_length: int = 64
    doc_max_length: int = 256
    batch_size: int = 32
    local_files_only: bool = False
    query_prefix: str = ""
    doc_prefix: str = ""
    query_model: Any = None
    doc_model: Any = None
    query_tokenizer: Any = None
    doc_tokenizer: Any = None
    doc_ids: List[str] | None = None
    emb: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.normalize = _as_bool(self.normalize, default=True)
        self.local_files_only = _as_bool(self.local_files_only, default=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_query_encoder(self) -> None:
        if self.query_model is not None and self.query_tokenizer is not None:
            return
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            self.query_model_name,
            local_files_only=self.local_files_only,
        )
        self.query_model = AutoModel.from_pretrained(
            self.query_model_name,
            local_files_only=self.local_files_only,
        )
        self.query_model.to(self.device)
        self.query_model.eval()

    def _load_doc_encoder(self) -> None:
        if self.doc_model is not None and self.doc_tokenizer is not None:
            return
        self.doc_tokenizer = AutoTokenizer.from_pretrained(
            self.doc_model_name,
            local_files_only=self.local_files_only,
        )
        self.doc_model = AutoModel.from_pretrained(
            self.doc_model_name,
            local_files_only=self.local_files_only,
        )
        self.doc_model.to(self.device)
        self.doc_model.eval()

    def _encode_texts(
        self,
        *,
        texts: list[str],
        tokenizer,
        model,
        max_length: int,
        prefix: str = "",
        desc: str = "Encoding texts",
        show_progress: bool = True,
    ) -> np.ndarray:
        vectors: list[np.ndarray] = []
        starts = range(0, len(texts), int(self.batch_size))
        total_batches = (len(texts) + int(self.batch_size) - 1) // int(self.batch_size)
        batches = (
            progress(starts, desc=desc, total=total_batches, unit="batch")
            if show_progress
            else starts
        )
        for start in batches:
            batch_texts = [
                f"{prefix}{text}" if prefix else text
                for text in texts[start : start + int(self.batch_size)]
            ]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=int(max_length),
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                out = model(**encoded)
                pooled = _pool_embeddings(out, encoded["attention_mask"], self.pooling)
                if self.normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.detach().cpu().numpy().astype(np.float32))

        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)

    def index(self, corpus_path: str) -> None:
        self._load_doc_encoder()
        doc_ids: list[str] = []
        texts: list[str] = []
        total = count_text_lines(corpus_path)
        with open(corpus_path, "r", encoding="utf-8") as f:
            rows = progress(
                f,
                desc="Reading bi-encoder index corpus",
                total=total,
                unit="doc",
            )
            for line in rows:
                if not line.strip():
                    continue
                row = json.loads(line)
                doc_id = row.get("doc_id")
                text = str(row.get("text") or "").strip()
                if not doc_id or not text:
                    continue
                title = str(row.get("title") or "").strip()
                if title:
                    text = f"{title}\n{text}"
                doc_ids.append(str(doc_id))
                texts.append(text)

        self.doc_ids = doc_ids
        self.emb = self._encode_texts(
            texts=texts,
            tokenizer=self.doc_tokenizer,
            model=self.doc_model,
            max_length=int(self.doc_max_length),
            prefix=str(self.doc_prefix),
            desc="Encoding bi-encoder documents",
            show_progress=True,
        )

    def retrieve(self, query: str, top_k: int) -> List[ScoredDoc]:
        if self.emb is None or self.doc_ids is None:
            raise ValueError("BiEncoderRetriever is not indexed or loaded.")
        if int(top_k) <= 0:
            return []

        self._load_query_encoder()
        q = self._encode_texts(
            texts=[query],
            tokenizer=self.query_tokenizer,
            model=self.query_model,
            max_length=int(self.query_max_length),
            prefix=str(self.query_prefix),
            desc="Encoding bi-encoder query",
            show_progress=False,
        )
        if q.size == 0:
            return []

        scores = (self.emb @ q.T).squeeze(1)
        k = min(int(top_k), len(self.doc_ids))
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [ScoredDoc(self.doc_ids[i], float(scores[i])) for i in idx]

    def save(self, path: str) -> None:
        if self.doc_ids is None or self.emb is None:
            raise ValueError("BiEncoderRetriever has no index to save.")
        payload = {
            "format": "medical-rag-reranker.bi-encoder-index",
            "version": 1,
            "query_model_name": self.query_model_name,
            "doc_model_name": self.doc_model_name,
            "pooling": self.pooling,
            "normalize": bool(self.normalize),
            "query_max_length": int(self.query_max_length),
            "doc_max_length": int(self.doc_max_length),
            "batch_size": int(self.batch_size),
            "local_files_only": bool(self.local_files_only),
            "query_prefix": self.query_prefix,
            "doc_prefix": self.doc_prefix,
            "doc_ids": self.doc_ids,
            "emb": self.emb,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if obj.get("format") != "medical-rag-reranker.bi-encoder-index":
            raise ValueError("Unsupported bi-encoder index format.")
        inst = cls(
            query_model_name=str(obj["query_model_name"]),
            doc_model_name=str(obj["doc_model_name"]),
            pooling=str(obj.get("pooling", "cls")),
            normalize=bool(obj.get("normalize", True)),
            query_max_length=int(obj.get("query_max_length", 64)),
            doc_max_length=int(obj.get("doc_max_length", 256)),
            batch_size=int(obj.get("batch_size", 32)),
            local_files_only=bool(obj.get("local_files_only", False)),
            query_prefix=str(obj.get("query_prefix", "")),
            doc_prefix=str(obj.get("doc_prefix", "")),
        )
        inst.doc_ids = list(obj["doc_ids"])
        inst.emb = np.asarray(obj["emb"], dtype=np.float32)
        return inst
