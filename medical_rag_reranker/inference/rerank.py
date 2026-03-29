from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from medical_rag_reranker.models.reranker_module import CrossEncoderReranker


@dataclass
class CandidateDoc:
    doc_id: str
    text: str
    retrieval_score: float
    source: str | None = None


@dataclass
class RerankedDoc:
    doc_id: str
    text: str
    retrieval_score: float
    reranker_score: float
    source: str | None = None


class CrossEncoderBatchReranker:
    """Batch reranker built on top of the trained Cross-Encoder checkpoint."""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        max_length: int,
        batch_size: int = 16,
        local_files_only: bool = False,
    ) -> None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Reranker checkpoint does not exist: {checkpoint}")

        self.model_name = str(model_name)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.local_files_only = bool(local_files_only)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self.model = CrossEncoderReranker.load_from_checkpoint(
            str(checkpoint),
            map_location=self.device,
        )
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_cfg(
        cls,
        cfg: DictConfig,
        checkpoint_path: str,
        batch_size: int = 16,
    ) -> "CrossEncoderBatchReranker":
        """Build reranker from the shared Hydra config and a checkpoint path."""
        return cls(
            model_name=str(cfg.model.model_name),
            checkpoint_path=checkpoint_path,
            max_length=int(cfg.model.max_length),
            batch_size=int(batch_size),
            local_files_only=bool(getattr(cfg.generation, "local_files_only", False)),
        )

    def rerank(
        self,
        question: str,
        candidates: Sequence[CandidateDoc],
        top_k: int | None = None,
    ) -> tuple[list[RerankedDoc], float]:
        """Score candidate docs for one question and return sorted results."""
        if not candidates:
            return [], 0.0

        start = perf_counter()
        reranked: list[RerankedDoc] = []

        for start_idx in range(0, len(candidates), self.batch_size):
            batch = list(candidates[start_idx : start_idx + self.batch_size])
            encoded = self.tokenizer(
                [question] * len(batch),
                [doc.text for doc in batch],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    token_type_ids=encoded.get("token_type_ids"),
                )
                scores = torch.sigmoid(logits).detach().cpu().tolist()

            for doc, score in zip(batch, scores):
                reranked.append(
                    RerankedDoc(
                        doc_id=doc.doc_id,
                        text=doc.text,
                        retrieval_score=float(doc.retrieval_score),
                        reranker_score=float(score),
                        source=doc.source,
                    )
                )

        reranked.sort(key=lambda item: item.reranker_score, reverse=True)
        if top_k is not None:
            reranked = reranked[: int(top_k)]

        elapsed_ms = (perf_counter() - start) * 1000.0
        return reranked, float(elapsed_ms)
