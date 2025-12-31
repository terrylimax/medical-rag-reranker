from __future__ import annotations

from typing import Optional

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from medical_rag_reranker.models.reranker_module import CrossEncoderReranker


def infer_from_cfg(
    cfg: DictConfig,
    query: str,
    document: str,
    checkpoint_path: Optional[str] = None,
) -> float:
    """Run a single (query, document) scoring pass.

    If checkpoint_path is provided, loads Lightning checkpoint weights.
    """
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model.model_name))
    encoded = tokenizer(
        query,
        document,
        truncation=True,
        max_length=int(cfg.model.max_length),
        padding="max_length",
        return_tensors="pt",
    )
    batch = {k: v for k, v in encoded.items()}

    if checkpoint_path:
        model = CrossEncoderReranker.load_from_checkpoint(checkpoint_path)
    else:
        model = CrossEncoderReranker(model_name=str(cfg.model.model_name))

    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        score = torch.sigmoid(logits).item()

    return float(score)
