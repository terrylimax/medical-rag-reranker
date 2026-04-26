from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from medical_rag_reranker.retrieval.bi_encoder import _pool_embeddings


class RetrieverTrainingDataset(Dataset):
    """Rows with one positive document and zero or more explicit negatives."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows = self._read_rows(self.path)
        if not self.rows:
            raise ValueError(f"No retriever training rows found in: {self.path}")

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                query = str(row.get("query") or "").strip()
                positive = str(row.get("positive_text") or "").strip()
                if not query or not positive:
                    continue
                negative_texts = [
                    str(text).strip()
                    for text in row.get("negative_texts", [])
                    if str(text).strip()
                ]
                copied = dict(row)
                copied["query"] = query
                copied["positive_text"] = positive
                copied["negative_texts"] = negative_texts
                rows.append(copied)
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def _collate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


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


def _resolve_device(device: str) -> torch.device:
    requested = str(device).strip().lower()
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _encode_texts(
    *,
    texts: list[str],
    tokenizer,
    model,
    max_length: int,
    device: torch.device,
    pooling: str,
    normalize: bool,
) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=int(max_length),
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = model(**encoded)
    embeddings = _pool_embeddings(output, encoded["attention_mask"], pooling)
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def _contrastive_step(
    *,
    rows: list[dict[str, Any]],
    query_tokenizer,
    doc_tokenizer,
    query_model,
    doc_model,
    device: torch.device,
    pooling: str,
    normalize: bool,
    query_max_length: int,
    doc_max_length: int,
    temperature: float,
) -> torch.Tensor:
    queries = [str(row["query"]) for row in rows]
    doc_texts = [str(row["positive_text"]) for row in rows]
    for row in rows:
        doc_texts.extend(str(text) for text in row.get("negative_texts", []))

    query_emb = _encode_texts(
        texts=queries,
        tokenizer=query_tokenizer,
        model=query_model,
        max_length=int(query_max_length),
        device=device,
        pooling=pooling,
        normalize=normalize,
    )
    doc_emb = _encode_texts(
        texts=doc_texts,
        tokenizer=doc_tokenizer,
        model=doc_model,
        max_length=int(doc_max_length),
        device=device,
        pooling=pooling,
        normalize=normalize,
    )
    if query_emb.shape[1] != doc_emb.shape[1]:
        raise ValueError(
            "Query and document encoder embedding dimensions do not match: "
            f"{query_emb.shape[1]} != {doc_emb.shape[1]}"
        )

    logits = torch.matmul(query_emb, doc_emb.T) / float(temperature)
    labels = torch.arange(len(rows), device=device)
    return F.cross_entropy(logits, labels)


def _run_train_epoch(
    *,
    loader: DataLoader,
    query_tokenizer,
    doc_tokenizer,
    query_model,
    doc_model,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pooling: str,
    normalize: bool,
    query_max_length: int,
    doc_max_length: int,
    temperature: float,
    max_grad_norm: float,
    log_every_n_steps: int,
    epoch: int,
) -> float:
    query_model.train()
    doc_model.train()
    total_loss = 0.0
    total_examples = 0

    for step, rows in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = _contrastive_step(
            rows=rows,
            query_tokenizer=query_tokenizer,
            doc_tokenizer=doc_tokenizer,
            query_model=query_model,
            doc_model=doc_model,
            device=device,
            pooling=pooling,
            normalize=normalize,
            query_max_length=query_max_length,
            doc_max_length=doc_max_length,
            temperature=temperature,
        )
        loss.backward()
        if float(max_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(
                list(query_model.parameters()) + list(doc_model.parameters()),
                float(max_grad_norm),
            )
        optimizer.step()

        batch_size = len(rows)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        if int(log_every_n_steps) > 0 and step % int(log_every_n_steps) == 0:
            avg = total_loss / max(1, total_examples)
            print(f"epoch={epoch} step={step} train_loss={avg:.6f}")

    return total_loss / max(1, total_examples)


@torch.no_grad()
def _run_eval_epoch(
    *,
    loader: DataLoader | None,
    query_tokenizer,
    doc_tokenizer,
    query_model,
    doc_model,
    device: torch.device,
    pooling: str,
    normalize: bool,
    query_max_length: int,
    doc_max_length: int,
    temperature: float,
) -> float | None:
    if loader is None:
        return None

    query_model.eval()
    doc_model.eval()
    total_loss = 0.0
    total_examples = 0
    for rows in loader:
        loss = _contrastive_step(
            rows=rows,
            query_tokenizer=query_tokenizer,
            doc_tokenizer=doc_tokenizer,
            query_model=query_model,
            doc_model=doc_model,
            device=device,
            pooling=pooling,
            normalize=normalize,
            query_max_length=query_max_length,
            doc_max_length=doc_max_length,
            temperature=temperature,
        )
        batch_size = len(rows)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size

    return total_loss / max(1, total_examples)


def train_retriever(
    *,
    train_path: str,
    val_path: str | None,
    output_dir: str,
    query_model_name: str,
    doc_model_name: str,
    pooling: str = "cls",
    normalize: bool = True,
    query_max_length: int = 64,
    doc_max_length: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    epochs: int = 1,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    temperature: float = 0.05,
    max_grad_norm: float = 1.0,
    device: str = "auto",
    local_files_only: bool = False,
    seed: int = 42,
    log_every_n_steps: int = 20,
) -> dict[str, Any]:
    _seed_everything(int(seed))
    resolved_device = _resolve_device(device)
    normalize = _as_bool(normalize, default=True)
    local_files_only = _as_bool(local_files_only, default=False)

    train_dataset = RetrieverTrainingDataset(train_path)
    val_dataset: RetrieverTrainingDataset | None = None
    if val_path and Path(val_path).exists():
        val_dataset = RetrieverTrainingDataset(val_path)

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=_collate_rows,
        generator=generator,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            collate_fn=_collate_rows,
        )
        if val_dataset is not None
        else None
    )

    query_tokenizer = AutoTokenizer.from_pretrained(
        query_model_name,
        local_files_only=local_files_only,
    )
    doc_tokenizer = AutoTokenizer.from_pretrained(
        doc_model_name,
        local_files_only=local_files_only,
    )
    query_model = AutoModel.from_pretrained(
        query_model_name,
        local_files_only=local_files_only,
    ).to(resolved_device)
    doc_model = AutoModel.from_pretrained(
        doc_model_name,
        local_files_only=local_files_only,
    ).to(resolved_device)

    optimizer = torch.optim.AdamW(
        list(query_model.parameters()) + list(doc_model.parameters()),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    history: list[dict[str, float | int | None]] = []
    for epoch in range(1, int(epochs) + 1):
        train_loss = _run_train_epoch(
            loader=train_loader,
            query_tokenizer=query_tokenizer,
            doc_tokenizer=doc_tokenizer,
            query_model=query_model,
            doc_model=doc_model,
            optimizer=optimizer,
            device=resolved_device,
            pooling=str(pooling),
            normalize=normalize,
            query_max_length=int(query_max_length),
            doc_max_length=int(doc_max_length),
            temperature=float(temperature),
            max_grad_norm=float(max_grad_norm),
            log_every_n_steps=int(log_every_n_steps),
            epoch=epoch,
        )
        val_loss = _run_eval_epoch(
            loader=val_loader,
            query_tokenizer=query_tokenizer,
            doc_tokenizer=doc_tokenizer,
            query_model=query_model,
            doc_model=doc_model,
            device=resolved_device,
            pooling=str(pooling),
            normalize=normalize,
            query_max_length=int(query_max_length),
            doc_max_length=int(doc_max_length),
            temperature=float(temperature),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": None if val_loss is None else float(val_loss),
            }
        )
        msg = f"epoch={epoch} train_loss={train_loss:.6f}"
        if val_loss is not None:
            msg += f" val_loss={val_loss:.6f}"
        print(msg)

    output_root = Path(output_dir)
    query_output_dir = output_root / "query_encoder"
    doc_output_dir = output_root / "doc_encoder"
    query_output_dir.mkdir(parents=True, exist_ok=True)
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    query_model.save_pretrained(query_output_dir)
    query_tokenizer.save_pretrained(query_output_dir)
    doc_model.save_pretrained(doc_output_dir)
    doc_tokenizer.save_pretrained(doc_output_dir)

    summary = {
        "output_dir": str(output_root),
        "query_encoder_dir": str(query_output_dir),
        "doc_encoder_dir": str(doc_output_dir),
        "train_path": str(train_path),
        "val_path": None if val_path is None else str(val_path),
        "query_model_name": str(query_model_name),
        "doc_model_name": str(doc_model_name),
        "pooling": str(pooling),
        "normalize": normalize,
        "query_max_length": int(query_max_length),
        "doc_max_length": int(doc_max_length),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "temperature": float(temperature),
        "device": str(resolved_device),
        "history": history,
    }
    summary_path = output_root / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def train_retriever_from_cfg(cfg: DictConfig) -> dict[str, Any]:
    run_cfg = cfg.run.train_retriever
    return train_retriever(
        train_path=str(run_cfg.train_path),
        val_path=str(run_cfg.val_path),
        output_dir=str(run_cfg.output_dir),
        query_model_name=str(run_cfg.query_model_name),
        doc_model_name=str(run_cfg.doc_model_name),
        pooling=str(run_cfg.pooling),
        normalize=bool(run_cfg.normalize),
        query_max_length=int(run_cfg.query_max_length),
        doc_max_length=int(run_cfg.doc_max_length),
        batch_size=int(run_cfg.batch_size),
        num_workers=int(run_cfg.num_workers),
        epochs=int(run_cfg.epochs),
        lr=float(run_cfg.lr),
        weight_decay=float(run_cfg.weight_decay),
        temperature=float(run_cfg.temperature),
        max_grad_norm=float(run_cfg.max_grad_norm),
        device=str(run_cfg.device),
        local_files_only=bool(run_cfg.local_files_only),
        seed=int(run_cfg.seed),
        log_every_n_steps=int(run_cfg.log_every_n_steps),
    )


def resolved_config_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg.run.train_retriever, resolve=True)  # type: ignore[return-value]
