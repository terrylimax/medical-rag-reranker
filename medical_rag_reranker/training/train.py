from __future__ import annotations

from omegaconf import DictConfig

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger

from medical_rag_reranker.data.datamodule import RerankerDataModule
from medical_rag_reranker.models.reranker_module import CrossEncoderReranker
from medical_rag_reranker.utils.git import get_git_commit_id


def train_from_cfg(cfg: DictConfig) -> None:
    """Train the reranker using a Hydra/OmegaConf config."""
    pl.seed_everything(int(cfg.train.seed), workers=True)

    mlf_logger = MLFlowLogger(
        tracking_uri=str(cfg.logging.tracking_uri),
        experiment_name=str(cfg.logging.experiment_name),
        run_name=None
        if cfg.logging.run_name in (None, "null")
        else str(cfg.logging.run_name),
    )

    # Log hyperparameters and code version
    hparams = {
        "data.raw_dir": str(cfg.data.raw_dir),
        "model.model_name": str(cfg.model.model_name),
        "model.max_length": int(cfg.model.max_length),
        "model.negatives_per_query": int(cfg.model.negatives_per_query),
        "train.seed": int(cfg.train.seed),
        "train.batch_size": int(cfg.train.batch_size),
        "train.num_workers": int(cfg.train.num_workers),
        "train.lr": float(cfg.train.lr),
        "train.weight_decay": float(cfg.train.weight_decay),
        "train.max_epochs": int(cfg.train.max_epochs),
        "train.accelerator": str(cfg.train.accelerator),
        "train.devices": int(cfg.train.devices),
        "train.limit_train_batches": float(
            getattr(cfg.train, "limit_train_batches", 1.0)
        ),
        "train.limit_val_batches": float(getattr(cfg.train, "limit_val_batches", 1.0)),
        "git.commit_id": get_git_commit_id(),
    }
    mlf_logger.log_hyperparams(hparams)

    datamodule = RerankerDataModule(
        raw_dir=str(cfg.data.raw_dir),
        model_name=str(cfg.model.model_name),
        max_length=int(cfg.model.max_length),
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        negatives_per_query=int(cfg.model.negatives_per_query),
        seed=int(cfg.train.seed),
    )

    model = CrossEncoderReranker(
        model_name=str(cfg.model.model_name),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator=str(cfg.train.accelerator),
        devices=int(cfg.train.devices),
        gradient_clip_val=float(cfg.train.gradient_clip_val),
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        limit_train_batches=float(getattr(cfg.train, "limit_train_batches", 1.0)),
        limit_val_batches=float(getattr(cfg.train, "limit_val_batches", 1.0)),
        logger=mlf_logger,
    )

    trainer.fit(model, datamodule=datamodule)
