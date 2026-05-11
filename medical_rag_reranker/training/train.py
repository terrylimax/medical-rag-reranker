from __future__ import annotations

from omegaconf import DictConfig

import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import MLFlowLogger

from medical_rag_reranker.data.datamodule import RerankerDataModule
from medical_rag_reranker.models.reranker_module import CrossEncoderReranker
from medical_rag_reranker.utils.git import get_git_commit_id


def train_from_cfg(cfg: DictConfig) -> None:
    """Train the reranker using a Hydra/OmegaConf config."""
    print("Reranker training: initialize seed", flush=True)
    pl.seed_everything(int(cfg.train.seed), workers=True)

    print("Reranker training: initialize MLflow logger", flush=True)
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
        "data.processed_dir": str(cfg.data.processed_dir),
        "data.prefer_prepared_artifacts": bool(cfg.data.prefer_prepared_artifacts),
        "model.model_name": str(cfg.model.model_name),
        "model.max_length": int(cfg.model.max_length),
        "model.negatives_per_query": int(cfg.model.negatives_per_query),
        "train.seed": int(cfg.train.seed),
        "train.batch_size": int(cfg.train.batch_size),
        "train.num_workers": int(cfg.train.num_workers),
        "train.hard_negative_pool_size": int(cfg.train.hard_negative_pool_size),
        "train.lr": float(cfg.train.lr),
        "train.weight_decay": float(cfg.train.weight_decay),
        "train.max_epochs": int(cfg.train.max_epochs),
        "train.accelerator": str(cfg.train.accelerator),
        "train.devices": int(cfg.train.devices),
        "train.limit_train_batches": float(
            getattr(cfg.train, "limit_train_batches", 1.0)
        ),
        "train.limit_val_batches": float(getattr(cfg.train, "limit_val_batches", 1.0)),
        "train.enable_progress_bar": bool(
            getattr(cfg.train, "enable_progress_bar", True)
        ),
        "train.progress_refresh_rate": int(
            getattr(cfg.train, "progress_refresh_rate", 10)
        ),
        "git.commit_id": get_git_commit_id(),
    }
    mlf_logger.log_hyperparams(hparams)
    print(
        "Reranker training config: "
        f"epochs={cfg.train.max_epochs}, "
        f"batch_size={cfg.train.batch_size}, "
        f"train_batches={getattr(cfg.train, 'limit_train_batches', 1.0)}, "
        f"val_batches={getattr(cfg.train, 'limit_val_batches', 1.0)}, "
        f"tracking_uri={cfg.logging.tracking_uri}",
        flush=True,
    )

    print("Reranker training: build datamodule", flush=True)
    datamodule = RerankerDataModule(
        raw_dir=str(cfg.data.raw_dir),
        processed_dir=str(cfg.data.processed_dir),
        model_name=str(cfg.model.model_name),
        max_length=int(cfg.model.max_length),
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        negatives_per_query=int(cfg.model.negatives_per_query),
        prefer_prepared_artifacts=bool(cfg.data.prefer_prepared_artifacts),
        hard_negative_pool_size=int(cfg.train.hard_negative_pool_size),
        seed=int(cfg.train.seed),
    )

    print("Reranker training: load cross-encoder model", flush=True)
    model = CrossEncoderReranker(
        model_name=str(cfg.model.model_name),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    callbacks = []
    if bool(getattr(cfg.train, "enable_progress_bar", True)):
        callbacks.append(
            TQDMProgressBar(
                refresh_rate=int(getattr(cfg.train, "progress_refresh_rate", 10))
            )
        )

    print("Reranker training: initialize trainer", flush=True)
    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator=str(cfg.train.accelerator),
        devices=int(cfg.train.devices),
        gradient_clip_val=float(cfg.train.gradient_clip_val),
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        limit_train_batches=float(getattr(cfg.train, "limit_train_batches", 1.0)),
        limit_val_batches=float(getattr(cfg.train, "limit_val_batches", 1.0)),
        logger=mlf_logger,
        enable_progress_bar=bool(getattr(cfg.train, "enable_progress_bar", True)),
        callbacks=callbacks,
    )

    print("Reranker training: start Trainer.fit", flush=True)
    trainer.fit(model, datamodule=datamodule)
    print("Reranker training: Trainer.fit completed", flush=True)
    checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
    best_path = getattr(checkpoint_callback, "best_model_path", "") or ""
    last_path = getattr(checkpoint_callback, "last_model_path", "") or ""
    if best_path:
        print(f"Best reranker checkpoint: {best_path}")
    elif last_path:
        print(f"Last reranker checkpoint: {last_path}")
