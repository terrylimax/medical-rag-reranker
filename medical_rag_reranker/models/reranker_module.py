from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch import nn
from torchmetrics.classification import BinaryAUROC, BinaryF1Score
from transformers import AutoModel


class CrossEncoderReranker(pl.LightningModule):
    def __init__(
        self, model_name: str, lr: float = 2e-5, weight_decay: float = 0.01
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.val_auroc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_type_ids is None:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        cls = out.last_hidden_state[:, 0]  # [CLS]
        logits = self.classifier(cls).squeeze(-1)
        return logits

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        model_inputs = {
            k: batch[k]
            for k in ("input_ids", "attention_mask", "token_type_ids")
            if k in batch
        }
        logits = self(**model_inputs)
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        model_inputs = {
            k: batch[k]
            for k in ("input_ids", "attention_mask", "token_type_ids")
            if k in batch
        }
        logits = self(**model_inputs)
        loss = self.loss_fn(logits, batch["labels"])
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        probs = torch.sigmoid(logits)
        targets = batch["labels"].int()
        self.val_auroc.update(probs, targets)
        self.val_f1.update(probs, targets)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log(
            "val/auroc",
            self.val_auroc.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/f1", self.val_f1.compute(), prog_bar=True, on_step=False, on_epoch=True
        )
        self.val_auroc.reset()
        self.val_f1.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
