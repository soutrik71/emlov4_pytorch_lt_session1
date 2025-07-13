from typing import Any

import lightning as L
import timm
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


class DogBreedClassifier(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        model_name: str = "resnet18",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        compile_model: bool = True,
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compile_model = compile_model
        self.label_smoothing = label_smoothing

        # Load pre-trained model with timm API
        self.model = timm.create_model(
            model_name,
            pretrained=True,  # Use pretrained for timm 1.0.9
            num_classes=num_classes,
            drop_rate=dropout,
        )

        # Compile model for better performance (PyTorch 2.0+)
        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # Define metrics using MetricCollection for better organization
        metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
                "precision": Precision(task="multiclass", num_classes=num_classes, average="macro"),
                "recall": Recall(task="multiclass", num_classes=num_classes, average="macro"),
                "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
            }
        )

        # Create separate metric collections for each phase
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> dict[str, Any]:
        """Shared step for train/val/test to avoid code duplication."""
        x, y = batch
        logits = self(x)

        # Use label smoothing for better generalization
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        # Get predictions (no need for softmax for accuracy)
        preds = torch.argmax(logits, dim=1)

        return {
            "loss": loss,
            "preds": preds,
            "targets": y,
            "logits": logits,
        }

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._shared_step(batch, "train")
        loss = outputs["loss"]

        # Update metrics
        self.train_metrics.update(outputs["preds"], outputs["targets"])

        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        outputs = self._shared_step(batch, "val")
        loss = outputs["loss"]

        # Update metrics
        self.val_metrics.update(outputs["preds"], outputs["targets"])

        # Log loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        outputs = self._shared_step(batch, "test")
        loss = outputs["loss"]

        # Update metrics
        self.test_metrics.update(outputs["preds"], outputs["targets"])

        # Log loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def on_training_epoch_end(self) -> None:
        """Log training metrics at epoch end."""
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Log test metrics at epoch end."""
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        # Choose optimizer
        if self.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
            )
        elif self.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        # Configure scheduler
        if self.scheduler.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=100,  # Should be set based on max_epochs
                eta_min=self.lr * 0.01,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif self.scheduler.lower() == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=self.lr * 0.001,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def configure_gradient_clipping(
        self,
        optimizer: optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Configure gradient clipping for training stability."""
        if gradient_clip_val is not None:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Prediction step for inference."""
        x, _ = batch
        logits = self(x)
        return F.softmax(logits, dim=1)

    def save_model(self, path: str) -> None:
        """Save model state dict."""
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Load model state dict."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def freeze_backbone(self) -> None:
        """Freeze backbone for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze classifier head
        if hasattr(self.model, "classifier"):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "head"):
            for param in self.model.head.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for full training."""
        for param in self.model.parameters():
            param.requires_grad = True
