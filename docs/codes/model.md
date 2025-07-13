# Dog Breed Classifier Model

The `DogBreedClassifier` class is the core model implementation for the dog breed classification project. It leverages PyTorch Lightning for structured training and evaluation.

## Overview

This model is designed for multi-class image classification of dog breeds. It uses a pre-trained backbone from the `timm` library (PyTorch Image Models) and adds custom training logic with PyTorch Lightning.

## Architecture

The model architecture consists of:

1. **Pre-trained Backbone**: A model from the `timm` library (default: `resnet18`)
2. **Classification Head**: Fully connected layer for the final classification
3. **Metrics Collection**: Comprehensive metrics tracking for training, validation, and testing

## Key Features

### Model Initialization

```python
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
```

### Metrics Tracking

The model uses `MetricCollection` from `torchmetrics` to organize and track multiple metrics:

```python
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
```

### Training Logic

The model implements a shared step function to avoid code duplication:

```python
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
```

### Optimizer Configuration

The model supports multiple optimizers and learning rate schedulers:

```python
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
```

### Learning Rate Schedulers

The model supports different learning rate schedulers:

1. **Cosine Annealing**: Gradually reduces learning rate following a cosine curve
2. **Reduce on Plateau**: Reduces learning rate when metrics plateau

```python
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
```

### Transfer Learning Support

The model includes methods for transfer learning:

```python
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
```

## Code Reference

```11:42:src/models/dogbreed_classifier.py
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
```

```46:63:src/models/dogbreed_classifier.py
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
``` 