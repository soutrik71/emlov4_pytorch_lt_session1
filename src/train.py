import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from datamodules.dogbreed_module import DogBreedImageDataModule
from models.dogbreed_classifier import DogBreedClassifier
from utils.logging_utils import setup_logger, task_wrapper


@task_wrapper
def train_and_save(
    data_module: DogBreedImageDataModule,
    model: DogBreedClassifier,
    trainer: L.Trainer,
    save_path: str | None = None,
):
    """Train the model and optionally save it."""
    # PyTorch Lightning will automatically call prepare_data() once
    # during trainer.fit() - no need to call it manually

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

    # Save the model if path is provided
    if save_path:
        model.save_model(save_path)
        print(f"Model saved to {save_path}")


class CustomModelCheckpoint(ModelCheckpoint):
    """Custom checkpoint callback with additional features."""

    def _save_checkpoint(self, trainer, filepath):
        # Remove the problematic attribute assignment
        super()._save_checkpoint(trainer, filepath)
        print(f"Checkpoint saved: {filepath}")


def main(args):
    """Main training function."""
    # Set up paths
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / args.data
    log_dir = base_dir / args.logs
    ckpt_path = base_dir / args.ckpt_path

    # Create directories if they don't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Set up logger
    setup_logger(log_dir / "train_log.log")

    # Initialize DataModule with optimized parameters
    data_module = DogBreedImageDataModule(
        dl_path=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_ref=args.dataset_ref,
        split_ratio=args.split_ratio,
    )

    # Initialize Model with optimized parameters
    model = DogBreedClassifier(
        num_classes=args.num_classes,
        model_name=args.model_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        compile_model=args.compile_model,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
    )

    # Set up callbacks
    checkpoint_callback = CustomModelCheckpoint(
        dirpath=ckpt_path,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=2)

    # Determine the best accelerator (prioritize MPS on macOS)
    import torch

    if torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        print("Using MPS (Metal Performance Shaders) acceleration")
    elif torch.cuda.is_available():
        accelerator = "gpu"
        devices = "auto"
        print("Using CUDA GPU acceleration")
    else:
        accelerator = "cpu"
        devices = "auto"
        print("Using CPU")

    # Configure precision based on accelerator
    if accelerator == "mps":
        # MPS doesn't support 16-bit mixed precision properly
        precision = "32-true"
    else:
        precision = args.precision

    # Initialize Trainer with optimized settings
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            lr_monitor,
            progress_bar,
            model_summary,
        ],
        accelerator=accelerator,
        devices=devices,
        logger=TensorBoardLogger(save_dir=log_dir, name="dogbreed_classification"),
        gradient_clip_val=args.gradient_clip_val,
        precision=precision,
        deterministic=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train and test the model
    save_path = ckpt_path / "final_model.ckpt" if args.save_model else None
    train_and_save(data_module, model, trainer, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DogBreed Classifier with PyTorch Lightning")

    # Data arguments
    parser.add_argument("--data", type=str, default="data", help="Path to data directory")
    parser.add_argument(
        "--dataset_ref", type=str, default="khushikhushikhushi/dog-breed-image-dataset", help="Kaggle dataset reference"
    )
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/validation split ratio")

    # Model arguments
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model architecture name")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"], help="Optimizer type"
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"], help="Learning rate scheduler"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")

    # Training arguments
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision")

    # System arguments
    parser.add_argument("--logs", type=str, default="logs", help="Path to logs directory")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints", help="Path to checkpoint directory")
    parser.add_argument("--compile_model", action="store_true", help="Compile model for better performance")
    parser.add_argument("--save_model", action="store_true", help="Save final model")

    args = parser.parse_args()
    main(args)
