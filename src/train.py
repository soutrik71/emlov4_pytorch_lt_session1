import argparse
import os
import sys
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from datamodules.dogbreed_module import DogBreedImageDataModule
from models.dogbreed_classifier import DogBreedClassifier
from utils.logging_utils import setup_logger, task_wrapper


def is_docker_environment():
    """Check if running in Docker container."""
    return os.getenv("DOCKER_CONTAINER") == "1" or os.path.exists("/.dockerenv")


@task_wrapper
def train_and_save(
    data_module: DogBreedImageDataModule,
    model: DogBreedClassifier,
    trainer: L.Trainer,
    save_path: str | None = None,
):
    """Train the model and optionally save it."""
    # PyTorch Lightning will automatically call prepare_data() and setup()
    trainer.fit(model, data_module)

    # Save the model if path is provided
    if save_path:
        trainer.save_checkpoint(save_path)
        print(f"Model saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DogBreed Classifier")

    # Data arguments
    parser.add_argument("--data", type=str, default="data", help="Data directory path")
    parser.add_argument(
        "--dataset", type=str, default="khushikhushikhushi/dog-breed-image-dataset", help="Kaggle dataset reference"
    )
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/validation split ratio")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of data loader workers (0 for Docker compatibility)"
    )

    # Model arguments
    parser.add_argument("--model_name", type=str, default="efficientnet_b0", help="Model architecture")
    parser.add_argument("--num_classes", type=int, default=120, help="Number of classes")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")

    # Training arguments
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum epochs")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator type (auto, cpu, mps, cuda)")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--precision", type=str, default="32", help="Training precision (16, 32, bf16)")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation batches")

    # Logging and checkpointing
    parser.add_argument("--logs", type=str, default="logs", help="Logs directory")
    parser.add_argument("--checkpoints", type=str, default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--log_every_n_steps", type=int, default=10, help="Log every n steps")
    parser.add_argument("--save_top_k", type=int, default=3, help="Save top k checkpoints")

    # Early stopping
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Early stopping minimum delta")

    # Optimizer and scheduler
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer (adam, adamw, sgd)")
    parser.add_argument("--scheduler", type=str, default="cosine", help="Scheduler (cosine, plateau, none)")

    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save final model")

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.logs, exist_ok=True)
    os.makedirs(args.checkpoints, exist_ok=True)

    # Set seed for reproducibility
    L.seed_everything(args.seed)

    # Setup logger - handle None return gracefully
    log_file = Path(args.logs) / "training.log"
    logger = setup_logger(str(log_file))

    # Check if running in Docker
    in_docker = is_docker_environment()

    # Print configuration (always to stdout for Docker visibility)
    print("🚀 Starting training with configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    if logger:
        logger.info("Starting training with configuration:")
        for key, value in vars(args).items():
            logger.info(f"  {key}: {value}")

    if in_docker:
        print("🐳 Docker Environment Detected")
        print("📊 Training Configuration:")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        if logger:
            logger.info("🐳 Running in Docker environment - optimizing for container logging")

    # Auto-detect accelerator and precision for different environments
    if args.accelerator == "auto":
        if in_docker:
            # In Docker, prefer CPU for compatibility
            args.accelerator = "cpu"
            args.precision = "32"
            print("🐳 Docker: Using CPU accelerator with 32-bit precision")
            if logger:
                logger.info("🐳 Docker: Using CPU accelerator with 32-bit precision")
        else:
            # Local environment - try MPS for Apple Silicon
            try:
                import torch

                if torch.backends.mps.is_available():
                    args.accelerator = "mps"
                    args.precision = "32"  # MPS works better with 32-bit
                    print("🍎 Using MPS accelerator with 32-bit precision")
                    if logger:
                        logger.info("🍎 Using MPS accelerator with 32-bit precision")
                else:
                    args.accelerator = "cpu"
                    args.precision = "32"
                    print("💻 Using CPU accelerator with 32-bit precision")
                    if logger:
                        logger.info("💻 Using CPU accelerator with 32-bit precision")
            except ImportError:
                args.accelerator = "cpu"
                args.precision = "32"

    # Initialize data module
    data_module = DogBreedImageDataModule(
        dl_path=args.data,
        dataset_ref=args.dataset,
        split_ratio=args.split_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Initialize model
    model = DogBreedClassifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        compile_model=args.compile,
    )

    # Setup callbacks
    callbacks = []

    # Progress bar - use TQDM for Docker, Rich for local
    if in_docker:
        callbacks.append(TQDMProgressBar(refresh_rate=1))
        print("📊 Using TQDM progress bar for Docker compatibility")
    else:
        callbacks.append(RichProgressBar())
        callbacks.append(RichModelSummary(max_depth=2))

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoints,
        filename="epoch={epoch:02d}-val_loss={val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        min_delta=args.min_delta,
        mode="min",
        verbose=True,
    )
    callbacks.append(early_stopping)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Setup logger
    tb_logger = TensorBoardLogger(
        save_dir=args.logs,
        name="dogbreed_classifier",
        version=None,
    )

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=args.deterministic,
        enable_model_summary=not in_docker,  # Disable for Docker to reduce logs
        enable_progress_bar=True,
        enable_checkpointing=True,
    )

    # Print training info
    print("\n🚀 Starting Training:")
    print(f"  📁 Data: {args.data}")
    print(f"  🏗️  Model: {args.model_name}")
    print(f"  📊 Classes: {args.num_classes}")
    print(f"  🔄 Epochs: {args.max_epochs}")
    print(f"  ⚡ Accelerator: {args.accelerator}")
    print(f"  🎯 Precision: {args.precision}")
    print(f"  📦 Batch Size: {args.batch_size}")
    print(f"  📈 Learning Rate: {args.learning_rate}")
    print()

    # Ensure stdout is flushed for Docker
    sys.stdout.flush()

    # Train the model
    try:
        train_and_save(data_module, model, trainer, args.save_path)
        print("✅ Training completed successfully!")

        # Print final metrics
        if trainer.callback_metrics:
            print("\n📊 Final Metrics:")
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")

        # Print best checkpoint info
        if hasattr(checkpoint_callback, "best_model_path"):
            print(f"\n🏆 Best model saved at: {checkpoint_callback.best_model_path}")
            print(f"🎯 Best validation loss: {checkpoint_callback.best_model_score:.4f}")

    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"❌ {error_msg}")
        if logger:
            logger.error(error_msg)
        raise


if __name__ == "__main__":
    main()
