import argparse
from pathlib import Path

import torch
from lightning import seed_everything
from lightning.pytorch import Trainer

from datamodules.dogbreed_module import DogBreedImageDataModule
from models.dogbreed_classifier import DogBreedClassifier


def evaluate_model(ckpt_path, data_dir, batch_size=32, num_workers=4):
    """Evaluate a trained model on the test dataset."""
    # Set seed for reproducibility
    seed_everything(42)

    # Determine the best accelerator (same as train.py)
    if torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        precision = "32-true"  # MPS doesn't support 16-bit mixed precision
        print("Using MPS (Metal Performance Shaders) acceleration")
    elif torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        precision = "16-mixed"
        print("Using CUDA GPU acceleration")
    else:
        accelerator = "cpu"
        devices = 1
        precision = "32-true"
        print("Using CPU")

    # Load the model from checkpoint
    print(f"Loading model from checkpoint: {ckpt_path}")
    model = DogBreedClassifier.load_from_checkpoint(ckpt_path)
    model.eval()  # Set the model to evaluation mode

    # Initialize the data module (same parameters as train.py)
    data_module = DogBreedImageDataModule(
        dl_path=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Initialize a trainer with same config as train.py
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        enable_checkpointing=False,  # No checkpointing during evaluation
        enable_progress_bar=True,
        enable_model_summary=False,  # No need for model summary
        logger=False,  # No logging during evaluation
    )

    # Run the test set
    print("Running evaluation on test dataset...")
    test_results = trainer.test(model, datamodule=data_module)

    # Print detailed results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    if test_results and len(test_results) > 0:
        results = test_results[0]
        print(f"Test Loss: {results.get('test_loss', 'N/A'):.4f}")
        print(f"Test Accuracy: {results.get('test_accuracy', 'N/A'):.4f}")
        print(f"Test F1 Score: {results.get('test_f1', 'N/A'):.4f}")
        print(f"Test Precision: {results.get('test_precision', 'N/A'):.4f}")
        print(f"Test Recall: {results.get('test_recall', 'N/A'):.4f}")
    else:
        print("No test results available")

    print("=" * 50)
    return test_results


def find_best_checkpoint(ckpt_dir):
    """Find the best checkpoint based on validation loss."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Look for epoch checkpoints (these contain val_loss in filename)
    epoch_ckpts = list(ckpt_path.glob("epoch=*-val_loss=*.ckpt"))

    if epoch_ckpts:
        # Sort by validation loss (lower is better)
        epoch_ckpts.sort(key=lambda x: float(x.stem.split("val_loss=")[1]))
        best_ckpt = epoch_ckpts[0]
        print(f"Found best checkpoint: {best_ckpt}")
        return str(best_ckpt)

    # Fallback to last checkpoint
    last_ckpt = ckpt_path / "last.ckpt"
    if last_ckpt.exists():
        print(f"Using last checkpoint: {last_ckpt}")
        return str(last_ckpt)

    # Fallback to final model
    final_ckpt = ckpt_path / "final_model.ckpt"
    if final_ckpt.exists():
        print(f"Using final model: {final_ckpt}")
        return str(final_ckpt)

    raise FileNotFoundError(f"No valid checkpoint found in {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained DogBreed Classifier")
    parser.add_argument("--data", type=str, default="data", help="Path to data directory")
    parser.add_argument("--ckpt_path", type=str, help="Path to specific model checkpoint")
    parser.add_argument(
        "--ckpt_dir", type=str, default="../checkpoints", help="Path to checkpoint directory (auto-finds best)"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")

    args = parser.parse_args()

    # Determine checkpoint path
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = find_best_checkpoint(args.ckpt_dir)

    # Run evaluation
    evaluate_model(
        ckpt_path=ckpt_path,
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
