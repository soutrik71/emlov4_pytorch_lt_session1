# Evaluation Script

The evaluation script (`eval.py`) is used to assess the performance of a trained dog breed classification model on the test dataset.

## Overview

This script loads a trained model checkpoint and evaluates its performance on the test dataset, providing detailed metrics including:

- Test Loss
- Test Accuracy
- Test F1 Score
- Test Precision
- Test Recall

## Key Components

### Model Evaluation

```python
def evaluate_model(ckpt_path, data_dir, batch_size=32, num_workers=4):
    """Evaluate a trained model on the test dataset."""
    # Set seed for reproducibility
    seed_everything(42)

    # Load the model from checkpoint
    print(f"Loading model from checkpoint: {ckpt_path}")
    model = DogBreedClassifier.load_from_checkpoint(ckpt_path)
    model.eval()  # Set the model to evaluation mode

    # Initialize the data module
    data_module = DogBreedImageDataModule(
        dl_path=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Initialize a trainer for evaluation
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
```

### Hardware Acceleration

Like the training script, the evaluation script automatically detects and uses the best available hardware:

```python
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
```

### Checkpoint Management

The script includes a function to automatically find the best checkpoint based on validation loss:

```python
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
```

## Command-line Arguments

The script supports the following command-line arguments:

| Argument | Description |
|----------|-------------|
| `--data` | Path to data directory |
| `--ckpt_path` | Path to specific model checkpoint |
| `--ckpt_dir` | Path to checkpoint directory (auto-finds best) |
| `--batch_size` | Batch size for evaluation |
| `--num_workers` | Number of data loader workers |

## Usage

Basic usage:

```bash
python src/eval.py
```

With custom parameters:

```bash
python src/eval.py --ckpt_path checkpoints/epoch=09-val_loss=0.51.ckpt --batch_size 64
```

## Code Reference

```11:13:src/eval.py
def evaluate_model(ckpt_path, data_dir, batch_size=32, num_workers=4):
    """Evaluate a trained model on the test dataset."""
```

```55:67:src/eval.py
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
``` 