# Training Script

The training script (`train.py`) is the main entry point for training the dog breed classification model. It handles the complete training pipeline, from data preparation to model checkpointing.

## Overview

This script orchestrates the training process using PyTorch Lightning, handling:

- Command-line argument parsing
- Data preparation and loading
- Model initialization and configuration
- Training, validation, and testing loops
- Checkpoint management
- Hardware acceleration detection

## Key Components

### Hardware Acceleration

The script automatically detects and uses the best available hardware:

```python
if torch.backends.mps.is_available():
    accelerator = "mps"  # Apple Silicon GPU
    devices = 1
    precision = "32-true"  # MPS doesn't support 16-bit mixed precision
elif torch.cuda.is_available():
    accelerator = "gpu"  # NVIDIA GPU
    devices = "auto"
    precision = args.precision
else:
    accelerator = "cpu"
    devices = "auto"
    precision = args.precision
```

### Training Function

```python
@task_wrapper
def train_and_save(
    data_module: DogBreedImageDataModule,
    model: DogBreedClassifier,
    trainer: L.Trainer,
    save_path: str | None = None,
):
    """Train the model and optionally save it."""
    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

    # Save the model if path is provided
    if save_path:
        model.save_model(save_path)
        print(f"Model saved to {save_path}")
```

### Callbacks

The script configures several callbacks to enhance the training process:

1. **Custom Checkpoint Callback**: Saves model checkpoints based on validation loss
2. **Early Stopping**: Prevents overfitting by stopping training when validation metrics plateau
3. **Learning Rate Monitor**: Tracks learning rate changes during training
4. **Rich Progress Bar**: Provides detailed training progress visualization
5. **Model Summary**: Displays model architecture information

### Command-line Arguments

The script supports extensive customization through command-line arguments:

| Category | Arguments |
|----------|-----------|
| **Data** | `--data`, `--dataset_ref`, `--split_ratio` |
| **Model** | `--num_classes`, `--model_name`, `--lr`, `--weight_decay`, `--optimizer`, `--scheduler`, `--dropout`, `--label_smoothing` |
| **Training** | `--max_epochs`, `--batch_size`, `--num_workers`, `--patience`, `--gradient_clip_val`, `--precision` |
| **System** | `--logs`, `--ckpt_path`, `--compile_model`, `--save_model` |

## Usage

Basic usage:

```bash
python src/train.py
```

With custom parameters:

```bash
python src/train.py --model_name resnet50 --batch_size 64 --max_epochs 20 --optimizer adam --lr 0.0005
```

## Code Reference

```12:15:src/train.py
@task_wrapper
def train_and_save(
    data_module: DogBreedImageDataModule,
    model: DogBreedClassifier,
    trainer: L.Trainer,
    save_path: str | None = None,
):
```

```91:119:src/train.py
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
``` 