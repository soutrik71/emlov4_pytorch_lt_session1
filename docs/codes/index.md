# Code Documentation

This section provides detailed documentation for all the source code components of the EMLO PyTorch Lightning Session 1 project.

## Project Structure

The project is organized into several modules:

```
src/
├── train.py                # Main training script
├── eval.py                 # Evaluation script
├── infer.py                # Inference script
├── datamodules/            # Data handling modules
│   ├── dogbreed_module.py  # DataModule for dog breed dataset
│   └── split_dataset.py    # Dataset splitting utility
├── models/                 # Model definitions
│   └── dogbreed_classifier.py  # Dog breed classifier model
└── utils/                  # Utility functions
    ├── kaggle_downloader.py  # Kaggle dataset downloader
    └── logging_utils.py      # Logging utilities
```

## Core Scripts

- [**Training Script**](train.md): Main script for training the dog breed classification model
- [**Evaluation Script**](eval.md): Script for evaluating model performance on test data
- [**Inference Script**](infer.md): Script for running inference on new images

## Data Handling

- [**Dog Breed DataModule**](datamodule.md): PyTorch Lightning DataModule for the dog breed dataset
- [**Dataset Splitter**](dataset_split.md): Utility for splitting datasets into train/validation sets

## Models

- [**Dog Breed Classifier**](model.md): PyTorch Lightning model for dog breed classification

## Utilities

- [**Kaggle Downloader**](kaggle_downloader.md): Utility for downloading datasets from Kaggle
- [**Logging Utilities**](logging_utils.md): Utilities for logging and progress tracking

## Key Features

The codebase implements several key features:

1. **Automated Data Pipeline**: Automatic download and preparation of datasets from Kaggle
2. **Hardware Acceleration**: Support for CUDA, MPS (Apple Silicon), and CPU
3. **Modern Training Techniques**: 
   - Mixed precision training
   - Learning rate scheduling
   - Gradient clipping
   - Label smoothing
4. **Comprehensive Metrics**: Tracking of accuracy, precision, recall, and F1 score
5. **Visualization**: Generation of prediction visualizations with confidence scores
6. **Checkpoint Management**: Automatic selection of best checkpoints based on validation metrics 