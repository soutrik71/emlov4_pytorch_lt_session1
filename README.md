# Dog Breed Classification with PyTorch Lightning

A comprehensive dog breed classification system built with PyTorch Lightning, featuring training, evaluation, and inference capabilities.

## Features

- 🔥 **PyTorch Lightning**: Modern deep learning framework with structured training loops
- 🐶 **Dog Breed Classification**: Multi-class image classification with pre-trained models
- 🚀 **Hardware Acceleration**: Support for CUDA, MPS (Apple Silicon), and CPU
- 📊 **Comprehensive Metrics**: Tracking of accuracy, precision, recall, and F1 score
- 📁 **Kaggle Integration**: Automatic dataset download and preparation
- 🐳 **Docker Support**: Containerized training, evaluation, and inference
- 🧪 **Testing**: Comprehensive test suite with pytest
- 📝 **Documentation**: Auto-generated documentation with MkDocs

## Quick Start

### Option 1: Using Poetry

```bash
# Install dependencies
poetry install --no-root

# Run training
poetry run python src/train.py

# Run evaluation
poetry run python src/eval.py

# Run inference
poetry run python src/infer.py --input_folder data/test_images
```

### Option 2: Using Docker

```bash
# Make the script executable
chmod +x run.sh

# Run the full pipeline
./run.sh all

# Or run individual services
./run.sh train
./run.sh eval
./run.sh infer
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **Getting Started**: Basic overview and quick start guide
- **Setup Guide**: Detailed setup instructions
- **Kaggle Guide**: How to use Kaggle datasets
- **Docker Guide**: Containerized deployment with Docker
- **Code Documentation**: Detailed documentation of all code components

Build and view the documentation:

```bash
poetry run mkdocs serve
```

## Project Structure

```
├── src/                    # Source code
│   ├── train.py            # Training script
│   ├── eval.py             # Evaluation script
│   ├── infer.py            # Inference script
│   ├── datamodules/        # Data handling modules
│   ├── models/             # Model definitions
│   └── utils/              # Utility functions
├── Docker/                 # Docker configuration
├── docs/                   # Documentation
├── pyproject.toml          # Project configuration
└── mkdocs.yml              # Documentation configuration
```

## License

This project is licensed under the MIT License.
