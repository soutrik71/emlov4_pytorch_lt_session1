# EMLO PyTorch Lightning Session 1

Welcome to the EMLO PyTorch Lightning Session 1 project documentation!

## Overview

This project is part of the EMLO (Extensive Machine Learning Operations) course and focuses on PyTorch Lightning fundamentals. It implements a dog breed classification system using modern deep learning practices.

## Features

- 🔥 **PyTorch Lightning**: Modern deep learning framework with structured training loops
- 🐶 **Dog Breed Classification**: Multi-class image classification with pre-trained models
- 🚀 **Hardware Acceleration**: Support for CUDA, MPS (Apple Silicon), and CPU
- 📊 **Comprehensive Metrics**: Tracking of accuracy, precision, recall, and F1 score
- 📁 **Kaggle Integration**: Automatic dataset download and preparation
- 🐳 **Docker Support**: Containerized training, evaluation, and inference
- 🧪 **Testing**: Comprehensive test suite with pytest
- 📝 **Documentation**: Auto-generated documentation with MkDocs
- 🎨 **Code Quality**: Black, Ruff, and isort for code formatting and linting
- 🔍 **Type Checking**: MyPy for static type analysis

## Quick Start

1. Clone the repository
2. Install dependencies with Poetry:
   ```bash
   poetry install --no-root --with dev
   ```
3. Run training:
   ```bash
   poetry run python src/train.py
   ```
4. Run evaluation:
   ```bash
   poetry run python src/eval.py
   ```
5. Run inference:
   ```bash
   poetry run python src/infer.py --input_folder data/test_images
   ```

### Docker Quick Start

Alternatively, use Docker to run the entire pipeline:

```bash
# Build and run with Docker
./run.sh all
```

## Documentation Sections

- [**Getting Started**](getting-started.md): Basic overview and quick start guide
- [**Setup Guide**](setup.md): Detailed setup instructions
- [**Kaggle Guide**](kaggle_guide.md): How to use Kaggle datasets
- [**Docker Guide**](docker.md): Containerized deployment with Docker
- [**Code Documentation**](codes/index.md): Detailed documentation of all code components

## Project Structure

```
├── src/                    # Source code
│   ├── train.py            # Training script
│   ├── eval.py             # Evaluation script
│   ├── infer.py            # Inference script
│   ├── datamodules/        # Data handling modules
│   ├── models/             # Model definitions
│   └── utils/              # Utility functions
├── tests/                  # Test files
├── docs/                   # Documentation
├── pyproject.toml          # Project configuration
└── mkdocs.yml              # Documentation configuration
```

## Development Tools

This project uses several development tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **isort**: Import sorting
- **pytest**: Testing framework
- **mypy**: Type checking
- **MkDocs**: Documentation generation

## License

This project is licensed under the MIT License. 