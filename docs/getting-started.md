# Getting Started

This guide will help you get up and running with this project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/soutrikchowdhury/emlov4_pytorch_lt_session1.git
   cd emlov4_pytorch_lt_session1
   ```

2. Set up the Python environment:
   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies with Poetry
   poetry install --no-root
   ```

## Project Structure

```
emlov4_pytorch_lt_session1/
  ├── src/               # Source code
  │   ├── datamodules/   # PyTorch Lightning data modules
  │   ├── models/        # Model definitions
  │   ├── utils/         # Utility functions
  │   ├── train.py       # Training script
  │   ├── eval.py        # Evaluation script
  │   └── infer.py       # Inference script
  ├── tests/             # Test files
  ├── docs/              # Documentation
  └── pyproject.toml     # Project configuration
```

## Quick Start

### Training a Model

```bash
poetry run python src/train.py
```

### Evaluating a Model

```bash
poetry run python src/eval.py
```

### Running Inference

```bash
poetry run python src/infer.py
```

## Next Steps

- Check out the [Setup Guide](setup.md) for detailed environment setup instructions
- Learn how to [Download Kaggle Datasets](kaggle_guide.md) for your experiments
- Explore the API documentation for more details on the available modules 