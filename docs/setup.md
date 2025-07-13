# Setup Guide

This document outlines the setup process we followed to configure this project environment.

## Environment Setup

### Python Virtual Environment

We initially created a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### Poetry Setup

We decided to use Poetry for dependency management. The process involved:

1. Creating a `pyproject.toml` file
2. Configuring dependencies
3. Resolving dependency conflicts, particularly for PyTorch on macOS

## Dependency Management

### PyTorch Installation

We faced challenges installing PyTorch for macOS (CPU version). After several attempts, we found a working configuration:

```bash
# Updated pyproject.toml with compatible versions
poetry lock
poetry install --no-root
```

The key was using the correct PyTorch and Lightning versions compatible with macOS:
- `torch>=2.5.0` (resolved to 2.7.1)
- `lightning>=2.5.0` (resolved to 2.5.2)
- `torchvision>=0.20.0` (resolved to 0.22.1)

### Development Dependencies

We added development dependencies in a separate group:

```bash
# Added to pyproject.toml under [tool.poetry.group.dev.dependencies]
poetry lock
poetry install --no-root --with dev
```

Development tools include:
- **Code formatting and linting**: black, ruff, isort
- **Testing**: pytest, pytest-cov, pytest-mock, pytest-xdist
- **Documentation**: mkdocs, mkdocs-material, mkdocstrings[python], mkdocs-jupyter
- **Type checking**: mypy

## Configuration Files

### Tool Configuration

We added tool configurations to `pyproject.toml` for:
- Black
- Ruff
- isort
- pytest
- mypy

### Documentation Setup

We created:
1. `mkdocs.yml` configuration file
2. Basic documentation structure in the `docs/` folder
3. Sample test file in `tests/` to verify the setup

## Verification

We verified our setup with:

```bash
# Test PyTorch and Lightning installation
poetry run pytest tests/ -v

# Verify code formatting tools
poetry run black --check tests/
poetry run isort --check-only tests/
poetry run ruff check tests/

# Build documentation
poetry run mkdocs build
```

## Apple Silicon (M1/M2/M3) Support

We confirmed that PyTorch can use MPS (Metal Performance Shaders) for GPU acceleration on Apple Silicon:
- MPS is available: True

## Current Environment

Final environment details:
- PyTorch: 2.7.1
- Lightning: 2.5.2
- torchvision: 0.22.1
- Python: 3.11+ 