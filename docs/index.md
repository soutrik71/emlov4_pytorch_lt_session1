# EMLO PyTorch Lightning Session 1

Welcome to the EMLO PyTorch Lightning Session 1 project documentation!

## Overview

This project is part of the EMLO (Extensive Machine Learning Operations) course and focuses on PyTorch Lightning fundamentals.

## Features

- 🔥 **PyTorch Lightning**: Modern deep learning framework
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
3. Run tests:
   ```bash
   poetry run pytest
   ```

## Development Tools

This project uses several development tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **isort**: Import sorting
- **pytest**: Testing framework
- **mypy**: Type checking
- **MkDocs**: Documentation generation

## Project Structure

```
├── src/                    # Source code
├── tests/                  # Test files
├── docs/                   # Documentation
├── pyproject.toml         # Project configuration
└── mkdocs.yml             # Documentation configuration
```

## Contributing

1. Format code: `poetry run black .`
2. Sort imports: `poetry run isort .`
3. Lint code: `poetry run ruff check .`
4. Type check: `poetry run mypy .`
5. Run tests: `poetry run pytest`

## License

This project is licensed under the MIT License. 