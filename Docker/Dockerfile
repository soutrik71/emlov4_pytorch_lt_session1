# Use PyTorch official image with Python 3.10+ support
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables for Docker detection and logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DOCKER_CONTAINER=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PYTHONPATH=/app/src

# Update package manager and install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Set work directory
WORKDIR /app

# Copy Poetry configuration files and README (needed for project setup)
COPY pyproject.toml poetry.lock README.md ./

# Configure Poetry: Don't create virtual environment, install dependencies only (no project)
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-root \
    && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p data logs checkpoints outputs

# Set the default command
CMD ["python", "-c", "print('Docker container is ready!')"] 