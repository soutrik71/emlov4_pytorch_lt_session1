# Docker Setup for Dog Breed Classification

This directory contains Docker configuration files for running the dog breed classification project in containerized environments.

## Files Overview

- `Dockerfile`: Docker image with Python 3.11 slim and Poetry
- `docker-compose.yml`: Orchestration file for training, evaluation, and inference services
- `.dockerignore`: Excludes unnecessary files from Docker build context
- `run.sh`: Helper script for running Docker services
- `README.md`: This documentation file

## Prerequisites

1. **Docker & Docker Compose**: Ensure Docker and Docker Compose are installed
2. **Kaggle Credentials**: Place `kaggle.json` in `~/.kaggle/` directory for dataset download
3. **Project Structure**: Run from the project root directory

## Quick Setup

The easiest way to run the Docker setup is using the provided helper script:

```bash
# Make the script executable
chmod +x Docker/run.sh

# Show available commands
./Docker/run.sh help

# Run the full pipeline
./Docker/run.sh all

# Run just the training
./Docker/run.sh train
```

## Services

### 1. Training Service (`train`)
- **Purpose**: Train the dog breed classification model
- **Script**: `src/train.py`
- **Volumes**: 
  - `data/` - Dataset storage and processing
  - `logs/` - Training logs and TensorBoard files
  - `checkpoints/` - Model checkpoints
  - `outputs/` - Additional training outputs
  - `~/.kaggle/` - Kaggle credentials (read-only)

### 2. Evaluation Service (`eval`)
- **Purpose**: Evaluate trained model performance
- **Script**: `src/eval.py`
- **Volumes**:
  - `data/` - Test dataset
  - `logs/` - Evaluation logs
  - `checkpoints/` - Trained model checkpoints
  - `outputs/` - Evaluation results
- **Dependencies**: Runs after training service

### 3. Inference Service (`infer`)
- **Purpose**: Run inference on test images
- **Script**: `src/infer.py`
- **Volumes**:
  - `data/` - Test images
  - `checkpoints/` - Trained model checkpoints
  - `outputs/` - Prediction results and visualizations
- **Dependencies**: Runs after training service

### 4. Interactive Service (`interactive`)
- **Purpose**: Manual operations and debugging
- **Volumes**: All directories mounted
- **Features**: Interactive shell with TTY support

## Manual Usage

If you prefer to use Docker Compose directly:

### Quick Start - Full Pipeline

Run the complete training, evaluation, and inference pipeline:

```bash
# Navigate to project root
cd /path/to/emlov4_pytorch_lt_session1

# Run the full pipeline
docker-compose up
```

### Individual Services

#### Training Only
```bash
docker-compose up train
```

#### Evaluation Only (after training)
```bash
docker-compose up eval
```

#### Inference Only (after training)
```bash
docker-compose up infer
```

#### Interactive Mode
```bash
docker-compose up -d interactive
docker exec -it dogbreed-interactive bash
```

### Custom Parameters

#### Training with Custom Parameters
```bash
docker-compose run --rm train \
  python src/train.py \
    --model_name resnet50 \
    --batch_size 64 \
    --max_epochs 20 \
    --lr 0.0005
```

#### Evaluation with Custom Checkpoint
```bash
docker-compose run --rm eval \
  python src/eval.py \
    --ckpt_path checkpoints/epoch=09-val_loss=0.51.ckpt \
    --batch_size 64
```

#### Inference with Custom Settings
```bash
docker-compose run --rm infer \
  python src/infer.py \
    --input_folder data/test_images \
    --output_folder outputs/custom_predictions \
    --num_images 10
```

## GPU Support

To enable GPU support (requires NVIDIA Docker runtime):

1. Uncomment the `runtime: nvidia` lines in `docker-compose.yml`
2. Ensure NVIDIA Container Toolkit is installed
3. Run with GPU support:

```bash
docker-compose up
```

## Volume Persistence

The setup uses bind mounts to ensure data persistence:

- **Training**: Downloads dataset, saves checkpoints and logs
- **Evaluation**: Uses saved checkpoints, generates evaluation metrics
- **Inference**: Uses saved checkpoints, generates prediction visualizations

All data persists on the host filesystem for reuse across container runs.

## Directory Structure

When running the containers, the following directories will be created and mounted:

```
emlov4_pytorch_lt_session1/
├── data/                  # Dataset storage
│   ├── dataset/           # Original dataset
│   ├── dataset_split/     # Split dataset (train/val)
│   └── test_images/       # Test images for inference
├── checkpoints/           # Model checkpoints
├── logs/                  # Training and evaluation logs
└── outputs/               # Prediction outputs and visualizations
```

## Troubleshooting

### Common Issues

1. **Kaggle Authentication Error**
   ```bash
   # Ensure kaggle.json is in the correct location
   ls ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Permission Issues**
   ```bash
   # Fix ownership of mounted directories
   sudo chown -R $USER:$USER data/ logs/ checkpoints/ outputs/
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size in docker-compose.yml
   --batch_size 16
   ```

4. **Context Path Issues**
   ```bash
   # Make sure docker-compose.yml is in the project root
   # Or run Docker Compose from the project root:
   docker-compose -f Docker/docker-compose.yml up
   ```

### Logs and Debugging

```bash
# View service logs
docker-compose logs train
docker-compose logs eval
docker-compose logs infer

# Follow logs in real-time
docker-compose logs -f train

# Debug interactive mode
docker-compose run --rm interactive bash
```

## Cleanup

```bash
# Using the helper script
./Docker/run.sh clean

# Or manually:
# Stop all services
docker-compose down

# Remove containers and networks
docker-compose down --remove-orphans

# Remove images (optional)
docker-compose down --rmi all

# Clean up volumes (WARNING: This will delete all data)
docker-compose down -v
```

## Performance Optimization

1. **Efficient Layering**: The Dockerfile uses efficient layering for faster builds
2. **Bind Mounts**: Direct filesystem access for better I/O performance
3. **Shared Dependencies**: Single image used across all services
4. **Optimized Base Image**: Python 3.11 slim for smaller footprint

## Security Considerations

1. **Read-only Mounts**: Kaggle credentials mounted as read-only
2. **Non-root User**: Consider adding non-root user for production
3. **Network Isolation**: Services run in isolated Docker network
4. **Minimal Base Image**: Uses slim Python image for reduced attack surface 