# Docker Setup

This guide covers the Docker setup for the dog breed classification project, allowing you to run training, evaluation, and inference in containerized environments.

## Docker Components

- **Dockerfile**: Uses `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` as base image with Python 3.10+ support
- **docker-compose.yml**: Orchestrates training, evaluation, and inference services
- **run.sh**: Helper script for running Docker services

## Prerequisites

1. **Docker & Docker Compose**: Ensure Docker and Docker Compose are installed
2. **Kaggle Credentials**: Place `kaggle.json` in `~/.kaggle/` directory with proper permissions (chmod 600)
3. **Project Structure**: Commands should be run from the project root directory

## Quick Start

The easiest way to run the Docker setup is using the provided helper script:

```bash
# Make the script executable
chmod +x run.sh

# Show available commands
./run.sh help

# Run the full pipeline
./run.sh all

# Run just the training
./run.sh train
```

## Docker Services

### 1. Training Service (`train`)
- **Purpose**: Train the dog breed classification model
- **Script**: `src/train.py`
- **Volumes**: 
  - `data/` - Dataset storage and processing
  - `logs/` - Training logs and TensorBoard files
  - `checkpoints/` - Model checkpoints
  - `outputs/` - Additional training outputs
  - `~/.kaggle/` - Kaggle credentials (read-only)
- **Features**:
  - Automatic Kaggle dataset download
  - Dataset splitting (train/val)
  - Optimized CPU training
  - Progress tracking with TQDM
  - Checkpoint saving

### 2. Evaluation Service (`eval`)
- **Purpose**: Evaluate trained model performance
- **Script**: `src/eval.py`
- **Volumes**:
  - `data/` - Test dataset
  - `logs/` - Evaluation logs
  - `checkpoints/` - Trained model checkpoints
  - `outputs/` - Evaluation results
- **Features**:
  - Automatic best checkpoint selection
  - Comprehensive metrics (accuracy, precision, recall, F1)
  - Detailed evaluation report

### 3. Inference Service (`infer`)
- **Purpose**: Run inference on test images
- **Script**: `src/infer.py`
- **Volumes**:
  - `data/test_images/` - Test images
  - `checkpoints/` - Trained model checkpoints
  - `outputs/predictions/` - Prediction results and visualizations
- **Features**:
  - Visual prediction outputs with confidence scores
  - Automatic class discovery
  - Batch processing of multiple images

### 4. Interactive Service (`interactive`)
- **Purpose**: Manual operations and debugging
- **Volumes**: All directories mounted
- **Features**: Interactive shell with TTY support

## Manual Usage

If you prefer to use Docker Compose directly:

### Quick Start - Full Pipeline

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
docker-compose run --rm interactive bash
```

## Custom Parameters

### Training with Custom Parameters
```bash
docker-compose run --rm train bash -c "cd src && python train.py --model_name resnet50 --batch_size 64 --max_epochs 20 --learning_rate 0.0005"
```

### Evaluation with Custom Checkpoint
```bash
docker-compose run --rm eval bash -c "cd src && python eval.py --ckpt_path ../checkpoints/epoch=09-val_loss=0.51.ckpt --batch_size 64"
```

### Inference with Custom Settings
```bash
docker-compose run --rm infer bash -c "cd src && python infer.py --input_folder ../data/test_images --output_folder ../outputs/custom_predictions --num_images 10"
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

4. **Checkpoint Parsing Issues**
   - The system handles checkpoint filenames with patterns like `epoch=epoch=02-val_loss=val_loss=0.96.ckpt`
   - No action needed as this is handled automatically

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
./run.sh clean

# Or manually:
# Stop all services
docker-compose down

# Remove containers and networks
docker-compose down --remove-orphans

# Remove images (optional)
docker-compose down --rmi all
```

## Performance Optimization

1. **PyTorch Base Image**: Uses official PyTorch image optimized for CPU performance
2. **Bind Mounts**: Direct filesystem access for better I/O performance
3. **Shared Dependencies**: Single image used across all services
4. **DataLoader Configuration**: Optimized for Docker environments (persistent_workers=False)
5. **Progress Tracking**: TQDM for Docker-friendly progress display 