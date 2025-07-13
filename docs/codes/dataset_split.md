# Dataset Splitter

The `split_dataset.py` module provides functionality to split a dataset into training and validation sets.

## Overview

This utility takes a source directory containing class folders and splits the images into train and validation sets according to a specified ratio. It maintains the class structure in both the train and validation directories.

## Key Components

### Main Function

```python
def split_dataset(source_dir, output_dir, split_ratio=0.8):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create train and val directories within the output directory
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Iterate through each class folder in the source directory
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Create corresponding class folders in train and val directories
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Get all files in the class folder
        files = os.listdir(class_dir)
        num_files = len(files)
        num_train = int(num_files * split_ratio)

        # Randomly shuffle the files
        random.shuffle(files)

        # Split files into train and val sets
        train_files = files[:num_train]
        val_files = files[num_train:]
```

### File Copying

The function copies files to their respective train and validation directories:

```python
# Copy files to respective directories
for file in train_files:
    src = os.path.join(class_dir, file)
    dst = os.path.join(train_class_dir, file)
    shutil.copy2(src, dst)

for file in val_files:
    src = os.path.join(class_dir, file)
    dst = os.path.join(val_class_dir, file)
    shutil.copy2(src, dst)

print(f"Class {class_name}: {len(train_files)} train, {len(val_files)} val")
```

### Command-line Interface

The module can be run as a standalone script:

```python
if __name__ == "__main__":
    source_dir = "data/dataset"
    output_dir = "data/dataset_split"
    split_ratio = 0.8

    split_dataset(source_dir, output_dir, split_ratio)
    print("Dataset split completed.")
```

## Usage

### As a Module

```python
from datamodules.split_dataset import split_dataset

# Split dataset with 80% training, 20% validation
split_dataset("data/dataset", "data/dataset_split", 0.8)
```

### As a Script

```bash
python src/datamodules/split_dataset.py
```

## Code Reference

```1:6:src/datamodules/split_dataset.py
import os
import random
import shutil
```

```9:24:src/datamodules/split_dataset.py
def split_dataset(source_dir, output_dir, split_ratio=0.8):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create train and val directories within the output directory
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Iterate through each class folder in the source directory
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
``` 