# Kaggle Dataset Downloader

The `kaggle_downloader.py` module provides functionality to download datasets from Kaggle.

## Overview

This utility handles the authentication with Kaggle API and downloads datasets. It ensures that Kaggle credentials are properly set up before importing the Kaggle API, which is a critical step to avoid authentication errors.

## Key Components

### Credential Setup

```python
def setup_kaggle_credentials():
    """Set up Kaggle credentials from kaggle.json file"""
    # Try multiple locations for kaggle.json
    possible_locations = [
        Path.home() / ".kaggle" / "kaggle.json",  # Default Kaggle location
        Path.cwd() / "kaggle.json",  # Current working directory
        Path(__file__).parent.parent.parent / "kaggle.json",  # Project root
    ]

    for location in possible_locations:
        if location.exists():
            try:
                with open(location) as f:
                    credentials = json.load(f)

                # Set environment variables
                os.environ["KAGGLE_USERNAME"] = credentials["username"]
                os.environ["KAGGLE_KEY"] = credentials["key"]
                print(f"Using credentials from {location}")
                return True
            except Exception as e:
                print(f"Error loading credentials from {location}: {e}")

    return False
```

### Download Function

```python
def download_kaggle_dataset(dataset_ref, download_path="./data"):
    """
    Download a Kaggle dataset

    Args:
        dataset_ref: Dataset reference like "owner/dataset-name"
        download_path: Where to save the dataset
    """
    # Create the API client
    api = KaggleApi()
    api.authenticate()

    # Create download directory if it doesn't exist
    Path(download_path).mkdir(parents=True, exist_ok=True)

    # Download dataset
    print(f"Downloading {dataset_ref}...")
    api.dataset_download_files(dataset_ref, path=download_path, unzip=True, force=True)

    print(f"Dataset downloaded to: {download_path}")
    return True
```

### Import Sequence

The module carefully manages the import sequence to ensure credentials are set before importing the Kaggle API:

```python
# Set up credentials before importing Kaggle
if not setup_kaggle_credentials():
    print("ERROR: Could not find or load kaggle.json file.")
    print("Please place it in ~/.kaggle/ or in the project root directory.")
    sys.exit(1)

# Only import Kaggle AFTER credentials are set
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("ERROR: Kaggle API not installed. Run 'pip install kaggle'.")
    sys.exit(1)
```

### Command-line Interface

The module can be run as a standalone script:

```python
if __name__ == "__main__":
    # Allow command-line specification of dataset
    if len(sys.argv) > 1:
        dataset_ref = sys.argv[1]
    else:
        dataset_ref = "khushikhushikhushi/dog-breed-image-dataset"

    # Optional: specify download path as second argument
    download_path = sys.argv[2] if len(sys.argv) > 2 else "./data"

    # Download the dataset
    download_kaggle_dataset(dataset_ref, download_path)
```

## Usage

### As a Module

```python
from utils.kaggle_downloader import download_kaggle_dataset

# Download a specific dataset
download_kaggle_dataset("khushikhushikhushi/dog-breed-image-dataset", "data")
```

### As a Script

```bash
# Download default dataset
python src/utils/kaggle_downloader.py

# Download specific dataset
python src/utils/kaggle_downloader.py username/dataset-name ./custom-path
```

## Credential Setup

Before using this module, you need to set up Kaggle credentials:

1. Create a Kaggle account if you don't have one
2. Go to your Kaggle account settings
3. Click on "Create New API Token" to download `kaggle.json`
4. Place the `kaggle.json` file in one of these locations:
   - `~/.kaggle/kaggle.json` (default Kaggle location)
   - In the project root directory
   - In the current working directory

## Code Reference

```15:36:src/utils/kaggle_downloader.py
def setup_kaggle_credentials():
    """Set up Kaggle credentials from kaggle.json file"""
    # Try multiple locations for kaggle.json
    possible_locations = [
        Path.home() / ".kaggle" / "kaggle.json",  # Default Kaggle location
        Path.cwd() / "kaggle.json",  # Current working directory
        Path(__file__).parent.parent.parent / "kaggle.json",  # Project root
    ]

    for location in possible_locations:
        if location.exists():
            try:
                with open(location) as f:
                    credentials = json.load(f)

                # Set environment variables
                os.environ["KAGGLE_USERNAME"] = credentials["username"]
                os.environ["KAGGLE_KEY"] = credentials["key"]
                print(f"Using credentials from {location}")
                return True
            except Exception as e:
                print(f"Error loading credentials from {location}: {e}")

    return False
``` 