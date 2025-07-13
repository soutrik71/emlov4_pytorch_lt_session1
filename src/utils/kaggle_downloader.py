#!/usr/bin/env python
# src/utils/kaggle_downloader_fixed.py

"""
Simple script to download Kaggle datasets.
This script ensures credentials are properly set up before importing Kaggle.
"""

import json
import os
import sys
from pathlib import Path


# Set up Kaggle credentials BEFORE importing the Kaggle API
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
