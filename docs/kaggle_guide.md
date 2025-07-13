# Downloading Datasets from Kaggle

This guide explains how to download datasets from Kaggle using our project's utility script.

## Prerequisites

Before you can download datasets from Kaggle, you need:

1. **A Kaggle Account**: Register at [kaggle.com](https://www.kaggle.com) if you don't have one
2. **Kaggle API Credentials**: Generate your API token
3. **Python and Required Packages**: Make sure you have the necessary packages installed

## Step 1: Get Your Kaggle API Credentials

1. Log in to your Kaggle account
2. Click on your profile picture (top right) → **Account**
3. Scroll down to the **API** section
4. Click **Create New API Token**
5. This will download a file named `kaggle.json` containing your credentials

## Step 2: Set Up Your Credentials

Our script looks for your credentials in these locations (in order):

1. `~/.kaggle/kaggle.json` (default Kaggle location)
2. `./kaggle.json` (current working directory)
3. Project root directory

Choose one of these methods to set up your credentials:

### Option A: Place in Default Location (Recommended)

```bash
# Create the directory if it doesn't exist
mkdir -p ~/.kaggle

# Copy your downloaded kaggle.json file
cp /path/to/downloaded/kaggle.json ~/.kaggle/

# Set proper permissions (important for security)
chmod 600 ~/.kaggle/kaggle.json
```

### Option B: Place in Project Root

```bash
# Copy your downloaded kaggle.json file to project root
cp /path/to/downloaded/kaggle.json /path/to/project/
```

## Step 3: Install Required Packages

Make sure you have the Kaggle API package installed:

```bash
# Using Poetry (recommended)
poetry add kaggle

# Or using pip
pip install kaggle
```

## Step 4: Using the Kaggle Downloader Script

Our project includes a utility script (`src/utils/kaggle_downloader.py`) that makes downloading datasets easy.

### Basic Usage

```bash
# Download the default dataset
poetry run python src/utils/kaggle_downloader.py

# Download a specific dataset
poetry run python src/utils/kaggle_downloader.py "username/dataset-name"

# Download to a specific location
poetry run python src/utils/kaggle_downloader.py "username/dataset-name" "./my_data_folder"
```

### Finding Dataset References

The dataset reference is in the format `username/dataset-name` and can be found in the dataset's URL on Kaggle:

- Example: For `https://www.kaggle.com/datasets/shivamb/netflix-shows`
- The reference is `shivamb/netflix-shows`

### Examples

```bash
# Download Netflix shows dataset
poetry run python src/utils/kaggle_downloader.py "shivamb/netflix-shows"

# Download dog breed images to a custom folder
poetry run python src/utils/kaggle_downloader.py "khushikhushikhushi/dog-breed-image-dataset" "./dog_images"
```

## Using the Downloader in Your Code

You can also import the downloader function in your Python code:

```python
from src.utils.kaggle_downloader import download_kaggle_dataset

# Download a dataset
download_kaggle_dataset("username/dataset-name", "./data_folder")
```

## Troubleshooting

### Common Issues

1. **Authentication Error**:
   - Make sure your `kaggle.json` file is in one of the expected locations
   - Check that the file has the correct permissions (`chmod 600`)

2. **Dataset Not Found**:
   - Verify the dataset reference format (`username/dataset-name`)
   - Check if the dataset is public or if you have access to it

3. **Import Error**:
   - Make sure you've installed the Kaggle API package (`pip install kaggle`)

### Checking Credentials

To verify your credentials are correctly set up:

```bash
# Check if the file exists
ls -la ~/.kaggle/kaggle.json

# Verify environment variables (after running the script)
echo $KAGGLE_USERNAME
echo $KAGGLE_KEY
```

## How Our Script Works

Our script improves on the basic Kaggle API by:

1. Setting up credentials **before** importing the Kaggle API
2. Looking for the credentials file in multiple locations
3. Providing better error handling and user feedback
4. Supporting command-line arguments for flexible usage

The key insight is that Kaggle API authentication happens during import, so environment variables must be set before importing the module.

## Additional Resources

- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Kaggle Competitions](https://www.kaggle.com/competitions) 