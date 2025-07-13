# Dog Breed DataModule

The `DogBreedImageDataModule` class is a PyTorch Lightning DataModule that handles data preparation, loading, and processing for the dog breed classification project.

## Overview

This DataModule manages the complete data pipeline, including:

1. **Data Download**: Automatically downloads the dataset from Kaggle
2. **Data Splitting**: Splits the dataset into train and validation sets
3. **Data Transformation**: Applies appropriate transforms for training and validation
4. **Data Loading**: Creates efficient data loaders with proper configurations

## Key Components

### Initialization

```python
def __init__(
    self,
    dl_path: str | Path = "data",
    num_workers: int = 0,
    batch_size: int = 8,
    dataset_ref: str = "khushikhushikhushi/dog-breed-image-dataset",
    split_ratio: float = 0.8,
):
    super().__init__()
    self.save_hyperparameters()

    self._dl_path = Path(dl_path)
    self._num_workers = num_workers
    self._batch_size = batch_size
    self._dataset_ref = dataset_ref
    self._split_ratio = split_ratio
```

### Data Preparation

The `prepare_data()` method is called once by PyTorch Lightning to download and prepare the dataset:

```python
def prepare_data(self):
    """Download and prepare data from Kaggle (called only once)."""
    # Download dataset from Kaggle
    print(f"Downloading dataset {self._dataset_ref} from Kaggle...")
    download_kaggle_dataset(self._dataset_ref, str(self._dl_path))

    # Check if the dataset was downloaded and extracted
    dataset_path = self._dl_path / "dataset"
    if not dataset_path.exists():
        # Try to find the extracted folder
        extracted_folders = [f for f in self._dl_path.iterdir() if f.is_dir() and f.name != "dataset_split"]
        if extracted_folders:
            dataset_path = extracted_folders[0]
        else:
            raise FileNotFoundError(f"Could not find extracted dataset in {self._dl_path}")

    # Split the dataset into train/val (only if not already split)
    split_output_dir = self._dl_path / "dataset_split"
    if not split_output_dir.exists():
        print(f"Splitting dataset with ratio {self._split_ratio}...")
        split_dataset(str(dataset_path), str(split_output_dir), self._split_ratio)
    else:
        print("Dataset already split, skipping split step.")
```

### Dataset Setup

The `setup()` method is called for each stage (fit, test, predict) to create the appropriate datasets:

```python
def setup(self, stage: str):
    """Set up datasets for different stages (fit, test, predict)."""
    # Assign train/val datasets for use in dataloaders
    if stage == "fit":
        train_path = self.data_path / "train"
        val_path = self.data_path / "val"

        self.train_dataset = ImageFolder(root=train_path, transform=self.train_transform)
        self.val_dataset = ImageFolder(root=val_path, transform=self.valid_transform)

    # Assign test dataset for use in dataloader(s)
    if stage == "test":
        test_path = self.data_path / "val"  # Using val as test
        self.test_dataset = ImageFolder(root=test_path, transform=self.valid_transform)

    # Assign predict dataset for use in dataloader(s)
    if stage == "predict":
        predict_path = self.data_path / "val"  # Using val for prediction
        self.predict_dataset = ImageFolder(root=predict_path, transform=self.valid_transform)
```

### Data Transformations

The DataModule defines separate transforms for training and validation:

```python
@property
def train_transform(self):
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ]
    )

@property
def valid_transform(self):
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), self.normalize_transform])
```

### DataLoaders

The DataModule creates optimized DataLoaders for each stage:

```python
def train_dataloader(self):
    return DataLoader(
        dataset=self.train_dataset,
        batch_size=self._batch_size,
        num_workers=self._num_workers,
        shuffle=True,
        persistent_workers=True if self._num_workers > 0 else False,
    )

def val_dataloader(self):
    return DataLoader(
        dataset=self.val_dataset,
        batch_size=self._batch_size,
        num_workers=self._num_workers,
        shuffle=False,
        persistent_workers=True if self._num_workers > 0 else False,
    )
```

## Integration with Kaggle Downloader

The DataModule integrates with the `kaggle_downloader.py` utility to automatically download the dataset:

```python
# Download dataset from Kaggle
print(f"Downloading dataset {self._dataset_ref} from Kaggle...")
download_kaggle_dataset(self._dataset_ref, str(self._dl_path))
```

## Integration with Dataset Splitter

The DataModule uses the `split_dataset.py` utility to split the dataset into train and validation sets:

```python
# Split the dataset into train/val (only if not already split)
split_output_dir = self._dl_path / "dataset_split"
if not split_output_dir.exists():
    print(f"Splitting dataset with ratio {self._split_ratio}...")
    split_dataset(str(dataset_path), str(split_output_dir), self._split_ratio)
```

## Code Reference

```22:33:src/datamodules/dogbreed_module.py
class DogBreedImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        dl_path: str | Path = "data",
        num_workers: int = 0,
        batch_size: int = 8,
        dataset_ref: str = "khushikhushikhushi/dog-breed-image-dataset",
        split_ratio: float = 0.8,
    ):
        super().__init__()
        self.save_hyperparameters()

        self._dl_path = Path(dl_path)
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._dataset_ref = dataset_ref
        self._split_ratio = split_ratio
```

```35:57:src/datamodules/dogbreed_module.py
def prepare_data(self):
    """Download and prepare data from Kaggle (called only once)."""
    # Download dataset from Kaggle
    print(f"Downloading dataset {self._dataset_ref} from Kaggle...")
    download_kaggle_dataset(self._dataset_ref, str(self._dl_path))

    # Check if the dataset was downloaded and extracted
    dataset_path = self._dl_path / "dataset"
    if not dataset_path.exists():
        # Try to find the extracted folder
        extracted_folders = [f for f in self._dl_path.iterdir() if f.is_dir() and f.name != "dataset_split"]
        if extracted_folders:
            dataset_path = extracted_folders[0]
        else:
            raise FileNotFoundError(f"Could not find extracted dataset in {self._dl_path}")

    # Split the dataset into train/val (only if not already split)
    split_output_dir = self._dl_path / "dataset_split"
    if not split_output_dir.exists():
        print(f"Splitting dataset with ratio {self._split_ratio}...")
        split_dataset(str(dataset_path), str(split_output_dir), self._split_ratio)
    else:
        print("Dataset already split, skipping split step.")
``` 