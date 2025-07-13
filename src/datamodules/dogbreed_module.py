from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Import our utilities
try:
    from ..utils.kaggle_downloader import download_kaggle_dataset
    from .split_dataset import split_dataset
except ImportError:
    # Fallback for when running as script
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from datamodules.split_dataset import split_dataset
    from utils.kaggle_downloader import download_kaggle_dataset


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

    @property
    def data_path(self):
        return self._dl_path / "dataset_split"

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            persistent_workers=True if self._num_workers > 0 else False,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            persistent_workers=True if self._num_workers > 0 else False,
        )

    def teardown(self, stage: str):
        """Clean up after training/testing/predicting."""
        # Clean up any resources if needed
        pass
