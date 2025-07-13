import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.dogbreed_classifier import DogBreedClassifier
from utils.logging_utils import get_rich_progress, setup_logger, task_wrapper


@task_wrapper
def load_image(image_path):
    """Load and preprocess an image for inference."""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return img, transform(img).unsqueeze(0)


def get_class_names(root_folder):
    """Get class names from folder structure."""
    # List only directories (folders) inside the root folder
    folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
    return sorted(folders)


@task_wrapper
def infer(model, image_tensor, class_labels, device):
    """Run inference on a single image."""
    model.eval()
    with torch.no_grad():
        # Move image to the same device as model
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    return predicted_label, confidence


@task_wrapper
def save_prediction_image(image, predicted_label, confidence, output_path):
    """Save prediction visualization."""
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def fetch_images_from_folder(root_folder, num_images=10):
    """Fetch images from the folder (handles both flat and nested structure)."""
    all_files = []

    # Check if root_folder contains image files directly
    if os.path.isdir(root_folder):
        # First, check for direct image files in root folder
        direct_files = [
            Path(os.path.join(root_folder, f))
            for f in os.listdir(root_folder)
            if os.path.isfile(os.path.join(root_folder, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        all_files.extend(direct_files)

        # Then check subfolders
        for item in os.listdir(root_folder):
            item_path = os.path.join(root_folder, item)
            if os.path.isdir(item_path):
                files_in_folder = [
                    Path(os.path.join(item_path, f))
                    for f in os.listdir(item_path)
                    if os.path.isfile(os.path.join(item_path, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                all_files.extend(files_in_folder)

    # If we have fewer images than requested, use all available
    if len(all_files) <= num_images:
        print(f"Found {len(all_files)} images, using all of them")
        return all_files

    # Randomly select images from the collection
    random_files = random.sample(all_files, num_images)
    return random_files


@task_wrapper
def main(args):
    """Main inference function."""
    # Set seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Determine the best accelerator (same as eval.py)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load the model from checkpoint
    print(f"Loading model from checkpoint: {args.ckpt_path}")
    model = DogBreedClassifier.load_from_checkpoint(checkpoint_path=args.ckpt_path)
    model = model.to(device)
    model.eval()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    # Verify input folder exists
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # Get class names from training data structure
    # Assuming the model was trained on data/dataset_split/train
    train_data_path = Path("data/dataset_split/train")
    if train_data_path.exists():
        class_labels = get_class_names(train_data_path)
        print(f"Found {len(class_labels)} classes: {class_labels}")
    else:
        # Fallback: try to get from input folder structure
        class_labels = get_class_names(input_folder)
        print(f"Using classes from input folder: {class_labels}")

    # Fetch images for inference
    image_files = fetch_images_from_folder(input_folder, args.num_images)
    print(f"Processing {len(image_files)} images...")

    if not image_files:
        print("No image files found in the input folder!")
        return

    with get_rich_progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))

        for image_file in image_files:
            try:
                img, img_tensor = load_image(image_file)
                predicted_label, confidence = infer(model, img_tensor, class_labels, device)

                output_file = output_folder / f"{image_file.stem}_prediction.png"
                save_prediction_image(img, predicted_label, confidence, output_file)

                progress.console.print(f"Processed {image_file.name}: {predicted_label} " f"({confidence:.2f})")
                progress.advance(task)

            except Exception as e:
                progress.console.print(f"Error processing {image_file.name}: {e}")
                progress.advance(task)

    print(f"Inference complete! Results saved to: {output_folder}")


def find_best_checkpoint(ckpt_dir):
    """Find the best checkpoint based on validation loss (same as eval.py)."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Look for epoch checkpoints (these contain val_loss in filename)
    epoch_ckpts = list(ckpt_path.glob("epoch=*-val_loss=*.ckpt"))

    if epoch_ckpts:
        # Sort by validation loss (lower is better)
        epoch_ckpts.sort(key=lambda x: float(x.stem.split("val_loss=")[1]))
        best_ckpt = epoch_ckpts[0]
        print(f"Found best checkpoint: {best_ckpt}")
        return str(best_ckpt)

    # Fallback to last checkpoint
    last_ckpt = ckpt_path / "last.ckpt"
    if last_ckpt.exists():
        print(f"Using last checkpoint: {last_ckpt}")
        return str(last_ckpt)

    # Fallback to final model
    final_ckpt = ckpt_path / "final_model.ckpt"
    if final_ckpt.exists():
        print(f"Using final model: {final_ckpt}")
        return str(final_ckpt)

    raise FileNotFoundError(f"No valid checkpoint found in {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer using trained DogBreed Classifier")
    parser.add_argument(
        "--input_folder", type=str, default="data/test_images", help="Path to input folder containing images"
    )
    parser.add_argument(
        "--output_folder", type=str, default="outputs/predictions", help="Path to output folder for predictions"
    )
    parser.add_argument("--ckpt_path", type=str, help="Path to specific model checkpoint")
    parser.add_argument(
        "--ckpt_dir", type=str, default="checkpoints", help="Path to checkpoint directory (auto-finds best)"
    )
    parser.add_argument("--num_images", type=int, default=10, help="Number of random images to process")

    args = parser.parse_args()

    # Set up logging
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    setup_logger(log_dir / "infer_log.log")

    # Determine checkpoint path
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = find_best_checkpoint(args.ckpt_dir)

    # Update args with determined checkpoint path
    args.ckpt_path = ckpt_path

    main(args)
