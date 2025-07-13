# Inference Script

The inference script (`infer.py`) allows you to run the trained dog breed classification model on new images to make predictions.

## Overview

This script loads a trained model checkpoint and runs inference on images from a specified folder. It visualizes the predictions with confidence scores and saves the results as images.

## Key Components

### Image Loading and Preprocessing

```python
@task_wrapper
def load_image(image_path):
    """Load and preprocess an image for inference."""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return img, transform(img).unsqueeze(0)
```

### Inference Process

```python
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
```

### Visualization

```python
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
```

### Flexible Image Discovery

The script can handle both flat directories with images and nested folder structures:

```python
def fetch_images_from_folder(root_folder, num_images=10):
    """Fetch images from the folder (handles both flat and nested structure)."""
    all_files = []

    # Check if root_folder contains image files directly
    if os.path.isdir(root_folder):
        # First, check for direct image files in root folder
        direct_files = [
            Path(os.path.join(root_folder, f))
            for f in os.listdir(root_folder)
            if os.path.isfile(os.path.join(root_folder, f)) and 
            f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        all_files.extend(direct_files)
        
        # Then check subfolders
        for item in os.listdir(root_folder):
            item_path = os.path.join(root_folder, item)
            if os.path.isdir(item_path):
                files_in_folder = [
                    Path(os.path.join(item_path, f))
                    for f in os.listdir(item_path)
                    if os.path.isfile(os.path.join(item_path, f)) and 
                    f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                all_files.extend(files_in_folder)
```

### Hardware Acceleration

Like the other scripts, the inference script automatically detects and uses the best available hardware:

```python
# Determine the best accelerator
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU")
```

## Command-line Arguments

The script supports the following command-line arguments:

| Argument | Description |
|----------|-------------|
| `--input_folder` | Path to input folder containing images |
| `--output_folder` | Path to output folder for predictions |
| `--ckpt_path` | Path to specific model checkpoint |
| `--ckpt_dir` | Path to checkpoint directory (auto-finds best) |
| `--num_images` | Number of random images to process |

## Usage

Basic usage:

```bash
python src/infer.py
```

With custom parameters:

```bash
python src/infer.py --input_folder data/test_images --num_images 5
```

## Code Reference

```139:154:src/infer.py
for image_file in image_files:
    try:
        img, img_tensor = load_image(image_file)
        predicted_label, confidence = infer(
            model, img_tensor, class_labels, device
        )

        output_file = output_folder / f"{image_file.stem}_prediction.png"
        save_prediction_image(
            img, predicted_label, confidence, output_file
        )

        progress.console.print(
            f"Processed {image_file.name}: {predicted_label} "
            f"({confidence:.2f})"
        )
``` 