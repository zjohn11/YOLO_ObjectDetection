import os
import zipfile
import yaml
from PIL import Image
from ultralytics import YOLO
import shutil

# Paths
image_dir = 'data/'  # Folder containing images after unzipping yolo_dataset.zip
label_dir = 'labels/'  # Folder containing labels (adjust if necessary)
yaml_path = 'dataset.yml'  # Path to dataset YAML file (likely needed for label mapping)

# Unzip the dataset if it's not already unzipped
def unzip_dataset(zip_path, extract_to):
    if not os.path.exists(extract_to):
        print(f"Unzipping dataset: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        print("Dataset already unzipped.")

# Parse YOLO labels
def parse_yolo_labels(label_file):
    """
    Parses a YOLO label file.
    Returns a list of objects with [class, x_center, y_center, width, height] normalized.
    """
    with open(label_file, 'r') as f:
        labels = []
        for line in f:
            parts = line.strip().split()
            label = list(map(float, parts))  # Convert each part to a float
            labels.append(label)
        return labels

# Load dataset information from dataset.yml (only if needed)
def load_dataset_info(yaml_path):
    # Only load if the file exists
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    else:
        print(f"Warning: {yaml_path} not found. Skipping dataset info.")
        return None

# Run tests
def run_tests():
    # Unzip the dataset if it's in a zip file
    unzip_dataset('yolo_dataset.zip', 'yolo_dataset')

    # Load dataset info from dataset.yml (if necessary for classes, paths, etc.)
    dataset_info = load_dataset_info(yaml_path)
    if dataset_info:
        print(f"Dataset info: {dataset_info}")
    else:
        print("No dataset info loaded.")

    # Initialize the YOLO model with the pre-trained 'best.pt' model from GitHub Releases
    model = YOLO("best.pt")  # The model will be downloaded by the workflow before running this

    # Loop through images and their corresponding label files
    for image_filename in os.listdir(image_dir):
        if image_filename.endswith(".jpg") or image_filename.endswith(".png"):
            image_path = os.path.join(image_dir, image_filename)
            label_filename = image_filename.replace(".jpg", ".txt").replace(".png", ".txt")
            label_path = os.path.join(label_dir, label_filename)

            # Load the image
            image = Image.open(image_path)

            # Run the image through your model to get predictions
            model_output = model.predict(image_path)  # YOLO model takes file path

            # Parse the YOLO labels for ground truth
            ground_truth = parse_yolo_labels(label_path)

            # Output comparison (for now, we'll just print the results)
            print(f"Image: {image_filename}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Model Output: {model_output.pandas().xywh}")  # Predicted coordinates and labels

            # Optional: Compare model output with ground truth (IoU or other metrics)
            # assert compare_predictions(model_output, ground_truth), f"Test failed for {image_filename}"

if __name__ == "__main__":
    run_tests()
