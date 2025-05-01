import os
from PIL import Image
import numpy as np
import yaml
from model import YourAIModel  # Replace with your actual model import

# Paths to images and labels
image_dir = 'dataset/data/'
label_dir = 'dataset/labels/'  # Assuming labels are in this folder
yaml_path = 'dataset/dataset.yml'

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

def load_dataset_info(yaml_path):
    """
    Loads dataset information from dataset.yml.
    """
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def run_tests():
    # Load dataset info from dataset.yml (if necessary for classes, paths, etc.)
    dataset_info = load_dataset_info(yaml_path)
    print(f"Dataset info: {dataset_info}")

    # Initialize your AI model
    model = YourAIModel()

    # Loop through images and their corresponding label files
    for image_filename in os.listdir(image_dir):
        if image_filename.endswith(".jpg") or image_filename.endswith(".png"):
            image_path = os.path.join(image_dir, image_filename)
            label_filename = image_filename.replace(".jpg", ".txt").replace(".png", ".txt")
            label_path = os.path.join(label_dir, label_filename)

            # Load the image
            image = Image.open(image_path)

            # Optionally, preprocess the image before running it through the model
            # Example: Resize image to match model input size
            # image = image.resize((224, 224))

            # Run the image through your model to get predictions
            model_output = model.predict(image)  # Replace with your actual model prediction

            # Parse the YOLO labels for ground truth
            ground_truth = parse_yolo_labels(label_path)

            # Output comparison (for now, we'll just print the results)
            print(f"Image: {image_filename}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Model Output: {model_output}")

            # Optional: Compare model output with ground truth (IoU or other metrics)
            # assert compare_predictions(model_output, ground_truth), f"Test failed for {image_filename}"

if __name__ == "__main__":
    run_tests()
