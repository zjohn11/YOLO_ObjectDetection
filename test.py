import os
import zipfile
import yaml
from PIL import Image
from ultralytics import YOLO

# Paths
zip_path = 'yolo_dataset.zip'
extract_to = 'yolo_dataset'
data_dir = os.path.join(extract_to, 'data')
yaml_path = os.path.join(extract_to, 'dataset.yml')
model_path = 'best.pt'  # Should be downloaded before this runs

def unzip_dataset(zip_path, extract_to):
    if not os.path.exists(extract_to):
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        print("Dataset already unzipped.")

def load_dataset_info(yaml_path):
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    print(f"Warning: {yaml_path} not found.")
    return None

def run_tests():
    # Step 1: Unzip
    unzip_dataset(zip_path, extract_to)

    # Step 2: Load metadata (optional)
    dataset_info = load_dataset_info(yaml_path)
    if dataset_info:
        print(f"Classes: {dataset_info.get('names')}")

    # Step 3: Load YOLO model
    model = YOLO(model_path)

    # Step 4: Run predictions
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(data_dir, filename)

            print(f"\nRunning inference on {filename}...")

            # Inference
            results = model.predict(image_path, verbose=False)
            boxes = results[0].boxes
            df = results[0].pandas().xywh[["class", "name", "xcenter", "ycenter", "width", "height", "confidence"]]

            # Output predictions
            print(df if not df.empty else "No objects detected.")

if __name__ == "__main__":
    run_tests()
