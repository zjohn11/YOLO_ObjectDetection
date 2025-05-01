import os
import zipfile
import yaml
from ultralytics import YOLO

# Paths
zip_path = 'yolo_dataset.zip'
extract_to = 'yolo_dataset'
yaml_path = os.path.join(extract_to, 'dataset.yml')
model_path = 'best.pt'

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

def find_image_dir(root_dir):
    # Recursively look for images
    for root, dirs, files in os.walk(root_dir):
        if any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in files):
            return root
    return None

def run_tests():
    unzip_dataset(zip_path, extract_to)

    dataset_info = load_dataset_info(yaml_path)
    if dataset_info:
        print(f"Classes: {dataset_info.get('names')}")

    model = YOLO(model_path)

    # Try to find the image directory dynamically
    image_dir = find_image_dir(extract_to)
    if not image_dir:
        raise FileNotFoundError("Could not find directory with images inside the unzipped dataset.")

    print(f"Using image directory: {image_dir}")

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            print(f"\nRunning inference on {filename}...")

            results = model.predict(image_path, verbose=False)
            df = results[0].to_df()
            if not df.empty:
                print(df[["class", "name", "xcenter", "ycenter", "width", "height", "confidence"]])
            else:
                print("No objects detected.")

            print(df if not df.empty else "No objects detected.")

if __name__ == "__main__":
    run_tests()
