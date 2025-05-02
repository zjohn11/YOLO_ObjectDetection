import os
import zipfile
import yaml
from ultralytics import YOLO

zip_path = 'yolo_dataset.zip'
extract_to = 'yolo_dataset'
yaml_path = os.path.join(extract_to, 'dataset.yml')  # use .yml as in your file or update accordingly
model_path = 'best.pt'  # on release

# --------------------------------------------------------------------------------
def unzip_dataset(zip_path, extract_to):
    """
    Extracts the dataset zip to a directory if not already present.
    """
    if not os.path.exists(extract_to):
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        print("Dataset already unzipped.")


def load_dataset_info(yaml_path):
    """
    Loads and returns the dataset YAML metadata.
    """
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    print(f"Warning: {yaml_path} not found.")
    return None


def find_image_dir(root_dir):
    """
    Recursively searches for a folder containing image files.
    """
    for root, dirs, files in os.walk(root_dir):
        if any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in files):
            return root
    return None


def main():
    # Step 1: Unzip dataset if needed
    unzip_dataset(zip_path, extract_to)

    # Step 2: Load dataset info and print classes
    dataset_info = load_dataset_info(yaml_path)
    if dataset_info:
        print(f"Classes: {dataset_info.get('names')}")

    # Step 3: Load YOLO model
    model = YOLO(model_path)

    # Step 4: Find images directory
    image_dir = find_image_dir(extract_to)
    if not image_dir:
        raise FileNotFoundError(
            "Could not find directory with images inside the unzipped dataset."
        )
    print(f"Using image directory: {image_dir}")

    # Step 5: Run inference on each image and print results
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            print(f"\nRunning inference on {filename}...")

            results = model.predict(image_path, verbose=False)
            df = results[0].to_df()

            if not df.empty:
                df["xcenter"] = (df["xmin"] + df["xmax"]) / 2
                df["ycenter"] = (df["ymin"] + df["ymax"]) / 2
                df["width"] = df["xmax"] - df["xmin"]
                df["height"] = df["ymax"] - df["ymin"]

                print(
                    df[
                        ["class", "name", "confidence",
                         "xcenter", "ycenter", "width", "height"]
                    ]
                )
            else:
                print("No objects detected.")

# --------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
