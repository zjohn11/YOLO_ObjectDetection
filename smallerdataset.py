import os
import shutil
import random
import zipfile
from pathlib import Path

source_root = Path("C:/Users/brenn/fiftyone/open-images-v6/train")
source_images = source_root / "data"
source_labels = source_root / "labels"
output_dir = Path("yolo_dataset")
num_samples = 50

# --------------------------------------------------------------------------------
def main():
    # Remove existing output directory if it exists
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Create data/ and labels/ folders
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Sample image files
    all_images = list(source_images.glob("*.jpg"))
    sample_images = random.sample(all_images, min(num_samples, len(all_images)))

    for img_path in sample_images:
        label_path = source_labels / (img_path.stem + ".txt")

        # Copy image
        shutil.copy(img_path, output_dir / "data" / img_path.name)

        # Copy corresponding label if present
        if label_path.exists():
            shutil.copy(label_path, output_dir / "labels" / label_path.name)

    # Create dataset.yaml
    dataset_yaml = f"""
train: ./yolo_dataset/data
val: ./yolo_dataset/data
nc: 5
names: ["Dog", "Cat", "Person", "Lightsaber", "Dalek"]
"""

    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(dataset_yaml)

    # Zip the folder
    zip_path = Path("yolo_dataset.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(output_dir.parent)
                zipf.write(full_path, arcname=rel_path)

    print(f"\n✅ Mini dataset created and zipped as: {zip_path}")

# --------------------------------------------------------------------------------

if __name__ == "__main__":
    main()