import fiftyone as fo
import fiftyone.zoo as foz
from icrawler.builtin import GoogleImageCrawler
import os
import cv2
import albumentations as A
from fiftyone.core.labels import Detection, Detections
import time

class_list = ["Dog", "Cat", "Person", "Lightsaber", "Dalek"]

# filtering out images that don't match desired class
def get_filtered_dataset(class_name, target_count, max_people=1):
    filtered_samples = []
    total_loaded = 0
    attempt = 1
    batch_size = target_count

    while len(filtered_samples) < target_count:
        print(f"Attempt {attempt}: Loading {batch_size} more samples...")

        try:
            raw_ds = foz.load_zoo_dataset(
                "open-images-v6",
                split="train",
                label_types=["detections"],
                classes=[class_name],
                max_samples=batch_size,
                shuffle=True,
                dataset_name=f"{class_name.lower()}_batch_{attempt}",
                reload=True
            )
        except Exception as e:
            print(f"Error during dataset loading: {e}")
            print("Retrying after 5 seconds...")
            time.sleep(5)  # wait a bit before retrying
            continue  # retry the current attempt

        for sample in raw_ds:
            detections = sample["ground_truth"].detections
            filtered = [d for d in detections if d.label == class_name]

            if class_name == "Person" and len(filtered) > max_people:
                continue  # Skip samples with too many people

            if filtered:
                sample["ground_truth"] = fo.Detections(detections=filtered)
                filtered_samples.append(sample)

            if len(filtered_samples) >= target_count:
                break

        total_loaded += batch_size
        attempt += 1

        print(f'Number of {class_name} samples: {len(filtered_samples)}')

    # Truncate the filtered_samples list to the target count
    filtered_samples = [sample.copy() for sample in filtered_samples[:target_count]]

    if f"{class_name}_filtered" in fo.list_datasets():
        fo.delete_dataset(f"{class_name}_filtered")

    # Create a new dataset with the filtered samples
    filtered_ds = fo.Dataset(name=f"{class_name}_filtered")
    filtered_ds.add_samples(filtered_samples)

    return filtered_ds


def main():
    # Load 600 images with at least one dog
    dog_ds = get_filtered_dataset('Dog', 10000)
    dog_ds.default_classes = class_list
    # Load cat images
    cat_ds = get_filtered_dataset('Cat', 10000)
    cat_ds.default_classes = class_list
    # load 600 human images
    human_ds = get_filtered_dataset('Person', 8000)
    human_ds.default_classes = class_list

    # Download Lightsaber images
    lightsaber_crawler = GoogleImageCrawler(storage={'root_dir': './lightsaber_images'})
    lightsaber_crawler.crawl(keyword='lightsaber', max_num=300, file_idx_offset=0)

    # Download Dalek images
    dalek_crawler = GoogleImageCrawler(storage={'root_dir': './dalek_images'})
    dalek_crawler.crawl(keyword='dalek', max_num=300, file_idx_offset=0)

    # Paths
    lightsaber_dir = "./lightsaber_images"
    dalek_dir = "./dalek_images"

    # Count how many images we already have of lightsaber
    existing_images = [f for f in os.listdir(lightsaber_dir) if f.lower().endswith(('.jpg', '.png'))]
    n_existing = len(existing_images)
    n_target = 1000
    n_needed = max(0, n_target - n_existing)

    # Define augmentations
    augment = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.GaussianBlur(p=0.2),
        A.RandomResizedCrop(size=(224, 224), scale=(0.95, 1.0), ratio=(0.9, 1.1), p=0.3),
        A.Resize(256, 256)
    ])

    # Apply augmentations to lightsaber
    save_count = 0
    i = 0
    while save_count < n_needed:
        img_path = os.path.join(lightsaber_dir, existing_images[i % n_existing])
        image = cv2.imread(img_path)

        if image is None:
            i += 1
            continue

        augmented = augment(image=image)['image']
        new_filename = f"aug_{save_count}_{existing_images[i % n_existing]}"
        cv2.imwrite(os.path.join(lightsaber_dir, new_filename), augmented)
        save_count += 1
        i += 1

    # Count how many images we already have of dalek
    existing_images = [f for f in os.listdir(dalek_dir) if f.lower().endswith(('.jpg', '.png'))]
    n_existing = len(existing_images)
    n_target = 1000
    n_needed = max(0, n_target - n_existing)

    # Apply augmentations to dalek
    save_count = 0
    i = 0
    while save_count < n_needed:
        img_path = os.path.join(dalek_dir, existing_images[i % n_existing])
        image = cv2.imread(img_path)

        if image is None:
            i += 1
            continue

        augmented = augment(image=image)['image']
        new_filename = f"aug_{save_count}_{existing_images[i % n_existing]}"
        cv2.imwrite(os.path.join(dalek_dir, new_filename), augmented)
        save_count += 1
        i += 1

    # Load lightsaber images into a dataset
    lightsaber_ds = fo.Dataset.from_dir(
        dataset_dir="./lightsaber_images",
        dataset_type=fo.types.ImageDirectory,
        name="lightsaber_dataset",
        classes=class_list
    )

    # Apply a classification label of "lightsaber" to each sample
    for sample in lightsaber_ds:
        # Example: adding a bounding box for a "lightsaber"
        detection = Detection(label="Lightsaber", bounding_box=[0.0, 0.0, 1.0, 1.0])  # [x, y, width, height]
        sample["ground_truth"] = Detections(detections=[detection])  # Add detection to the sample
        sample.save()

    # Load dalek images into a dataset
    dalek_ds = fo.Dataset.from_dir(
        dataset_dir="./dalek_images",
        dataset_type=fo.types.ImageDirectory,
        name="dalek_dataset",
        classes=class_list
    )

    # Apply a classification label of "dalek" to each sample
    for sample in dalek_ds:
        # Example: adding a bounding box for a "lightsaber"
        detection = Detection(label="Dalek", bounding_box=[0.0, 0.0, 1.0, 1.0])  # [x, y, width, height]
        sample["ground_truth"] = Detections(detections=[detection])  # Add detection to the sample
        sample.save()

    if 'combined_ds' in fo.list_datasets():
        fo.delete_dataset('combined_ds')

    # combining all datasets
    combined_ds = fo.Dataset(name='combined_ds')

    # List of datasets to merge
    datasets = [dog_ds, cat_ds, human_ds, lightsaber_ds, dalek_ds]

    # Add samples from each dataset to the combined dataset
    for ds in datasets:
        # Copy and detach samples to prevent conflicts
        samples = [s.copy() for s in ds]
        combined_ds.add_samples(samples)

    # getting number of imgs per class
    print(combined_ds.count_values("ground_truth.detections.label"))

    # Shuffle the dataset
    shuffled = combined_ds.shuffle(seed=42)

    # Get total number of samples
    total = len(shuffled)
    split_idx = int(0.8 * total)

    # Slice into train and val
    train_view = shuffled[:split_idx]
    val_view = shuffled[split_idx:]

    # Export train split
    train_view.export(
        export_dir="yolo_dataset",
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="train",
        classes=class_list,
    )

    # Export val split
    val_view.export(
        export_dir="yolo_dataset",
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="val",
        classes=class_list,
    )

if __name__ == "__main__":
    main()
