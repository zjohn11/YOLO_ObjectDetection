from ultralytics import YOLO
import torch

if __name__ == "__main__":

    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)

    model = YOLO('yolov8l.pt').to(device)

    model.train(
        data='./yolo_dataset/dataset.yaml',
        epochs = 30,
        batch = 4,
        imgsz = 416,
        cache=True,
        workers=0,
        device=device
    )

    # Evaluate the trained model on the validation set
    results = model.val(data='./yolo_dataset/dataset.yaml')
    print(results)