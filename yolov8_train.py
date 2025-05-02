from ultralytics import YOLO
import torch

# -----------------------------------------------------------------------------
# TRAIN & EVALUATION SCRIPT
# -----------------------------------------------------------------------------
def main():
    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)

    model = YOLO('yolov8x.pt').to(device)

    model.train(
        data='./yolo_dataset/dataset.yaml',
        epochs=50,
        batch=16,
        imgsz=416,
        cache=False,
        workers=0,
        device=device
    )

    # Evaluate the trained model on the validation set
    results = model.val(data='./yolo_dataset/dataset.yaml')
    print(results)

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
