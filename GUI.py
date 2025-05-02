import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
import os

# Load your trained YOLO model here
# (will instantiate under the guard below)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_path = os.path.splitext(video_path)[0] + "_annotated.mp4"
    out = cv2.VideoWriter(
        out_path, 
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, 
        (width, height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, imgsz=416, conf=0.5, device=0)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"[✔] Saved annotated video to: {out_path}")


def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4")]
    )
    if file_path:
        process_video(file_path)


if __name__ == "__main__":
    # Load your trained YOLO model here
    model = YOLO("yolov8n.pt")  # update with model name/path

    root = tk.Tk()
    root.title("OSU YOLO Video Annotator")

    root.geometry("800x400")
    root.configure(bg="#FE5C00")

    label = tk.Label(
        root, 
        text="Select a .mp4 video to annotate with YOLO",
        bg="#FE5C00", 
        fg="white", 
        font=("Helvetica", 12)
    )
    label.pack(pady=20)

    btn = tk.Button(
        root, 
        text="Choose Video", 
        command=open_file,
        bg="black", 
        fg="white", 
        font=("Helvetica", 12)
    )
    btn.pack(pady=10)

    logo = tk.PhotoImage(file="OSU.png")
    logo_label = tk.Label(root, image=logo, bg="#FE5C00")
    logo_label.pack(pady=10)

    root.mainloop()
