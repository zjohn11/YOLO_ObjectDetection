# YOLO_ObjectDetection
Fine-Tuning a Pretrained YOLOv8 model on a dataset with Dogs, Cats, Humans, Lightsabers, and Daleks

Link to Download Trained YOLO Model: [https://drive.google.com/file/d/1AtdU6g_ihLcNoyWy8Mj7X_WfqVEXle-p/view?usp=sharing]


# Downloading and using the GUI
Dependencies:
tkinter
ultralytics
cv2
os

Since the GUI is a Python script, you will also need an appropriate version of python installed to be able to run it. We used Python 3.12.10

Once all dependencies are installed, you can clone the repo to your local machine and download the model from the link above or the repo releases. The most important files are the model, GUI, and OSU image for the GUI.

Now that things are downloaded, simply navigate to the parent directory and run the GUI.py file. The GUI will open and prompt you to select a video to test with, and will output a video of the same name +_annotated when it is finished. 
