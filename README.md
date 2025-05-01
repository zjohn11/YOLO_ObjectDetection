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

# Autodocumentation

# Test plan
To read the test plan document, open testPlan

# Test scripts
The github workflow pulles in requirements from requirements.txt and executes test.py to verify results. It's having formatting issues and doesn't execute smoothly as one unit right now. The test uses a smaller dataset generated from the full yolo training set of data. smallerdataser.py outputs this new dataset as yolo_dataset.zip. The workflow pulls in and unzips this and uses it to test best.pt that has been released on this repo. The workflow is activated when anything is pushed or pulled to the repo and sends an email with the result when complete. 
We had attempted to use Jenkins but had several issues so pivoted last minute and had more success with Github workflows.

# Graphs

# Results
