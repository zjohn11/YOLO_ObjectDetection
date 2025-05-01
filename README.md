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

# Test Plan
We did a variety of things to test the model. The most important test was run right after training. We had a second set of data that was kept seperate from the training data that we fed into the model and compared against the labels to determine accuracy in both location and classification. The first models weren't as good as we needed, so we trained the model for longer and added more photos to our data sets. Once we got the accuracy suitably high on the downloaded data sets, we set up and ran the model on our local devices and pulled more images from the internet to verify each category was still being properly recognized. Once we confirmed each goal was met, we moved on to videos and created our own test video to verify and display the accuracy and consistency of the model on a variety of clips with a range of people and things in them. 

# Test scripts
The github workflow pulles in requirements from requirements.txt and executes test.py to verify results. It's having formatting issues and doesn't execute smoothly as one unit right now. The test uses a smaller dataset generated from the full yolo training set of data. smallerdataser.py outputs this new dataset as yolo_dataset.zip. The workflow pulls in and unzips this and uses it to test best.pt that has been released on this repo. The workflow is activated when anything is pushed or pulled to the repo and sends an email with the result when complete. 
We had attempted to use Jenkins but had several issues so pivoted last minute and had more success with Github workflows.

# Graphs
results.png shows the graphs for both training and validation that depict bounding box regression, classification, and the distribution of focal loss as the model developed. 

# Results
The results confirming that each category is recognized with at least 75% accuracy are graphed in PR_curve.png 
