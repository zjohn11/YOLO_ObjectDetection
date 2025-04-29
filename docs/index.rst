YOLO Object Detection Project
==============================

.. toctree::
   :maxdepth: 2
   :titlesonly:

   build_dataset
   yolov8_train
   GUI
   usage_instructions
   results_discussion
   conclusion

Introduction
------------

This project develops a YOLO-based object detection system, including:

- **Dataset Preparation** (`build_dataset.py`)
- **Model Training** (`yolov8_train.py`)
- **Video Annotation GUI** (`GUI.py`)

System Overview
---------------

The pipeline consists of:

1. Downloading, filtering & augmenting images  
2. Training a YOLOv8 detector  
3. Running a Tkinter GUI to annotate new videos  

