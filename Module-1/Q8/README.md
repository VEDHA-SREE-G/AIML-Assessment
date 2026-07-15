# Manufacturing Defect Detection using CNN

## Overview

This project classifies steel surface defects using the NEU Surface Defect Dataset.

The system uses a Convolutional Neural Network (CNN) to identify six defect categories:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

## Technologies Used

- Python 3.12
- TensorFlow/Keras
- NumPy
- Scikit-Learn

## Dataset

NEU Surface Defect Database

Classes:

1. crazing
2. inclusion
3. patches
4. pitted_surface
5. rolled-in_scale
6. scratches

## Preprocessing

- Resize images to 128x128
- Normalize pixel values
- Data augmentation

## Model Architecture

CNN consisting of:

- Conv2D
- MaxPooling2D
- Dense Layers
- Dropout

## How to Run

Install dependencies:

pip install tensorflow numpy scikit-learn pillow

Run:

py Q8.py

## Output

- Validation Accuracy
- Classification Report
- Saved Model

Model file:

neu_defect_detection_model.h5