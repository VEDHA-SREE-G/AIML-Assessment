# Face Mask Detection Using MobileNetV2

## Project Overview

This project implements a Face Mask Detection system using MobileNetV2 and Transfer Learning. The model classifies facial images into two categories:

* With Mask
* Without Mask

The objective is to develop an efficient and lightweight image classification model suitable for real-time applications and edge-device deployment.

## Dataset Structure

dataset/
├── with_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── without_mask/
    ├── image1.jpg
    ├── image2.jpg
    └── ...

## Technologies Used

* Python 3.11+
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn

## Installation

Install the required libraries:

pip install tensorflow opencv-python numpy matplotlib scikit-learn

## How to Run

Place the dataset folder in the project directory.

Run the training script:

python Q9.py

## Project Workflow

1. Load image dataset
2. Resize images to 128 × 128
3. Normalize pixel values
4. Encode labels
5. Split data into training and testing sets
6. Apply data augmentation
7. Load pretrained MobileNetV2
8. Add custom classification layers
9. Train the model
10. Evaluate performance
11. Generate plots and save the trained model


## Model Architecture

### Base Model

* MobileNetV2 (Pretrained on ImageNet)

### Custom Layers

* AveragePooling2D
* Flatten
* Dense (128, ReLU)
* Dropout (0.5)
* Dense (2, Softmax)


## Training Parameters

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Image Size    | 128 × 128                       |
| Batch Size    | 16                              |
| Epochs        | 5                               |
| Optimizer     | Adam                            |
| Learning Rate | 0.0001                          |
| Loss Function | Sparse Categorical Crossentropy |



## Output Files

After execution, the following files are generated:

mask_detector.h5
accuracy_plot.png
loss_plot.png

### mask_detector.h5

Trained MobileNetV2 model.

### accuracy_plot.png

Training and validation accuracy graph.

### loss_plot.png

Training and validation loss graph.



## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix



## Edge Deployment Optimizations

The following optimizations can be applied:

* TensorFlow Lite Conversion
* Quantization (INT8)
* Model Pruning
* Reduced Image Resolution
* Lightweight MobileNetV2 Architecture

These optimizations improve inference speed and reduce memory consumption on edge devices.



## Applications

* Smart Surveillance Systems
* Airports
* Hospitals
* Public Transportation
* Educational Institutions
* Workplace Safety Monitoring



## Conclusion

This project demonstrates the use of Transfer Learning with MobileNetV2 for face mask detection. The model achieves efficient image classification while maintaining low computational requirements, making it suitable for deployment on mobile and embedded platforms.
