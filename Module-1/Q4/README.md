# MNIST Digit Recognition using Logistic Regression

## Objective

The objective of this project is to build a handwritten digit recognition system using the MNIST dataset and a Logistic Regression classifier.

## Dataset

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9).

Image Size:

* 28 × 28 pixels

Features:

* 784 pixel values per image

Target:

* Digit labels from 0 to 9

## Requirements

Install required libraries:

pip install scikit-learn pandas matplotlib

## How to Run

Run the script:

python Q4.py

## Workflow

1. Load MNIST dataset from OpenML
2. Split dataset into training and testing sets
3. Scale features using StandardScaler
4. Train Logistic Regression model
5. Evaluate training and testing accuracy
6. Visualize correct and incorrect predictions

## Output

The program generates:

* Training Accuracy
* Testing Accuracy
* Five correctly classified digit images
* Five incorrectly classified digit images

## Technologies Used

* Python 3.x
* Scikit-learn
* Pandas
* Matplotlib

## Author

Vedha Sree G
