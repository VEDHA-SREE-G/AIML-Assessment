# Image Classification Preprocessing Pipeline

## Objective

The objective of this project is to build a reusable image preprocessing pipeline that prepares image datasets for machine learning and deep learning tasks.

## Dataset Structure

The dataset is organized into folders where each folder represents a class label.

Example:

dataset/

├── cats/

│   ├── cat1.jpg

│   ├── cat2.jpg

│

├── dogs/

│   ├── dog1.jpg

│   ├── dog2.jpg

## Project Workflow

1. Load images from dataset folders.
2. Convert images to RGB format.
3. Resize images to a fixed size of 128 × 128 pixels.
4. Normalize pixel values to the range 0–1.
5. Store image labels.
6. Create metadata for all images.
7. Split data into training and validation sets.
8. Export metadata to a CSV file.

## Requirements

Install the required libraries:

pip install pillow pandas numpy scikit-learn

## How to Run

Place the dataset folder in the project directory and execute:

python Q7.py

## Output

The program generates:

* Processed image arrays
* Training and validation datasets
* metadata.csv file

## Technologies Used

* Python 3.x
* NumPy
* Pandas
* Pillow (PIL)
* Scikit-learn

## Author

Vedha Sree G
