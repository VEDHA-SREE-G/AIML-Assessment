# Spam Email Classification using TF-IDF and Naive Bayes

## Objective

The objective of this project is to classify messages as spam or non-spam (ham) using Natural Language Processing (NLP) techniques and machine learning.

## Dataset

The dataset contains text messages labeled as:

* ham (non-spam)
* spam

Example:

ham → Hey, are you free today?

spam → Congratulations! You have won a prize.

## Project Workflow

1. Load the dataset.
2. Extract message text and labels.
3. Convert text into numerical features using TF-IDF.
4. Split data into training and testing sets.
5. Train a Multinomial Naive Bayes classifier.
6. Evaluate the model using accuracy, precision, recall, and F1-score.

## Requirements

Install required libraries:

pip install pandas scikit-learn

## How to Run

Place the dataset file in the project directory and run:

python Q6.py

## Output

The program displays:

* Accuracy
* Precision
* Recall
* F1-score

## Technologies Used

* Python 3.x
* Pandas
* Scikit-learn

## Author

Vedha Sree G
