# Customer Churn Prediction Using Machine Learning

## Project Overview

This project predicts customer churn using machine learning techniques. The complete workflow includes data cleaning, exploratory data analysis, preprocessing, model training, evaluation, error analysis, and business insights.

## Dataset

Dataset: Telco Customer Churn Dataset

Target Variable:

* Churn (0 = No, 1 = Yes)

## Technologies Used

* Python 3.11+
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib

## Project Workflow

1. Load Dataset
2. Clean Data
3. Perform Exploratory Data Analysis
4. Encode Categorical Features
5. Split Training and Testing Data
6. Train Logistic Regression Model
7. Train Random Forest Model
8. Compare Model Performance
9. Generate Evaluation Metrics
10. Perform Error Analysis
11. Save Best Model

## Installation

Install required packages:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

## Running the Project

Place the dataset file inside:

```text
dataset/
└── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Run:

```bash
py Q10.py
```

or

```bash
python Q10.py
```

## Generated Files

After successful execution:

```text
churn_distribution.png
contract_distribution.png
tenure_distribution.png
confusion_matrix.png
churn_model.pkl
```

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

## Business Insights

* Customers with shorter tenure are more likely to churn.
* Month-to-month contracts show higher churn.
* High monthly charges increase churn probability.
* Long-term contracts improve retention.

## Conclusion

This project demonstrates a complete end-to-end machine learning workflow for customer churn prediction and provides actionable insights for improving customer retention.
