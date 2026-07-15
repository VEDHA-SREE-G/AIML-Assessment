
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(
    "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("\nDataset Shape:")
print(df.shape)

# =====================================================
# DATA CLEANING
# =====================================================

# Remove customerID
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Remove duplicates
duplicates = df.duplicated().sum()

print("\nDuplicate Rows:", duplicates)

df.drop_duplicates(inplace=True)

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================================
# EDA VISUALIZATIONS
# =====================================================

# Churn Distribution
plt.figure(figsize=(6,4))
df["Churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.tight_layout()
plt.savefig("churn_distribution.png")
plt.close()

# Contract Distribution
plt.figure(figsize=(6,4))
df["Contract"].value_counts().plot(kind="bar")
plt.title("Contract Type Distribution")
plt.tight_layout()
plt.savefig("contract_distribution.png")
plt.close()

# Tenure Distribution
plt.figure(figsize=(6,4))
plt.hist(df["tenure"], bins=20)
plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig("tenure_distribution.png")
plt.close()

print("\nEDA plots saved.")

# =====================================================
# LABEL ENCODING
# =====================================================

encoder = LabelEncoder()

for col in df.columns:

    if str(df[col].dtype) in ["object", "string", "str"]:

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

print("\nData Types After Encoding:")
print(df.dtypes)

# =====================================================
# FEATURES / TARGET
# =====================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# =====================================================
# LOGISTIC REGRESSION
# =====================================================

lr = LogisticRegression(
    max_iter=1000
)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_accuracy = accuracy_score(
    y_test,
    lr_pred
)

print("\n==============================")
print("LOGISTIC REGRESSION RESULTS")
print("==============================")

print("Accuracy:", lr_accuracy)

print(
    classification_report(
        y_test,
        lr_pred
    )
)

# =====================================================
# RANDOM FOREST
# =====================================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print("\n==============================")
print("RANDOM FOREST RESULTS")
print("==============================")

print("Accuracy:", rf_accuracy)

print(
    classification_report(
        y_test,
        rf_pred
    )
)

# =====================================================
# MODEL COMPARISON
# =====================================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    f"Logistic Regression Accuracy : {lr_accuracy:.4f}"
)

print(
    f"Random Forest Accuracy       : {rf_accuracy:.4f}"
)

# =====================================================
# SELECT BEST MODEL
# =====================================================

if rf_accuracy >= lr_accuracy:

    best_model = rf
    best_predictions = rf_pred
    model_name = "Random Forest"

else:

    best_model = lr
    best_predictions = lr_pred
    model_name = "Logistic Regression"

print("\nBest Model:", model_name)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    f"Confusion Matrix - {model_name}"
)

plt.savefig(
    "confusion_matrix.png"
)

plt.close()

print("Confusion matrix saved.")

# =====================================================
# ERROR ANALYSIS
# =====================================================

errors = X_test[
    y_test != best_predictions
]

print(
    "\nMisclassified Records:",
    len(errors)
)

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    best_model,
    "churn_model.pkl"
)

print(
    "\nModel saved as churn_model.pkl"
)

# =====================================================
# BUSINESS INSIGHTS
# =====================================================

print("\n==============================")
print("BUSINESS INSIGHTS")
print("==============================")

print(
    "1. Customers with shorter tenure are more likely to churn."
)

print(
    "2. Month-to-month contracts generally show higher churn."
)

print(
    "3. Customers with higher monthly charges tend to leave more often."
)

print(
    "4. Long-term contracts improve customer retention."
)

print(
    "5. Retention efforts should focus on new customers."
)

print("\nProject Completed Successfully.")

