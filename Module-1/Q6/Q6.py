# Spam Email Classification

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Read the file
df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label","message"])

# Extract the features and target
X = df["message"]
y = df["label"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Convert email text to TF-IDF features
X_tfidf = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf,y,test_size=0.2,random_state=42)

# Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# Evaluate Accuracy
acc = accuracy_score(y_test,y_pred)
print("Accuracy: ",acc)

# Evaluate precision, recall & F1-score
print(classification_report(y_test,y_pred))
