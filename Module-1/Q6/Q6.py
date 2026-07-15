import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("sms.tsv",sep="\t",header=None,names=["label","message"])
X = df["message"]
y = df["label"]
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf,y,test_size=0.2,random_state=42)
model = MultinomialNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print("Accuracy: ",acc)
print(classification_report(y_test,y_pred))