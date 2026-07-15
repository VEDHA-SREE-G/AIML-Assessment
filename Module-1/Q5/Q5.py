import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

titanic = pd.read_csv("titanic.csv")

#Basic EDA
print(titanic.shape)
print(titanic.info())
print(titanic.describe())

#Clean Missing Values
titanic.drop(columns=["deck"],inplace=True)
titanic = titanic.dropna()

#Engineer Simple Features
titanic["familysize"] = (titanic["sibsp"] + titanic["parch"] + 1)
titanic["sex"] = titanic["sex"].map({"male":0,"female":1})
titanic["embarked"] = titanic["embarked"].map({"S":0,"C":1,"Q":2})

#Select Features
X = titanic[["pclass","sex","age","fare","familysize","embarked"]]
y = titanic["survived"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train,y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test,lr_pred)
print("Accuracy Score of Logistic Regression Model: ",lr_acc)

#Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train,y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test,rf_pred)
print("Accuracy Score of Random Forest Model: ",rf_acc)


