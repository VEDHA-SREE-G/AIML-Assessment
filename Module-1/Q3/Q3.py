# Simple Iris Classifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load iris dataset
iris = load_iris()

# Extract features and target
X = iris.data
y = iris.target

# Split into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Decision Tree Classifier Model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train,y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate Accuracy
accuracy = accuracy_score(y_test, predictions)

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print("Accuracy: ",accuracy)
print("Confusion Matrix: ")
print(cm)
