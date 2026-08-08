from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = fetch_openml("mnist_784",version=1, as_frame=False) 

# Extract the features and targets
X = mnist.data
y = mnist.target.astype(int)

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

# Training Accuracy
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print("Training Accuracy: ",train_acc)

# Testing Accuracy
test_pred =  model.predict(X_test)
test_acc = accuracy_score(y_test,test_pred)
print("Testing Accuracy: ",test_acc)

# 5 Correct Predictions
correct = []
count = 0
for i in range(len(y_test)):
    if y_test[i] == test_pred[i]:
        count += 1
        correct.append(i)
        plt.imshow(X_test[i].reshape(28,28),cmap="gray")
        plt.title(f"Actual={y_test[i]} Predicted={test_pred[i]}")
        plt.show()
        if count == 5:
            break
            
# 5 Incorrect Predictions
wrong = []
count = 0
for i in range(len(y_test)):
    if y_test[i] != test_pred[i]:
        count += 1
        wrong.append(i)
        plt.imshow(X_test[i].reshape(28,28),cmap="gray")
        plt.title(f"Actual={y_test[i]} Predicted={test_pred[i]}")
        plt.show()
        if count == 5:
            break



    
