
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    AveragePooling2D,
    Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "dataset"
IMAGE_SIZE = 128
MAX_PER_CLASS = 1000
BATCH_SIZE = 16
EPOCHS = 5

# ==========================================
# LOAD DATASET (BALANCED)
# ==========================================

data = []
labels = []

categories = ["with_mask", "without_mask"]

for category in categories:

    folder = os.path.join(DATASET_PATH, category)

    if not os.path.exists(folder):
        print("Folder not found:", folder)
        continue

    loaded = 0

    for file_name in os.listdir(folder):

        if loaded >= MAX_PER_CLASS:
            break

        image_path = os.path.join(folder, file_name)

        try:
            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            image = cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            image = image.astype("float32") / 255.0

            data.append(image)
            labels.append(category)

            loaded += 1

        except Exception:
            pass

    print(f"{category}: {loaded} images loaded")

# ==========================================
# NUMPY ARRAYS
# ==========================================

data = np.array(data, dtype=np.float16)
labels = np.array(labels)

print("\nTotal Images:", len(data))

# ==========================================
# LABEL ENCODING
# ==========================================

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

print("\nClasses:", encoder.classes_)

unique, counts = np.unique(
    labels,
    return_counts=True
)

print("\nLabel Distribution:")

for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

print("Train Classes:", np.unique(y_train))
print("Test Classes :", np.unique(y_test))

# ==========================================
# DATA AUGMENTATION
# ==========================================

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ==========================================
# MOBILENETV2
# ==========================================

baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
)

# ==========================================
# CUSTOM HEAD
# ==========================================

headModel = baseModel.output

headModel = AveragePooling2D(
    pool_size=(4, 4)
)(headModel)

headModel = Flatten()(headModel)

headModel = Dense(
    128,
    activation="relu"
)(headModel)

headModel = Dropout(0.5)(headModel)

headModel = Dense(
    2,
    activation="softmax"
)(headModel)

model = Model(
    inputs=baseModel.input,
    outputs=headModel
)

# ==========================================
# FREEZE BASE MODEL
# ==========================================

for layer in baseModel.layers:
    layer.trainable = False

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# TRAIN
# ==========================================

history = model.fit(
    aug.flow(
        X_train,
        y_train,
        batch_size=BATCH_SIZE
    ),
    validation_data=(
        X_test,
        y_test
    ),
    epochs=EPOCHS
)

# ==========================================
# SAVE MODEL
# ==========================================

model.save("mask_detector.h5")

print("\nModel Saved Successfully")

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    predicted_classes
)

print("\nAccuracy:")
print(accuracy)

# ==========================================
# REPORT
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predicted_classes,
        labels=[0, 1],
        target_names=[
            "Mask",
            "No Mask"
        ],
        zero_division=0
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predicted_classes,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

# ==========================================
# ACCURACY PLOT
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("accuracy_plot.png")
plt.show()

# ==========================================
# LOSS PLOT
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("loss_plot.png")
plt.show()