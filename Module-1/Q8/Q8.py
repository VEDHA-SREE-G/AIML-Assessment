# Manufacturing Defect Detection

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import numpy as np

# Dataset Path
DATASET_PATH = "NEU-DET/train/images"

# Configuration
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 3

# Data Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training Data
train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

# Validation Data
val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


# CNN Model
model = Sequential([
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(128,128,3)
    ),

    MaxPooling2D(2,2),

    Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(2,2),

    Conv2D(
        128,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(2,2),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        train_data.num_classes,
        activation="softmax"
    )
])

# Compile Model
model.compile(
    optimizer=Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Model Summary
model.summary()

# Train Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Evaluate
loss, accuracy = model.evaluate(val_data)

print("\nValidation Accuracy:", accuracy)

# Predictions
y_pred = model.predict(val_data)

y_pred_classes = np.argmax(
    y_pred,
    axis=1
)

y_true = val_data.classes

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred_classes,
        target_names=list(
            train_data.class_indices.keys()
        )
    )
)

# Save Model
model.save(
    "neu_defect_detection_model.h5"
)

print("\nModel Saved Successfully")
