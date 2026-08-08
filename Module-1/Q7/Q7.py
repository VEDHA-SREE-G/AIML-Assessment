# Image Classification Pipeline

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Configuration
DATASET_PATH = "dataset"
IMAGE_SIZE = (128, 128)

# Storage Lists
images = []
labels = []
metadata = []

# Load Images
for label in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, label)

    if os.path.isdir(folder_path):

        for file in os.listdir(folder_path):

            image_path = os.path.join(folder_path, file)

            try:
                # Open Image
                img = Image.open(image_path)

                # Convert to RGB
                img = img.convert("RGB")

                # Resize Image
                img = img.resize(IMAGE_SIZE)

                # Convert Image to Array
                img_array = np.array(img)

                # Normalize Pixel Values (0-255 → 0-1)
                img_array = img_array / 255.0

                # Store Data
                images.append(img_array)
                labels.append(label)

                # Store Metadata
                metadata.append([
                    file,
                    label,
                    image_path,
                    img_array.shape
                ])

            except Exception as e:
                print(f"Error loading {image_path}: {e}")

# Features and Targets
X = np.array(images)
y = np.array(labels)

# Train / Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Metadata DataFrame
metadata_df = pd.DataFrame(
    metadata,
    columns=[
        "filename",
        "label",
        "path",
        "image_shape"
    ]
)

# Save Metadata CSV
metadata_df.to_csv(
    "metadata.csv",
    index=False
)

# Display Results
print("Total Images:", len(X))
print("Training Images:", len(X_train))
print("Validation Images:", len(X_val))

print("\nMetadata Preview:")
print(metadata_df.head())

print("\nMetadata saved as metadata.csv")
