import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model_definition import SimpleCNN


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True
    )

    model = SimpleCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 3

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Loss={running_loss/len(train_loader):.4f}"
        )

    torch.save(model.state_dict(), "model.pth")

    print("Saved model.pth")

    os.makedirs("calib", exist_ok=True)

    for i in range(50):
        sample, _ = test_data[i]
        npy_path = f"calib/{i}.npy"
        import numpy as np
        np.save(npy_path, sample.numpy())

    print("Saved calibration data")


if __name__ == "__main__":
    train()