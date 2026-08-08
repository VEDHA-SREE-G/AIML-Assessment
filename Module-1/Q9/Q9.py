#  Face Mask Detection

import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train = DataLoader(
    datasets.ImageFolder("dataset", transform),
    batch_size=32,
    shuffle=True
)

val = DataLoader(
    datasets.ImageFolder("dataset", transform),
    batch_size=32
)

model = models.mobilenet_v3_small(weights="DEFAULT")
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    print("Starting epoch", epoch + 1)
    model.train()
    for x, y in train:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()

model.eval()
true, pred = [], []

with torch.no_grad():
    for x, y in val:
        pred.extend(model(x.to(device)).argmax(1).cpu().numpy())
        true.extend(y.numpy())

print("Precision:", precision_score(true, pred))
print("Recall:", recall_score(true, pred))

x, _ = next(iter(val))
x = x[:1].to(device)

start = time.time()
with torch.no_grad():
    for _ in range(100):
        model(x)

print("Inference:", (time.time() - start) / 100 * 1000, "ms")
