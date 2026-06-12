"""
Train Fashion-MNIST CNN and generate Cost + Accuracy plot.
Matches the FashionMNISTProject.ipynb cells 25-37.
"""

import time
import json
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt

# --- Configuration ---
IMAGE_SIZE = 28
BATCH_SIZE = 100
N_EPOCHS = 5
LR = 0.1
SEED = 42

# Reproducibility
torch.manual_seed(SEED)

# --- Dataset ---
data_dir = os.path.join(os.path.dirname(__file__), ".fashion", "data")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = dsets.FashionMNIST(
    root=data_dir, train=True, download=True, transform=transform
)
val_dataset = dsets.FashionMNIST(
    root=data_dir, train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# --- Model: CNN_batch (matching notebook cell 25) ---
class CNN_batch(nn.Module):
    def __init__(self, out_1=16, out_2=32, number_of_classes=10):
        super(CNN_batch, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(out_1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        self.conv2_bn = nn.BatchNorm2d(out_2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(out_2 * 7 * 7, number_of_classes)
        self.bn_fc1 = nn.BatchNorm1d(number_of_classes)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = torch.relu(x)
        x = self.maxpool1(x)

        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        return x


model = CNN_batch(out_1=16, out_2=32, number_of_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# --- Training & Validation ---
N_val = len(val_dataset)
cost_list = []
accuracy_list = []

print(f"Training on CPU, {len(train_dataset)} train, {N_val} val samples")
print(f"Epochs: {N_EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE}")
print("-" * 50)

start_time = time.time()

for epoch in range(N_EPOCHS):
    model.train()
    epoch_cost = 0.0

    for x, y in train_loader:
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        epoch_cost += loss.item()

    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for x_test, y_test in val_loader:
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()

    accuracy = correct / N_val
    cost_list.append(epoch_cost)
    accuracy_list.append(accuracy)

    print(f"Epoch {epoch+1}/{N_EPOCHS} — Cost: {epoch_cost:.4f} — Accuracy: {accuracy*100:.2f}%")

elapsed = time.time() - start_time
print("-" * 50)
print(f"Training complete in {elapsed:.1f}s — Final accuracy: {accuracy_list[-1]*100:.2f}%")

# Save metrics
metrics = {
    "cost_list": cost_list,
    "accuracy_list": accuracy_list,
    "epochs": list(range(1, N_EPOCHS + 1)),
    "final_accuracy": accuracy_list[-1],
}

metrics_path = os.path.join(os.path.dirname(__file__), "fashionmnist_first3_images", "training_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics to {metrics_path}")

# --- Plot: Cost vs Accuracy ---
fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:red'
ax1.plot(cost_list, color=color, marker='o', linewidth=2, label='Cost')
ax1.set_xlabel('Epoch', color='black', fontsize=12)
ax1.set_ylabel('Cost', color=color, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(range(N_EPOCHS))
ax1.set_xticklabels([f"{i+1}" for i in range(N_EPOCHS)])

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(accuracy_list, color=color, marker='s', linewidth=2, label='Accuracy')
ax2.set_ylabel('Accuracy', color=color, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1.05)

# Add a horizontal line at 85%
ax2.axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='85% target')

fig.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), "fashionmnist_first3_images", "cost_accuracy_plot.png")
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to {plot_path}")

# Also save a clean version without the target line for the notebook
color = 'tab:red'
ax1.clear()
ax2.clear()
fig.clf()
plt.close(fig)

fig, ax1 = plt.subplots(figsize=(8, 5))
color = 'tab:red'
ax1.plot(cost_list, color=color, marker='o', linewidth=2)
ax1.set_xlabel('epoch', color=color)
ax1.set_ylabel('Cost', color=color)
ax1.tick_params(axis='y', color=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)
ax2.set_xlabel('epoch', color=color)
ax2.plot(accuracy_list, color=color, marker='s', linewidth=2)
ax2.tick_params(axis='y', color=color)
ax2.set_ylim(0, 1.05)

fig.tight_layout()
clean_plot_path = os.path.join(os.path.dirname(__file__), "fashionmnist_first3_images", "cost_accuracy_clean.png")
fig.savefig(clean_plot_path, dpi=150, bbox_inches='tight')
print(f"Saved clean plot to {clean_plot_path}")

print("\nDone.")
