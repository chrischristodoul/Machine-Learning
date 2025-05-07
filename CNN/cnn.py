import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np
from itertools import product

# ----------------------------- CONFIG -----------------------------
train_dir = "train-images/train"
test_dir = "TEST_images/TEST_images"
csv_path = "Test-IDs.csv"
image_size = 64
epochs = 75
learning_rate = 0.0001
results_log_path = "cnn_results.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- TRANSFORMS -----------------------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# ----------------------------- DATASETS -----------------------------
class TrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label, subdir in enumerate(["pleasant", "unpleasant"]):
            full_path = os.path.join(root_dir, subdir)
            for fname in os.listdir(full_path):
                if fname.lower().endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(full_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = str(self.df.iloc[idx, 1])
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, filename

# ----------------------------- MODEL -----------------------------
class CNNClassifier(nn.Module):
    def __init__(self, conv_layers=3, fc_layers=[128], dropout=0.5):
        super().__init__()
        channels = [1, 32, 64, 128, 256, 512]
        conv = []
        for i in range(conv_layers):
            conv += [
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            ]
        self.conv = nn.Sequential(*conv)

        conv_out_size = image_size // (2 ** conv_layers)
        in_features = channels[conv_layers] * (conv_out_size ** 2)
        fc = []
        for h in fc_layers:
            fc += [nn.Linear(in_features, h), nn.ReLU(), nn.Dropout(dropout)]
            in_features = h
        fc.append(nn.Linear(in_features, 2))
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----------------------------- METRICS FUNCTION -----------------------------
def evaluate_metrics(y_true, y_pred, y_probs, set_name="Train",
                     conv_layers=None, fc_layers=None, dropout=None, batch_size=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    with open(results_log_path, "a") as log_file:
        log_file.write(f"\n[{set_name} Metrics]\n")
        log_file.write(
            f"Conv Layers: {conv_layers}, FC Layers: {fc_layers}, Dropout: {dropout}, Batch Size: {batch_size}\n")
        log_file.write(
            f"Accuracy: {acc:.4f} (conv={conv_layers}, fc={fc_layers}, dropout={dropout}, batch={batch_size})\n")
        log_file.write(
            f"Precision: {prec:.4f} (conv={conv_layers}, fc={fc_layers}, dropout={dropout}, batch={batch_size})\n")
        log_file.write(
            f"Recall: {rec:.4f} (conv={conv_layers}, fc={fc_layers}, dropout={dropout}, batch={batch_size})\n")
        log_file.write(
            f"F1-Score: {f1:.4f} (conv={conv_layers}, fc={fc_layers}, dropout={dropout}, batch={batch_size})\n")
        log_file.write(
            f"AUC: {roc_auc:.4f} (conv={conv_layers}, fc={fc_layers}, dropout={dropout}, batch={batch_size})\n")
        log_file.write("Confusion Matrix:\n")
        log_file.write(str(cm) + "\n")
        log_file.write("-" * 60 + "\n")

    return f1


# ----------------------------- SEARCH SPACE -----------------------------
conv_options = [2, 3, 4, 5]
fc_options = [[128], [128, 64], [256, 128, 64]]
dropout_options = [0.3, 0.5]
batch_options = [64, 128]

best_f1 = 0
best_predictions = None

if os.path.exists(results_log_path):
    os.remove(results_log_path)

# ----------------------------- GRID SEARCH -----------------------------
for conv_layers, fc_layers, dropout_rate, batch_size in product(conv_options, fc_options, dropout_options, batch_options):
    print(f"\n🔍 Testing: conv={conv_layers}, fc={fc_layers}, dropout={dropout_rate}, batch={batch_size}")

    full_dataset = TrainDataset(train_dir, transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNNClassifier(conv_layers=conv_layers, fc_layers=fc_layers, dropout=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}")

    # Validation
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)

    f1 = evaluate_metrics(
        y_true, y_pred, np.array(y_probs),
        set_name="Validation",
        conv_layers=conv_layers,
        fc_layers=fc_layers,
        dropout=dropout_rate,
        batch_size=batch_size
    )
    print(f"Validation F1 Score: {f1:.4f}")

    if f1 > best_f1:
        print("✅ New best model!")
        best_f1 = f1

        test_dataset = TestDataset(csv_path, test_dir, transform)
        test_loader = DataLoader(test_dataset, batch_size=1)
        predictions = []
        with torch.no_grad():
            for X, fname in tqdm(test_loader, desc="Predicting"):
                X = X.to(device)
                output = model(X)
                pred = output.argmax(dim=1).item()
                predictions.append((fname[0], pred))
        best_predictions = predictions

# Save best predictions
pd.DataFrame(best_predictions, columns=["filename", "predicted_label"]).to_csv("best_submission.csv", index=False)
print(f"🎯 Best F1: {best_f1:.4f} — Predictions saved to best_submission.csv")
