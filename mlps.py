import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Settings
train_dir = "train-images/train"
test_dir = "TEST_images/TEST_images"
csv_path = "Test-IDs.csv"

image_size = 64
batch_size = 128
epochs = 75
learning_rate = 0.0001
hidden_layers_options = [
    [512],
    [512, 256],
    [512, 256, 128],
    [512, 256, 128, 64]
]
activation_functions = ["relu", "sigmoid", "tanh"]
optimizers = ["adam", "sgd"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# Dataset for training
class TrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label, subdir in enumerate(["pleasant", "unpleasant"]):
            full_path = os.path.join(root_dir, subdir)
            for fname in os.listdir(full_path):
                if fname.lower().endswith((".jpg", ".png")):
                    self.samples.append((os.path.join(full_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image.view(-1), label

# Dataset for test set
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
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image.view(-1), filename

# Custom subset dataset to split train/validation
class CustomSubset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image.view(-1), label

# MLP model definition
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, activation):
        super().__init__()
        act_fn = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }[activation]

        layers = []
        in_dim = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_fn)
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Split train/validation sets
full_dataset = TrainDataset(train_dir, transform)
train_samples, val_samples = train_test_split(full_dataset.samples, test_size=0.2, random_state=42)
train_dataset = CustomSubset(train_samples, transform)
val_dataset = CustomSubset(val_samples, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

input_dim = image_size * image_size

best_accuracy = 0
best_predictions = None
best_config_str = ""
best_model = None

results_log_path = "all_results.txt"
if os.path.exists(results_log_path):
    os.remove(results_log_path)

# Grid search over configurations
for hidden_layers in hidden_layers_options:
    for activation_type in activation_functions:
        for optimizer_type in optimizers:
            config_str = f"Hidden Layers: {hidden_layers} | Activation: {activation_type} | Optimizer: {optimizer_type}"
            print(f"\n🔧 Training with: {config_str}")

            model = MLP(input_size=input_dim, hidden_layers=hidden_layers, activation=activation_type).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate) if optimizer_type == "adam" else optim.SGD(model.parameters(), lr=learning_rate)

            for epoch in range(epochs):
                model.train()
                total_loss = 0
                y_true, y_pred = [], []

                for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                    X_batch, y_batch = X_batch.to(device), torch.tensor(y_batch).to(device)
                    optimizer.zero_grad()
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    loss.backward()
                    optimizer.step()

                    _, predicted = torch.max(preds, 1)
                    y_true.extend(y_batch.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                    total_loss += loss.item()

                epoch_acc = accuracy_score(y_true, y_pred)
                print(f"Epoch {epoch + 1} ➤ Loss: {total_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")

                # Evaluate on validation set
                model.eval()
                val_true, val_pred = [], []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(device), torch.tensor(y_batch).to(device)
                        outputs = model(X_batch)
                        _, predicted = torch.max(outputs, 1)
                        val_true.extend(y_batch.cpu().numpy())
                        val_pred.extend(predicted.cpu().numpy())

                val_acc = accuracy_score(val_true, val_pred)
                print(f"Validation Accuracy: {val_acc:.4f}")

            # Predict on test set
            test_dataset = TestDataset(csv_path, test_dir, transform)
            test_loader = DataLoader(test_dataset, batch_size=1)
            model.eval()
            predictions = []
            with torch.no_grad():
                for inputs, fname in tqdm(test_loader, desc="Predicting"):
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predictions.append((fname[0], predicted.item()))
                    print(f"Prediction: {fname[0]} ➤ Predicted label: {predicted.item()}")

            # Save results to log file
            with open(results_log_path, "a") as log_file:
                log_file.write(f"{config_str}\n")
                log_file.write(f"Train Accuracy: {epoch_acc:.4f}, Validation Accuracy: {val_acc:.4f}\n")
                log_file.write("Test predictions completed.\n")
                log_file.write("-" * 60 + "\n")

            # Save best model based on validation accuracy
            if best_predictions is None or val_acc > best_accuracy:
                best_predictions = predictions
                best_config_str = config_str
                best_accuracy = val_acc
                best_model = model
                torch.save(best_model.state_dict(), "best_model.pth")
                print("🌟 Best model saved as best_model.pth")

# Save final predictions
if best_predictions is not None:
    df_out = pd.DataFrame(best_predictions, columns=["filename", "predicted_label"])
    df_out.to_csv("submission.csv", index=False)
    print(f"🌟 submission.csv saved for → {best_config_str}")
