import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
import datetime
from imblearn.over_sampling import RandomOverSampler
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Paths ===
basedir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(basedir, "train-images", "train")
test_path = os.path.join(basedir, "TEST_images")

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Dataset / DataLoader ===
train_data = datasets.ImageFolder(train_path, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
class_names = train_data.classes

# === Custom Dataset for Test Set (no labels) ===
class CustomDatasetWithoutLabels(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                            if fname.lower().endswith((".jpg", ".png", ".jpeg"))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

test_data = CustomDatasetWithoutLabels(test_path, transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# === CNN Feature Extractors ===
cnn_models = {
    "resnet18": nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-1]),
    "vgg16": nn.Sequential(*list(models.vgg16(weights=models.VGG16_Weights.DEFAULT).features),
                           nn.AdaptiveAvgPool2d((7, 7))),
    "efficientnet_b0": nn.Sequential(
        *list(models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).children())[:-1],
        nn.AdaptiveAvgPool2d((1, 1))
    )
}

# === Classifiers with class balancing ===
classifiers = {
    "SVM": SVC(class_weight='balanced'),
    "RandomForest": RandomForestClassifier(class_weight='balanced'),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced')
}

# === Logging ===
log_file = open("model_training_log.txt", "w", encoding="utf-8")
log_file.write(f"Log started at {datetime.datetime.now()}\n\n")

best_f1_global = 0
best_result = {}

# === Feature extraction for train set ===
def extract_features(loader, model):
    model.eval()
    model.to(device)
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting train features"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            outputs = outputs.view(outputs.size(0), -1).cpu().numpy()
            features.extend(outputs)
            labels.extend(lbls.numpy())
    return np.array(features), np.array(labels)

# === Feature extraction for test set (no labels) ===
def extract_features_no_labels(loader, model):
    model.eval()
    model.to(device)
    features, filenames = [], []
    with torch.no_grad():
        for imgs, names in tqdm(loader, desc="Extracting test features"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            outputs = outputs.view(outputs.size(0), -1).cpu().numpy()
            features.extend(outputs)
            filenames.extend(names)
    return np.array(features), filenames

for cnn_name, feature_extractor in cnn_models.items():
    log_file.write(f"\n### CNN Model: {cnn_name.upper()} ###\n")

    # === Train Feature Extraction ===
    X_train, y_train = extract_features(train_loader, feature_extractor)
    log_file.write(f"Original Train shape: {X_train.shape}\n")

    # === PCA if needed ===
    pca = None
    if X_train.shape[1] > 512:
        pca = PCA(n_components=200)
        X_train = pca.fit_transform(X_train)
        log_file.write(f"Post-PCA Train shape: {X_train.shape}\n")
    else:
        log_file.write(f"Post-PCA skipped\n")

    # === Oversampling ===
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    log_file.write(f"Train resampled shape: {X_train_resampled.shape}\n")

    for clf_name, clf in classifiers.items():
        clf.fit(X_train_resampled, y_train_resampled)
        preds = clf.predict(X_train_resampled)
        acc = accuracy_score(y_train_resampled, preds)
        f1_macro = f1_score(y_train_resampled, preds, average='macro')

        log_file.write(f"{clf_name}: Train Accuracy = {acc:.4f}, F1_macro = {f1_macro:.4f}\n")
        print(f"[{cnn_name.upper()} - {clf_name}] Train Accuracy: {acc:.4f} | F1_macro: {f1_macro:.4f}")

        if f1_macro > best_f1_global:
            best_f1_global = f1_macro
            best_result = {
                "cnn": cnn_name,
                "classifier": clf_name,
                "f1": f1_macro,
                "model": feature_extractor,
                "clf": clf,
                "pca": pca if pca is not None else None
            }

# === Predict test set with best model ===
print("\n--- Predicting on TEST SET ---")
log_file.write(f"\n\n=== BEST OVERALL RESULT ===\n")
log_file.write(f"CNN: {best_result['cnn']}\n")
log_file.write(f"Classifier: {best_result['classifier']}\n")
log_file.write(f"Train F1 Score: {best_result['f1']:.4f}\n\n")

X_test, filenames = extract_features_no_labels(test_loader, best_result["model"])

if best_result.get("pca") is not None:
    print(f"[DEBUG] Applying PCA to test set (shape before: {X_test.shape})")
    X_test = best_result["pca"].transform(X_test)
    print(f"[DEBUG] Test shape after PCA: {X_test.shape}")
else:
    print(f"[DEBUG] PCA not applied to test set (shape: {X_test.shape})")

try:
    y_test_pred = best_result["clf"].predict(X_test)
except ValueError as e:
    print(f"[ERROR] Prediction failed: {e}")
    log_file.write(f"[ERROR] Prediction failed: {e}\n")
    log_file.close()
    raise SystemExit(1)

with open("test_predictions.txt", "w", encoding="utf-8") as f:
    for name, pred in zip(filenames, y_test_pred):
        f.write(f"{name},{class_names[pred]}\n")

log_file.write("Predictions saved to test_predictions.txt\n")
log_file.write(f"\nLog finished at {datetime.datetime.now()}\n")
log_file.close()
