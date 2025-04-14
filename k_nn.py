import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# === CONFIG ===
IMAGE_SIZE = (64, 64)
TRAIN_DIR = r"D:\db\cs.uoi\Mixaniki mathisi\machine-learning-undergraduate-course-cse-uoi-gr\train-images\train"
TEST_IDS_PATH = r"D:\db\cs.uoi\Mixaniki mathisi\machine-learning-undergraduate-course-cse-uoi-gr\Test-IDs.csv"
TEST_BASE_DIR = r"D:\db\cs.uoi\Mixaniki mathisi\machine-learning-undergraduate-course-cse-uoi-gr\TEST_images\TEST_images"

# === Load images ===
def load_images(folder, label):
    X, y = [], []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized = cv2.resize(img, IMAGE_SIZE)
            X.append(resized.flatten())
            y.append(label)
    return X, y

pleasant_X, pleasant_y = load_images(os.path.join(TRAIN_DIR, "pleasant"), 1)
unpleasant_X, unpleasant_y = load_images(os.path.join(TRAIN_DIR, "unpleasant"), 0)

X = np.array(pleasant_X + unpleasant_X)
y = np.array(pleasant_y + unpleasant_y)

# === Preprocessing ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)

# === Split data ===
X_train, X_val, y_train, y_val = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# === Grid search ===
k_range = range(1, 21)
metrics = ['euclidean', 'cosine']
best_f1 = 0
best_k = None
best_metric = None
best_model = None

print("\n🔍 Grid Search Results:")
for metric in metrics:
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        print(f"k={k:2d}, metric={metric:<9s} → F1 Score = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_k = k
            best_metric = metric
            best_model = model

# === Validation metrics for best model ===
y_val_pred = best_model.predict(X_val)
acc = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred)
rec = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print("\n✅ Best Model:")
print(f"Best k      : {best_k}")
print(f"Best metric : {best_metric}")
print(f"Best F1     : {best_f1:.4f}")

print("\n📊 Validation Metrics (Best Model):")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# === Load test data ===
test_ids_df = pd.read_csv(TEST_IDS_PATH, header=None, names=['id', 'filename'])
X_test = []

for fname in test_ids_df['filename']:
    img_path = os.path.join(TEST_BASE_DIR, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        resized = cv2.resize(img, IMAGE_SIZE)
        X_test.append(resized.flatten())
    else:
        print(f"⚠️ Could not read image: {fname}")
        X_test.append(np.zeros(IMAGE_SIZE[0] * IMAGE_SIZE[1]))  # Placeholder

X_test = np.array(X_test)
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# === Predict using best model ===
y_test_pred = best_model.predict(X_test_pca)

# === Save predictions ===
output_df = pd.DataFrame({'Filename': test_ids_df['filename'], 'Predicted_Label': y_test_pred})
output_df.to_csv("knn_best_predictions.csv", index=False)
print("\n📁 Predictions saved to: knn_best_predictions.csv")
