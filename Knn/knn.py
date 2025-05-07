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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import warnings
warnings.filterwarnings("ignore")

# === CONFIG ===
#path of the data ,size of images
IMAGE_SIZE = (64, 64)
TRAIN_DIR = "train-images/train"
TEST_IDS_PATH = "Test-IDs.csv"
TEST_BASE_DIR = "TEST_images/TEST_images"



# === Load images ===
def load_images(folder, label):
    X, y = [], []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  #turn them to gray-scale
        if img is not None:
            resized = cv2.resize(img, IMAGE_SIZE) #change to image size
            normalized = resized / 255.0  #have it smaller
            X.append(normalized.flatten())  #flatten it
            y.append(label)
    return X, y

pleasant_X, pleasant_y = load_images(os.path.join(TRAIN_DIR, "pleasant"), 1)   #class 1 -pleasant faces loading
unpleasant_X, unpleasant_y = load_images(os.path.join(TRAIN_DIR, "unpleasant"), 0) #class 0 -unpleasant faces loading

X = np.array(pleasant_X + unpleasant_X)  #combines data from the 2 in a single numpy array to have the full dataset
y = np.array(pleasant_y + unpleasant_y)

# === Preprocessing ===
scaler = StandardScaler()   #This will normalize the data to have mean = 0 and standard deviation = 1 for each feature 
X_scaled = scaler.fit_transform(X)  #Fits the scaler to X and transforms the data

pca = PCA(n_components=100)  #principal component  analysis to Reduce noise and redundancy.and training faster
X_pca = pca.fit_transform(X_scaled)  #Applies PCA to the scaled data

# === Split data ===
X_train, X_val, y_train, y_val = train_test_split(       #splits the data to training set 80% and validation set 20%
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# === Grid search, parameters  ===
k_range = range(1, 21)
metrics = ['euclidean', 'cosine','manhattan']
best_f1 = 0
best_k = None
best_metric = None
best_model = None

print("\n Grid Search Results:")
for metric in metrics:
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')  #implements for every metric to a weight(distance)
        model.fit(X_train, y_train)     
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)  #calculates f1-score and keep the higher 
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

# === ROC Curve drawing it  ===
y_val_probs = best_model.predict_proba(X_val)[:, 1]  # πιθανότητες για κλάση 1
fpr, tpr, thresholds = roc_curve(y_val, y_val_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - ROC')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# === Load test data ===

test_ids_df = pd.read_csv(TEST_IDS_PATH)  # has columns 'ID' and 'Filename'

X_test = []
ids = []

for _, row in test_ids_df.iterrows():
    id_    = row['ID']
    fname  = row['Filename']
    img_path = os.path.join(TEST_BASE_DIR, fname)

    if not os.path.exists(img_path):
        print(f" File not found: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        resized = cv2.resize(img, IMAGE_SIZE)
        normalized = resized / 255.0  # Normalize test images too
        X_test.append(normalized.flatten())
       
    else:
        print(f" Could not read image: {fname}")
        X_test.append(np.zeros(IMAGE_SIZE[0] * IMAGE_SIZE[1]))

    ids.append(id_)

X_test = np.array(X_test)   #converts to a test the data in a single numpy aray
X_test_scaled = scaler.transform(X_test)  #scaling the images
X_test_pca = pca.transform(X_test_scaled)  #apply the pca for the same in  this test data

# === Predict using best model ===
y_test_pred = best_model.predict(X_test_pca)  #assing them 0 or 1 about is claas there are

# === Save predictions in required format  .csv ===
output_df = pd.DataFrame({
    'ID': ids,
    'LABEL': y_test_pred.astype(int)
})
output_df.to_csv("knn_best_predictions.csv", index=False)

print("\n✅ Saved predictions to: knn_best_predictions.csv")



