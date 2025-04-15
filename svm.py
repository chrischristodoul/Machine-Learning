import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# === CONFIG ===
IMAGE_SIZE = (64, 64)
TRAIN_DIR = r"D:\db\cs.uoi\Mixaniki mathisi\machine-learning-undergraduate-course-cse-uoi-gr\train-images\train"
TEST_IDS_PATH = r"D:\db\cs.uoi\Mixaniki mathisi\machine-learning-undergraduate-course-cse-uoi-gr\Test-IDs.csv"
TEST_BASE_DIR = r"D:\db\cs.uoi\Mixaniki mathisi\machine-learning-undergraduate-course-cse-uoi-gr\TEST_images\TEST_images"
C_values = [0.1, 1, 10, 100]
kernels = ['linear', 'rbf', 'cosine']

# === Load train images ===
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

# === Split into train/validation ===
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Grid Search ===
best_f1 = 0
best_model = None
best_kernel = None
best_C = None
results = []

print("\n🔍 Grid Search Results:")
for kernel in kernels:
    for C in C_values:
        if kernel == 'cosine':
            K_train = cosine_similarity(X_train, X_train)
            K_val = cosine_similarity(X_val, X_train)
            model = SVC(kernel='precomputed', C=C)
            model.fit(K_train, y_train)
            y_pred = model.predict(K_val)
        else:
            model = SVC(kernel=kernel, C=C)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

        f1 = f1_score(y_val, y_pred)
        print(f"Kernel={kernel:<7s}, C={C:<4} → F1 Score = {f1:.4f}")
        results.append({'Kernel': kernel, 'C': C, 'F1_Score': f1})

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_kernel = kernel
            best_C = C

# === Validation metrics ===
if best_kernel == 'cosine':
    K_val = cosine_similarity(X_val, X_train)
    y_val_pred = best_model.predict(K_val)
else:
    y_val_pred = best_model.predict(X_val)

acc = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred)
rec = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print("\n✅ Best Model:")
print(f"Kernel     : {best_kernel}")
print(f"C          : {best_C}")
print(f"Best F1    : {f1:.4f}")

print("\n📊 Validation Metrics (Best Model):")
print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f}")
print(f"F1 Score   : {f1:.4f}")

# === Save grid search results ===
results_df = pd.DataFrame(results)
results_df.to_csv("svm_results.csv", index=False)

# === Load test set from CSV ===
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

# === Predict on test set ===
if best_kernel == 'cosine':
    K_test = cosine_similarity(X_test_scaled, X_train)
    y_test_pred = best_model.predict(K_test)
else:
    y_test_pred = best_model.predict(X_test_scaled)

# === Save predictions ===
# Ensure ID is included from the CSV
output_df = pd.DataFrame({
    'ID': test_ids_df['id'],
    'LABEL': y_test_pred.astype(int)  # Ensure integer labels
})

output_df.to_csv("svm_best_predictions.csv", index=False)
print("\n📁 Final predictions saved to: svm_best_predictions.csv")

