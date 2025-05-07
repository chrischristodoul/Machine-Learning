import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# === CONFIGURATION ===
IMAGE_SIZE = (64, 64)
TRAIN_DIR = "train-images/train"
TEST_IDS_PATH = "Test-IDs.csv"
TEST_BASE_DIR = "TEST_images/TEST_images"

# === Load Training Images ===
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

pleasant_X, pleasant_y = load_images(os.path.join(TRAIN_DIR, "pleasant"), 1)  #loading pleassnt,unplesant classes
unpleasant_X, unpleasant_y = load_images(os.path.join(TRAIN_DIR, "unpleasant"), 0)

X = np.array(pleasant_X + unpleasant_X)
y = np.array(pleasant_y + unpleasant_y)

# === Preprocessing ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)



# === Bagging Classifier with Linear SVM Base (Cross-Validation) ===
n_estimators_list = [10,15 ,20]
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
best_f1 = 0
best_model = None
bagging_results = []

print("\n Bagging Classifier with Linear SVM (Cross-Validation):")
for kernel in kernels:
    for n in n_estimators_list:
        base_svm = SVC(kernel=kernel, probability=False)
        model = BaggingClassifier(estimator=base_svm, n_estimators=n, random_state=42, n_jobs=-1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        f1 = f1_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)

        print(f"kernel={kernel:<7} | n_estimators={n:<3} → F1 Score = {f1:.4f}")
        bagging_results.append({
            'kernel': kernel,
            'n_estimators': n,
            'F1_Score': f1,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_n = n
            best_kernel = kernel

# Retrain the best model on the full training data before test prediction
best_model.fit(X_scaled, y)


# === Save grid search results ===
pd.DataFrame(bagging_results).to_csv("bagging_results.csv", index=False)

# === Evaluate Best Model ===
y_val_pred = best_model.predict(X_val)
acc = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred)
rec = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print("\n Best Bagging Model:")
print(f"n_estimators : {best_n}")
print(f"F1 Score     : {f1:.4f}")

print("\n Validation Metrics:")
print(f"Accuracy     : {acc:.4f}")
print(f"Precision    : {prec:.4f}")
print(f"Recall       : {rec:.4f}")
print(f"F1 Score     : {f1:.4f}")

# === Load Test Set ===
test_ids_df = pd.read_csv(TEST_IDS_PATH, header=None, names=['id', 'filename'])
X_test = []

for fname in test_ids_df['filename']:
    path = os.path.join(TEST_BASE_DIR, fname)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        resized = cv2.resize(img, IMAGE_SIZE)
        X_test.append(resized.flatten())
    else:
        print(f" Could not read image: {fname}")
        X_test.append(np.zeros(IMAGE_SIZE[0] * IMAGE_SIZE[1]))

X_test = np.array(X_test)
X_test_scaled = scaler.transform(X_test)

# === Predict on Test Set ===
y_test_pred = best_model.predict(X_test_scaled)

# === Save Final Predictions ===
output_df = pd.DataFrame({
    'ID': test_ids_df['id'],
    'LABEL': y_test_pred.astype(int)
})
output_df.to_csv("bagging_best_predictions.csv", index=False)
print("\n Predictions saved to: bagging_best_predictions.csv")
