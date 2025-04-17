import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# === CONFIGURATION ===
IMAGE_SIZE = (64, 64)
TRAIN_DIR = r"D:\db\cs.uoi\Mixaniki mathisi\machine-learning-undergraduate-course-cse-uoi-gr\train-images\train"
TEST_IDS_PATH = r"D:\db\cs.uoi\Mixaniki mathisi\machine-learning-undergraduate-course-cse-uoi-gr\Test-IDs.csv"
TEST_BASE_DIR = r"D:\db\cs.uoi\Mixaniki mathisi\machine-learning-undergraduate-course-cse-uoi-gr\TEST_images\TEST_images"

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

pleasant_X, pleasant_y = load_images(os.path.join(TRAIN_DIR, "pleasant"), 1)
unpleasant_X, unpleasant_y = load_images(os.path.join(TRAIN_DIR, "unpleasant"), 0)

X = np.array(pleasant_X + unpleasant_X)
y = np.array(pleasant_y + unpleasant_y)

# === Preprocessing ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/Test Split ===
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Random Forest Grid Search ===
n_estimators_list = [10, 50, 100, 200]
max_depth_list = [5, 10, 20, None]

best_f1 = 0
best_model = None
rf_results = []

print("\n🌲 Random Forest Grid Search:")
for n in n_estimators_list:
    for depth in max_depth_list:
        model = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)

        print(f"n_estimators={n:<3}, max_depth={str(depth):<5} → F1 Score = {f1:.4f}")
        rf_results.append({'n_estimators': n, 'max_depth': depth, 'F1_Score': f1})

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_n = n
            best_depth = depth

# === Save grid search results ===
pd.DataFrame(rf_results).to_csv("rf_results.csv", index=False)

# === Evaluate Best Model ===
y_val_pred = best_model.predict(X_val)
acc = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred)
rec = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print("\n✅ Best Random Forest Model:")
print(f"n_estimators : {best_n}")
print(f"max_depth    : {best_depth}")
print(f"F1 Score     : {f1:.4f}")

print("\n📊 Validation Metrics:")
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
        print(f"⚠️ Could not read image: {fname}")
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
output_df.to_csv("rf_best_predictions.csv", index=False)
print("\n📁 Predictions saved to: rf_best_predictions.csv")
