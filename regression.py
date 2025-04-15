import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Paths
train_dir = "train-images/train"
test_dir = "TEST_images/TEST_images"
csv_path = "Test-IDs.csv"
log_path = "log_results.txt"
submission_path = "submission.csv"

# Load train data
def load_data(root_dir):
    X, y = [], []
    for label, subdir in enumerate(["pleasant", "unpleasant"]):
        folder = os.path.join(root_dir, subdir)
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".png")):
                img_path = os.path.join(folder, fname)
                img = Image.open(img_path).convert("L").resize((64, 64))
                img_array = np.array(img).flatten()
                X.append(img_array)
                y.append(label)
    return np.array(X), np.array(y)

# Load test data
def load_test_data(csv_path, test_dir):
    df = pd.read_csv(csv_path)
    X_test, filenames = [], []
    for i in range(len(df)):
        fname = df.iloc[i, 1]
        path = os.path.join(test_dir, fname)
        img = Image.open(path).convert("L").resize((64, 64))
        img_array = np.array(img).flatten()
        X_test.append(img_array)
        filenames.append(fname)
    return np.array(X_test), filenames

# Prepare data
from PIL import Image

X, y = load_data(train_dir)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Test data
X_test, test_filenames = load_test_data(csv_path, test_dir)
X_test = scaler.transform(X_test)

# Hyperparameter grid
C_values = [0.01, 0.1, 1, 10, 100]
best_val_acc = 0
best_submission = None
best_config = ""

# Clear old log
if os.path.exists(log_path):
    os.remove(log_path)

# Train and evaluate models
for C in C_values:
    print(f"\nTraining Logistic Regression with C={C}")
    clf = LogisticRegression(C=C, max_iter=1000)
    clf.fit(X_train, y_train)

    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)

    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Log results
    with open(log_path, "a") as f:
        f.write(f"C={C} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}\n")

    # If best, save submission
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_config = f"C={C}"
        test_preds = clf.predict(X_test)
        best_submission = pd.DataFrame({
            "filename": test_filenames,
            "predicted_label": test_preds
        })

# Save best submission
if best_submission is not None:
    best_submission.to_csv(submission_path, index=False)
    print(f"\n✅ Best model saved: {best_config} with val acc: {best_val_acc:.4f}")
    print(f"📁 Submission file saved as: {submission_path}")
