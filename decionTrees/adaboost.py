import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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






pleasant_X, pleasant_y = load_images(os.path.join(TRAIN_DIR, "pleasant"), 1)
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

# === AdaBoost with Decision Tree Base ===
n_estimators_list = [100,125,150,175, 200,250]
best_f1 = 0
best_model = None
adaboost_results = []

print("\n AdaBoost with Decision Tree base:")
for n in n_estimators_list:
    base_tree = DecisionTreeClassifier(max_depth=1)
    model = AdaBoostClassifier(estimator=base_tree, n_estimators=n, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)

    print(f"n_estimators={n:<3} → F1 Score = {f1:.4f}")
    adaboost_results.append({'n_estimators': n, 'F1_Score': f1})

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_n = n

# === Save grid search results ===
pd.DataFrame(adaboost_results).to_csv("adaboost_results.csv", index=False)

# === Evaluate Best Model ===
y_val_pred = best_model.predict(X_val)
acc = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred)
rec = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print("\n Best AdaBoost Model:")
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
output_df.to_csv("adaboost_best_predictions.csv", index=False)
print("\n Predictions saved to: adaboost_best_predictions.csv")

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# === Plot and save the best individual decision tree from the best AdaBoost model ===
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Find the most influential tree (highest weight)
best_tree_index = np.argmax(best_model.estimator_weights_)
best_tree = best_model.estimators_[best_tree_index]

# Plot and save it
plt.figure(figsize=(10, 6))
plot_tree(best_tree, filled=True, class_names=["unpleasant", "pleasant"])
plt.title(f"Best Single Tree in Best AdaBoost (index={best_tree_index}, weight={best_model.estimator_weights_[best_tree_index]:.4f})")
plt.savefig("best_individual_tree.png")
plt.close()


def visualize_adaboost_trees(model, n_trees=3):
    for i in range(min(n_trees, len(model.estimators_))):
        estimator = model.estimators_[i]
        plt.figure(figsize=(10, 6))
        plot_tree(estimator, filled=True, class_names=["unpleasant", "pleasant"])
        plt.title(f"Decision Tree {i+1} in AdaBoost Ensemble")
        plt.show()

# Call the function to visualize
visualize_adaboost_trees(model, n_trees=3)

