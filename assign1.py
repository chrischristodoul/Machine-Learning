import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# === SETTINGS ===
IMAGE_SIZE = (64, 64)
DATA_DIR = r"D:\db\cs.uoi\Mixaniki mathisi\machine-learning-undergraduate-course-cse-uoi-gr\train-images\train"

# === Load images and assign labels ===
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, IMAGE_SIZE)
            images.append(img_resized.flatten())
            labels.append(label)
    return images, labels

# Load both classes
X, y = [], []
pleasant_path = os.path.join(DATA_DIR, "pleasant")
unpleasant_path = os.path.join(DATA_DIR, "unpleasant")

pleasant_imgs, pleasant_lbls = load_images_from_folder(pleasant_path, 1)
unpleasant_imgs, unpleasant_lbls = load_images_from_folder(unpleasant_path, 0)

X.extend(pleasant_imgs)
y.extend(pleasant_lbls)
X.extend(unpleasant_imgs)
y.extend(unpleasant_lbls)

X = np.array(X)
y = np.array(y)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Test different Ks and distance metrics ===
k_values = [1, 3, 5, 7, 9, 11, 13]
distance_metrics = ['euclidean', 'cosine']
results = []

for metric in distance_metrics:
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({'K': k, 'Metric': metric, 'Accuracy': acc})
        print(f"Metric: {metric}, K: {k}, Accuracy: {acc:.4f}")

# === Plot accuracy for each setting ===
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
for metric in distance_metrics:
    subset = results_df[results_df['Metric'] == metric]
    plt.plot(subset['K'], subset['Accuracy'], marker='o', label=f"{metric.title()} Distance")

plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.title("K-NN Accuracy vs K for Different Distance Metrics")
plt.legend()
plt.grid(True)
plt.show()

# === Final model for detailed evaluation (e.g., best config) ===
best_result = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]
best_k = best_result["K"]
best_metric = best_result["Metric"]

print(f"\nBest config → Metric: {best_metric}, K: {best_k}")

knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Unpleasant (0)", "Pleasant (1)"],
            yticklabels=["Unpleasant (0)", "Pleasant (1)"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# === Show sample predictions ===
def show_predictions(images, true_labels, predicted_labels, num=6):
    plt.figure(figsize=(12, 6))
    for i in range(num):
        index = i
        img = images[index].reshape(IMAGE_SIZE)
        plt.subplot(2, num//2, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true_labels[index]} | Pred: {predicted_labels[index]}")
        plt.axis('off')
    plt.suptitle("Sample Test Predictions")
    plt.show()

show_predictions(X_test, y_test, y_pred)

