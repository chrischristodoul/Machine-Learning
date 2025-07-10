# Machine Learning Classification Methods

## Overview
In this assignment, you will **implement, analyze, and compare** the performance of various machine learning classification algorithms on the provided dataset. Your goal is to systematically evaluate these methods by experimenting with different hyperparameters and settings.

---

## 1. K-Nearest Neighbors (K-NN)
- Implement the K-NN classifier and study how performance changes with:
  - Different values of **K** (number of neighbors)
  - Various **distance metrics** (Euclidean, Cosine)

---

## 2. Shallow and Deep Neural Networks (Fully Connected MLPs)
- Implement feedforward neural networks with:
  - **1 to 4 hidden layers**
  - Varying hidden units per layer: **512, 256, 128, or 64 neurons**
  - Different **activation functions** (ReLU, Sigmoid, Tanh)
  - Different **optimization algorithms** (Adam, SGD)

- Compare the impact of **depth** (number of layers) and **width** (number of neurons) on classification performance.

---

## 3. Support Vector Machines (SVMs)
- Train SVM classifiers with different **kernel functions** (linear, cosine, or RBF kernel).
- Experiment with different values of the regularization hyperparameter (**C**): `0.1, 1, 10, 100`
- Discuss how these affect the decision boundaries and performance.

---

## 4. Logistic Regression
- Experiment with different **regularization strengths** to study **overfitting** and model performance.

---

## 5. Ensemble Learning Methodologies
Implement and compare different ensemble techniques that combine multiple models to improve performance:
- **Random Forest Classifier**  
  - Vary the number of decision trees and maximum depth of individual trees.
- **Bagging Classifier**  
  - Use a linear SVM as the base classifier with 10–20 estimators.
- **AdaBoost Classifier**  
  - Use a decision tree as the base learner with 100–200 estimators.

---

## 6. Convolutional Neural Networks (CNN)
Implement and study a CNN-based classifier that works directly on images:
- **2 to 5 convolutional layers** using 3×3 kernels
- **Batch Normalization** (optional but recommended)
- **ReLU activation** after each convolutional layer
- **2×2 max pooling** layers to reduce dimensionality
- A fully connected **classification head** with 1 to 3 dense layers
- Final output layer with **softmax activation**

*Optional:* Experiment with different architectures, **dropout rates**, and **batch sizes** to optimize performance.

---

## Evaluation & Submission
- The primary metric for evaluating your models will be the **F1-score**, which will determine your ranking.
- Your submitted code must also include other standard classification metrics:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix
  - ROC Curve

These metrics should be calculated on the training set and included in your final report.

---

## Deliverables
- Well-documented code for each classifier.
- A report comparing the models, discussing experiments and results.
- Plots and metrics for each method.
- Submission file with your final predictions and evaluation metrics.

---

Happy coding and good luck!

