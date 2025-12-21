"""
CSE480: Machine Vision — Lab Assignment #06
Author: (Your Name)
Repository: vision-labs

Task 1 (Scikit-Learn):
1) Load the digits dataset.
2) Display the first 50 images in a grid.
3) Train/Test split (test_size=0.25, random_state=42),
   apply standardization correctly,
   train a k-NN classifier with k=3,
   and report the test accuracy.

Run:
    python lab06_vision.py
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def show_first_50_images(digits):
    """
    Show the first 50 images (8x8) from the digits dataset,
    arranged as 5 rows x 10 columns.
    """
    fig, axes = plt.subplots(5, 10, figsize=(10, 5))
    axes = axes.ravel()

    for i in range(50):
        axes[i].imshow(digits.data[i].reshape(8, 8), cmap="gray")
        axes[i].set_title(str(digits.target[i]), fontsize=9)
        axes[i].axis("off")

    plt.suptitle("Digits Dataset — First 50 Images", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    # 1) Load dataset
    digits = load_digits()

    print("Dataset keys:", digits.keys())
    print("Data shape (flattened):", digits.data.shape)      # (n_samples, 64)
    print("Images shape:", digits.images.shape)              # (n_samples, 8, 8)
    print("Targets shape:", digits.target.shape)             # (n_samples,)

    # 2) Show first 50 images
    show_first_50_images(digits)

    # 3) Train-test split
    X = digits.data  # flattened features (64 per image)
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Standardization: fit on TRAIN only, transform both train and test
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # k-NN with k=3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_std, y_train)

    # Evaluate
    y_pred = knn.predict(X_test_std)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy (k=3): {acc:.4f}")


if __name__ == "__main__":
    main()
