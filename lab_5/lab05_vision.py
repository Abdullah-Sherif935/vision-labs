"""
CSE480: Machine Vision — Lab Assignment #05
Author: (Your Name)
Repository: vision-labs

This script implements:
Task 1) A shallow CNN for Fashion-MNIST classification.
Task 2) Transfer Learning on CIFAR-10 using a pre-trained model (MobileNetV2 by default).

Run:
    python lab05_vision.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Shared Utilities
# -----------------------------
def plot_history(history, title_prefix="Model"):
    """Plot training/validation loss and accuracy curves."""
    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    # Loss
    if "loss" in hist:
        plt.figure()
        plt.plot(epochs, hist["loss"], label="train_loss")
        if "val_loss" in hist:
            plt.plot(epochs, hist["val_loss"], label="val_loss")
        plt.title(f"{title_prefix} - Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

    # Accuracy (key names may vary by TF version)
    acc_key = "accuracy" if "accuracy" in hist else ("acc" if "acc" in hist else None)
    val_acc_key = "val_accuracy" if "val_accuracy" in hist else ("val_acc" if "val_acc" in hist else None)

    if acc_key is not None:
        plt.figure()
        plt.plot(epochs, hist[acc_key], label="train_accuracy")
        if val_acc_key is not None:
            plt.plot(epochs, hist[val_acc_key], label="val_accuracy")
        plt.title(f"{title_prefix} - Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

    plt.show()


def show_sample_predictions(model, x, y_true, class_names, n=12, title="Sample Predictions"):
    """Display sample images with predicted and true labels."""
    idx = np.random.choice(len(x), size=n, replace=False)
    x_s = x[idx]
    y_s = y_true[idx]

    preds = model.predict(x_s, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    cols = 4
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(12, 3 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        img = x_s[i]
        if img.shape[-1] == 1:
            plt.imshow(img.squeeze(), cmap="gray")
        else:
            plt.imshow(img)
        true_label = class_names[int(y_s[i])]
        pred_label = class_names[int(y_pred[i])]
        color = "green" if y_pred[i] == y_s[i] else "red"
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=color, fontsize=10)
        plt.axis("off")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Task 1: Shallow CNN on Fashion-MNIST
# -----------------------------
def build_fashion_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Shallow CNN:
    Conv -> MaxPool -> Conv -> MaxPool -> Flatten -> Dense -> Output
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def run_task1_fashion_mnist(epochs=10, batch_size=128):
    print("\n============================")
    print("Task 1: CNN on Fashion-MNIST")
    print("============================")

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalize to [0,1] and add channel dimension
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]  # (N, 28, 28, 1)
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    model = build_fashion_cnn(input_shape=x_train.shape[1:], num_classes=10)
    model.summary()

    t0 = time.time()
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    train_time = time.time() - t0

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[Fashion-MNIST] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    print(f"[Fashion-MNIST] Training Time: {train_time:.2f} seconds")

    plot_history(history, title_prefix="Fashion-MNIST CNN")
    show_sample_predictions(model, x_test, y_test, class_names, n=12, title="Fashion-MNIST: Sample Predictions")

    return test_acc, train_time


# -----------------------------
# Task 2: Transfer Learning on CIFAR-10
# -----------------------------
def build_transfer_model(base_name="MobileNetV2", input_shape=(32, 32, 3), num_classes=10):
    """
    Load a pre-trained model without top layers (include_top=False),
    freeze its convolutional base, and add custom head for CIFAR-10.
    """
    if base_name == "VGG16":
        base = keras.applications.VGG16(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        preprocess = keras.applications.vgg16.preprocess_input
    elif base_name == "ResNet50":
        base = keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        preprocess = keras.applications.resnet50.preprocess_input
    else:
        # Default: MobileNetV2 (lighter and usually faster)
        base = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        preprocess = keras.applications.mobilenet_v2.preprocess_input

    base.trainable = False  # freeze base

    inputs = keras.Input(shape=input_shape)
    x = preprocess(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def run_task2_transfer_cifar10(base_name="MobileNetV2", epochs=8, batch_size=128):
    print("\n============================================")
    print(f"Task 2: Transfer Learning on CIFAR-10 ({base_name})")
    print("============================================")

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Keep raw uint8 range [0..255] and let preprocess_input handle scaling/normalization.
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    model = build_transfer_model(base_name=base_name, input_shape=(32, 32, 3), num_classes=10)
    model.summary()

    t0 = time.time()
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    train_time = time.time() - t0

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[CIFAR-10 TL] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    print(f"[CIFAR-10 TL] Training Time: {train_time:.2f} seconds")

    plot_history(history, title_prefix=f"CIFAR-10 Transfer Learning ({base_name})")
    show_sample_predictions(model, x_test / 255.0, y_test, class_names, n=12,
                            title=f"CIFAR-10: Sample Predictions ({base_name})")

    return test_acc, train_time


if __name__ == "__main__":
    # Lab requirement hints
    task1_acc, task1_time = run_task1_fashion_mnist(epochs=10, batch_size=128)
    task2_acc, task2_time = run_task2_transfer_cifar10(base_name="MobileNetV2", epochs=8, batch_size=128)

    print("\n============================")
    print("Summary")
    print("============================")
    print(f"Task 1 (Fashion-MNIST CNN): Accuracy={task1_acc:.4f}, Time={task1_time:.2f}s")
    print(f"Task 2 (CIFAR-10 TL):       Accuracy={task2_acc:.4f}, Time={task2_time:.2f}s")
    print("\nCompare Task 2 results with your Lab Exercise 1 (Dense-only CIFAR-10) in terms of accuracy and training time.")
