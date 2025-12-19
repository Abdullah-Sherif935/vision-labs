import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


def plot_history(history, title_prefix="Model"):
    """
    Plots training & validation loss and accuracy curves over epochs.
    Works for any keras History object that has accuracy/loss keys.
    """
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

    # Accuracy
    # Keras key might be 'accuracy' or 'acc' depending on version
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


def build_dense_classifier(input_dim, num_classes, hidden_units=(256, 128), dropout=0.2):
    """
    Fully-connected (Dense-only) classifier.
    input_dim: flattened input dimension
    num_classes: number of output classes
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
        if dropout and dropout > 0:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_mnist(epochs=10, batch_size=128):
    # Task 1: MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize + Flatten
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape((x_train.shape[0], -1))  # (N, 784)
    x_test = x_test.reshape((x_test.shape[0], -1))

    model = build_dense_classifier(
        input_dim=x_train.shape[1],
        num_classes=10,
        hidden_units=(256, 128),
        dropout=0.2
    )

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[MNIST] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Required: plot training/validation loss & accuracy
    plot_history(history, title_prefix="MNIST Dense NN")


def run_fashion_mnist(epochs=10, batch_size=128):
    # Task 2: Fashion-MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # 1) Normalize input images (required)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten before Dense (required)
    x_train = x_train.reshape((x_train.shape[0], -1))  # (N, 784)
    x_test = x_test.reshape((x_test.shape[0], -1))

    model = build_dense_classifier(
        input_dim=x_train.shape[1],
        num_classes=10,
        hidden_units=(256, 128),
        dropout=0.3
    )

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[Fashion-MNIST] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Required: plot loss and accuracy curves
    plot_history(history, title_prefix="Fashion-MNIST Dense NN")


def run_cifar10(epochs=15, batch_size=128):
    # Task 3: CIFAR-10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # y is shape (N, 1) -> flatten to (N,)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Preprocess: normalize + flatten (required)
    x_train = x_train.astype("float32") / 255.0  # (N, 32, 32, 3)
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape((x_train.shape[0], -1))  # (N, 3072)
    x_test = x_test.reshape((x_test.shape[0], -1))

    model = build_dense_classifier(
        input_dim=x_train.shape[1],
        num_classes=10,
        hidden_units=(512, 256, 128),
        dropout=0.4
    )

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[CIFAR-10] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Required: plot training and validation accuracy curves
    # (We also plot loss automatically if present)
    plot_history(history, title_prefix="CIFAR-10 Dense NN")


if __name__ == "__main__":
    # Run tasks one by one
    run_mnist(epochs=10)
    run_fashion_mnist(epochs=10)

    # CIFAR-10 Dense-only is harder -> I used 15 epochs by default
    # If you want to reduce time, set epochs=10
    run_cifar10(epochs=15)
