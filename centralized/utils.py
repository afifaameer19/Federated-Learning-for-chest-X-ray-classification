import os
import json
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_metrics(metrics: dict, path):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {path}")


def plot_curves(history, save_path):

    directory = os.path.dirname(save_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    epochs = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    # Loss
    axes[0].plot(
        epochs,
        train_loss,
        label="Train Loss"
    )

    axes[0].plot(
        epochs,
        val_loss,
        label="Val Loss"
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()

    # Accuracy
    axes[1].plot(
        epochs,
        train_acc,
        label="Train Accuracy"
    )

    axes[1].plot(
        epochs,
        val_acc,
        label="Validation Accuracy"
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Plot saved to {save_path}")