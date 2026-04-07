"""Visualization utilities for the Sign Language Prediction System."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import setup_logger

logger = setup_logger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Plot confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    return fig


def plot_training_history(history: Dict, title: str = "Training History") -> plt.Figure:
    """Plot training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.get("accuracy", []), label="Train")
    if "val_accuracy" in history:
        axes[0].plot(history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.get("loss", []), label="Train")
    if "val_loss" in history:
        axes[1].plot(history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_class_distribution(
    labels: np.ndarray, class_names: Optional[List[str]] = None
) -> plt.Figure:
    """Plot distribution of classes in the dataset."""
    unique, counts = np.unique(labels, return_counts=True)
    if class_names:
        labels_display = [class_names[i] if i < len(class_names) else str(i) for i in unique]
    else:
        labels_display = [str(i) for i in unique]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(labels_display, counts, color="steelblue", edgecolor="black")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Gesture Class")
    ax.set_ylabel("Sample Count")
    ax.bar_label(bars)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_model_comparison(results: Dict) -> plt.Figure:
    """Plot comparison of model accuracies."""
    models = list(results.keys())
    accuracies = [
        results[m].get("accuracy", results[m].get("train_accuracy", 0)) for m in models
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, accuracies, color=colors, edgecolor="black")
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.bar_label(bars, fmt="%.3f")
    plt.tight_layout()
    return fig
