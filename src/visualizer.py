"""
visualizer.py
-------------
Reusable plotting utilities for ML projects.
  - Confusion matrix heatmap
  - Feature importance bar chart
  - Training loss/accuracy curves
  - Cluster scatter (2D / PCA-reduced)
  - Regression: actual vs predicted
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import List, Optional

PALETTE = 'viridis'


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = 'Confusion Matrix',
    figsize: tuple = (6, 5),
) -> None:
    """Heatmap confusion matrix with count and percentage annotations."""
    fig, ax = plt.subplots(figsize=figsize)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    labels = [[f'{v}\n({p:.1f}%)' for v, p in zip(row_v, row_p)]
               for row_v, row_p in zip(cm, cm_pct)]
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax,
                xticklabels=class_names or 'auto',
                yticklabels=class_names or 'auto')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    title: str = 'Feature Importances',
    top_n: int = 15,
    figsize: tuple = (8, 5),
) -> None:
    """Sorted horizontal bar chart of feature importances."""
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette(PALETTE, len(idx))
    ax.barh([feature_names[i] for i in idx], importances[idx], color=colors)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    figsize: tuple = (12, 4),
) -> None:
    """Loss (and optional accuracy) curves for neural network training."""
    cols = 2 if train_accs is not None else 1
    fig, axes = plt.subplots(1, cols, figsize=figsize)
    if cols == 1:
        axes = [axes]

    axes[0].plot(train_losses, label='Train Loss', color='royalblue')
    if val_losses:
        axes[0].plot(val_losses, label='Val Loss', color='tomato', linestyle='--')
    axes[0].set_title('Loss Curve', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    if train_accs is not None:
        axes[1].plot(train_accs, label='Train Acc', color='seagreen')
        if val_accs:
            axes[1].plot(val_accs, label='Val Acc', color='darkorange', linestyle='--')
        axes[1].set_title('Accuracy Curve', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    centers: Optional[np.ndarray] = None,
    title: str = 'Cluster Visualization',
    figsize: tuple = (7, 5),
) -> None:
    """2D scatter (PCA-reduced if needed) with cluster coloring."""
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
        xlabel, ylabel = 'PC 1', 'PC 2'
        if centers is not None:
            centers = pca.transform(centers)
    else:
        X2 = X
        xlabel, ylabel = 'Feature 1', 'Feature 2'

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap=PALETTE, alpha=0.7, s=40)
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200,
                   zorder=5, label='Centroids')
        ax.legend()
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Actual vs Predicted',
    figsize: tuple = (6, 5),
) -> None:
    """Scatter plot with perfect-prediction diagonal line."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, alpha=0.6, color='steelblue', edgecolors='white', s=50)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.legend()
    plt.tight_layout()
    plt.show()
