"""
visualization.py
-----------------
Reusable, consistently-styled plotting functions for EDA and model evaluation.
All charts are saved to /images/plots as PNG files for use in the README and reports.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110


def save_fig(fig, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_title("Confusion Matrix - Churn Prediction Model")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    save_fig(fig, out_path)


def plot_roc_curve(fpr, tpr, auc, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Churn Prediction Model")
    ax.legend(loc="lower right")
    save_fig(fig, out_path)


def plot_feature_importance(importances: pd.Series, out_path, top_n=15):
    top = importances.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7, 6))
    top.plot(kind="barh", ax=ax, color=sns.color_palette("deep")[0])
    ax.set_title(f"Top {top_n} Feature Importances (Churn Model)")
    ax.set_xlabel("Importance")
    save_fig(fig, out_path)


def plot_correlation_heatmap(df: pd.DataFrame, out_path):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Correlation Heatmap - Numeric Features")
    save_fig(fig, out_path)
