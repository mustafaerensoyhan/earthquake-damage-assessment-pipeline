"""
Visualization utilities for the two-stage damage assessment pipeline.

Generates publication-quality plots for:
  - Accuracy-latency tradeoff scatter plots (Pareto frontier)
  - Confusion matrices (heatmaps)
  - Per-class accuracy bar charts
  - Training history curves
  - Cross-domain comparison charts
  - Summary tables (LaTeX and Markdown)

All plots are saved to outputs/figures/ and use a consistent style.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

from src.utils.config import FIGURES_DIR, RESULTS_DIR, XBD_DAMAGE_CLASSES, TEBDE_CLASSES

logger = logging.getLogger(__name__)

# Consistent style for all plots
matplotlib.rcParams.update({
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Color palette for models
MODEL_COLORS = {
    "yolov8n": "#2196F3",
    "yolov8s": "#FF9800",
    "yolov8m": "#4CAF50",
    "efficientnet_b0": "#9C27B0",
    "resnet34": "#F44336",
}

RESOLUTION_MARKERS = {
    640: "o",
    800: "s",
}


# ---------------------------------------------------------------------------
# Detector plots
# ---------------------------------------------------------------------------
def plot_detector_comparison(
    results: List[dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart comparing mAP@0.5 and mAP@0.5:0.95 across detector configs.
    """
    tags = [r["tag"] for r in results]
    map50 = [r["mAP50"] for r in results]
    map50_95 = [r["mAP50-95"] for r in results]

    x = np.arange(len(tags))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, map50, width, label="mAP@0.5", color="#2196F3")
    bars2 = ax.bar(x + width / 2, map50_95, width, label="mAP@0.5:0.95", color="#FF9800")

    ax.set_xlabel("Configuration")
    ax.set_ylabel("mAP Score")
    ax.set_title("Detector Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        logger.info("Saved detector comparison → %s", save_path)
    return fig


def plot_accuracy_latency_tradeoff(
    results: List[dict],
    metric_key: str = "mAP50",
    latency_key: str = "median_ms",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Scatter plot with accuracy on Y-axis and latency on X-axis.
    Highlights Pareto-optimal configurations.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for r in results:
        model = r.get("model", "unknown")
        res = r.get("resolution", 640)
        color = MODEL_COLORS.get(model, "#999999")
        marker = RESOLUTION_MARKERS.get(res, "^")
        prec = r.get("precision_mode", "fp32")
        edge = "black" if prec == "fp16" else color

        latency = r.get(latency_key, 0)
        accuracy = r.get(metric_key, 0)

        ax.scatter(
            latency, accuracy,
            c=color, marker=marker, s=120, edgecolors=edge,
            linewidths=2, zorder=5,
        )
        ax.annotate(
            r["tag"], (latency, accuracy),
            textcoords="offset points", xytext=(8, 5),
            fontsize=8, alpha=0.8,
        )

    # Pareto frontier
    points = [(r.get(latency_key, 0), r.get(metric_key, 0), r["tag"]) for r in results]
    points.sort(key=lambda p: p[0])
    pareto = []
    best_acc = -1
    for lat, acc, tag in points:
        if acc > best_acc:
            pareto.append((lat, acc))
            best_acc = acc
    if len(pareto) > 1:
        px, py = zip(*pareto)
        ax.plot(px, py, "--", color="gray", alpha=0.5, label="Pareto frontier")

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel(metric_key)
    ax.set_title("Accuracy vs Latency Tradeoff")
    ax.grid(alpha=0.3)

    # Custom legend
    from matplotlib.lines import Line2D
    handles = []
    for model, color in MODEL_COLORS.items():
        if any(r.get("model") == model for r in results):
            handles.append(Line2D([0], [0], marker="o", color=color,
                                  label=model, markersize=8, linestyle=""))
    for res, marker in RESOLUTION_MARKERS.items():
        if any(r.get("resolution") == res for r in results):
            handles.append(Line2D([0], [0], marker=marker, color="gray",
                                  label=f"{res}px", markersize=8, linestyle=""))
    ax.legend(handles=handles, loc="lower right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        logger.info("Saved tradeoff plot → %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Classifier plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_plot = cm.astype(float) / row_sums
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmax = None

    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, vmin=0, vmax=vmax, square=True,
        cbar_kws={"label": "Rate" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        logger.info("Saved confusion matrix → %s", save_path)
    return fig


def plot_classifier_comparison(
    results: List[dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart of accuracy and macro F1 for each classifier config."""
    tags = [r["tag"] for r in results]
    accs = [r.get("val_accuracy", 0) for r in results]
    f1s = [r.get("val_macro_f1", 0) for r in results]

    x = np.arange(len(tags))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="#9C27B0")
    ax.bar(x + width / 2, f1s, width, label="Macro F1", color="#F44336")

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score")
    ax.set_title("Classifier Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        logger.info("Saved classifier comparison → %s", save_path)
    return fig


def plot_per_class_accuracy(
    report: dict,
    class_names: List[str],
    title: str = "Per-Class Performance",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart showing precision, recall, and F1 per class."""
    precision = [report.get(c, {}).get("precision", 0) for c in class_names]
    recall = [report.get(c, {}).get("recall", 0) for c in class_names]
    f1 = [report.get(c, {}).get("f1-score", 0) for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#2196F3")
    ax.bar(x, recall, width, label="Recall", color="#FF9800")
    ax.bar(x + width, f1, width, label="F1-Score", color="#4CAF50")

    ax.set_xlabel("Damage Class")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        logger.info("Saved per-class accuracy → %s", save_path)
    return fig


def plot_training_history(
    history: List[dict],
    title: str = "Training History",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot training/validation loss and accuracy curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_f1 = [h["val_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss", color="#2196F3")
    ax1.plot(epochs, val_loss, label="Val Loss", color="#F44336")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy & F1
    ax2.plot(epochs, val_acc, label="Val Accuracy", color="#4CAF50")
    ax2.plot(epochs, val_f1, label="Val Macro F1", color="#FF9800")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title(f"{title} — Metrics")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.0)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        logger.info("Saved training history → %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Cross-domain plots
# ---------------------------------------------------------------------------
def plot_cross_domain_comparison(
    xbd_results: dict,
    tebde_results: dict,
    title: str = "Cross-Domain Performance",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Side-by-side bars: xBD val accuracy vs TEBDE accuracy."""
    models = list(xbd_results.keys())
    xbd_acc = [xbd_results[m].get("accuracy", 0) for m in models]
    tebde_acc = [tebde_results[m].get("accuracy", 0) for m in models]
    delta = [x - t for x, t in zip(xbd_acc, tebde_acc)]

    x = np.arange(len(models))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, xbd_acc, width, label="xBD Val", color="#2196F3")
    bars2 = ax.bar(x + width / 2, tebde_acc, width, label="UAVs-TEBDE", color="#F44336")

    ax.set_xlabel("Classifier")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Annotate delta
    for i, (b1, b2, d) in enumerate(zip(bars1, bars2, delta)):
        ax.annotate(
            f"Δ = {d:+.3f}",
            xy=(x[i], max(b1.get_height(), b2.get_height()) + 0.03),
            ha="center", fontsize=9, color="gray",
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        logger.info("Saved cross-domain comparison → %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------
def generate_detector_table_latex(results: List[dict]) -> str:
    """Generate a LaTeX table of detector results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Detector Performance Comparison}",
        r"\label{tab:detector_results}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Configuration & mAP@0.5 & mAP@0.5:0.95 & Precision & Recall \\",
        r"\hline",
    ]
    for r in sorted(results, key=lambda x: x["tag"]):
        lines.append(
            f"{r['tag']} & {r['mAP50']:.4f} & {r['mAP50-95']:.4f} & "
            f"{r.get('precision', 0):.4f} & {r.get('recall', 0):.4f} \\\\"
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_classifier_table_latex(results: List[dict]) -> str:
    """Generate a LaTeX table of classifier results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Classifier Performance Comparison}",
        r"\label{tab:classifier_results}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Configuration & Accuracy & Macro F1 & Best Epoch \\",
        r"\hline",
    ]
    for r in sorted(results, key=lambda x: x["tag"]):
        lines.append(
            f"{r['tag']} & {r['val_accuracy']:.4f} & "
            f"{r['val_macro_f1']:.4f} & {r.get('best_epoch', '-')} \\\\"
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Master plot generator
# ---------------------------------------------------------------------------
def generate_all_figures(
    figures_dir: Path = FIGURES_DIR,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """
    Read all saved results and generate every figure for the paper.

    Call this after all training and evaluation is complete.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Detector figures ---
    det_path = results_dir / "detector_training_results.json"
    if det_path.exists():
        with open(det_path) as f:
            det_results = json.load(f)
        logger.info("Generating detector figures from %d configs...", len(det_results))

        plot_detector_comparison(
            det_results,
            save_path=figures_dir / "detector_comparison.png",
        )

        latex = generate_detector_table_latex(det_results)
        (figures_dir / "detector_table.tex").write_text(latex)
        logger.info("Saved LaTeX table → detector_table.tex")
    else:
        logger.warning("No detector results found at %s", det_path)

    # --- Classifier figures ---
    clf_path = results_dir / "classifier_training_results.json"
    if clf_path.exists():
        with open(clf_path) as f:
            clf_results = json.load(f)
        logger.info("Generating classifier figures from %d configs...", len(clf_results))

        plot_classifier_comparison(
            clf_results,
            save_path=figures_dir / "classifier_comparison.png",
        )

        for r in clf_results:
            tag = r["tag"]
            # Confusion matrix
            if "confusion_matrix" in r:
                cm = np.array(r["confusion_matrix"])
                plot_confusion_matrix(
                    cm, XBD_DAMAGE_CLASSES,
                    title=f"Confusion Matrix — {tag}",
                    save_path=figures_dir / f"cm_{tag}.png",
                )
            # Per-class accuracy
            if "classification_report" in r:
                plot_per_class_accuracy(
                    r["classification_report"], XBD_DAMAGE_CLASSES,
                    title=f"Per-Class Performance — {tag}",
                    save_path=figures_dir / f"per_class_{tag}.png",
                )
            # Training history
            if "history" in r:
                plot_training_history(
                    r["history"],
                    title=tag,
                    save_path=figures_dir / f"history_{tag}.png",
                )

        latex = generate_classifier_table_latex(clf_results)
        (figures_dir / "classifier_table.tex").write_text(latex)
        logger.info("Saved LaTeX table → classifier_table.tex")
    else:
        logger.warning("No classifier results found at %s", clf_path)

    # --- Latency figures ---
    lat_path = results_dir / "latency_results.json"
    if lat_path.exists():
        with open(lat_path) as f:
            lat_results = json.load(f)
        logger.info("Generating latency tradeoff plot...")
        plot_accuracy_latency_tradeoff(
            lat_results,
            save_path=figures_dir / "accuracy_latency_tradeoff.png",
        )
    else:
        logger.info("No latency results yet — will generate after benchmarking")

    # --- Cross-domain figures ---
    cd_path = results_dir / "cross_domain_results.json"
    if cd_path.exists():
        with open(cd_path) as f:
            cd_results = json.load(f)
        for r in cd_results:
            tag = r["tag"]
            if "confusion_matrix" in r:
                cm = np.array(r["confusion_matrix"])
                plot_confusion_matrix(
                    cm, TEBDE_CLASSES,
                    title=f"Cross-Domain CM — {tag}",
                    save_path=figures_dir / f"cm_cross_domain_{tag}.png",
                )

    logger.info("=== All figures saved to %s ===", figures_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    generate_all_figures()
