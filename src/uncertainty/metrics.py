"""
Calibration and accuracy metrics: Accuracy, Expected Calibration Error (ECE),
Negative Log-Likelihood (NLL), paired/standard bootstrap CI for ECE, and
per-class recall (for the safety-critical Collapsed class).

Two entry points:
  - compute_all_metrics(probs, labels): top-label calibration where the
    prediction is argmax(probs). Used for in-domain 4-class evaluation.
  - compute_all_metrics_pred(probs, labels, preds): the prediction is supplied
    explicitly. Confidence is the probability the model assigns to its predicted
    class. Used cross-domain so that the prediction is the 4-class argmax mapped
    to 3 classes (temperature-invariant), while ECE/NLL still move with
    temperature. This keeps accuracy fixed across calibration states.

Adapted from the author's uncertainty-tta-medmnist study.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _to_numpy(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    probs, labels = _to_numpy(probs), _to_numpy(labels).ravel()
    return float((probs.argmax(axis=1) == labels).mean())


def per_class_recall(preds: np.ndarray, labels: np.ndarray,
                     num_classes: int) -> Dict[int, float]:
    """Recall per class: fraction of each true class that is predicted correctly."""
    preds, labels = _to_numpy(preds).ravel(), _to_numpy(labels).ravel()
    out = {}
    for c in range(num_classes):
        mask = labels == c
        out[c] = float((preds[mask] == c).mean()) if mask.any() else float("nan")
    return out


def _ece_from_conf(confidences: np.ndarray, correct: np.ndarray,
                   n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        c = int(mask.sum())
        if c == 0:
            continue
        ece += (c / n) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    probs, labels = _to_numpy(probs), _to_numpy(labels).ravel()
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    return _ece_from_conf(conf, correct, n_bins)


def negative_log_likelihood(probs, labels, eps: float = 1e-12) -> float:
    probs, labels = _to_numpy(probs), _to_numpy(labels).ravel().astype(np.int64)
    n = len(labels)
    return float(-np.mean(np.log(probs[np.arange(n), labels] + eps)))


def _bootstrap_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int,
                   n_boot: int, ci: float, seed: int) -> Tuple[float, float, float]:
    point = _ece_from_conf(conf, correct, n_bins)
    n = len(conf)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = _ece_from_conf(conf[idx], correct[idx], n_bins)
    lo = float(np.percentile(boots, (1.0 - ci) / 2.0 * 100.0))
    hi = float(np.percentile(boots, (1.0 + ci) / 2.0 * 100.0))
    return point, lo, hi


def bootstrap_ece_ci(probs, labels, n_bins: int = 10, n_boot: int = 2000,
                     ci: float = 0.95, seed: int = 0) -> Tuple[float, float, float]:
    probs, labels = _to_numpy(probs), _to_numpy(labels).ravel()
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    return _bootstrap_ece(conf, correct, n_bins, n_boot, ci, seed)


def compute_all_metrics(probs, labels, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Top-label metrics where the prediction is argmax(probs)."""
    ece, lo, hi = bootstrap_ece_ci(probs, labels, n_boot=n_boot, seed=seed)
    return {
        "n": int(len(_to_numpy(labels).ravel())),
        "accuracy": accuracy(probs, labels),
        "ece": ece, "ece_ci_low": lo, "ece_ci_high": hi,
        "nll": negative_log_likelihood(probs, labels),
    }


def compute_all_metrics_pred(probs, labels, preds, n_boot: int = 2000,
                             seed: int = 0) -> Dict[str, float]:
    """
    Metrics where the prediction is supplied explicitly. Confidence is the
    probability assigned to the predicted class, correctness is preds==labels.
    Accuracy reflects the supplied decision (temperature-invariant when preds
    come from the 4-class argmax), while ECE and NLL move with the probabilities.
    """
    probs = _to_numpy(probs)
    labels = _to_numpy(labels).ravel().astype(np.int64)
    preds = _to_numpy(preds).ravel().astype(np.int64)
    n = len(labels)
    conf = probs[np.arange(n), preds]
    correct = (preds == labels).astype(np.float64)
    ece, lo, hi = _bootstrap_ece(conf, correct, 10, n_boot, 0.95, seed)
    return {
        "n": int(n),
        "accuracy": float(correct.mean()),
        "ece": ece, "ece_ci_low": lo, "ece_ci_high": hi,
        "nll": negative_log_likelihood(probs, labels),
    }
