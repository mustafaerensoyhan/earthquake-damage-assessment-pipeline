"""
Selective prediction and safety-asymmetric triage for cross-domain damage
assessment.

Two ideas:
  1. Risk-coverage: abstain on the least confident predictions and auto-assess
     the rest. Sweeping the confidence threshold traces a risk-coverage curve;
     AURC summarises it. Calibration does not change the ranking much, but it
     makes a CONFIDENCE THRESHOLD meaningful: a calibrated model accepted at
     conf>=0.9 is right ~90% of the time, an uncalibrated one is not.
  2. Safety-asymmetric triage: missing a collapsed building is far costlier than
     a false alarm, so we never auto-clear a building that carries non-trivial
     collapse probability. We rank buildings for human review by their collapse
     risk (when not already predicted Collapsed) and measure the residual
     collapsed-miss-rate against the human-review budget, versus a confidence-only
     triage.

All functions take harmonised 3-class probabilities and labels; the prediction
is supplied (the temperature-invariant 4-class argmax mapped to 3 classes).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

COLLAPSED = 2  # TEBDE 3-class index


def confidence_correct(probs3: np.ndarray, labels3: np.ndarray,
                       preds3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(labels3)
    conf = probs3[np.arange(n), preds3]
    correct = (preds3 == labels3).astype(np.float64)
    return conf, correct


def risk_coverage_curve(conf: np.ndarray, correct: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Sort by confidence desc; return (coverage, risk) where risk is the error
    rate among the most-confident `coverage` fraction."""
    order = np.argsort(-conf)
    c = correct[order]
    k = np.arange(1, len(c) + 1)
    coverage = k / len(c)
    risk = 1.0 - np.cumsum(c) / k
    return coverage, risk


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration, version-safe (np.trapz removed in numpy 2.0)."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # numpy < 2.0


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    cov, risk = risk_coverage_curve(conf, correct)
    return _trapz(risk, cov)


def selective_accuracy_at_coverage(conf, correct, target_cov: float) -> float:
    order = np.argsort(-conf)
    k = max(1, int(round(target_cov * len(conf))))
    return float(correct[order][:k].mean())


def coverage_at_risk(conf: np.ndarray, correct: np.ndarray,
                     target_risk: float) -> float:
    """Largest auto-assess coverage whose error rate stays at or below
    target_risk. Returns 0.0 if even the single most-confident prediction
    exceeds the target. Threshold-free, so it works for raw and calibrated
    confidence identically (the ranking is what matters)."""
    cov, risk = risk_coverage_curve(conf, correct)
    ok = np.where(risk <= target_risk)[0]
    return float(cov[ok.max()]) if len(ok) else 0.0


def threshold_operating_point(conf, correct, tau: float) -> Dict[str, float]:
    """At a fixed confidence threshold tau: coverage and selective accuracy of
    the accepted set. Shows why an uncalibrated threshold over-accepts."""
    acc_mask = conf >= tau
    cov = float(acc_mask.mean())
    sel_acc = float(correct[acc_mask].mean()) if acc_mask.any() else float("nan")
    return {"tau": tau, "coverage": cov, "selective_accuracy": sel_acc,
            "selective_risk": (1.0 - sel_acc) if acc_mask.any() else float("nan")}


# --- safety-asymmetric triage ---------------------------------------------
def review_priority_confidence(conf: np.ndarray) -> np.ndarray:
    """Review the least confident first."""
    return 1.0 - conf


def review_priority_collapse(probs3: np.ndarray, preds3: np.ndarray) -> np.ndarray:
    """Review by collapse risk: highest P(Collapsed) among buildings NOT already
    predicted Collapsed (those are already treated as collapsed, so safe)."""
    prio = probs3[:, COLLAPSED].copy()
    prio[preds3 == COLLAPSED] = -1.0
    return prio


def collapsed_miss_rate_vs_budget(probs3, labels3, preds3, priority,
                                  budgets) -> np.ndarray:
    """For each review budget (fraction sent to humans, highest priority first),
    the residual collapsed-miss-rate: true Collapsed buildings that are
    auto-assessed AND not predicted Collapsed."""
    n = len(labels3)
    order = np.argsort(-priority)
    is_coll = labels3 == COLLAPSED
    n_coll = max(1, int(is_coll.sum()))
    out = []
    for b in budgets:
        k = int(round(b * n))
        flagged = np.zeros(n, dtype=bool)
        flagged[order[:k]] = True
        auto = ~flagged
        missed = is_coll & auto & (preds3 != COLLAPSED)
        out.append(missed.sum() / n_coll)
    return np.array(out, dtype=np.float64)
