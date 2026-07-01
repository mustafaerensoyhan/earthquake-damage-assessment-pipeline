#!/usr/bin/env python
"""
Phase 3 figures for the headline backbone:
  Fig A: risk-coverage curve, raw vs target-T calibrated confidence.
  Fig B: collapsed-miss-rate vs human-review budget, confidence-only triage vs
         the safety-asymmetric collapse-aware triage.

Usage:
    python scripts/make_selective_figs.py --model efficientnet_b0 --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uncertainty.harmonize import (
    harmonize_probs_4to3, harmonize_labels_4to3, stratified_split,
)
from src.uncertainty.temperature import fit_temperature_logits, apply_temperature
from src.uncertainty.selective import (
    confidence_correct, risk_coverage_curve, review_priority_confidence,
    review_priority_collapse, collapsed_miss_rate_vs_budget,
)
from src.utils.config import OUTPUT_DIR

ACCENT, WARN = "#2E5A88", "#C44536"


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-frac", type=float, default=0.3)
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    unc = Path(OUTPUT_DIR) / "uncertainty"
    teb = np.load(unc / f"logits_{args.model}_seed{args.seed}_tebde.npz")
    tl, ty = teb["logits"], teb["labels"].astype(np.int64)
    cal, test = stratified_split(ty, frac=args.split_frac, seed=args.split_seed)
    T = fit_temperature_logits(tl[cal], ty[cal], harmonize=True)

    lg, y = tl[test], ty[test]
    preds3 = harmonize_labels_4to3(_softmax(lg).argmax(1))
    probs_raw = harmonize_probs_4to3(_softmax(lg))
    probs_cal = apply_temperature(lg, T, harmonize=True)
    conf_raw, correct = confidence_correct(probs_raw, y, preds3)
    conf_cal, _ = confidence_correct(probs_cal, y, preds3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Fig A: risk-coverage
    cov_r, risk_r = risk_coverage_curve(conf_raw, correct)
    cov_c, risk_c = risk_coverage_curve(conf_cal, correct)
    ax1.plot(cov_r, risk_r, color="#888", lw=2, label="raw confidence")
    ax1.plot(cov_c, risk_c, color=ACCENT, lw=2, label=f"target-T (T={T:.2f})")
    ax1.set_xlabel("coverage (fraction auto-assessed)")
    ax1.set_ylabel("risk (error rate on accepted)")
    ax1.set_title("Risk-coverage")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, max(risk_r.max(), risk_c.max()) * 1.05)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.25)

    # Fig B: collapsed-miss-rate vs review budget
    budgets = np.linspace(0, 0.6, 13)
    prio_conf = review_priority_confidence(conf_cal)
    prio_coll = review_priority_collapse(probs_cal, preds3)
    miss_conf = collapsed_miss_rate_vs_budget(probs_cal, y, preds3, prio_conf, budgets)
    miss_coll = collapsed_miss_rate_vs_budget(probs_cal, y, preds3, prio_coll, budgets)
    ax2.plot(budgets, miss_conf, color="#888", lw=2, marker="o", ms=3, label="confidence-only triage")
    ax2.plot(budgets, miss_coll, color=WARN, lw=2, marker="o", ms=3, label="collapse-aware triage")
    ax2.set_xlabel("human-review budget (fraction flagged)")
    ax2.set_ylabel("collapsed-miss-rate")
    ax2.set_title("Safety-asymmetric triage")
    ax2.set_xlim(0, 0.6); ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.25)

    fig.suptitle(f"Selective prediction: {args.model} (seed {args.seed})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    figs = unc / "figs"; figs.mkdir(parents=True, exist_ok=True)
    out = figs / f"selective_{args.model}_seed{args.seed}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[make_selective_figs] wrote {out}")


if __name__ == "__main__":
    main()
