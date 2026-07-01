#!/usr/bin/env python
"""
Figure 1: reliability diagrams for the headline backbone, showing the
in-domain vs cross-domain calibration gap and the target-temperature repair.

Three panels: in-domain xBD (4-class top-label), cross-domain TEBDE raw, and
cross-domain TEBDE after target-fit temperature. Each panel plots bin accuracy
against bin confidence with the y=x diagonal, and annotates ECE.

Usage:
    python scripts/make_reliability.py --model efficientnet_b0 --seed 42
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
from src.utils.config import OUTPUT_DIR


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def reliability_bins(conf, correct, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        centers.append((edges[i] + edges[i + 1]) / 2)
        if m.any():
            accs.append(correct[m].mean()); confs.append(conf[m].mean()); counts.append(int(m.sum()))
        else:
            accs.append(np.nan); confs.append(np.nan); counts.append(0)
    ece = float(np.nansum([(c / max(1, sum(counts))) * abs(a - cf)
                for a, cf, c in zip(accs, confs, counts) if c > 0]))
    return np.array(centers), np.array(accs), ece


def panel(ax, conf, correct, title):
    centers, accs, ece = reliability_bins(conf, correct)
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
    ax.bar(centers, accs, width=0.09, color="#2E5A88", edgecolor="white", alpha=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE = {ece:.3f}", fontsize=10)
    ax.set_aspect("equal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-frac", type=float, default=0.3)
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    unc = Path(OUTPUT_DIR) / "uncertainty"
    xbd = np.load(unc / f"logits_{args.model}_seed{args.seed}_xbdval.npz")
    teb = np.load(unc / f"logits_{args.model}_seed{args.seed}_tebde.npz")
    xl, xy = xbd["logits"], xbd["labels"].astype(np.int64)
    tl, ty = teb["logits"], teb["labels"].astype(np.int64)

    cal, test = stratified_split(ty, frac=args.split_frac, seed=args.split_seed)
    T_tgt = fit_temperature_logits(tl[cal], ty[cal], harmonize=True)

    # in-domain (4-class top-label)
    px = _softmax(xl)
    conf_in, corr_in = px.max(1), (px.argmax(1) == xy).astype(float)

    # cross-domain raw (3-class, prediction = 4-argmax mapped, temp-invariant)
    p3_raw = harmonize_probs_4to3(_softmax(tl[test]))
    pred3 = harmonize_labels_4to3(_softmax(tl[test]).argmax(1))
    conf_raw = p3_raw[np.arange(len(pred3)), pred3]
    corr_cd = (pred3 == ty[test]).astype(float)

    # cross-domain target-T
    p3_t = apply_temperature(tl[test], T_tgt, harmonize=True)
    conf_t = p3_t[np.arange(len(pred3)), pred3]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    panel(axes[0], conf_in, corr_in, "In-domain xBD (4-class)")
    panel(axes[1], conf_raw, corr_cd, "Cross-domain TEBDE (raw)")
    panel(axes[2], conf_t, corr_cd, f"Cross-domain TEBDE (+target-T={T_tgt:.2f})")
    fig.suptitle(f"Reliability diagrams: {args.model} (seed {args.seed})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    figs = unc / "figs"; figs.mkdir(parents=True, exist_ok=True)
    out = figs / f"reliability_{args.model}_seed{args.seed}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[make_reliability] wrote {out}")


if __name__ == "__main__":
    main()
