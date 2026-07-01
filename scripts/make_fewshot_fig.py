#!/usr/bin/env python
"""
Few-shot data-efficiency figure with FT vs LP-FT overlay.

Reads fewshot_{model}_seed*.json (full fine-tune, FT) and
fewshot_{model}_seed*_lpft.json (linear-probe-then-fine-tune) and plots three
panels (accuracy, Collapsed recall, ECE) vs labeled shots per class, overlaying
the two modes with mean +/- std bands across seeds and draws. The 0-shot source
baseline is the shared leftmost point.

Usage:
    python scripts/make_fewshot_fig.py --model efficientnet_b0
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import OUTPUT_DIR

MODE_STYLE = {
    "ft":   {"color": "#C44536", "ls": "--", "label": "full fine-tune (FT)"},
    "lpft": {"color": "#2E5A88", "ls": "-",  "label": "linear-probe then FT (LP-FT)"},
}
METRICS = [("accuracy", "accuracy", (0, 1)),
           ("collapsed_recall", "Collapsed recall", (0, 1)),
           ("ece", "ECE", None)]


def mode_of(path):
    stem = path.stem
    if stem.endswith("_lpft"):
        return "lpft"
    if stem.endswith("_ft"):
        return "ft"
    return "ft"  # legacy untagged files are full fine-tune


def collect(unc, model):
    buckets = {"ft": defaultdict(lambda: defaultdict(list)),
               "lpft": defaultdict(lambda: defaultdict(list))}
    found = set()
    for f in sorted(unc.glob(f"fewshot_{model}_seed*.json")):
        if f.stem.startswith(f"fewshot_probs_"):
            continue
        mode = mode_of(f)
        d = json.loads(f.read_text())
        b = d["baseline_0shot"]
        for m, _, _ in METRICS:
            buckets[mode][0][m].append(b[m])
        for r in d["results"]:
            for m, _, _ in METRICS:
                buckets[mode][r["k"]][m].append(r[m])
        found.add(mode)
    return buckets, found


def series(by_k, metric):
    ks = sorted(by_k)
    mean = np.array([np.nanmean(by_k[k][metric]) for k in ks])
    std = np.array([np.nanstd(by_k[k][metric]) for k in ks])
    return np.array(ks), mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    args = ap.parse_args()

    unc = Path(OUTPUT_DIR) / "uncertainty"
    buckets, found = collect(unc, args.model)
    if not found:
        print(f"[make_fewshot_fig] no fewshot_{args.model}_seed*.json found; run fewshot_adapt.py")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (metric, label, ylim) in zip(axes, METRICS):
        for mode in ("ft", "lpft"):
            by_k = buckets[mode]
            if not by_k:
                continue
            st = MODE_STYLE[mode]
            ks, mean, std = series(by_k, metric)
            ax.plot(ks, mean, marker="o", ms=4, color=st["color"], ls=st["ls"],
                    label=st["label"])
            ax.fill_between(ks, mean - std, mean + std, color=st["color"], alpha=0.13)
        ax.set_xlabel("labeled UAV shots per class")
        ax.set_ylabel(label)
        ax.set_title(label)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.axvline(0, color="#bbb", lw=1, ls=":")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Few-shot data efficiency, FT vs LP-FT: {args.model}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    figs = unc / "figs"; figs.mkdir(parents=True, exist_ok=True)
    out = figs / f"fewshot_{args.model}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[make_fewshot_fig] wrote {out}  (modes present: {sorted(found)})")


if __name__ == "__main__":
    main()
