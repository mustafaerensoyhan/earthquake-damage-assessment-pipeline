#!/usr/bin/env python
"""
Deep-ensemble report (Lever 2). Averaging the per-seed predictive distributions
is a standard, principled way to improve accuracy and calibration together under
shift, and it reuses models you already trained, so it is not cherry-picking: it
is applied uniformly and reported in full.

Section A (cross-domain): averages the per-seed TEBDE softmax distributions for
each backbone and compares single-seed (mean over seeds), the ensemble, and each
plus target-temperature calibration, on the common seeded test split.

Section B (few-shot): if fewshot_probs_*.npz dumps exist, averages the per-seed,
per-draw few-shot distributions at each budget and compares the single-model mean
against the ensemble.

Writes outputs/uncertainty/ensemble_report.json

Usage:
    python scripts/ensemble_report.py
    python scripts/ensemble_report.py --models efficientnet_b0 resnet34 deit_tiny --mode lpft
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uncertainty.metrics import compute_all_metrics_pred, per_class_recall
from src.uncertainty.harmonize import (
    harmonize_probs_4to3, harmonize_labels_4to3, stratified_split, TEBDE_CLASSES,
)
from src.uncertainty.temperature import fit_temperature_logits
from src.utils.config import OUTPUT_DIR

COLLAPSED = 2
EPS = 1e-8


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _metrics_from_probs4(probs4, labels3, boot_seed):
    probs3 = harmonize_probs_4to3(probs4)
    pred3 = harmonize_labels_4to3(probs4.argmax(1))
    m = compute_all_metrics_pred(probs3, labels3, pred3, seed=boot_seed)
    m["collapsed_recall"] = per_class_recall(pred3, labels3, 3)[COLLAPSED]
    return m


def _metrics_from_probs3(probs3, labels3, boot_seed):
    pred3 = probs3.argmax(1)
    m = compute_all_metrics_pred(probs3, labels3, pred3, seed=boot_seed)
    m["collapsed_recall"] = per_class_recall(pred3, labels3, 3)[COLLAPSED]
    return m


def cross_domain_ensemble(unc, model, split_frac, split_seed, boot_seed):
    dumps = sorted(unc.glob(f"logits_{model}_seed*_tebde.npz"))
    if len(dumps) < 2:
        return None
    per_seed_probs4, labels = [], None
    for d in dumps:
        z = np.load(d)
        per_seed_probs4.append(_softmax(z["logits"]))
        labels = z["labels"].astype(np.int64)
    cal, test = stratified_split(labels, frac=split_frac, seed=split_seed)
    y_test = labels[test]

    # single-seed mean (raw, and +target-T per seed); collect calibrated probs too
    single_raw, single_T, cal_probs4 = [], [], []
    for d, probs4 in zip(dumps, per_seed_probs4):
        single_raw.append(_metrics_from_probs4(probs4[test], y_test, boot_seed))
        logits = np.load(d)["logits"]
        T = fit_temperature_logits(logits[cal], labels[cal], harmonize=True)
        cal_probs4.append(_softmax(logits / T))
        single_T.append(_metrics_from_probs4(cal_probs4[-1][test], y_test, boot_seed))

    # ensemble = mean of per-seed predictive distributions
    ens4 = np.mean(per_seed_probs4, axis=0)
    ens_raw = _metrics_from_probs4(ens4[test], y_test, boot_seed)
    # calibrated ensemble = mean of per-seed target-T-calibrated distributions
    # (the ID-calibrated ensemble of Kumar et al. 2022; no extra temperature scale)
    ens_cal4 = np.mean(cal_probs4, axis=0)
    ens_T = _metrics_from_probs4(ens_cal4[test], y_test, boot_seed)

    def avg(rows, key):
        return float(np.mean([r[key] for r in rows]))

    return {
        "n_seeds": len(dumps),
        "single_raw": {k: avg(single_raw, k) for k in ("accuracy", "ece", "nll", "collapsed_recall")},
        "single_target_T": {k: avg(single_T, k) for k in ("accuracy", "ece", "nll", "collapsed_recall")},
        "ensemble_raw": {k: ens_raw[k] for k in ("accuracy", "ece", "nll", "collapsed_recall")},
        "ensemble_calibrated": {k: ens_T[k] for k in ("accuracy", "ece", "nll", "collapsed_recall")},
    }


def fewshot_ensemble(unc, model, mode, boot_seed):
    tag = "" if mode == "ft" else f"_{mode}"
    files = sorted(unc.glob(f"fewshot_probs_{model}_seed*{tag}.npz"))
    # keep only files whose tag matches exactly (ft must not pick up _lpft)
    files = [f for f in files if (f.stem.endswith(tag) if tag else not f.stem.endswith("_lpft"))]
    if len(files) < 2:
        return None
    labels = None
    budget_draws = defaultdict(list)  # k -> list of (Ntest,3) prob arrays across seeds+draws
    for f in files:
        z = np.load(f)
        labels = z["labels"].astype(np.int64)
        for key in z.files:
            if key.startswith("k") and "_d" in key:
                k = int(key[1:key.index("_d")])
                budget_draws[k].append(z[key])
    rows = []
    for k in sorted(budget_draws):
        stack = np.stack(budget_draws[k], axis=0)  # (n_models, Ntest, 3)
        single_mean = float(np.mean([(_metrics_from_probs3(p, labels, boot_seed)["accuracy"]) for p in stack]))
        single_cr = float(np.nanmean([(_metrics_from_probs3(p, labels, boot_seed)["collapsed_recall"]) for p in stack]))
        single_ece = float(np.mean([(_metrics_from_probs3(p, labels, boot_seed)["ece"]) for p in stack]))
        ens = stack.mean(axis=0)
        em = _metrics_from_probs3(ens, labels, boot_seed)
        rows.append({"k": k, "n_members": int(stack.shape[0]),
                     "single_acc": single_mean, "ensemble_acc": em["accuracy"],
                     "single_collapsed": single_cr, "ensemble_collapsed": em["collapsed_recall"],
                     "single_ece": single_ece, "ensemble_ece": em["ece"]})
    return rows


def print_cross_domain(results):
    print("\n=== Section A: cross-domain deep ensemble (TEBDE test split) ===")
    hdr = f"{'model':<16}{'method':<18}{'acc':>7}{'ECE':>7}{'NLL':>7}{'Collapsed rec':>15}"
    print(hdr); print("-" * len(hdr))
    names = [("single_raw", "single raw"), ("single_target_T", "single +target-T"),
             ("ensemble_raw", "ensemble raw"), ("ensemble_calibrated", "ensemble calibrated")]
    for model, r in results.items():
        if r is None:
            print(f"{model:<16}(need >=2 seeds to ensemble)"); continue
        for key, name in names:
            m = r[key]
            print(f"{model:<16}{name:<18}{m['accuracy']:>7.3f}{m['ece']:>7.3f}"
                  f"{m['nll']:>7.3f}{m['collapsed_recall']:>15.3f}")


def print_fewshot(fs_results, mode):
    any_rows = any(v for v in fs_results.values())
    if not any_rows:
        print(f"\n=== Section B: few-shot ensemble ({mode}) ===\n  (no fewshot_probs_*.npz dumps yet; run fewshot_adapt.py to enable)")
        return
    print(f"\n=== Section B: few-shot deep ensemble across seeds+draws ({mode}) ===")
    hdr = (f"{'model':<16}{'k/cls':>6}{'single acc':>12}{'ens acc':>9}"
           f"{'single col':>12}{'ens col':>9}{'single ECE':>12}{'ens ECE':>9}")
    print(hdr); print("-" * len(hdr))
    for model, rows in fs_results.items():
        if not rows:
            continue
        for r in rows:
            print(f"{model:<16}{r['k']:>6}{r['single_acc']:>12.3f}{r['ensemble_acc']:>9.3f}"
                  f"{r['single_collapsed']:>12.3f}{r['ensemble_collapsed']:>9.3f}"
                  f"{r['single_ece']:>12.3f}{r['ensemble_ece']:>9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["efficientnet_b0", "resnet34", "deit_tiny"])
    ap.add_argument("--mode", choices=["ft", "lpft"], default="ft",
                    help="which few-shot probability dumps to ensemble in Section B")
    ap.add_argument("--split-frac", type=float, default=0.3)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    unc = Path(OUTPUT_DIR) / "uncertainty"
    cross = {m: cross_domain_ensemble(unc, m, args.split_frac, args.split_seed, args.boot_seed)
             for m in args.models}
    fewshot = {m: fewshot_ensemble(unc, m, args.mode, args.boot_seed) for m in args.models}

    print_cross_domain(cross)
    print_fewshot(fewshot, args.mode)
    out = {"cross_domain": cross, "few_shot": {"mode": args.mode, "results": fewshot}}
    (unc / "ensemble_report.json").write_text(json.dumps(out, indent=2))
    print(f"\n[ensemble_report] wrote {unc / 'ensemble_report.json'}")
    print(f"TEBDE classes: {TEBDE_CLASSES}  (Collapsed = index {COLLAPSED})")


if __name__ == "__main__":
    main()
