#!/usr/bin/env python
"""
Phase 3 report: selective prediction and safety-asymmetric triage on UAVs-TEBDE,
on the common seeded test split, using target-T calibrated probabilities (the
recalibration that the calibration study showed is the prerequisite).

Prints, per (backbone, seed) and aggregated:
  - AURC and selective accuracy at coverage 0.5 / 0.7 / 0.9
  - operating points at confidence thresholds 0.80 / 0.90: coverage and selective
    accuracy, RAW vs target-T (shows the uncalibrated model over-accepts)
  - collapsed-miss-rate at review budgets 0 / 0.1 / 0.2 / 0.3 / 0.5 for a
    confidence-only triage vs the collapse-aware (safety-asymmetric) triage

Needs dump_all.py outputs.

Usage:
    python scripts/selective_report.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uncertainty.harmonize import (
    harmonize_probs_4to3, harmonize_labels_4to3, stratified_split, TEBDE_CLASSES,
)
from src.uncertainty.temperature import fit_temperature_logits, apply_temperature
from src.uncertainty.selective import (
    confidence_correct, aurc, selective_accuracy_at_coverage, coverage_at_risk,
    review_priority_confidence, review_priority_collapse,
    collapsed_miss_rate_vs_budget,
)
from src.utils.config import OUTPUT_DIR

BUDGETS = [0.0, 0.1, 0.2, 0.3, 0.5]
RISK_TARGETS = [0.35, 0.40, 0.45]


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def discover_seeds(unc, model):
    seeds = []
    for p in unc.glob(f"logits_{model}_seed*_tebde.npz"):
        s = p.stem.rsplit("_seed", 1)[1].split("_")[0]
        if s.isdigit():
            seeds.append(int(s))
    return sorted(set(seeds))


def analyse(unc, model, seed, split_frac, split_seed):
    teb = np.load(unc / f"logits_{model}_seed{seed}_tebde.npz")
    tl, ty = teb["logits"], teb["labels"].astype(np.int64)
    cal, test = stratified_split(ty, frac=split_frac, seed=split_seed)
    T = fit_temperature_logits(tl[cal], ty[cal], harmonize=True)

    logits_test, y = tl[test], ty[test]
    preds3 = harmonize_labels_4to3(_softmax(logits_test).argmax(1))
    probs_raw = harmonize_probs_4to3(_softmax(logits_test))
    probs_cal = apply_temperature(logits_test, T, harmonize=True)

    conf_raw, correct = confidence_correct(probs_raw, y, preds3)
    conf_cal, _ = confidence_correct(probs_cal, y, preds3)

    rc = {
        "aurc_raw": aurc(conf_raw, correct),
        "aurc_cal": aurc(conf_cal, correct),
        "selacc@0.5": selective_accuracy_at_coverage(conf_cal, correct, 0.5),
        "selacc@0.7": selective_accuracy_at_coverage(conf_cal, correct, 0.7),
        "selacc@0.9": selective_accuracy_at_coverage(conf_cal, correct, 0.9),
        "base_acc": float(correct.mean()),
        "cov_at_risk": {str(r): coverage_at_risk(conf_cal, correct, r) for r in RISK_TARGETS},
    }

    prio_conf = review_priority_confidence(conf_cal)
    prio_coll = review_priority_collapse(probs_cal, preds3)
    miss = {
        "budgets": BUDGETS,
        "confidence_only": collapsed_miss_rate_vs_budget(probs_cal, y, preds3, prio_conf, BUDGETS).tolist(),
        "collapse_aware": collapsed_miss_rate_vs_budget(probs_cal, y, preds3, prio_coll, BUDGETS).tolist(),
    }
    return {"model": model, "seed": seed, "T_target": T, "risk_coverage": rc, "collapsed_miss": miss}


def print_risk_coverage(results):
    print("\n=== Risk-coverage (target-T calibrated confidence) ===")
    hdr = (f"{'model':<16}{'seed':>5}  {'AURC raw':>9}{'AURC cal':>9}"
           f"{'selAcc@.5':>10}{'selAcc@.7':>10}{'selAcc@.9':>10}")
    print(hdr); print("-" * len(hdr))
    for r in results:
        rc = r["risk_coverage"]
        print(f"{r['model']:<16}{r['seed']:>5}  {rc['aurc_raw']:>9.3f}{rc['aurc_cal']:>9.3f}"
              f"{rc['selacc@0.5']:>10.3f}{rc['selacc@0.7']:>10.3f}{rc['selacc@0.9']:>10.3f}")


def print_operating(results):
    print("\n=== Max coverage at a target error rate (calibrated confidence) ===")
    print("    (largest fraction auto-assessable while staying under each error target)")
    hdr = (f"{'model':<16}{'seed':>5}  {'baseAcc':>8}" +
           "".join(f"{'cov@risk' + str(r):>14}" for r in RISK_TARGETS))
    print(hdr); print("-" * len(hdr))
    for r in results:
        rc = r["risk_coverage"]
        row = f"{r['model']:<16}{r['seed']:>5}  {rc['base_acc']:>8.3f}"
        row += "".join(f"{rc['cov_at_risk'][str(t)]:>14.3f}" for t in RISK_TARGETS)
        print(row)


def print_collapsed(results):
    print("\n=== Collapsed-miss-rate vs review budget (mean over seeds) ===")
    print("  budget:        " + "".join(f"{b:>8.0%}" for b in BUDGETS))
    by = defaultdict(lambda: defaultdict(list))
    for r in results:
        by[r["model"]]["conf"].append(r["collapsed_miss"]["confidence_only"])
        by[r["model"]]["coll"].append(r["collapsed_miss"]["collapse_aware"])
    for model, d in by.items():
        conf = np.mean(d["conf"], axis=0); coll = np.mean(d["coll"], axis=0)
        print(f"{model:<16} conf-only " + "".join(f"{x:>8.3f}" for x in conf))
        print(f"{'':<16} collapse  " + "".join(f"{x:>8.3f}" for x in coll))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["efficientnet_b0", "resnet34", "deit_tiny"])
    ap.add_argument("--split-frac", type=float, default=0.3)
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    unc = Path(OUTPUT_DIR) / "uncertainty"
    results = []
    for model in args.models:
        for seed in discover_seeds(unc, model):
            results.append(analyse(unc, model, seed, args.split_frac, args.split_seed))
    if not results:
        print("[selective_report] no dumps found; run dump_all.py first."); return
    print_risk_coverage(results)
    print_operating(results)
    print_collapsed(results)
    (unc / "selective_report.json").write_text(json.dumps(results, indent=2))
    print(f"\n[selective_report] wrote {unc / 'selective_report.json'}")
    print(f"TEBDE classes: {TEBDE_CLASSES}  (Collapsed = index 2)")
    print("Note: collapse-aware triage flags high P(Collapsed) buildings for human review first.")


if __name__ == "__main__":
    main()
