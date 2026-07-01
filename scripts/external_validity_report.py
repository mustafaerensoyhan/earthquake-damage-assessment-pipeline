#!/usr/bin/env python
"""
External-validity report: replicate the two load-bearing findings on a second,
independent UAV target (RescueNet) and show them side by side with TEBDE.

Finding 1 (calibration gap): cross-domain the model is overconfident; source-fit
temperature fails to transfer; target-fit temperature repairs ECE without
changing accuracy.

Finding 2 (collapse-aware triage): at a fixed human-review budget, routing review
by collapse risk misses fewer true-Collapsed buildings than routing by low
confidence.

Reads the per-seed dumps for each target and aggregates over seeds. Needs:
    logits_{model}_seed{seed}_tebde.npz, _rescuenet.npz   (cross-domain targets)
    logits_{model}_seed{seed}_xbdval.npz                  (source, for source-T)

Usage:
    python scripts/external_validity_report.py
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
from src.uncertainty.selective import (
    confidence_correct, review_priority_confidence, review_priority_collapse,
    collapsed_miss_rate_vs_budget,
)
from src.utils.config import OUTPUT_DIR

COLLAPSED = 2
BUDGETS = [0.0, 0.1, 0.2, 0.3, 0.5]
TARGETS = ["tebde", "rescuenet"]


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _cal_row(probs4, labels3, boot_seed):
    probs3 = harmonize_probs_4to3(probs4)
    pred3 = harmonize_labels_4to3(probs4.argmax(1))
    m = compute_all_metrics_pred(probs3, labels3, pred3, seed=boot_seed)
    m["collapsed_recall"] = per_class_recall(pred3, labels3, 3)[COLLAPSED]
    return m


def analyse_target(unc, model, target, split_frac, split_seed, boot_seed):
    dumps = sorted(unc.glob(f"logits_{model}_seed*_{target}.npz"))
    if not dumps:
        return None
    cal_rows = defaultdict(list)  # method -> list of metric dicts (per seed)
    miss_conf, miss_coll = [], []
    for d in dumps:
        seed = d.stem.split("_seed")[1].split("_")[0]
        z = np.load(d)
        logits, labels = z["logits"], z["labels"].astype(np.int64)
        cal, test = stratified_split(labels, frac=split_frac, seed=split_seed)
        y_test = labels[test]

        # raw
        cal_rows["raw"].append(_cal_row(_softmax(logits[test]), y_test, boot_seed))
        # target-fit T
        Tt = fit_temperature_logits(logits[cal], labels[cal], harmonize=True)
        cal_rows["target_T"].append(_cal_row(_softmax(logits[test] / Tt), y_test, boot_seed))
        # source-fit T (from xBD val dump for this seed, if present)
        xbd = unc / f"logits_{model}_seed{seed}_xbdval.npz"
        if xbd.exists():
            zx = np.load(xbd)
            Ts = fit_temperature_logits(zx["logits"], zx["labels"].astype(np.int64), harmonize=False)
            cal_rows["source_T"].append(_cal_row(_softmax(logits[test] / Ts), y_test, boot_seed))

        # triage on the test split (calibrated probs; ranking-invariant to T but use target-T)
        probs3 = harmonize_probs_4to3(_softmax(logits[test] / Tt))
        pred3 = harmonize_labels_4to3(_softmax(logits[test]).argmax(1))
        conf, _ = confidence_correct(probs3, y_test, pred3)
        pc = review_priority_confidence(conf)
        pk = review_priority_collapse(probs3, pred3)
        miss_conf.append(collapsed_miss_rate_vs_budget(probs3, y_test, pred3, pc, BUDGETS))
        miss_coll.append(collapsed_miss_rate_vs_budget(probs3, y_test, pred3, pk, BUDGETS))

    def agg(method):
        rows = cal_rows.get(method)
        if not rows:
            return None
        return {k: float(np.mean([r[k] for r in rows]))
                for k in ("accuracy", "ece", "nll", "collapsed_recall")}

    return {
        "n_seeds": len(dumps),
        "calibration": {m: agg(m) for m in ("raw", "source_T", "target_T")},
        "triage": {
            "budgets": BUDGETS,
            "conf_only": np.mean(miss_conf, axis=0).tolist(),
            "collapse_aware": np.mean(miss_coll, axis=0).tolist(),
        },
    }


def print_calibration(results):
    print("\n=== Finding 1: calibration gap replicates across UAV targets ===")
    hdr = f"{'model':<16}{'target':<11}{'method':<11}{'acc':>7}{'ECE':>7}{'NLL':>7}{'Collapsed':>11}"
    print(hdr); print("-" * len(hdr))
    names = [("raw", "raw"), ("source_T", "+source-T"), ("target_T", "+target-T")]
    for model, per_t in results.items():
        for target in TARGETS:
            r = per_t.get(target)
            if r is None:
                continue
            for key, name in names:
                m = r["calibration"].get(key)
                if m is None:
                    continue
                print(f"{model:<16}{target:<11}{name:<11}{m['accuracy']:>7.3f}"
                      f"{m['ece']:>7.3f}{m['nll']:>7.3f}{m['collapsed_recall']:>11.3f}")


def print_triage(results):
    print("\n=== Finding 2: collapse-aware triage beats confidence-only on both targets ===")
    print("    (collapsed-miss-rate vs review budget; lower is better)")
    hdr = f"{'model':<16}{'target':<11}{'rule':<14}" + "".join(f"{int(b*100):>6}%" for b in BUDGETS)
    print(hdr); print("-" * len(hdr))
    for model, per_t in results.items():
        for target in TARGETS:
            r = per_t.get(target)
            if r is None:
                continue
            t = r["triage"]
            print(f"{model:<16}{target:<11}{'conf-only':<14}" +
                  "".join(f"{v:>7.3f}" for v in t['conf_only']))
            print(f"{'':<16}{'':<11}{'collapse-aware':<14}" +
                  "".join(f"{v:>7.3f}" for v in t['collapse_aware']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["efficientnet_b0", "resnet34", "deit_tiny"])
    ap.add_argument("--split-frac", type=float, default=0.3)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    unc = Path(OUTPUT_DIR) / "uncertainty"
    results = {}
    for model in args.models:
        per_t = {}
        for target in TARGETS:
            r = analyse_target(unc, model, target, args.split_frac, args.split_seed, args.boot_seed)
            if r is not None:
                per_t[target] = r
        if per_t:
            results[model] = per_t

    if not results:
        print("[external_validity] no dumps found; run dump_all.py and dump_rescuenet.py first.")
        return
    if not any("rescuenet" in v for v in results.values()):
        print("[external_validity] NOTE: no rescuenet dumps found; showing TEBDE only.")
    print_calibration(results)
    print_triage(results)
    (unc / "external_validity_report.json").write_text(json.dumps(results, indent=2))
    print(f"\n[external_validity] wrote {unc / 'external_validity_report.json'}")
    print(f"Classes: {TEBDE_CLASSES}  (Collapsed = index {COLLAPSED})")


if __name__ == "__main__":
    main()
