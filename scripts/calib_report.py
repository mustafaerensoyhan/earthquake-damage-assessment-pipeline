#!/usr/bin/env python
"""
Calibration table across all (backbone, seed) dumps. For each model it reports,
on a common seeded TEBDE test split:
  - in-domain xBD validation calibration (4-class, top-label),
  - cross-domain TEBDE raw, +source-fit T, +target-fit T (3-class harmonised).

The cross-domain PREDICTION is the 4-class argmax mapped to 3 classes, which is
temperature-invariant, so accuracy is identical across raw/source-T/target-T and
only ECE and NLL move. Results are also aggregated across seeds per backbone.

Usage:
    python scripts/calib_report.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uncertainty.metrics import compute_all_metrics, compute_all_metrics_pred
from src.uncertainty.harmonize import (
    harmonize_probs_4to3, harmonize_labels_4to3, stratified_split, TEBDE_CLASSES,
)
from src.uncertainty.temperature import fit_temperature_logits, apply_temperature
from src.utils.config import OUTPUT_DIR


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _cross_domain_row(logits4, labels3, T, boot_seed):
    """Harmonised 3-class metrics with a temperature-invariant prediction."""
    probs3 = apply_temperature(logits4, T, harmonize=True)   # softmax(/T) then 4->3
    pred3 = harmonize_labels_4to3(_softmax(logits4).argmax(1))  # argmax is T-invariant
    return compute_all_metrics_pred(probs3, labels3, pred3, seed=boot_seed)


def discover(unc_dir):
    grid = defaultdict(set)
    for p in unc_dir.glob("logits_*_seed*_tebde.npz"):
        # logits_{model}_seed{seed}_tebde.npz
        stem = p.stem[len("logits_"):-len("_tebde")]
        if "_seed" not in stem:
            continue  # skip stale non-seed dumps from Phase 1
        model, seed = stem.rsplit("_seed", 1)
        if not seed.isdigit():
            continue
        grid[model].add(int(seed))
    return {m: sorted(s) for m, s in grid.items()}


def analyse(unc_dir, model, seed, split_frac, split_seed, boot_seed):
    xbd = np.load(unc_dir / f"logits_{model}_seed{seed}_xbdval.npz")
    teb = np.load(unc_dir / f"logits_{model}_seed{seed}_tebde.npz")
    xl, xy = xbd["logits"], xbd["labels"].astype(np.int64)
    tl, ty = teb["logits"], teb["labels"].astype(np.int64)

    cal, test = stratified_split(ty, frac=split_frac, seed=split_seed)
    T_src = fit_temperature_logits(xl, xy, harmonize=False)
    T_tgt = fit_temperature_logits(tl[cal], ty[cal], harmonize=True)

    return {
        "model": model, "seed": seed, "T_source": T_src, "T_target": T_tgt,
        "rows": {
            "xbd_val_4class": compute_all_metrics(_softmax(xl), xy, seed=boot_seed),
            "tebde_raw": _cross_domain_row(tl[test], ty[test], 1.0, boot_seed),
            "tebde_source_T": _cross_domain_row(tl[test], ty[test], T_src, boot_seed),
            "tebde_target_T": _cross_domain_row(tl[test], ty[test], T_tgt, boot_seed),
        },
    }


def print_table(results):
    label = {"xbd_val_4class": "xBD val (4cls)", "tebde_raw": "TEBDE raw",
             "tebde_source_T": "TEBDE +source-T", "tebde_target_T": "TEBDE +target-T"}
    hdr = f"{'model':<16}{'seed':>5}  {'set':<18}{'acc':>7}{'ECE':>7}{'ECE 95% CI':>18}{'NLL':>7}"
    print("\n" + hdr); print("-" * len(hdr))
    for r in results:
        for key, name in label.items():
            m = r["rows"][key]
            ci = f"[{m['ece_ci_low']:.3f},{m['ece_ci_high']:.3f}]"
            print(f"{r['model']:<16}{r['seed']:>5}  {name:<18}{m['accuracy']:>7.3f}"
                  f"{m['ece']:>7.3f}{ci:>18}{m['nll']:>7.3f}")
        print(f"{'':<21}T_source={r['T_source']:.2f}  T_target={r['T_target']:.2f}")


def print_aggregate(results):
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)
    print("\n=== aggregated across seeds (mean) ===")
    hdr = f"{'model':<16}{'set':<18}{'acc':>7}{'ECE':>7}{'NLL':>7}{'seeds':>7}"
    print(hdr); print("-" * len(hdr))
    keys = ["xbd_val_4class", "tebde_raw", "tebde_source_T", "tebde_target_T"]
    label = {"xbd_val_4class": "xBD val", "tebde_raw": "TEBDE raw",
             "tebde_source_T": "TEBDE +srcT", "tebde_target_T": "TEBDE +tgtT"}
    for model, rs in by_model.items():
        for k in keys:
            acc = np.mean([r["rows"][k]["accuracy"] for r in rs])
            ece = np.mean([r["rows"][k]["ece"] for r in rs])
            nll = np.mean([r["rows"][k]["nll"] for r in rs])
            print(f"{model:<16}{label[k]:<18}{acc:>7.3f}{ece:>7.3f}{nll:>7.3f}{len(rs):>7}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-frac", type=float, default=0.3)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    unc = Path(OUTPUT_DIR) / "uncertainty"
    grid = discover(unc)
    if not grid:
        print("[calib_report] no dumps found; run dump_all.py first."); return
    results = []
    for model, seeds in grid.items():
        for seed in seeds:
            results.append(analyse(unc, model, seed, args.split_frac,
                                   args.split_seed, args.boot_seed))
    print_table(results)
    print_aggregate(results)
    (unc / "calibration_report.json").write_text(json.dumps(results, indent=2))
    print(f"\n[calib_report] wrote {unc / 'calibration_report.json'}")
    print(f"TEBDE classes: {TEBDE_CLASSES}")


if __name__ == "__main__":
    main()
