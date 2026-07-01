#!/usr/bin/env python
"""
Retraining-free remedies on UAVs-TEBDE, compared on the common seeded test split.

For each (backbone, seed) it reports four inference modes:
    raw            single forward pass
    +target-T      target-fit temperature scaling
    +TTA(entropy)  uncertainty-weighted test-time augmentation (safe pool)
    +MC-dropout    T stochastic passes averaged
For each: accuracy (4-class argmax mapped to 3, so it reflects the decision),
ECE, NLL, and Collapsed recall (the safety-critical class). A short per-fusion
scan is also printed so the best TTA strategy is visible.

Needs dumps from dump_all.py, tta_dump.py, and mc_dump.py.

Usage:
    python scripts/remedies_report.py
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
from src.uncertainty.temperature import fit_temperature_logits, apply_temperature
from src.uncertainty.tta import fuse, FUSION_FNS
from src.uncertainty.mc_dropout import mc_dropout_fuse
from src.utils.config import OUTPUT_DIR

COLLAPSED = 2  # TEBDE class index for Collapsed


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _row_from_probs4(probs4, labels3, boot_seed):
    """Harmonise 4-class probs to 3, predict via 4-class argmax, score."""
    probs3 = harmonize_probs_4to3(probs4)
    pred3 = harmonize_labels_4to3(probs4.argmax(1))
    m = compute_all_metrics_pred(probs3, labels3, pred3, seed=boot_seed)
    m["collapsed_recall"] = per_class_recall(pred3, labels3, 3)[COLLAPSED]
    return m


def discover_seeds(unc, model):
    return sorted(int(p.stem.rsplit("_seed", 1)[1].split("_")[0])
                  for p in unc.glob(f"logits_{model}_seed*_tebde.npz"))


def analyse(unc, model, seed, split_frac, split_seed, boot_seed):
    teb = np.load(unc / f"logits_{model}_seed{seed}_tebde.npz")
    tl, ty = teb["logits"], teb["labels"].astype(np.int64)
    cal, test = stratified_split(ty, frac=split_frac, seed=split_seed)
    T_tgt = fit_temperature_logits(tl[cal], ty[cal], harmonize=True)

    rows = {
        "raw": _row_from_probs4(_softmax(tl[test]), ty[test], boot_seed),
        "target_T": _row_from_probs4(_softmax(tl[test] / T_tgt), ty[test], boot_seed),
    }

    # TTA: load per-view probs, scan fusions on the test split, keep entropy primary
    tta_p = unc / f"tta_{model}_seed{seed}_tebde.npz"
    fusion_scan = {}
    if tta_p.exists():
        pv = np.load(tta_p)["per_view"][test]            # (n_test, S, 4)
        for strat in FUSION_FNS:
            fused = fuse(pv, strat)
            fusion_scan[strat] = _row_from_probs4(fused, ty[test], boot_seed)
        rows["tta_entropy"] = fusion_scan["entropy"]
    # MC-dropout
    mc_p = unc / f"mc_{model}_seed{seed}_tebde.npz"
    if mc_p.exists():
        per_pass = np.load(mc_p)["per_pass"][:, test, :]  # (T, n_test, 4)
        rows["mc_dropout"] = _row_from_probs4(mc_dropout_fuse(per_pass), ty[test], boot_seed)

    return {"model": model, "seed": seed, "T_target": T_tgt,
            "rows": rows, "fusion_scan": fusion_scan}


def print_table(results):
    order = [("raw", "raw"), ("target_T", "+target-T"),
             ("tta_entropy", "+TTA(entropy)"), ("mc_dropout", "+MC-dropout")]
    hdr = (f"{'model':<16}{'seed':>5}  {'mode':<16}{'acc':>7}{'ECE':>7}"
           f"{'NLL':>7}{'Collapsed rec':>15}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in results:
        for key, name in order:
            if key not in r["rows"]:
                continue
            m = r["rows"][key]
            cr = m["collapsed_recall"]
            print(f"{r['model']:<16}{r['seed']:>5}  {name:<16}{m['accuracy']:>7.3f}"
                  f"{m['ece']:>7.3f}{m['nll']:>7.3f}{cr:>15.3f}")


def print_aggregate(results):
    by = defaultdict(lambda: defaultdict(list))
    for r in results:
        for k, m in r["rows"].items():
            by[r["model"]][k].append(m)
    order = [("raw", "raw"), ("target_T", "+target-T"),
             ("tta_entropy", "+TTA(entropy)"), ("mc_dropout", "+MC-dropout")]
    print("\n=== aggregated across seeds (mean) ===")
    hdr = f"{'model':<16}{'mode':<16}{'acc':>7}{'ECE':>7}{'NLL':>7}{'Collapsed rec':>15}"
    print(hdr); print("-" * len(hdr))
    for model, modes in by.items():
        for key, name in order:
            if key not in modes:
                continue
            ms = modes[key]
            acc = np.mean([m["accuracy"] for m in ms])
            ece = np.mean([m["ece"] for m in ms])
            nll = np.mean([m["nll"] for m in ms])
            cr = np.nanmean([m["collapsed_recall"] for m in ms])
            print(f"{model:<16}{name:<16}{acc:>7.3f}{ece:>7.3f}{nll:>7.3f}{cr:>15.3f}")


def print_fusion_scan(results):
    print("\n=== TTA fusion scan (ECE, mean over seeds) ===")
    by = defaultdict(lambda: defaultdict(list))
    for r in results:
        for strat, m in r.get("fusion_scan", {}).items():
            by[r["model"]][strat].append(m["ece"])
    if not any(by.values()):
        print("(no TTA dumps found)"); return
    strategies = sorted({s for m in by.values() for s in m})
    print(f"{'model':<16}" + "".join(f"{s:>14}" for s in strategies))
    for model, d in by.items():
        print(f"{model:<16}" + "".join(
            f"{np.mean(d[s]):>14.3f}" if s in d else f"{'-':>14}" for s in strategies))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["efficientnet_b0", "resnet34", "deit_tiny"])
    ap.add_argument("--split-frac", type=float, default=0.3)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    unc = Path(OUTPUT_DIR) / "uncertainty"
    results = []
    for model in args.models:
        for seed in discover_seeds(unc, model):
            results.append(analyse(unc, model, seed, args.split_frac,
                                   args.split_seed, args.boot_seed))
    if not results:
        print("[remedies_report] no dumps found; run dump_all/tta_dump/mc_dump first."); return
    print_table(results)
    print_aggregate(results)
    print_fusion_scan(results)
    (unc / "remedies_report.json").write_text(json.dumps(results, indent=2))
    print(f"\n[remedies_report] wrote {unc / 'remedies_report.json'}")
    print(f"TEBDE classes: {TEBDE_CLASSES}  (Collapsed = index {COLLAPSED})")


if __name__ == "__main__":
    main()
