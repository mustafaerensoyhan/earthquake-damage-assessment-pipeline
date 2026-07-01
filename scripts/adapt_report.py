#!/usr/bin/env python
"""
Label-free test-time adaptation on UAVs-TEBDE, compared on the common seeded test
split against the raw model and target-temperature calibration.

Methods per (backbone, seed):
    raw         single forward pass (from dumped logits)
    target-T    target-fit temperature scaling (from dumped logits)
    BN-adapt    recompute BN statistics on the unlabeled target stream
    TENT        entropy minimisation over normalisation affine params (no labels)

For each: accuracy (4-class argmax mapped to 3), ECE, NLL, Collapsed recall. The
point is to see whether label-free adaptation improves accuracy and what it does
to calibration. BN-adapt is skipped for DeiT (no BatchNorm); TENT on DeiT adapts
LayerNorm affine parameters and is flagged as a variant.

Needs dump_all.py outputs and the trained checkpoints.

Usage:
    python scripts/adapt_report.py
    python scripts/adapt_report.py --models efficientnet_b0 resnet34 --tent-steps 1
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tebde_dataset import TEBDEDataset
from src.data.xbd_classifier_dataset import get_val_transforms
from src.models.classifier import DamageClassifier
from src.uncertainty.metrics import compute_all_metrics_pred, per_class_recall
from src.uncertainty.harmonize import (
    harmonize_probs_4to3, harmonize_labels_4to3, stratified_split, TEBDE_CLASSES,
)
from src.uncertainty.temperature import fit_temperature_logits, apply_temperature
from src.uncertainty.test_time_adapt import bn_adapt, tent_adapt, has_batchnorm
from src.utils.config import MODEL_DIR, OUTPUT_DIR, TEBDE_ROOT

COLLAPSED = 2


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _row_from_probs4(probs4, labels3, boot_seed):
    probs3 = harmonize_probs_4to3(probs4)
    pred3 = harmonize_labels_4to3(probs4.argmax(1))
    m = compute_all_metrics_pred(probs3, labels3, pred3, seed=boot_seed)
    m["collapsed_recall"] = per_class_recall(pred3, labels3, 3)[COLLAPSED]
    return m


@torch.no_grad()
def _infer_logits(model, loader, device):
    model.eval()
    out, ys = [], []
    for img, lab in loader:
        out.append(model(img.to(device)).cpu().numpy())
        ys.append(np.asarray(lab).reshape(-1))
    return np.concatenate(out), np.concatenate(ys)


def discover_seeds(unc, model):
    seeds = []
    for p in unc.glob(f"logits_{model}_seed*_tebde.npz"):
        s = p.stem.rsplit("_seed", 1)[1].split("_")[0]
        if s.isdigit():
            seeds.append(int(s))
    return sorted(set(seeds))


def analyse(unc, model_name, seed, test_ds_val, split_frac, split_seed,
            tent_steps, tent_lr, batch_size, device, boot_seed):
    teb = np.load(unc / f"logits_{model_name}_seed{seed}_tebde.npz")
    tl, ty = teb["logits"], teb["labels"].astype(np.int64)
    cal, test = stratified_split(ty, frac=split_frac, seed=split_seed)
    y_test = ty[test]

    rows = {}
    rows["raw"] = _row_from_probs4(_softmax(tl[test]), y_test, boot_seed)
    T = fit_temperature_logits(tl[cal], ty[cal], harmonize=True)
    rows["target_T"] = _row_from_probs4(_softmax(tl[test] / T), y_test, boot_seed)

    test_subset = Subset(test_ds_val, test.tolist())
    eval_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    adapt_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True, num_workers=2)

    src = Path(MODEL_DIR) / f"{model_name}_fp32_seed{seed}_best.pt"

    # BN-adapt (skip if no BatchNorm)
    m_bn = DamageClassifier.load(model_name, src).to(device)
    if has_batchnorm(m_bn):
        bn_adapt(m_bn, eval_loader, device)
        logits, ys = _infer_logits(m_bn, eval_loader, device)
        rows["bn_adapt"] = _row_from_probs4(_softmax(logits), ys.astype(np.int64), boot_seed)

    # TENT
    m_tent = DamageClassifier.load(model_name, src).to(device)
    tent_adapt(m_tent, adapt_loader, device, steps=tent_steps, lr=tent_lr)
    logits, ys = _infer_logits(m_tent, eval_loader, device)
    rows["tent"] = _row_from_probs4(_softmax(logits), ys.astype(np.int64), boot_seed)

    return {"model": model_name, "seed": seed, "T_target": T,
            "tent_steps": tent_steps, "rows": rows}


def print_table(results):
    order = [("raw", "raw"), ("target_T", "+target-T"),
             ("bn_adapt", "BN-adapt"), ("tent", "TENT")]
    hdr = (f"{'model':<16}{'seed':>5}  {'method':<12}{'acc':>7}{'ECE':>7}"
           f"{'NLL':>7}{'Collapsed rec':>15}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in results:
        for key, name in order:
            if key not in r["rows"]:
                continue
            m = r["rows"][key]
            print(f"{r['model']:<16}{r['seed']:>5}  {name:<12}{m['accuracy']:>7.3f}"
                  f"{m['ece']:>7.3f}{m['nll']:>7.3f}{m['collapsed_recall']:>15.3f}")


def print_aggregate(results):
    by = defaultdict(lambda: defaultdict(list))
    for r in results:
        for k, m in r["rows"].items():
            by[r["model"]][k].append(m)
    order = [("raw", "raw"), ("target_T", "+target-T"),
             ("bn_adapt", "BN-adapt"), ("tent", "TENT")]
    print("\n=== aggregated across seeds (mean) ===")
    hdr = f"{'model':<16}{'method':<12}{'acc':>7}{'ECE':>7}{'NLL':>7}{'Collapsed rec':>15}"
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
            print(f"{model:<16}{name:<12}{acc:>7.3f}{ece:>7.3f}{nll:>7.3f}{cr:>15.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["efficientnet_b0", "resnet34", "deit_tiny"])
    ap.add_argument("--split-frac", type=float, default=0.3)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--tent-steps", type=int, default=1)
    ap.add_argument("--tent-lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unc = Path(OUTPUT_DIR) / "uncertainty"
    test_ds_val = TEBDEDataset(root=TEBDE_ROOT, transform=get_val_transforms())

    results = []
    for model_name in args.models:
        for seed in discover_seeds(unc, model_name):
            print(f"[adapt] {model_name} seed{seed} ...")
            results.append(analyse(unc, model_name, seed, test_ds_val,
                                   args.split_frac, args.split_seed,
                                   args.tent_steps, args.tent_lr,
                                   args.batch_size, device, args.boot_seed))
    if not results:
        print("[adapt_report] no dumps found; run dump_all.py first."); return
    print_table(results)
    print_aggregate(results)
    (unc / "adapt_report.json").write_text(json.dumps(results, indent=2))
    print(f"\n[adapt_report] wrote {unc / 'adapt_report.json'}")
    print(f"TEBDE classes: {TEBDE_CLASSES}  (Collapsed = index {COLLAPSED})")
    print("Note: BN-adapt and TENT use no target labels. BN-adapt is skipped for DeiT (no BatchNorm);")
    print("      TENT on DeiT adapts LayerNorm affine params (variant).")


if __name__ == "__main__":
    main()
