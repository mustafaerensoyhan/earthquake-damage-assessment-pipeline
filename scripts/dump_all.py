#!/usr/bin/env python
"""
Dump per-sample single-pass LOGITS for every trained 4-class classifier on the
in-domain xBD validation split and the cross-domain UAVs-TEBDE set.

Iterates the (backbone, seed) grid by naming convention:
    outputs/models/{model}_fp32_seed{seed}_best.pt
and writes, under outputs/uncertainty/:
    logits_{model}_seed{seed}_xbdval.npz   keys: logits (N,4), labels (N,)
    logits_{model}_seed{seed}_tebde.npz    keys: logits (N,4), labels (N,)  [3-class]

Usage:
    python scripts/dump_all.py
    python scripts/dump_all.py --models efficientnet_b0 --seeds 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tebde_dataset import TEBDEDataset
from src.data.xbd_classifier_dataset import PatchFolderDataset, get_val_transforms
from src.models.classifier import DamageClassifier
from src.utils.config import CLASSIFIER_PATCHES_DIR, MODEL_DIR, OUTPUT_DIR, TEBDE_ROOT


@torch.no_grad()
def _dump(model, loader, device):
    model.eval()
    logits, labels = [], []
    for img, label in loader:
        img = img.to(device, non_blocking=True)
        logits.append(model(img).cpu().numpy())
        labels.append(np.asarray(label).reshape(-1))
    return np.concatenate(logits), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["efficientnet_b0", "resnet34", "deit_tiny"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(OUTPUT_DIR) / "uncertainty"
    out.mkdir(parents=True, exist_ok=True)
    tf = get_val_transforms()

    xbd = PatchFolderDataset(Path(CLASSIFIER_PATCHES_DIR) / "val", transform=tf)
    teb = TEBDEDataset(root=TEBDE_ROOT, transform=tf)
    xbd_loader = DataLoader(xbd, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    teb_loader = DataLoader(teb, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    for model_name in args.models:
        for seed in args.seeds:
            w = Path(MODEL_DIR) / f"{model_name}_fp32_seed{seed}_best.pt"
            if not w.exists():
                print(f"[dump_all] SKIP missing {w.name}")
                continue
            print(f"[dump_all] {model_name} seed{seed} on {device}")
            model = DamageClassifier.load(model_name, w).to(device)
            lg, lb = _dump(model, xbd_loader, device)
            np.savez(out / f"logits_{model_name}_seed{seed}_xbdval.npz", logits=lg, labels=lb)
            lg, lb = _dump(model, teb_loader, device)
            np.savez(out / f"logits_{model_name}_seed{seed}_tebde.npz", logits=lg, labels=lb)
            print(f"           wrote xbdval + tebde dumps")
    print("[dump_all] done")


if __name__ == "__main__":
    main()
