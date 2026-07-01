#!/usr/bin/env python
"""
Dump per-sample single-pass logits for every trained 4-class classifier on the
extracted RescueNet patches, mirroring dump_all.py. Writes, under
outputs/uncertainty/:
    logits_{model}_seed{seed}_rescuenet.npz   keys: logits (N,4), labels (N,) [3-class]

Run scripts/extract_rescuenet_patches.py first to produce the patch folders.

Usage:
    python scripts/dump_rescuenet.py --root rescuenet_patches
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rescuenet_dataset import RescueNetDataset
from src.data.xbd_classifier_dataset import get_val_transforms
from src.models.classifier import DamageClassifier
from src.utils.config import MODEL_DIR, OUTPUT_DIR


@torch.no_grad()
def _dump(model, loader, device):
    model.eval()
    logits, labels = [], []
    for img, label in loader:
        logits.append(model(img.to(device, non_blocking=True)).cpu().numpy())
        labels.append(np.asarray(label).reshape(-1))
    return np.concatenate(logits), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("rescuenet_patches"))
    ap.add_argument("--models", nargs="+",
                    default=["efficientnet_b0", "resnet34", "deit_tiny"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(OUTPUT_DIR) / "uncertainty"
    out.mkdir(parents=True, exist_ok=True)

    ds = RescueNetDataset(root=args.root, transform=get_val_transforms())
    if len(ds) == 0:
        print(f"[dump_rescuenet] no patches under {args.root}; run extract_rescuenet_patches.py first")
        return
    print(f"[dump_rescuenet] {len(ds)} RescueNet patches")
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    for model_name in args.models:
        for seed in args.seeds:
            w = Path(MODEL_DIR) / f"{model_name}_fp32_seed{seed}_best.pt"
            if not w.exists():
                print(f"[dump_rescuenet] SKIP missing {w.name}")
                continue
            model = DamageClassifier.load(model_name, w).to(device)
            lg, lb = _dump(model, loader, device)
            np.savez(out / f"logits_{model_name}_seed{seed}_rescuenet.npz", logits=lg, labels=lb)
            print(f"[dump_rescuenet] {model_name} seed{seed}: wrote rescuenet dump ({len(lb)} patches)")
    print("[dump_rescuenet] done")


if __name__ == "__main__":
    main()
