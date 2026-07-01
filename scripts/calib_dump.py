#!/usr/bin/env python
"""
Seed-aware single-model logit dumper (companion to dump_all.py). Dumps one
backbone+seed at a time, useful for re-running a single config.

    python scripts/calib_dump.py --model efficientnet_b0 --seed 42 \
        --weights outputs/models/efficientnet_b0_fp32_seed42_best.pt
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
    lg, lb = [], []
    for img, label in loader:
        lg.append(model(img.to(device, non_blocking=True)).cpu().numpy())
        lb.append(np.asarray(label).reshape(-1))
    return np.concatenate(lg), np.concatenate(lb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["efficientnet_b0", "resnet34", "deit_tiny"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--weights", default=None, help="defaults to the seed-tagged checkpoint")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    weights = Path(args.weights) if args.weights else \
        Path(MODEL_DIR) / f"{args.model}_fp32_seed{args.seed}_best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(OUTPUT_DIR) / "uncertainty"; out.mkdir(parents=True, exist_ok=True)
    tf = get_val_transforms()

    model = DamageClassifier.load(args.model, weights).to(device)
    xbd = PatchFolderDataset(Path(CLASSIFIER_PATCHES_DIR) / "val", transform=tf)
    teb = TEBDEDataset(root=TEBDE_ROOT, transform=tf)
    for name, ds in [("xbdval", xbd), ("tebde", teb)]:
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)
        lg, lb = _dump(model, loader, device)
        p = out / f"logits_{args.model}_seed{args.seed}_{name}.npz"
        np.savez(p, logits=lg, labels=lb)
        print(f"[calib_dump] wrote {p.name}  logits={lg.shape}")


if __name__ == "__main__":
    main()
