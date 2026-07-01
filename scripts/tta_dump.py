#!/usr/bin/env python
"""
Dump TTA per-view softmax probabilities on UAVs-TEBDE for every (backbone, seed),
using the safe UAV augmentation pool. The per-view tensor is reused by the
remedies report to compare fusion strategies without re-running the model.

Writes, under outputs/uncertainty/:
    tta_{model}_seed{seed}_tebde.npz   keys: per_view (N,S,4), labels (N,), views (S,)

Usage:
    python scripts/tta_dump.py
    python scripts/tta_dump.py --n-views 8 --models deit_tiny --seeds 42
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
from src.models.classifier import DamageClassifier
from src.uncertainty.augmentations import get_tta_pipeline, get_tta_transforms
from src.uncertainty.tta import tta_per_view_probs
from src.utils.config import MODEL_DIR, OUTPUT_DIR, TEBDE_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["efficientnet_b0", "resnet34", "deit_tiny"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    ap.add_argument("--n-views", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for stochastic augs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random as _r; _r.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(OUTPUT_DIR) / "uncertainty"
    out.mkdir(parents=True, exist_ok=True)

    augs = get_tta_pipeline(n_views=args.n_views, pool="safe")
    view_names = [name for _fn, name in augs]
    print(f"[tta_dump] views ({len(augs)}): {view_names}")

    # Un-normalised loader: augmentations + normalisation happen in the TTA loop.
    teb = TEBDEDataset(root=TEBDE_ROOT, transform=get_tta_transforms())
    loader = DataLoader(teb, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    for model_name in args.models:
        for seed in args.seeds:
            w = Path(MODEL_DIR) / f"{model_name}_fp32_seed{seed}_best.pt"
            if not w.exists():
                print(f"[tta_dump] SKIP missing {w.name}")
                continue
            print(f"[tta_dump] {model_name} seed{seed} on {device} ...")
            model = DamageClassifier.load(model_name, w).to(device)
            per_view, labels = tta_per_view_probs(model, loader, augs, device)
            np.savez(out / f"tta_{model_name}_seed{seed}_tebde.npz",
                     per_view=per_view, labels=labels, views=np.array(view_names))
            print(f"           per_view={per_view.shape} -> saved")
    print("[tta_dump] done")


if __name__ == "__main__":
    main()
