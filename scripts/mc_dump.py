#!/usr/bin/env python
"""
Dump MC-dropout per-pass softmax probabilities on UAVs-TEBDE for every
(backbone, seed). T stochastic passes with dropout active on the head input.

Writes, under outputs/uncertainty/:
    mc_{model}_seed{seed}_tebde.npz   keys: per_pass (T,N,4), labels (N,), T, p

Usage:
    python scripts/mc_dump.py
    python scripts/mc_dump.py --passes 20 --p 0.2
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
from src.data.xbd_classifier_dataset import get_val_transforms
from src.models.classifier import DamageClassifier
from src.uncertainty.mc_dropout import mc_dropout_per_pass_probs
from src.utils.config import MODEL_DIR, OUTPUT_DIR, TEBDE_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["efficientnet_b0", "resnet34", "deit_tiny"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--p", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(OUTPUT_DIR) / "uncertainty"
    out.mkdir(parents=True, exist_ok=True)

    teb = TEBDEDataset(root=TEBDE_ROOT, transform=get_val_transforms())
    loader = DataLoader(teb, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    for model_name in args.models:
        for seed in args.seeds:
            w = Path(MODEL_DIR) / f"{model_name}_fp32_seed{seed}_best.pt"
            if not w.exists():
                print(f"[mc_dump] SKIP missing {w.name}")
                continue
            print(f"[mc_dump] {model_name} seed{seed}  T={args.passes} p={args.p} ...")
            model = DamageClassifier.load(model_name, w).to(device)
            per_pass, labels = mc_dropout_per_pass_probs(
                model, loader, device, T=args.passes, p=args.p)
            var = float(per_pass.var(axis=0).mean())
            np.savez(out / f"mc_{model_name}_seed{seed}_tebde.npz",
                     per_pass=per_pass, labels=labels, T=args.passes, p=args.p)
            flag = "OK" if var > 1e-8 else "WARN: near-zero variance (dropout inactive?)"
            print(f"          per_pass={per_pass.shape}  mean pass-variance={var:.2e}  [{flag}]")
    print("[mc_dump] done")


if __name__ == "__main__":
    main()
