#!/usr/bin/env python
"""
Phase 0 sanity check: verify the environment and that everything Phase 1 needs
is present, without changing anything. Run from the repo root:

    python scripts/check_env.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OK, MISS = "  OK  ", "MISSING"


def line(flag, msg):
    print(f"[{flag}] {msg}")


def count_classes(root, classes):
    root = Path(root)
    out = {}
    for c in classes:
        d = root / c
        out[c] = len(list(d.glob("*.png")) + list(d.glob("*.jpg")) +
                     list(d.glob("*.jpeg"))) if d.is_dir() else "no dir"
    return out


def main():
    print("=" * 64)
    print("ENVIRONMENT")
    print("=" * 64)
    line(OK, f"python {sys.version.split()[0]}")
    try:
        import torch
        line(OK, f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            line(OK, f"gpu: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        line(MISS, f"torch import failed: {e}")

    from src.utils.config import (
        CLASSIFIER_PATCHES_DIR, TEBDE_ROOT, MODEL_DIR, OUTPUT_DIR,
    )

    print("\n" + "=" * 64)
    print("DATA")
    print("=" * 64)
    val_dir = Path(CLASSIFIER_PATCHES_DIR) / "val"
    line(OK if val_dir.is_dir() else MISS, f"xBD val dir: {val_dir}")
    if val_dir.is_dir():
        line(OK, f"  xBD val per-class: "
                 f"{count_classes(val_dir, ['no-damage','minor-damage','major-damage','destroyed'])}")
    line(OK if Path(TEBDE_ROOT).is_dir() else MISS, f"TEBDE root: {TEBDE_ROOT}")
    if Path(TEBDE_ROOT).is_dir():
        line(OK, f"  TEBDE per-class: "
                 f"{count_classes(TEBDE_ROOT, ['Intact','Damaged','Collapsed'])}")

    print("\n" + "=" * 64)
    print("CLASSIFIER WEIGHTS (needed for Phase 1, no retraining)")
    print("=" * 64)
    for name in ["efficientnet_b0_fp32_best.pt", "resnet34_fp32_best.pt"]:
        p = Path(MODEL_DIR) / name
        line(OK if p.exists() else MISS, str(p))
    md = Path(MODEL_DIR)
    if md.is_dir():
        pts = sorted(x.name for x in md.glob("*.pt"))
        line(OK, f"all .pt in {md}: {pts if pts else 'none'}")
    else:
        line(MISS, f"model dir does not exist: {md}")

    print("\n" + "=" * 64)
    print("OUTPUT TARGET")
    print("=" * 64)
    line(OK, f"results will be written under: {Path(OUTPUT_DIR) / 'uncertainty'}")
    print("\nIf xBD val, TEBDE, and the two .pt weights are OK, run calib_dump next.")


if __name__ == "__main__":
    main()
