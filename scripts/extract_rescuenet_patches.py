#!/usr/bin/env python
"""
Extract per-building classification patches from RescueNet pixel masks into
class folders (Intact / Damaged / Collapsed), so RescueNet can be used as a
second cross-domain UAV target with the existing reports.

STEP 1 - verify the label ids in YOUR download first:
    python scripts/extract_rescuenet_patches.py --inspect \
        --images-dir <...>/train-org-img --masks-dir <...>/train-label-img

  This prints, for a few masks, the unique pixel values and counts, plus the
  image<->mask pairing it found. Confirm which pixel values are the four building
  classes (no-damage / minor / major / total-destruction) and pass them via
  --class-map if they differ from the default 2:0,3:1,4:1,5:2.

STEP 2 - extract:
    python scripts/extract_rescuenet_patches.py \
        --images-dir <...>/train-org-img --masks-dir <...>/train-label-img \
        --out-dir rescuenet_patches

Masks are expected to be single-channel index PNGs (palette/L mode), which is
what --inspect will confirm. If your masks are colour RGB, tell me and I will add
a colour->class map.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rescuenet_extract import (
    extract_from_pair, inspect_mask_values, RESCUENET_BUILDING_CLASSES, TARGET_NAMES,
)

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def parse_class_map(s):
    if not s:
        return RESCUENET_BUILDING_CLASSES
    out = {}
    for pair in s.split(","):
        raw, tgt = pair.split(":")
        out[int(raw)] = int(tgt)
    return out


def find_mask(img_path: Path, masks_dir: Path, suffix: str):
    stem = img_path.stem
    candidates = [stem + suffix, stem, stem.replace("_org", "") + suffix]
    for cand in candidates:
        for ext in (".png", ".tif", ".tiff", ".jpg"):
            p = masks_dir / f"{cand}{ext}"
            if p.exists():
                return p
    # fallback: any mask whose stem starts with the image stem
    for p in masks_dir.iterdir():
        if p.stem.startswith(stem):
            return p
    return None


def list_images(images_dir: Path):
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--masks-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("rescuenet_patches"))
    ap.add_argument("--mask-suffix", default="_lab")
    ap.add_argument("--class-map", default="", help="e.g. 2:0,3:1,4:1,5:2 (raw_value:target3)")
    ap.add_argument("--min-area", type=int, default=500)
    ap.add_argument("--pad-frac", type=float, default=0.08)
    ap.add_argument("--limit", type=int, default=0, help="process only N images (0 = all)")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    images = list_images(args.images_dir)
    if not images:
        print(f"[rescuenet] no images found in {args.images_dir}"); return

    if args.inspect:
        print(f"[inspect] {len(images)} images in {args.images_dir}")
        agg = {}
        for img_path in images[:6]:
            mp = find_mask(img_path, args.masks_dir, args.mask_suffix)
            if mp is None:
                print(f"  {img_path.name:<40} -> NO MASK FOUND"); continue
            mask = np.array(Image.open(mp))
            vc = inspect_mask_values(mask)
            for v, c in vc.items():
                agg[v] = agg.get(v, 0) + c
            print(f"  {img_path.name:<40} -> {mp.name:<40} mask shape {mask.shape}, "
                  f"mode {Image.open(mp).mode}")
            print(f"      values: {dict(sorted(vc.items()))}")
        print(f"\n[inspect] aggregated pixel values across sampled masks:")
        for v, c in sorted(agg.items()):
            print(f"      value {v:>3}: {c} px")
        print(f"\nDefault building class map (raw->3class): {RESCUENET_BUILDING_CLASSES}")
        print("0=Intact, 1=Damaged, 2=Collapsed. Pass --class-map to override.")
        print("If 'mode' above is RGB rather than P/L, the masks are colour-coded; tell me.")
        return

    class_map = parse_class_map(args.class_map)
    for name in TARGET_NAMES:
        (args.out_dir / name).mkdir(parents=True, exist_ok=True)

    counts = {0: 0, 1: 0, 2: 0}
    missing = 0
    todo = images if args.limit == 0 else images[:args.limit]
    for i, img_path in enumerate(todo):
        mp = find_mask(img_path, args.masks_dir, args.mask_suffix)
        if mp is None:
            missing += 1; continue
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mp))
        if mask.ndim == 3:
            print("[rescuenet] mask is RGB, not index; aborting. Re-run --inspect and tell me.")
            return
        for patch, label3 in extract_from_pair(image, mask, class_map,
                                               args.min_area, args.pad_frac):
            out = args.out_dir / TARGET_NAMES[label3] / f"{img_path.stem}_{counts[label3]:05d}.png"
            Image.fromarray(patch).save(out)
            counts[label3] += 1
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(todo)}  patches so far: {counts}")

    print(f"\n[rescuenet] done. images with no mask: {missing}")
    print(f"[rescuenet] patches: Intact={counts[0]}, Damaged={counts[1]}, Collapsed={counts[2]}")
    print(f"[rescuenet] written under {args.out_dir}/")


if __name__ == "__main__":
    main()
