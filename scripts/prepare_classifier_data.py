#!/usr/bin/env python
"""
Prepare building patches for damage-classifier training.

Steps:
  1. Scan xBD post-disaster labels for EARTHQUAKE events only
     (mexico-earthquake, palu-tsunami).
  2. Build a per-building manifest with bounding boxes and damage labels.
  3. Crop patches from post-disaster images, resize to 224×224.
  4. Stratified train/val split (80/20 by disaster).
  5. Save patches to disk in ImageFolder layout.
  6. Write a CSV manifest for reproducibility.

Run from the project root:
    python scripts/prepare_classifier_data.py
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    XBD_TRAIN_IMAGES,
    XBD_TRAIN_LABELS,
    CLASSIFIER_PATCHES_DIR,
    CLASSIFIER_INPUT_SIZE,
    EARTHQUAKE_PREFIXES,
    VALIDATION_SPLIT,
    XBD_DAMAGE_CLASSES,
    ensure_dirs,
    save_config_snapshot,
)
from src.data.xbd_classifier_dataset import (
    build_classifier_manifest,
    extract_patches_to_disk,
    save_manifest_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract earthquake building patches for classifier training"
    )
    parser.add_argument(
        "--patch-size", type=int, default=CLASSIFIER_INPUT_SIZE,
        help=f"Resize patches to this square size (default: {CLASSIFIER_INPUT_SIZE})",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=VALIDATION_SPLIT,
        help=f"Validation split ratio (default: {VALIDATION_SPLIT})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Sanity checks
    if not XBD_TRAIN_LABELS.is_dir():
        logger.error("xBD labels directory not found: %s", XBD_TRAIN_LABELS)
        logger.error("Update XBD_ROOT in src/utils/config.py to match your setup.")
        sys.exit(1)

    ensure_dirs()

    # Step 1-2: Build manifest
    logger.info("=== Building classifier manifest ===")
    logger.info("Earthquake prefixes: %s", EARTHQUAKE_PREFIXES)
    t0 = time.time()

    records = build_classifier_manifest(
        labels_dir=XBD_TRAIN_LABELS,
        images_dir=XBD_TRAIN_IMAGES,
        disaster_prefixes=EARTHQUAKE_PREFIXES,
    )
    logger.info("Found %d building patches", len(records))

    # Report class distribution
    from collections import Counter
    dist = Counter(r["damage_label"] for r in records)
    logger.info("Class distribution:")
    for cls in XBD_DAMAGE_CLASSES:
        count = dist.get(cls, 0)
        pct = 100 * count / max(len(records), 1)
        logger.info("  %-15s  %5d  (%.1f%%)", cls, count, pct)

    # Save manifest CSV
    manifest_path = CLASSIFIER_PATCHES_DIR / "manifest.csv"
    save_manifest_csv(records, manifest_path)

    # Step 3-5: Extract patches
    logger.info("=== Extracting patches (resize to %d×%d) ===", args.patch_size, args.patch_size)
    train_dir, val_dir = extract_patches_to_disk(
        records,
        output_dir=CLASSIFIER_PATCHES_DIR,
        patch_size=args.patch_size,
        val_ratio=args.val_ratio,
    )

    elapsed = time.time() - t0
    logger.info("Extraction completed in %.1f seconds", elapsed)

    # Validate output
    for split, split_dir in [("train", train_dir), ("val", val_dir)]:
        total = 0
        for cls in XBD_DAMAGE_CLASSES:
            cls_dir = split_dir / cls
            n = len(list(cls_dir.glob("*.png"))) if cls_dir.exists() else 0
            total += n
            logger.info("  %s/%-15s  %5d patches", split, cls, n)
        logger.info("  %s total: %d patches", split, total)

    # Save config snapshot
    snapshot_path = CLASSIFIER_PATCHES_DIR / "config_snapshot.yaml"
    save_config_snapshot(snapshot_path)

    logger.info("=== Done. Ready for classifier training. ===")


if __name__ == "__main__":
    main()
