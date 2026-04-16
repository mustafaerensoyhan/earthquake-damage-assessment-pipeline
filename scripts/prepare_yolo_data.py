#!/usr/bin/env python
"""
Prepare the YOLO-format dataset for detector training.

Steps:
  1. Parse all xBD post-disaster labels (ALL disaster types).
  2. Convert building polygons to YOLO bounding boxes.
  3. Create a stratified train/val split (80/20 by disaster type).
  4. Copy images and write label .txt files.
  5. Generate ``dataset.yaml`` for Ultralytics.

Run from the project root:
    python scripts/prepare_yolo_data.py [--no-copy]
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
    YOLO_DATASET_DIR,
    VALIDATION_SPLIT,
    RANDOM_SEED,
    ensure_dirs,
    save_config_snapshot,
)
from src.data.xbd_yolo_converter import convert_xbd_to_yolo
from src.data.xbd_parser import get_disaster_names, count_buildings_by_damage


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from xBD")
    parser.add_argument(
        "--no-copy", action="store_true",
        help="Symlink images instead of copying (saves disk space)",
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

    # Report dataset contents
    logger.info("=== xBD Dataset Summary ===")
    disasters = get_disaster_names(XBD_TRAIN_LABELS)
    logger.info("Disaster events found: %s", sorted(disasters))

    n_post = len(list(XBD_TRAIN_LABELS.glob("*_post_disaster.json")))
    logger.info("Post-disaster label files: %d", n_post)

    # Convert to YOLO format
    logger.info("=== Converting to YOLO format ===")
    t0 = time.time()

    yaml_path = convert_xbd_to_yolo(
        images_dir=XBD_TRAIN_IMAGES,
        labels_dir=XBD_TRAIN_LABELS,
        output_dir=YOLO_DATASET_DIR,
        val_ratio=args.val_ratio,
        copy_images=not args.no_copy,
    )

    elapsed = time.time() - t0
    logger.info("Conversion completed in %.1f seconds", elapsed)
    logger.info("Dataset YAML: %s", yaml_path)

    # Save config snapshot
    snapshot_path = YOLO_DATASET_DIR / "config_snapshot.yaml"
    save_config_snapshot(snapshot_path)
    logger.info("Config snapshot saved to %s", snapshot_path)

    # Quick validation
    for split in ("train", "val"):
        img_dir = YOLO_DATASET_DIR / "images" / split
        lbl_dir = YOLO_DATASET_DIR / "labels" / split
        n_img = len(list(img_dir.glob("*.png")))
        n_lbl = len(list(lbl_dir.glob("*.txt")))
        logger.info("  %s: %d images, %d label files", split, n_img, n_lbl)

    logger.info("=== Done. Ready for detector training. ===")


if __name__ == "__main__":
    main()
