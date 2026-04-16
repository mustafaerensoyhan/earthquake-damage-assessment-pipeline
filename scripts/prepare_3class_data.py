#!/usr/bin/env python
"""
Prepare 3-class building patches for classifier training.

Merges xBD 4-class labels into 3 classes:
    no-damage -> Intact | minor+major -> Damaged | destroyed -> Collapsed

Usage:
    python scripts/prepare_3class_data.py
"""

import logging
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.three_class_dataset import (
    THREE_CLASS_NAMES,
    build_3class_manifest,
    extract_3class_patches,
)
from src.utils.config import (
    CLASSIFIER_INPUT_SIZE,
    EARTHQUAKE_PREFIXES,
    ROOT,
    VALIDATION_SPLIT,
    XBD_TRAIN_IMAGES,
    XBD_TRAIN_LABELS,
    ensure_dirs,
)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    ensure_dirs()

    output_dir = ROOT / "classifier_patches_3class"

    if not XBD_TRAIN_LABELS.is_dir():
        logger.error("xBD labels not found: %s", XBD_TRAIN_LABELS)
        sys.exit(1)

    logger.info("=== Building 3-class manifest ===")
    logger.info("Earthquake prefixes: %s", EARTHQUAKE_PREFIXES)
    t0 = time.time()

    records = build_3class_manifest(
        labels_dir=XBD_TRAIN_LABELS,
        images_dir=XBD_TRAIN_IMAGES,
        disaster_prefixes=EARTHQUAKE_PREFIXES,
    )
    logger.info("Found %d building patches", len(records))

    dist = Counter(r["class_name"] for r in records)
    logger.info("3-class distribution:")
    for cls in THREE_CLASS_NAMES:
        count = dist.get(cls, 0)
        pct = 100 * count / max(len(records), 1)
        logger.info("  %-12s  %5d  (%.1f%%)", cls, count, pct)

    logger.info("=== Extracting patches (resize to %dx%d) ===",
                CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE)

    train_dir, val_dir = extract_3class_patches(
        records, output_dir=output_dir,
        patch_size=CLASSIFIER_INPUT_SIZE,
        val_ratio=VALIDATION_SPLIT,
    )

    elapsed = time.time() - t0
    logger.info("Extraction completed in %.1f seconds", elapsed)

    for split, split_dir in [("train", train_dir), ("val", val_dir)]:
        total = 0
        for cls in THREE_CLASS_NAMES:
            cls_dir = split_dir / cls
            n = len(list(cls_dir.glob("*.png"))) if cls_dir.exists() else 0
            total += n
            logger.info("  %s/%-12s  %5d patches", split, cls, n)
        logger.info("  %s total: %d patches", split, total)

    logger.info("=== Done. Ready for 3-class classifier training. ===")


if __name__ == "__main__":
    main()
