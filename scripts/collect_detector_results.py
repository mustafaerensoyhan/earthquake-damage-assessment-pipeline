#!/usr/bin/env python
"""
Collect results from all trained detector models.

Scans outputs/models/ for completed training runs, validates each,
and writes a consolidated JSON with mAP metrics.  Safe to re-run —
it merges with any existing results file.

Usage
-----
    python scripts/collect_detector_results.py
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import YOLODetector
from src.utils.config import MODEL_DIR, RESULTS_DIR, YOLO_DATASET_DIR, ensure_dirs


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    ensure_dirs()

    data_yaml = YOLO_DATASET_DIR / "dataset.yaml"
    if not data_yaml.exists():
        logger.error("Dataset YAML not found: %s", data_yaml)
        sys.exit(1)

    results_path = RESULTS_DIR / "detector_training_results.json"

    # Load existing results
    existing = []
    if results_path.exists():
        with open(results_path, "r") as f:
            existing = json.load(f)
    existing_tags = {r["tag"]: r for r in existing}

    # Scan for trained models
    model_dir = Path(MODEL_DIR)
    found_runs = sorted(model_dir.glob("yolov8*_r*/weights/best.pt"))

    if not found_runs:
        logger.warning("No trained models found in %s", model_dir)
        sys.exit(0)

    logger.info("Found %d trained detector(s)", len(found_runs))

    for best_pt in found_runs:
        # Parse tag from directory name: e.g. yolov8n_r640
        tag = best_pt.parent.parent.name
        parts = tag.split("_r")
        if len(parts) != 2:
            logger.warning("Skipping unrecognised directory: %s", tag)
            continue

        model_name = parts[0]  # e.g. "yolov8n"
        resolution = int(parts[1])  # e.g. 640

        logger.info("Validating %s ...", tag)

        try:
            detector = YOLODetector(model_name, weights_path=best_pt)
            metrics = detector.validate(data_yaml=str(data_yaml), imgsz=resolution)
        except Exception as exc:
            logger.error("Failed to validate %s: %s", tag, exc)
            continue

        entry = {
            "tag": tag,
            "model": model_name,
            "resolution": resolution,
            "best_weights": str(best_pt),
            "mAP50": metrics["mAP50"],
            "mAP50-95": metrics["mAP50-95"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }

        existing_tags[tag] = entry
        logger.info(
            "  %s — mAP@0.5=%.4f  mAP@0.5:0.95=%.4f  P=%.3f  R=%.3f",
            tag, metrics["mAP50"], metrics["mAP50-95"],
            metrics["precision"], metrics["recall"],
        )

    # Write consolidated results
    all_results = list(existing_tags.values())
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("")
    logger.info("Consolidated %d result(s) → %s", len(all_results), results_path)
    logger.info("")
    logger.info("%-25s  %8s  %12s  %8s  %8s", "Config", "mAP@0.5", "mAP@0.5:0.95", "Prec", "Recall")
    logger.info("-" * 70)
    for r in sorted(all_results, key=lambda x: x["tag"]):
        logger.info(
            "%-25s  %8.4f  %12.4f  %8.4f  %8.4f",
            r["tag"], r["mAP50"], r["mAP50-95"],
            r.get("precision", 0), r.get("recall", 0),
        )
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
