#!/usr/bin/env python
"""
Train YOLO detectors for building localization (Stage 1).

Trains each (model_size, resolution) pair sequentially.  FP16 is applied
at inference time, so we only need one training run per pair.

Usage
-----
    python scripts/train_detector.py                    # train all 6 combos
    python scripts/train_detector.py --model yolov8s    # single model
    python scripts/train_detector.py --model yolov8n --resolution 640
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import YOLODetector
from src.utils.config import (
    DETECTOR_EPOCHS,
    DETECTOR_PATIENCE,
    MODEL_DIR,
    RESULTS_DIR,
    YOLO_DATASET_DIR,
    ensure_dirs,
)


def main():
    parser = argparse.ArgumentParser(description="Train YOLO building detectors")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["yolov8n", "yolov8s", "yolov8m"],
        help="Train a single model size (default: train all three)",
    )
    parser.add_argument(
        "--resolution", type=int, default=None,
        choices=[640, 800],
        help="Train at a single resolution (default: both 640 and 800)",
    )
    parser.add_argument("--epochs", type=int, default=DETECTOR_EPOCHS)
    parser.add_argument("--patience", type=int, default=DETECTOR_PATIENCE)
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size (default: auto per model size)",
    )
    args = parser.parse_args()

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
        logger.error("Run 'python scripts/prepare_yolo_data.py' first.")
        sys.exit(1)

    # Determine which configurations to train
    model_names = [args.model] if args.model else ["yolov8n", "yolov8s", "yolov8m"]
    resolutions = [args.resolution] if args.resolution else [640, 800]

    results_summary = []

    for model_name in model_names:
        for res in resolutions:
            tag = f"{model_name}_r{res}"
            logger.info("=" * 60)
            logger.info("Training: %s", tag)
            logger.info("=" * 60)

            detector = YOLODetector(model_name)
            t0 = time.time()

            train_kwargs = {}
            if args.batch_size is not None:
                train_kwargs["batch"] = args.batch_size

            best_weights = detector.train(
                data_yaml=str(data_yaml),
                epochs=args.epochs,
                imgsz=res,
                patience=args.patience,
                project=str(MODEL_DIR),
                name=tag,
                **train_kwargs,
            )

            train_time = time.time() - t0

            # Validate and record metrics
            detector_trained = YOLODetector(model_name, weights_path=best_weights)
            logger.info("Validating %s ...", tag)
            metrics = detector_trained.validate(
                data_yaml=str(data_yaml), imgsz=res
            )

            entry = {
                "tag": tag,
                "model": model_name,
                "resolution": res,
                "epochs_trained": args.epochs,
                "train_time_s": round(train_time, 1),
                "best_weights": str(best_weights),
                **metrics,
            }
            results_summary.append(entry)

            logger.info(
                "%s — mAP@0.5=%.4f  mAP@0.5:0.95=%.4f  (%.0fs)",
                tag, metrics["mAP50"], metrics["mAP50-95"], train_time,
            )

    # Append to existing results (don't overwrite previous runs)
    results_path = RESULTS_DIR / "detector_training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if results_path.exists():
        with open(results_path, "r") as f:
            existing = json.load(f)
    # Replace entries with same tag, append new ones
    existing_tags = {r["tag"]: i for i, r in enumerate(existing)}
    for entry in results_summary:
        if entry["tag"] in existing_tags:
            existing[existing_tags[entry["tag"]]] = entry
        else:
            existing.append(entry)
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Print summary table
    logger.info("")
    logger.info("%-25s  %8s  %12s  %10s", "Config", "mAP@0.5", "mAP@0.5:0.95", "Time (s)")
    logger.info("-" * 65)
    for r in results_summary:
        logger.info(
            "%-25s  %8.4f  %12.4f  %10.1f",
            r["tag"], r["mAP50"], r["mAP50-95"], r["train_time_s"],
        )

    logger.info("=== Detector training complete ===")


if __name__ == "__main__":
    main()
