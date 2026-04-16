#!/usr/bin/env python
"""
Evaluate all trained models in FP16 mode (inference only).

Runs validation on existing trained detectors and classifiers with
half=True to measure accuracy impact of reduced precision.
Results are appended to existing JSON files.

Usage
-----
    python scripts/eval_fp16.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import YOLODetector
from src.models.classifier import DamageClassifier
from src.data.xbd_classifier_dataset import PatchFolderDataset, get_val_transforms
from src.utils.config import (
    CLASSIFIER_PATCHES_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    YOLO_DATASET_DIR,
    XBD_DAMAGE_CLASSES,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


def eval_detectors_fp16():
    """Evaluate all trained detectors in FP16 mode."""
    data_yaml = YOLO_DATASET_DIR / "dataset.yaml"
    if not data_yaml.exists():
        logger.warning("Dataset YAML not found, skipping detector FP16 eval")
        return []

    model_dir = Path(MODEL_DIR)
    results = []

    for run_dir in sorted(model_dir.glob("yolov8*_r*/weights/best.pt")):
        tag = run_dir.parent.parent.name
        parts = tag.split("_r")
        model_name = parts[0]
        resolution = int(parts[1])

        fp16_tag = f"{tag}_fp16"
        logger.info("Evaluating detector: %s (FP16)", tag)

        detector = YOLODetector(model_name, weights_path=run_dir)
        metrics = detector.validate(
            data_yaml=str(data_yaml), imgsz=resolution, half=True,
        )

        entry = {
            "tag": fp16_tag,
            "model": model_name,
            "resolution": resolution,
            "precision_mode": "fp16",
            "best_weights": str(run_dir),
            "mAP50": metrics["mAP50"],
            "mAP50-95": metrics["mAP50-95"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
        results.append(entry)

        logger.info(
            "  %s — mAP@0.5=%.4f  mAP@0.5:0.95=%.4f",
            fp16_tag, metrics["mAP50"], metrics["mAP50-95"],
        )

    return results


def eval_classifiers_fp16():
    """Evaluate all trained classifiers in FP16 mode."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dir = CLASSIFIER_PATCHES_DIR / "val"
    if not val_dir.exists():
        logger.warning("Validation patches not found, skipping classifier FP16 eval")
        return []

    val_ds = PatchFolderDataset(val_dir, transform=get_val_transforms())
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
    )

    model_dir = Path(MODEL_DIR)
    results = []

    for pt_file in sorted(model_dir.glob("*_fp32_best.pt")):
        base_tag = pt_file.stem.replace("_best", "")  # e.g. "efficientnet_b0_fp32"
        model_name = base_tag.replace("_fp32", "")
        fp16_tag = f"{model_name}_fp16"

        logger.info("Evaluating classifier: %s (FP16)", model_name)

        model = DamageClassifier.load(model_name, pt_file).to(device)
        model.eval()
        model = model.half()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True).half()
                logits = model(images)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds).tolist()
        report = classification_report(
            all_labels, all_preds,
            target_names=XBD_DAMAGE_CLASSES,
            output_dict=True,
            zero_division=0,
        )

        entry = {
            "tag": fp16_tag,
            "model": model_name,
            "precision_mode": "fp16",
            "best_weights": str(pt_file),
            "val_accuracy": round(float(acc), 4),
            "val_macro_f1": round(float(f1), 4),
            "classification_report": report,
            "confusion_matrix": cm,
        }
        results.append(entry)

        logger.info("  %s — Accuracy=%.4f  Macro F1=%.4f", fp16_tag, acc, f1)

        logger.info("\n%s", classification_report(
            all_labels, all_preds,
            target_names=XBD_DAMAGE_CLASSES,
            zero_division=0,
        ))

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ensure_dirs()

    logger.info("=== FP16 Accuracy Evaluation ===")

    # --- Detectors ---
    logger.info("\n--- Detector FP16 Evaluation ---")
    det_fp16 = eval_detectors_fp16()

    # Merge with existing detector results
    det_path = RESULTS_DIR / "detector_training_results.json"
    existing_det = []
    if det_path.exists():
        with open(det_path) as f:
            existing_det = json.load(f)

    existing_tags = {r["tag"]: i for i, r in enumerate(existing_det)}
    for entry in det_fp16:
        if entry["tag"] in existing_tags:
            existing_det[existing_tags[entry["tag"]]] = entry
        else:
            existing_det.append(entry)

    with open(det_path, "w") as f:
        json.dump(existing_det, f, indent=2)
    logger.info("Detector results updated → %s", det_path)

    # --- Classifiers ---
    logger.info("\n--- Classifier FP16 Evaluation ---")
    clf_fp16 = eval_classifiers_fp16()

    # Merge with existing classifier results
    clf_path = RESULTS_DIR / "classifier_training_results.json"
    existing_clf = []
    if clf_path.exists():
        with open(clf_path) as f:
            existing_clf = json.load(f)

    existing_tags = {r["tag"]: i for i, r in enumerate(existing_clf)}
    for entry in clf_fp16:
        if entry["tag"] in existing_tags:
            existing_clf[existing_tags[entry["tag"]]] = entry
        else:
            existing_clf.append(entry)

    with open(clf_path, "w") as f:
        json.dump(existing_clf, f, indent=2)
    logger.info("Classifier results updated → %s", clf_path)

    # --- Summary ---
    logger.info("\n" + "=" * 70)
    logger.info("FP16 vs FP32 COMPARISON — DETECTORS")
    logger.info("=" * 70)
    logger.info("%-25s  %8s  %12s", "Config", "mAP@0.5", "mAP@0.5:0.95")
    logger.info("-" * 50)
    for r in sorted(existing_det, key=lambda x: x["tag"]):
        logger.info(
            "%-25s  %8.4f  %12.4f",
            r["tag"], r.get("mAP50", 0), r.get("mAP50-95", 0),
        )

    logger.info("\n" + "=" * 70)
    logger.info("FP16 vs FP32 COMPARISON — CLASSIFIERS")
    logger.info("=" * 70)
    logger.info("%-25s  %8s  %8s", "Config", "Accuracy", "Macro F1")
    logger.info("-" * 45)
    for r in sorted(existing_clf, key=lambda x: x["tag"]):
        logger.info(
            "%-25s  %8.4f  %8.4f",
            r["tag"], r.get("val_accuracy", 0), r.get("val_macro_f1", 0),
        )

    logger.info("\n=== FP16 evaluation complete ===")


if __name__ == "__main__":
    main()
