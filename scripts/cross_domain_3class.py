#!/usr/bin/env python
"""
Cross-domain evaluation for 3-class classifiers on UAVs-TEBDE.

Since the 3-class classifier directly outputs Intact/Damaged/Collapsed,
no label harmonisation is needed — predictions map 1:1 to TEBDE classes.

Usage:
    python scripts/cross_domain_3class.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tebde_dataset import TEBDEDataset
from src.data.xbd_classifier_dataset import get_val_transforms
from src.data.three_class_dataset import THREE_CLASS_NAMES
from src.utils.config import MODEL_DIR, RESULTS_DIR, TEBDE_ROOT, TEBDE_CLASSES, ensure_dirs

logger = logging.getLogger(__name__)


def build_model(model_name, num_classes=3):
    import torchvision.models as models
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    if not TEBDE_ROOT.exists():
        logger.error("TEBDE not found: %s", TEBDE_ROOT)
        sys.exit(1)

    transform = get_val_transforms()
    dataset = TEBDEDataset(root=TEBDE_ROOT, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    logger.info("TEBDE: %d images", len(dataset))

    # Load 4-class xBD results for comparison
    clf_4class_path = RESULTS_DIR / "classifier_training_results.json"
    xbd_4class_acc = {}
    if clf_4class_path.exists():
        with open(clf_4class_path) as f:
            for r in json.load(f):
                xbd_4class_acc[r["model"]] = r.get("val_accuracy", 0)

    # Load 3-class xBD results
    clf_3class_path = RESULTS_DIR / "classifier_3class_results.json"
    xbd_3class_acc = {}
    if clf_3class_path.exists():
        with open(clf_3class_path) as f:
            for r in json.load(f):
                xbd_3class_acc[r["model"]] = r.get("val_accuracy", 0)

    # Find trained 3-class weights
    model_dir = Path(MODEL_DIR)
    weight_files = sorted(model_dir.glob("*_3class_fp32_best.pt"))
    if not weight_files:
        logger.error("No 3-class weights found in %s", model_dir)
        sys.exit(1)

    all_results = []

    for wf in weight_files:
        tag = wf.stem.replace("_best", "")
        model_name = tag.replace("_3class_fp32", "").replace("_3class_fp16", "")
        logger.info("=" * 50)
        logger.info("Evaluating %s on TEBDE (3-class, no harmonisation needed)", tag)
        logger.info("=" * 50)

        model = build_model(model_name, num_classes=3)
        state = torch.load(str(wf), map_location="cpu")
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                preds = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds).tolist()
        report = classification_report(all_labels, all_preds, target_names=TEBDE_CLASSES, output_dict=True, zero_division=0)

        per_class_acc = {}
        for i, cls in enumerate(TEBDE_CLASSES):
            mask = all_labels == i
            per_class_acc[cls] = float((all_preds[mask] == i).mean()) if mask.sum() > 0 else 0.0

        xbd_3c = xbd_3class_acc.get(model_name, 0)
        xbd_4c = xbd_4class_acc.get(model_name, 0)
        delta_3c = xbd_3c - acc

        result = {
            "tag": tag, "model": model_name, "num_classes": 3,
            "tebde_accuracy": round(float(acc), 4),
            "tebde_macro_f1": round(float(f1), 4),
            "xbd_3class_accuracy": round(xbd_3c, 4),
            "xbd_4class_accuracy": round(xbd_4c, 4),
            "delta_3class": round(delta_3c, 4),
            "confusion_matrix": cm,
            "classification_report": report,
            "per_class_accuracy": per_class_acc,
        }
        all_results.append(result)

        logger.info("TEBDE Accuracy: %.4f  |  Macro F1: %.4f", acc, f1)
        logger.info("xBD 3-class Acc: %.4f  |  Delta: %+.4f", xbd_3c, delta_3c)
        logger.info("xBD 4-class Acc: %.4f (for reference)", xbd_4c)
        logger.info("Per-class: %s", per_class_acc)
        logger.info("\n%s", classification_report(all_labels, all_preds, target_names=TEBDE_CLASSES, zero_division=0))

    # Save
    results_path = RESULTS_DIR / "cross_domain_3class_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Summary
    logger.info("")
    logger.info("=" * 65)
    logger.info("3-CLASS CROSS-DOMAIN SUMMARY")
    logger.info("=" * 65)
    logger.info("%-30s  %10s  %10s  %8s", "Config", "xBD 3c Acc", "TEBDE Acc", "Delta")
    logger.info("-" * 65)
    for r in all_results:
        logger.info("%-30s  %10.4f  %10.4f  %+8.4f",
                     r["tag"], r["xbd_3class_accuracy"], r["tebde_accuracy"], r["delta_3class"])
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
