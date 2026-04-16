#!/usr/bin/env python
"""
Collect results from all trained classifier models.

Re-evaluates each trained classifier on the validation set and writes
a consolidated JSON. Useful when separate training runs overwrote
each other's results.

Usage
-----
    python scripts/collect_classifier_results.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.xbd_classifier_dataset import PatchFolderDataset, get_val_transforms
from src.models.classifier import DamageClassifier
from src.utils.config import (
    CLASSIFIER_PATCHES_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    XBD_DAMAGE_CLASSES,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load validation dataset
    val_dir = CLASSIFIER_PATCHES_DIR / "val"
    if not val_dir.exists():
        logger.error("Validation patches not found: %s", val_dir)
        sys.exit(1)

    val_ds = PatchFolderDataset(val_dir, transform=get_val_transforms())
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
    )
    logger.info("Validation set: %d samples", len(val_ds))

    # Find trained classifier weights
    model_dir = Path(MODEL_DIR)
    weight_files = sorted(model_dir.glob("*_fp32_best.pt"))

    if not weight_files:
        logger.error("No trained classifier weights found in %s", model_dir)
        sys.exit(1)

    logger.info("Found %d trained classifier(s)", len(weight_files))

    all_results = []

    for wf in weight_files:
        tag = wf.stem.replace("_best", "")  # e.g. "efficientnet_b0_fp32"
        model_name = tag.replace("_fp32", "").replace("_fp16", "")

        logger.info("Evaluating %s ...", tag)

        model = DamageClassifier.load(model_name, wf).to(device)
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
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
            "tag": tag,
            "model": model_name,
            "precision_mode": "fp32",
            "best_weights": str(wf),
            "val_accuracy": round(float(acc), 4),
            "val_macro_f1": round(float(f1), 4),
            "classification_report": report,
            "confusion_matrix": cm,
        }
        all_results.append(entry)

        logger.info(
            "  %s — Accuracy=%.4f  Macro F1=%.4f", tag, acc, f1,
        )
        logger.info("\n%s", classification_report(
            all_labels, all_preds,
            target_names=XBD_DAMAGE_CLASSES,
            zero_division=0,
        ))

    # Save
    results_path = RESULTS_DIR / "classifier_training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("Consolidated %d result(s) → %s", len(all_results), results_path)
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
