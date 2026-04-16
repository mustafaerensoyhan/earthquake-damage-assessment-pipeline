#!/usr/bin/env python
"""
Cross-domain evaluation: apply xBD-trained classifiers to UAVs-TEBDE.

Measures transfer performance from satellite imagery (xBD) to UAV imagery
(TEBDE) without any fine-tuning. Reports accuracy, confusion matrices,
and per-class breakdown using the 3-class harmonised labels.

Usage
-----
    python scripts/cross_domain_eval.py
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

from src.data.tebde_dataset import TEBDEDataset
from src.data.xbd_classifier_dataset import get_val_transforms
from src.models.classifier import DamageClassifier
from src.utils.config import (
    HARMONISATION_MAP,
    MODEL_DIR,
    RESULTS_DIR,
    TEBDE_CLASSES,
    TEBDE_ROOT,
    XBD_DAMAGE_CLASSES,
    ensure_dirs,
)

logger = logging.getLogger(__name__)

# xBD 4-class index → TEBDE 3-class index
# no-damage(0) → Intact(0), minor-damage(1) → Damaged(1),
# major-damage(2) → Damaged(1), destroyed(3) → Collapsed(2)
XBD_TO_TEBDE = {0: 0, 1: 1, 2: 1, 3: 2}


def find_trained_classifiers():
    """Find all trained classifier weights."""
    classifiers = []
    model_dir = Path(MODEL_DIR)
    for pt_file in sorted(model_dir.glob("*_fp32_best.pt")):
        tag = pt_file.stem.replace("_best", "")
        model_name = tag.replace("_fp32", "")
        classifiers.append({
            "tag": tag,
            "model_name": model_name,
            "weights": pt_file,
        })
    return classifiers


def evaluate_on_tebde(
    model_name: str,
    weights_path: Path,
    dataset: TEBDEDataset,
    device: torch.device,
    batch_size: int = 32,
):
    """
    Evaluate a trained xBD classifier on TEBDE.

    The classifier outputs 4-class xBD predictions which are harmonised
    to 3-class TEBDE labels for comparison.
    """
    model = DamageClassifier.load(model_name, weights_path).to(device)
    model.eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    all_preds_xbd = []  # Raw 4-class predictions
    all_preds_3class = []  # Harmonised 3-class predictions
    all_labels = []  # TEBDE ground truth (3-class)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds_4class = logits.argmax(dim=1).cpu().numpy()

            # Harmonise 4-class → 3-class
            preds_3class = np.array([XBD_TO_TEBDE[p] for p in preds_4class])

            all_preds_xbd.extend(preds_4class)
            all_preds_3class.extend(preds_3class)
            all_labels.extend(labels.numpy())

    all_preds_3class = np.array(all_preds_3class)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds_3class)
    f1 = f1_score(all_labels, all_preds_3class, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds_3class)

    report = classification_report(
        all_labels, all_preds_3class,
        target_names=TEBDE_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    # Also compute per-class accuracy
    per_class_acc = {}
    for i, cls_name in enumerate(TEBDE_CLASSES):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[cls_name] = float((all_preds_3class[mask] == i).mean())
        else:
            per_class_acc[cls_name] = 0.0

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "per_class_accuracy": per_class_acc,
        "raw_4class_predictions": np.bincount(
            np.array(all_preds_xbd), minlength=4
        ).tolist(),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Check TEBDE dataset exists
    if not TEBDE_ROOT.exists():
        logger.error("TEBDE dataset not found: %s", TEBDE_ROOT)
        logger.error("Update TEBDE_ROOT in src/utils/config.py")
        sys.exit(1)

    # Load TEBDE dataset
    transform = get_val_transforms()
    dataset = TEBDEDataset(root=TEBDE_ROOT, transform=transform)
    logger.info("TEBDE dataset: %d images", len(dataset))

    # Find trained classifiers
    classifiers = find_trained_classifiers()
    if not classifiers:
        logger.error("No trained classifiers found in %s", MODEL_DIR)
        sys.exit(1)

    logger.info("Found %d classifier(s) to evaluate", len(classifiers))

    # Load xBD validation accuracy for comparison
    clf_results_path = RESULTS_DIR / "classifier_training_results.json"
    xbd_accuracies = {}
    if clf_results_path.exists():
        with open(clf_results_path) as f:
            clf_results = json.load(f)
        for r in clf_results:
            xbd_accuracies[r["tag"]] = r.get("val_accuracy", 0)

    # Evaluate each classifier
    all_results = []
    for clf_info in classifiers:
        tag = clf_info["tag"]
        logger.info("=" * 50)
        logger.info("Evaluating %s on TEBDE", tag)
        logger.info("=" * 50)

        metrics = evaluate_on_tebde(
            clf_info["model_name"], clf_info["weights"],
            dataset, device,
        )

        xbd_acc = xbd_accuracies.get(tag, 0)
        delta = xbd_acc - metrics["accuracy"]

        result = {
            "tag": tag,
            "model": clf_info["model_name"],
            "tebde_accuracy": metrics["accuracy"],
            "tebde_macro_f1": metrics["macro_f1"],
            "xbd_accuracy": xbd_acc,
            "delta": round(delta, 4),
            "confusion_matrix": metrics["confusion_matrix"],
            "classification_report": metrics["classification_report"],
            "per_class_accuracy": metrics["per_class_accuracy"],
            "raw_4class_predictions": metrics["raw_4class_predictions"],
        }
        all_results.append(result)

        # Print results
        logger.info("\nTEBDE Accuracy: %.4f", metrics["accuracy"])
        logger.info("TEBDE Macro F1: %.4f", metrics["macro_f1"])
        logger.info("xBD Val Accuracy: %.4f", xbd_acc)
        logger.info("Delta (xBD - TEBDE): %+.4f", delta)
        logger.info("\nPer-class accuracy:")
        for cls, acc in metrics["per_class_accuracy"].items():
            logger.info("  %-12s  %.4f", cls, acc)

        logger.info("\n%s", classification_report(
            # Reconstruct from confusion matrix for display
            *_labels_from_cm(metrics["confusion_matrix"]),
            target_names=TEBDE_CLASSES,
            zero_division=0,
        ) if False else "")  # Use stored report instead

        # Print classification report
        report = metrics["classification_report"]
        logger.info("\n%-12s  %8s  %8s  %8s  %8s", "Class", "Prec", "Recall", "F1", "Support")
        logger.info("-" * 50)
        for cls in TEBDE_CLASSES:
            r = report.get(cls, {})
            logger.info(
                "%-12s  %8.2f  %8.2f  %8.2f  %8.0f",
                cls, r.get("precision", 0), r.get("recall", 0),
                r.get("f1-score", 0), r.get("support", 0),
            )

        logger.info("\nConfusion Matrix:")
        cm = np.array(metrics["confusion_matrix"])
        header = "            " + "  ".join(f"{c:>10}" for c in TEBDE_CLASSES)
        logger.info(header)
        for i, cls in enumerate(TEBDE_CLASSES):
            row = "  ".join(f"{cm[i][j]:>10}" for j in range(len(TEBDE_CLASSES)))
            logger.info("%-10s  %s", cls, row)

    # Save results
    results_path = RESULTS_DIR / "cross_domain_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nResults saved to %s", results_path)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CROSS-DOMAIN SUMMARY")
    logger.info("=" * 60)
    logger.info("%-25s  %10s  %10s  %8s", "Config", "xBD Acc", "TEBDE Acc", "Delta")
    logger.info("-" * 60)
    for r in all_results:
        logger.info(
            "%-25s  %10.4f  %10.4f  %+8.4f",
            r["tag"], r["xbd_accuracy"], r["tebde_accuracy"], r["delta"],
        )

    logger.info("=== Cross-domain evaluation complete ===")


if __name__ == "__main__":
    main()
