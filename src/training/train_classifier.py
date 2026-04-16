#!/usr/bin/env python
"""
Train damage-severity classifiers (Stage 2).

Trains EfficientNet-B0 and ResNet-34 on the pre-extracted earthquake
building patches with:
  - Class-weighted CrossEntropyLoss (handles severe imbalance)
  - Warm-up + cosine-decay learning rate schedule
  - Optional FP16 mixed-precision training
  - Checkpointing at best validation F1

Usage
-----
    python scripts/train_classifier.py                        # train both
    python scripts/train_classifier.py --model efficientnet_b0
    python scripts/train_classifier.py --model resnet34 --fp16
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.xbd_classifier_dataset import (
    PatchFolderDataset,
    get_train_transforms,
    get_val_transforms,
)
from src.models.classifier import DamageClassifier
from src.utils.config import (
    CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_INPUT_SIZE,
    CLASSIFIER_LR,
    CLASSIFIER_WARMUP_EPOCHS,
    CLASSIFIER_WEIGHT_DECAY,
    CLASSIFIER_PATCHES_DIR,
    MODEL_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    XBD_DAMAGE_CLASSES,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warm-up + cosine decay
# ---------------------------------------------------------------------------
def get_lr_lambda(warmup_epochs: int, total_epochs: int):
    """Return a lambda for torch.optim.lr_scheduler.LambdaLR."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear warm-up
        # Cosine decay from 1.0 to 0.0 over remaining epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, use_amp
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = running_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, f1, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train_classifier(
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    use_amp: bool,
    device: torch.device,
):
    """Train a single classifier configuration and return metrics."""
    tag = f"{model_name}_{'fp16' if use_amp else 'fp32'}"
    logger.info("=" * 60)
    logger.info("Training: %s", tag)
    logger.info("=" * 60)

    # Datasets
    train_dir = CLASSIFIER_PATCHES_DIR / "train"
    val_dir = CLASSIFIER_PATCHES_DIR / "val"

    train_ds = PatchFolderDataset(train_dir, transform=get_train_transforms())
    val_ds = PatchFolderDataset(val_dir, transform=get_val_transforms())

    logger.info("Train samples: %d  |  Val samples: %d", len(train_ds), len(val_ds))

    # Class weights
    class_weights = train_ds.get_class_weights().to(device)
    logger.info("Class weights: %s", class_weights.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    model = DamageClassifier(model_name).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=CLASSIFIER_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, get_lr_lambda(CLASSIFIER_WARMUP_EPOCHS, epochs)
    )
    scaler = GradScaler(enabled=use_amp)

    # Training loop
    best_f1 = 0.0
    best_epoch = 0
    save_path = MODEL_DIR / f"{tag}_best.pt"
    history = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device, use_amp
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "lr": round(current_lr, 6),
        })

        logger.info(
            "Epoch %3d/%d — lr=%.1e  train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f  val_f1=%.4f",
            epoch, epochs, current_lr,
            train_loss, train_acc, val_loss, val_acc, val_f1,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            model.save(save_path)
            logger.info("  ↑ New best F1=%.4f — saved to %s", best_f1, save_path)

    train_time = time.time() - t0

    # Final evaluation with best checkpoint
    logger.info("Loading best checkpoint (epoch %d) for final evaluation...", best_epoch)
    best_model = DamageClassifier.load(model_name, save_path).to(device)
    _, final_acc, final_f1, final_preds, final_labels = validate(
        best_model, val_loader, criterion, device, use_amp
    )

    # Classification report
    report = classification_report(
        final_labels, final_preds,
        target_names=XBD_DAMAGE_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(final_labels, final_preds).tolist()

    logger.info("\n%s", classification_report(
        final_labels, final_preds,
        target_names=XBD_DAMAGE_CLASSES,
        zero_division=0,
    ))

    return {
        "tag": tag,
        "model": model_name,
        "precision_mode": "fp16" if use_amp else "fp32",
        "epochs": epochs,
        "best_epoch": best_epoch,
        "train_time_s": round(train_time, 1),
        "best_weights": str(save_path),
        "val_accuracy": round(final_acc, 4),
        "val_macro_f1": round(final_f1, 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train damage classifiers")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["efficientnet_b0", "resnet34"],
        help="Train a single architecture (default: train both)",
    )
    parser.add_argument("--epochs", type=int, default=CLASSIFIER_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=CLASSIFIER_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=CLASSIFIER_LR)
    parser.add_argument(
        "--fp16", action="store_true",
        help="Enable FP16 mixed-precision training",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    seed_everything()
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Check patches exist
    train_dir = CLASSIFIER_PATCHES_DIR / "train"
    if not train_dir.exists():
        logger.error("Classifier patches not found: %s", train_dir)
        logger.error("Run 'python scripts/prepare_classifier_data.py' first.")
        sys.exit(1)

    # Determine which models to train
    model_names = [args.model] if args.model else ["efficientnet_b0", "resnet34"]
    all_results = []

    for model_name in model_names:
        result = train_classifier(
            model_name=model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_amp=args.fp16,
            device=device,
        )
        all_results.append(result)

    # Append to existing results (don't overwrite previous runs)
    results_path = RESULTS_DIR / "classifier_training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if results_path.exists():
        with open(results_path, "r") as f:
            existing = json.load(f)
    existing_tags = {r["tag"]: i for i, r in enumerate(existing)}
    for entry in all_results:
        if entry["tag"] in existing_tags:
            existing[existing_tags[entry["tag"]]] = entry
        else:
            existing.append(entry)
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Summary
    logger.info("")
    logger.info("%-30s  %8s  %8s  %10s", "Config", "Accuracy", "Macro F1", "Time (s)")
    logger.info("-" * 65)
    for r in all_results:
        logger.info(
            "%-30s  %8.4f  %8.4f  %10.1f",
            r["tag"], r["val_accuracy"], r["val_macro_f1"], r["train_time_s"],
        )

    logger.info("=== Classifier training complete ===")


if __name__ == "__main__":
    main()
