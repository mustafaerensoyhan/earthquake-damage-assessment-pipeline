#!/usr/bin/env python
"""
Train 3-class damage classifiers (Intact / Damaged / Collapsed).

Usage:
    python scripts/train_3class_classifier.py
    python scripts/train_3class_classifier.py --model efficientnet_b0
    python scripts/train_3class_classifier.py --model resnet34
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
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.three_class_dataset import ThreeClassDataset, THREE_CLASS_NAMES
from src.data.xbd_classifier_dataset import get_train_transforms, get_val_transforms
from src.utils.config import (
    CLASSIFIER_BATCH_SIZE, CLASSIFIER_EPOCHS, CLASSIFIER_LR,
    CLASSIFIER_WARMUP_EPOCHS, CLASSIFIER_WEIGHT_DECAY,
    MODEL_DIR, RANDOM_SEED, RESULTS_DIR, ROOT, ensure_dirs,
)

logger = logging.getLogger(__name__)
NUM_CLASSES = 3


def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_name, num_classes=NUM_CLASSES):
    """Build a 3-class classifier."""
    import torchvision.models as models
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def get_lr_lambda(warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = running_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1, all_preds, all_labels


def train_classifier(model_name, epochs, batch_size, lr, use_amp, device):
    tag = f"{model_name}_3class_{'fp16' if use_amp else 'fp32'}"
    logger.info("=" * 60)
    logger.info("Training 3-class: %s", tag)
    logger.info("=" * 60)

    patches_dir = ROOT / "classifier_patches_3class"
    train_ds = ThreeClassDataset(patches_dir / "train", transform=get_train_transforms())
    val_ds = ThreeClassDataset(patches_dir / "val", transform=get_val_transforms())
    logger.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    class_weights = train_ds.get_class_weights().to(device)
    logger.info("Class weights: %s", class_weights.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(model_name, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CLASSIFIER_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda(CLASSIFIER_WARMUP_EPOCHS, epochs))
    scaler = GradScaler(enabled=use_amp)

    best_f1, best_epoch = 0.0, 0
    save_path = MODEL_DIR / f"{tag}_best.pt"
    history = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4), "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4), "val_f1": round(val_f1, 4),
            "lr": round(current_lr, 6),
        })

        logger.info(
            "Epoch %3d/%d — lr=%.1e  train_loss=%.4f  val_acc=%.4f  val_f1=%.4f",
            epoch, epochs, current_lr, train_loss, val_acc, val_f1,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info("  New best F1=%.4f — saved", best_f1)

    train_time = time.time() - t0

    # Final eval with best checkpoint
    model.load_state_dict(torch.load(str(save_path), map_location=device))
    _, final_acc, final_f1, final_preds, final_labels = validate(model, val_loader, criterion, device, use_amp)

    report = classification_report(final_labels, final_preds, target_names=THREE_CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(final_labels, final_preds).tolist()

    logger.info("\n%s", classification_report(final_labels, final_preds, target_names=THREE_CLASS_NAMES, zero_division=0))

    return {
        "tag": tag, "model": model_name, "num_classes": 3,
        "precision_mode": "fp16" if use_amp else "fp32",
        "epochs": epochs, "best_epoch": best_epoch,
        "train_time_s": round(train_time, 1),
        "best_weights": str(save_path),
        "val_accuracy": round(final_acc, 4),
        "val_macro_f1": round(final_f1, 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train 3-class classifiers")
    parser.add_argument("--model", type=str, default=None, choices=["efficientnet_b0", "resnet34"])
    parser.add_argument("--epochs", type=int, default=CLASSIFIER_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=CLASSIFIER_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=CLASSIFIER_LR)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    seed_everything()
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    patches_dir = ROOT / "classifier_patches_3class" / "train"
    if not patches_dir.exists():
        logger.error("3-class patches not found. Run: python scripts/prepare_3class_data.py")
        sys.exit(1)

    model_names = [args.model] if args.model else ["efficientnet_b0", "resnet34"]
    all_results = []

    for model_name in model_names:
        result = train_classifier(model_name, args.epochs, args.batch_size, args.lr, args.fp16, device)
        all_results.append(result)

    # Save results
    results_path = RESULTS_DIR / "classifier_3class_results.json"
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

    logger.info("")
    logger.info("%-35s  %8s  %8s", "Config", "Accuracy", "Macro F1")
    logger.info("-" * 55)
    for r in all_results:
        logger.info("%-35s  %8.4f  %8.4f", r["tag"], r["val_accuracy"], r["val_macro_f1"])
    logger.info("=== 3-class classifier training complete ===")


if __name__ == "__main__":
    main()
