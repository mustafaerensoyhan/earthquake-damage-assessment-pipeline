"""
PyTorch Dataset for the damage-severity classifier (Stage 2).

Provides two usage modes:

1. **On-the-fly cropping** — ``XBDClassifierDataset`` reads pre-built patch
   manifests (CSV) and crops patches from the original xBD images at
   ``__getitem__`` time.  Good for prototyping.

2. **Pre-extracted patches** — ``extract_patches_to_disk()`` writes every
   building patch as a separate file organised by class.  The resulting
   tree can be loaded with ``torchvision.datasets.ImageFolder`` or with
   ``PatchFolderDataset`` defined here.  Preferred for training speed.

Both modes filter for earthquake-only events using the disaster prefixes
defined in config.
"""

from __future__ import annotations

import csv
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.xbd_parser import iter_xbd_labels, BuildingAnnotation
from src.utils.config import (
    CLASSIFIER_INPUT_SIZE,
    CLASSIFIER_PATCHES_DIR,
    EARTHQUAKE_PREFIXES,
    RANDOM_SEED,
    VALIDATION_SPLIT,
    XBD_DAMAGE_CLASSES,
    XBD_TRAIN_IMAGES,
    XBD_TRAIN_LABELS,
)

logger = logging.getLogger(__name__)

# Label string → integer index
DAMAGE_TO_IDX: Dict[str, int] = {
    cls: idx for idx, cls in enumerate(XBD_DAMAGE_CLASSES)
}


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------
def build_classifier_manifest(
    labels_dir: Path = XBD_TRAIN_LABELS,
    images_dir: Path = XBD_TRAIN_IMAGES,
    disaster_prefixes: tuple = EARTHQUAKE_PREFIXES,
) -> List[Dict]:
    """
    Scan xBD labels and build a list of per-building records.

    Each record is a dict with keys:
        image_path, bbox (minx, miny, maxx, maxy), damage_label,
        damage_idx, disaster, uid.

    Only post-disaster files matching *disaster_prefixes* are included.
    """
    records = []
    seen_uids = set()

    for bld in iter_xbd_labels(
        labels_dir, post_only=True, disaster_prefixes=disaster_prefixes
    ):
        if bld.damage_label not in DAMAGE_TO_IDX:
            continue

        # Deduplicate by UID (safety check)
        if bld.uid in seen_uids:
            continue
        seen_uids.add(bld.uid)

        # Resolve the image path
        img_name = f"{bld.image_id}_post_disaster.png"
        img_path = Path(images_dir) / img_name
        if not img_path.exists():
            logger.debug("Image not found: %s", img_path)
            continue

        records.append({
            "image_path": str(img_path),
            "bbox": bld.bbox,
            "damage_label": bld.damage_label,
            "damage_idx": DAMAGE_TO_IDX[bld.damage_label],
            "disaster": bld.disaster,
            "uid": bld.uid,
        })

    # Log class distribution
    dist = Counter(r["damage_label"] for r in records)
    logger.info("Classifier manifest: %d patches — %s", len(records), dict(dist))
    return records


def save_manifest_csv(records: List[Dict], path: Path) -> None:
    """Write the manifest to CSV for inspection and reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "minx", "miny", "maxx", "maxy",
                   "damage_label", "damage_idx", "disaster", "uid"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {
                "image_path": r["image_path"],
                "minx": r["bbox"][0],
                "miny": r["bbox"][1],
                "maxx": r["bbox"][2],
                "maxy": r["bbox"][3],
                "damage_label": r["damage_label"],
                "damage_idx": r["damage_idx"],
                "disaster": r["disaster"],
                "uid": r["uid"],
            }
            writer.writerow(row)
    logger.info("Manifest saved to %s", path)


# ---------------------------------------------------------------------------
# Pre-extraction to disk
# ---------------------------------------------------------------------------
def extract_patches_to_disk(
    records: List[Dict],
    output_dir: Path = CLASSIFIER_PATCHES_DIR,
    patch_size: int = CLASSIFIER_INPUT_SIZE,
    val_ratio: float = VALIDATION_SPLIT,
    seed: int = RANDOM_SEED,
) -> Tuple[Path, Path]:
    """
    Crop building patches from xBD images and save to disk.

    Output structure::

        output_dir/
        ├── train/
        │   ├── no-damage/
        │   ├── minor-damage/
        │   ├── major-damage/
        │   └── destroyed/
        └── val/
            ├── no-damage/ ...

    Returns (train_dir, val_dir).
    """
    output_dir = Path(output_dir)
    rng = random.Random(seed)

    # Create class directories
    for split in ("train", "val"):
        for cls in XBD_DAMAGE_CLASSES:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    # Stratified split: group by disaster, then split within each group
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[r["disaster"]].append(r)

    train_records, val_records = [], []
    for disaster, recs in sorted(groups.items()):
        rng.shuffle(recs)
        n_val = max(1, int(len(recs) * val_ratio))
        val_records.extend(recs[:n_val])
        train_records.extend(recs[n_val:])

    # Cache open images to avoid re-reading the same 1024² image per building
    img_cache: Dict[str, Image.Image] = {}

    def _crop_and_save(rec: Dict, split: str) -> bool:
        img_path = rec["image_path"]
        if img_path not in img_cache:
            try:
                img_cache[img_path] = Image.open(img_path).convert("RGB")
            except Exception as exc:
                logger.warning("Cannot open %s: %s", img_path, exc)
                return False

        img = img_cache[img_path]
        minx, miny, maxx, maxy = rec["bbox"]

        # Clip to image bounds
        w, h = img.size
        minx = max(0, int(minx))
        miny = max(0, int(miny))
        maxx = min(w, int(maxx))
        maxy = min(h, int(maxy))

        if maxx - minx < 2 or maxy - miny < 2:
            return False

        patch = img.crop((minx, miny, maxx, maxy))
        patch = patch.resize((patch_size, patch_size), Image.BILINEAR)

        out_path = (
            output_dir / split / rec["damage_label"] / f"{rec['uid']}.png"
        )
        patch.save(out_path)
        return True

    saved = 0
    for rec in train_records:
        if _crop_and_save(rec, "train"):
            saved += 1
        # Evict cache periodically to manage memory
        if len(img_cache) > 50:
            img_cache.clear()

    for rec in val_records:
        if _crop_and_save(rec, "val"):
            saved += 1
        if len(img_cache) > 50:
            img_cache.clear()

    logger.info(
        "Extracted %d patches (%d train, %d val) to %s",
        saved, len(train_records), len(val_records), output_dir,
    )
    return output_dir / "train", output_dir / "val"


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------
def get_train_transforms(input_size: int = CLASSIFIER_INPUT_SIZE):
    """Augmentation transforms for classifier training."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transforms(input_size: int = CLASSIFIER_INPUT_SIZE):
    """Deterministic transforms for validation / inference."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class PatchFolderDataset(Dataset):
    """
    Load pre-extracted patches from an ImageFolder-style directory.

    Expects:  root/<class_name>/<uid>.png

    This is a thin wrapper that enforces the canonical class ordering
    defined in ``XBD_DAMAGE_CLASSES`` (index 0 = no-damage, etc.).
    """

    def __init__(
        self,
        root: Path,
        transform=None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.classes = XBD_DAMAGE_CLASSES
        self.class_to_idx = DAMAGE_TO_IDX

        # Collect all image paths
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                logger.warning("Missing class directory: %s", cls_dir)
                continue
            for img_path in sorted(cls_dir.glob("*.png")):
                self.samples.append((img_path, self.class_to_idx[cls_name]))

        logger.info(
            "PatchFolderDataset: %d samples from %s",
            len(self.samples), self.root,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for CrossEntropyLoss."""
        counts = Counter(label for _, label in self.samples)
        total = sum(counts.values())
        weights = []
        for idx in range(len(self.classes)):
            c = counts.get(idx, 1)
            weights.append(total / (len(self.classes) * c))
        return torch.tensor(weights, dtype=torch.float32)
