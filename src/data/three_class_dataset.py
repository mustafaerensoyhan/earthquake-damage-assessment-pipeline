"""
3-class classifier dataset for damage severity.

Merges the xBD 4-class labels into 3 classes that align with TEBDE:
    - no-damage      → Intact   (0)
    - minor-damage   → Damaged  (1)
    - major-damage   → Damaged  (1)
    - destroyed      → Collapsed (2)

This increases training samples for the intermediate class and directly
matches the cross-domain evaluation scheme.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.xbd_parser import iter_xbd_labels
from src.utils.config import (
    CLASSIFIER_INPUT_SIZE,
    EARTHQUAKE_PREFIXES,
    RANDOM_SEED,
    VALIDATION_SPLIT,
    XBD_TRAIN_IMAGES,
    XBD_TRAIN_LABELS,
)

logger = logging.getLogger(__name__)

# 3-class scheme
THREE_CLASS_NAMES = ["Intact", "Damaged", "Collapsed"]
THREE_CLASS_MAP = {
    "no-damage": 0,     # Intact
    "minor-damage": 1,  # Damaged
    "major-damage": 1,  # Damaged
    "destroyed": 2,     # Collapsed
}


def build_3class_manifest(
    labels_dir: Path = XBD_TRAIN_LABELS,
    images_dir: Path = XBD_TRAIN_IMAGES,
    disaster_prefixes: tuple = EARTHQUAKE_PREFIXES,
) -> List[Dict]:
    """Build manifest with 3-class labels."""
    records = []
    seen_uids = set()

    for bld in iter_xbd_labels(
        labels_dir, post_only=True, disaster_prefixes=disaster_prefixes
    ):
        if bld.damage_label not in THREE_CLASS_MAP:
            continue
        if bld.uid in seen_uids:
            continue
        seen_uids.add(bld.uid)

        img_name = f"{bld.image_id}_post_disaster.png"
        img_path = Path(images_dir) / img_name
        if not img_path.exists():
            continue

        records.append({
            "image_path": str(img_path),
            "bbox": bld.bbox,
            "original_label": bld.damage_label,
            "class_idx": THREE_CLASS_MAP[bld.damage_label],
            "class_name": THREE_CLASS_NAMES[THREE_CLASS_MAP[bld.damage_label]],
            "disaster": bld.disaster,
            "uid": bld.uid,
        })

    dist = Counter(r["class_name"] for r in records)
    logger.info("3-class manifest: %d patches — %s", len(records), dict(dist))
    return records


def extract_3class_patches(
    records: List[Dict],
    output_dir: Path,
    patch_size: int = CLASSIFIER_INPUT_SIZE,
    val_ratio: float = VALIDATION_SPLIT,
    seed: int = RANDOM_SEED,
) -> Tuple[Path, Path]:
    """Extract patches organised by 3-class labels."""
    output_dir = Path(output_dir)
    rng = random.Random(seed)

    for split in ("train", "val"):
        for cls in THREE_CLASS_NAMES:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    # Stratified split by disaster
    groups = defaultdict(list)
    for r in records:
        groups[r["disaster"]].append(r)

    train_records, val_records = [], []
    for disaster, recs in sorted(groups.items()):
        rng.shuffle(recs)
        n_val = max(1, int(len(recs) * val_ratio))
        val_records.extend(recs[:n_val])
        train_records.extend(recs[n_val:])

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
        w, h = img.size
        minx, miny = max(0, int(minx)), max(0, int(miny))
        maxx, maxy = min(w, int(maxx)), min(h, int(maxy))

        if maxx - minx < 2 or maxy - miny < 2:
            return False

        patch = img.crop((minx, miny, maxx, maxy))
        patch = patch.resize((patch_size, patch_size), Image.BILINEAR)
        out_path = output_dir / split / rec["class_name"] / f"{rec['uid']}.png"
        patch.save(out_path)
        return True

    saved = 0
    for rec in train_records:
        if _crop_and_save(rec, "train"):
            saved += 1
        if len(img_cache) > 50:
            img_cache.clear()

    for rec in val_records:
        if _crop_and_save(rec, "val"):
            saved += 1
        if len(img_cache) > 50:
            img_cache.clear()

    logger.info(
        "Extracted %d 3-class patches (%d train, %d val) to %s",
        saved, len(train_records), len(val_records), output_dir,
    )
    return output_dir / "train", output_dir / "val"


class ThreeClassDataset(Dataset):
    """Load 3-class patches from ImageFolder-style directory."""

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = THREE_CLASS_NAMES
        self.class_to_idx = {cls: i for i, cls in enumerate(THREE_CLASS_NAMES)}

        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                logger.warning("Missing class directory: %s", cls_dir)
                continue
            for img_path in sorted(cls_dir.glob("*.png")):
                self.samples.append((img_path, self.class_to_idx[cls_name]))

        logger.info("ThreeClassDataset: %d samples from %s", len(self.samples), self.root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_weights(self) -> torch.Tensor:
        counts = Counter(label for _, label in self.samples)
        total = sum(counts.values())
        weights = []
        for idx in range(len(self.classes)):
            c = counts.get(idx, 1)
            weights.append(total / (len(self.classes) * c))
        return torch.tensor(weights, dtype=torch.float32)
