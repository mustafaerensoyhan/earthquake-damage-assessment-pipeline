"""
PyTorch Dataset for RescueNet building-damage patches, used as a second
cross-domain UAV target. Mirrors TEBDEDataset so the existing dump and report
scripts work unchanged.

Expects the class-folder layout produced by scripts/extract_rescuenet_patches.py:

    <root>/Intact/      *.png
    <root>/Damaged/     *.png
    <root>/Collapsed/   *.png

Labels use the same canonical ordering as TEBDE: 0 = Intact, 1 = Damaged,
2 = Collapsed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.config import TEBDE_CLASSES

logger = logging.getLogger(__name__)

RESCUENET_CLASSES = TEBDE_CLASSES  # ["Intact", "Damaged", "Collapsed"]
RESCUENET_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(RESCUENET_CLASSES)}


class RescueNetDataset(Dataset):
    """Load extracted RescueNet building patches for cross-domain evaluation."""

    def __init__(self, root: Path, transform: Optional[transforms.Compose] = None):
        self.root = Path(root)
        self.transform = transform
        self.classes = RESCUENET_CLASSES
        self.class_to_idx = RESCUENET_CLASS_TO_IDX

        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                logger.warning("Missing RescueNet class directory: %s", cls_dir)
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in sorted(cls_dir.glob(ext)):
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        logger.info("Loaded %d RescueNet patches from %s", len(self.samples), self.root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
