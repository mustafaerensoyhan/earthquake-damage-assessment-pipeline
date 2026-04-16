"""
PyTorch Dataset for the UAVs-TEBDE cross-domain evaluation.

The dataset provides pre-cropped building patches from UAV imagery of the
2023 Türkiye earthquakes, organised into three folders:

    Collapsed/   (474 images)
    Damaged/     (664 images)
    Intact/      (498 images)

This module loads these images and provides labels compatible with both
the native 3-class scheme and the harmonised xBD mapping.

Usage
-----
    from src.data.tebde_dataset import TEBDEDataset

    ds = TEBDEDataset(transform=get_val_transforms())
    img, label = ds[0]   # label is an int in {0, 1, 2}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.config import (
    CLASSIFIER_INPUT_SIZE,
    TEBDE_ROOT,
    TEBDE_CLASSES,
)

logger = logging.getLogger(__name__)

# Canonical class ordering: 0 = Intact, 1 = Damaged, 2 = Collapsed
TEBDE_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(TEBDE_CLASSES)}


class TEBDEDataset(Dataset):
    """
    Load UAVs-TEBDE building patches for cross-domain evaluation.

    Parameters
    ----------
    root : Path
        Path to the *Original Dataset* directory containing Collapsed/,
        Damaged/, and Intact/ sub-folders.
    transform : callable, optional
        Torchvision transform applied to each image.
    """

    def __init__(
        self,
        root: Path = TEBDE_ROOT,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.classes = TEBDE_CLASSES
        self.class_to_idx = TEBDE_CLASS_TO_IDX

        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                logger.warning("Missing TEBDE class directory: %s", cls_dir)
                continue
            # Accept both .jpg and .png
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in sorted(cls_dir.glob(ext)):
                    self.samples.append(
                        (img_path, self.class_to_idx[cls_name])
                    )

        logger.info(
            "TEBDEDataset: %d images from %s — %s",
            len(self.samples), self.root,
            {cls: sum(1 for _, l in self.samples if l == i)
             for i, cls in enumerate(self.classes)},
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
