"""
Core extraction logic for turning RescueNet pixel-level masks into per-building
classification patches mapped to the 3-class UAV scheme (Intact/Damaged/Collapsed).

RescueNet annotates building damage at the pixel level with separate classes for
no-damage / minor / major / total-destruction. We take connected components of
each building-damage class, crop the bounding box (with padding) from the RGB
image, and label the patch by that class mapped to 3 classes:

    no-damage              -> Intact    (0)
    minor + major damage   -> Damaged   (1)
    total destruction      -> Collapsed (2)

The mapping from raw mask pixel VALUES to these classes is dataset-version
dependent, so it is passed in explicitly (see RESCUENET_BUILDING_CLASSES default
and the --inspect mode of the CLI to verify the values in your download).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage

# Default raw-mask pixel value -> 3-class target (VERIFY against your download
# with the CLI --inspect mode; RescueNet label ids vary between releases).
# Common RescueNet scheme: 2=building-no-damage, 3=minor, 4=major, 5=total-destruction.
RESCUENET_BUILDING_CLASSES: Dict[int, int] = {2: 0, 3: 1, 4: 1, 5: 2}
TARGET_NAMES = ["Intact", "Damaged", "Collapsed"]


def extract_from_pair(image: np.ndarray, mask: np.ndarray,
                      class_map: Dict[int, int] = None,
                      min_area: int = 500, pad_frac: float = 0.08
                      ) -> List[Tuple[np.ndarray, int]]:
    """
    Extract labelled building patches from one (image, mask) pair.

    image: (H, W, 3) uint8 RGB. mask: (H, W) integer label map.
    Returns a list of (patch_rgb, label3) tuples. label3 in {0,1,2}.
    """
    if class_map is None:
        class_map = RESCUENET_BUILDING_CLASSES
    H, W = mask.shape[:2]
    patches: List[Tuple[np.ndarray, int]] = []

    for raw_val, label3 in class_map.items():
        binary = (mask == raw_val)
        if not binary.any():
            continue
        labelled, n = ndimage.label(binary)
        if n == 0:
            continue
        slices = ndimage.find_objects(labelled)
        for sl in slices:
            if sl is None:
                continue
            ys, xs = sl
            area = (ys.stop - ys.start) * (xs.stop - xs.start)
            if area < min_area:
                continue
            ph = ys.stop - ys.start
            pw = xs.stop - xs.start
            py = int(round(ph * pad_frac))
            px = int(round(pw * pad_frac))
            y0 = max(0, ys.start - py); y1 = min(H, ys.stop + py)
            x0 = max(0, xs.start - px); x1 = min(W, xs.stop + px)
            patch = image[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            patches.append((patch, int(label3)))
    return patches


def inspect_mask_values(mask: np.ndarray) -> Dict[int, int]:
    """Return {pixel_value: count} for a mask, to help verify the class map."""
    vals, counts = np.unique(mask, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}
