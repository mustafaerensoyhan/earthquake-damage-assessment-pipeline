"""
Test-time augmentation pool for oblique UAV building patches, with a safety
table. Augmentation functions operate on a [0, 1] CHW tensor and return a
[0, 1] tensor; ImageNet normalisation is applied AFTER augmentation by the TTA
loop (so photometric ops act on raw pixel values, not normalised ones).

Domain reasoning (this is the part that differs from a generic / medical pool):
oblique UAV damage cues are geometric (wall cracks, floor pancaking, leaning)
and textural (rubble, debris). So geometric flips/rotations and mild
photometric jitter are safe, but operations that distort structure (elastic) or
heavily shift colour/hue (which corrupts rubble and crack cues) are unsafe and
are excluded from the default pool. The safety table is reported in the paper.
"""

from __future__ import annotations

import random
from typing import Callable, List, Tuple

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

AugFn = Callable[[torch.Tensor], torch.Tensor]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _clamp01(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(0.0, 1.0)


def normalize_imagenet(t: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalise a [0,1] tensor or batch (channel dim at -3)."""
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(-1, 1, 1)
    return (t - mean) / std


def get_tta_transforms(size: int = 224):
    """Loader transform for the TTA path: resize + ToTensor, NO normalisation.
    Augmentations and normalisation are applied later, per view, on the GPU."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # -> [0,1] CHW
    ])


# --- augmentation functions (act on [0,1] CHW or B,C,H,W) -------------------
def aug_identity(img):  return img
def aug_hflip(img):     return TF.hflip(img)
def aug_vflip(img):     return TF.vflip(img)


def aug_rotate(img):
    return TF.rotate(img, random.uniform(-10.0, 10.0))


def aug_brightness(img):
    return _clamp01(TF.adjust_brightness(img, random.uniform(0.85, 1.15)))


def aug_contrast(img):
    return _clamp01(TF.adjust_contrast(img, random.uniform(0.9, 1.1)))


def aug_gaussian_noise(img):
    return _clamp01(img + torch.randn_like(img) * 0.01)


def aug_center_crop(img):
    h, w = img.shape[-2], img.shape[-1]
    ch, cw = int(round(h * 0.9)), int(round(w * 0.9))
    return TF.resize(TF.center_crop(img, [ch, cw]), [h, w], antialias=True)


def aug_color_jitter(img):
    img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
    img = TF.adjust_contrast(img, random.uniform(0.85, 1.15))
    img = TF.adjust_saturation(img, random.uniform(0.85, 1.15))
    img = TF.adjust_hue(img, random.uniform(-0.08, 0.08))
    return _clamp01(img)


def aug_elastic(img):
    try:
        from torchvision.transforms import ElasticTransform
        return _clamp01(ElasticTransform(alpha=20.0, sigma=4.0)(img))
    except Exception:
        return _clamp01(TF.affine(img, angle=0, translate=[0, 0], scale=1.0,
                                  shear=[random.uniform(-5.0, 5.0), 0.0]))


# Safety table: category and rationale per augmentation. SAFE ops form the
# default pool; CAUTION/RISKY are available but excluded by default.
AUGMENTATION_SAFETY = {
    "identity":       ("safe",    "original view; calibration anchor"),
    "hflip":          ("safe",    "buildings are roughly bilaterally symmetric"),
    "vflip":          ("safe",    "oblique viewpoint makes vertical flip plausible"),
    "rotate":         ("safe",    "small +/-10 deg covers UAV roll/heading jitter"),
    "brightness":     ("safe",    "+/-15% covers exposure and time-of-day variation"),
    "contrast":       ("safe",    "+/-10% covers haze and sensor differences"),
    "gaussian_noise": ("safe",    "sigma 0.01 mimics mild sensor noise"),
    "center_crop":    ("safe",    "0.9 crop covers small scale/altitude changes"),
    "color_jitter":   ("risky",   "hue/saturation shifts corrupt rubble and debris cues"),
    "elastic":        ("risky",   "warps structural geometry, the damage signal itself"),
}

_FN = {
    "identity": aug_identity, "hflip": aug_hflip, "vflip": aug_vflip,
    "rotate": aug_rotate, "brightness": aug_brightness, "contrast": aug_contrast,
    "gaussian_noise": aug_gaussian_noise, "center_crop": aug_center_crop,
    "color_jitter": aug_color_jitter, "elastic": aug_elastic,
}

# Ordered SAFE pool (identity is added separately as view 0).
SAFE_POOL_ORDER = ["hflip", "vflip", "rotate", "brightness",
                   "contrast", "gaussian_noise", "center_crop"]


def get_tta_pipeline(n_views: int = 8, pool: str = "safe"
                     ) -> List[Tuple[AugFn, str]]:
    """
    Return [(fn, name), ...] of length n_views, with identity as view 0.
    pool='safe' draws only from augmentations marked safe in the table.
    """
    if pool == "safe":
        names = ["identity"] + SAFE_POOL_ORDER
    else:
        names = ["identity"] + [n for n in _FN if n != "identity"]
    names = names[:n_views]
    return [(_FN[n], n) for n in names]
