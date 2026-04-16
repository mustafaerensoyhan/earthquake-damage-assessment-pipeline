"""
CNN classifier for building damage severity (Stage 2).

Supports EfficientNet-B0 and ResNet-34, both initialized with ImageNet
weights and fine-tuned on xBD earthquake building patches.

Usage
-----
    from src.models.classifier import DamageClassifier

    clf = DamageClassifier("efficientnet_b0")
    logits = clf(batch_of_images)         # (B, 4) tensor
    preds  = clf.predict(single_image)    # int in {0, 1, 2, 3}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

from src.utils.config import CLASSIFIER_INPUT_SIZE, XBD_DAMAGE_CLASSES

logger = logging.getLogger(__name__)

NUM_CLASSES = len(XBD_DAMAGE_CLASSES)  # 4


def _build_efficientnet_b0(num_classes: int = NUM_CLASSES) -> nn.Module:
    """EfficientNet-B0 with replaced classification head."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_resnet34(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-34 with replaced fully-connected head."""
    weights = models.ResNet34_Weights.IMAGENET1K_V1
    model = models.resnet34(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


_BUILDERS = {
    "efficientnet_b0": _build_efficientnet_b0,
    "resnet34": _build_resnet34,
}


class DamageClassifier(nn.Module):
    """
    Wrapper around EfficientNet-B0 or ResNet-34 for 4-class damage
    classification.

    Parameters
    ----------
    model_name : str
        ``"efficientnet_b0"`` or ``"resnet34"``.
    num_classes : int
        Number of output classes (default 4).
    weights_path : Path or None
        If provided, load trained checkpoint instead of ImageNet init.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = NUM_CLASSES,
        weights_path: Optional[Path] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.classes = XBD_DAMAGE_CLASSES

        builder = _BUILDERS.get(model_name)
        if builder is None:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from {list(_BUILDERS)}"
            )
        self.backbone = builder(num_classes)

        if weights_path is not None:
            state = torch.load(str(weights_path), map_location="cpu")
            self.backbone.load_state_dict(state)
            logger.info("Loaded classifier weights from %s", weights_path)
        else:
            logger.info("Initialized %s with ImageNet weights", model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (B, num_classes)."""
        return self.backbone(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices for a batch."""
        self.eval()
        logits = self.forward(x)
        return logits.argmax(dim=1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (B, num_classes)."""
        self.eval()
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def save(self, path: Path) -> None:
        """Save model weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.backbone.state_dict(), path)
        logger.info("Classifier saved to %s", path)

    @classmethod
    def load(cls, model_name: str, weights_path: Path, **kwargs):
        """Convenience constructor: build model and load weights."""
        return cls(model_name=model_name, weights_path=weights_path, **kwargs)
