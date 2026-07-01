"""
MC-dropout uncertainty. Whereas TTA draws its uncertainty signal from
augmentation views, MC-dropout draws it from the model: the same un-augmented
image is passed T times with dropout active on the classifier head's input, and
the softmax vectors are averaged.

A forward PRE-hook injects functional dropout on the head's input, so this works
on all three backbones (ResNet has no native dropout, EfficientNet has 0.2 in
its head, DeiT has a head) with no retraining and no state_dict change. The model
stays in eval() so BatchNorm keeps its running statistics. Ported from the
author's uncertainty-tta-medmnist study.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _find_head(model: nn.Module) -> nn.Module:
    """Locate the classifier head, descending into a .backbone wrapper first."""
    target = getattr(model, "backbone", model)
    for attr in ("fc", "classifier", "head"):
        if hasattr(target, attr):
            return getattr(target, attr)
    raise AttributeError("no .fc/.classifier/.head found on model(.backbone)")


@torch.no_grad()
def mc_dropout_per_pass_probs(model, loader: DataLoader, device,
                              T: int = 20, p: float = 0.2
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """T stochastic passes with dropout on the head input. Returns (T,N,C), labels.
    loader must be the NORMALISED eval loader (no augmentation) and shuffle=False."""
    model.eval()
    head = _find_head(model)

    def _pre_hook(_m, inputs):
        return (F.dropout(inputs[0], p=p, training=True),) + inputs[1:]

    handle = head.register_forward_pre_hook(_pre_hook)
    labels_out = None
    try:
        per_pass = []
        for t in range(T):
            probs_b, lab_b = [], []
            for imgs, labels in loader:
                imgs = imgs.to(device, non_blocking=True)
                probs_b.append(F.softmax(model(imgs), dim=1).cpu().numpy())
                if t == 0:
                    lab_b.append(np.asarray(labels).reshape(-1))
            per_pass.append(np.concatenate(probs_b, axis=0))
            if t == 0:
                labels_out = np.concatenate(lab_b, axis=0)
    finally:
        handle.remove()
    return np.stack(per_pass, axis=0), labels_out


def mc_dropout_fuse(per_pass: np.ndarray) -> np.ndarray:
    """Mean over the T passes. (T,N,C) -> (N,C)."""
    return per_pass.mean(axis=0)
