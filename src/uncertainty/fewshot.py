"""
Helpers for few-shot domain adaptation from the source 4-class classifier to the
3-class UAV target.

The head is replaced with a 3-class head warm-started from the source head via
the same harmonisation used elsewhere (Intact = no-damage, Damaged = mean of
minor+major, Collapsed = destroyed), so the 0-shot starting point already
matches the harmonised source model and fine-tuning only has to refine it.

Few-shot training samples are drawn BALANCED (k per class) from a supplied pool
of indices (the held-out calibration pool), never from the test split.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def replace_head_3class(model: nn.Module, warm: bool = True) -> nn.Module:
    """Swap the classifier head for a 3-output Linear, warm-started from the
    4-class head when possible. Works for ResNet (.fc), EfficientNet
    (.classifier[-1]) and DeiT (.head)."""
    bb = getattr(model, "backbone", model)

    if hasattr(bb, "fc") and isinstance(bb.fc, nn.Linear):
        old = bb.fc; set_head = lambda m: setattr(bb, "fc", m)
    elif hasattr(bb, "classifier"):
        seq = bb.classifier
        if isinstance(seq, nn.Linear):
            old = seq; set_head = lambda m: setattr(bb, "classifier", m)
        else:
            old = seq[-1]; last = len(seq) - 1
            set_head = lambda m: seq.__setitem__(last, m)
    elif hasattr(bb, "head") and isinstance(bb.head, nn.Linear):
        old = bb.head; set_head = lambda m: setattr(bb, "head", m)
    else:
        raise AttributeError("no .fc/.classifier/.head Linear found to replace")

    new = nn.Linear(old.in_features, 3)
    if warm and old.out_features == 4:
        with torch.no_grad():
            new.weight.copy_(torch.stack(
                [old.weight[0], 0.5 * (old.weight[1] + old.weight[2]), old.weight[3]]))
            new.bias.copy_(torch.stack(
                [old.bias[0], 0.5 * (old.bias[1] + old.bias[2]), old.bias[3]]))
    set_head(new)
    return model


def find_head(model: nn.Module) -> nn.Linear:
    """Return the current classifier head Linear (after replace_head_3class)."""
    bb = getattr(model, "backbone", model)
    if hasattr(bb, "fc") and isinstance(bb.fc, nn.Linear):
        return bb.fc
    if hasattr(bb, "classifier"):
        seq = bb.classifier
        return seq if isinstance(seq, nn.Linear) else seq[-1]
    if hasattr(bb, "head") and isinstance(bb.head, nn.Linear):
        return bb.head
    raise AttributeError("no head Linear found")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze everything except the head (for LP-FT phase control)."""
    head = find_head(model)
    head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad_(trainable)
    for p in head.parameters():
        p.requires_grad_(True)


def sample_fewshot(labels: np.ndarray, pool_idx: np.ndarray,
                   k_per_class: int, seed: int) -> np.ndarray:
    """Draw k indices per class from pool_idx (balanced few-shot training set)."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(labels):
        idx = pool_idx[labels[pool_idx] == c]
        rng.shuffle(idx)
        chosen.extend(idx[:k_per_class].tolist())
    return np.array(sorted(chosen), dtype=np.int64)
