"""
Test-time augmentation: per-view inference plus fusion strategies.

The expensive step (tta_per_view_probs) runs the model once per augmentation
view and returns (N, S, C) softmax probabilities. Fusion is cheap numpy on top:
equal-weight averaging, plus uncertainty-weighted variants that down-weight or
up-weight views by their per-view confidence/entropy. Ported from the author's
uncertainty-tta-medmnist study.
"""

from __future__ import annotations

from functools import partial
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .augmentations import AugFn, normalize_imagenet


@torch.no_grad()
def tta_per_view_probs(model, loader: DataLoader, augmentations: List[Tuple[AugFn, str]],
                       device, temperature: float = 1.0
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run one forward pass per augmentation view over an UN-normalised [0,1] loader.

    Returns:
        per_view_probs: (N, S, C) softmax probabilities, S = len(augmentations)
        labels:         (N,)
    """
    model.eval()
    view_stacks = [[] for _ in augmentations]
    labels_all = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)  # [0,1]
        labels_all.append(np.asarray(labels).reshape(-1))
        for s, (fn, _name) in enumerate(augmentations):
            aug = normalize_imagenet(fn(imgs))
            logits = model(aug) / temperature
            view_stacks[s].append(F.softmax(logits, dim=1).cpu().numpy())
    per_view = np.stack([np.concatenate(v, axis=0) for v in view_stacks], axis=1)
    return per_view, np.concatenate(labels_all, axis=0)


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -np.sum(probs * np.log(probs + eps), axis=-1)


def fuse_equal_weight(per_view: np.ndarray) -> np.ndarray:
    """Plain mean over views. (N,S,C) -> (N,C)."""
    return per_view.mean(axis=1)


def fuse_maxprob(per_view: np.ndarray) -> np.ndarray:
    """Pick, per image, the single view with the highest max-confidence."""
    conf = per_view.max(axis=2)               # (N,S)
    best = conf.argmax(axis=1)                 # (N,)
    return per_view[np.arange(per_view.shape[0]), best]


def fuse_entropy(per_view: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Weight views by inverse entropy (confident views count more)."""
    ent = _entropy(per_view, eps)             # (N,S)
    w = 1.0 / (ent + eps)
    w = w / w.sum(axis=1, keepdims=True)
    return np.einsum("ns,nsc->nc", w, per_view)


def fuse_variance_inv(per_view: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Weight views by inverse predictive variance across classes."""
    var = per_view.var(axis=2)                # (N,S)
    w = 1.0 / (var + eps)
    w = w / w.sum(axis=1, keepdims=True)
    return np.einsum("ns,nsc->nc", w, per_view)


def fuse_top_k(per_view: np.ndarray, k: int, eps: float = 1e-12) -> np.ndarray:
    """Average the k most confident views per image."""
    conf = per_view.max(axis=2)               # (N,S)
    k = min(k, per_view.shape[1])
    topk = np.argsort(-conf, axis=1)[:, :k]   # (N,k)
    rows = np.arange(per_view.shape[0])[:, None]
    return per_view[rows, topk].mean(axis=1)


FUSION_FNS = {
    "equal": fuse_equal_weight,
    "maxprob": fuse_maxprob,
    "entropy": fuse_entropy,
    "variance_inv": fuse_variance_inv,
    "top3": partial(fuse_top_k, k=3),
}


def fuse(per_view: np.ndarray, strategy: str) -> np.ndarray:
    if strategy not in FUSION_FNS:
        raise ValueError(f"unknown fusion '{strategy}'; valid: {', '.join(FUSION_FNS)}")
    return FUSION_FNS[strategy](per_view)
