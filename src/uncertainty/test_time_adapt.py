"""
Label-free, source-free test-time adaptation: BN-statistic adaptation and TENT.

BN-adapt recomputes batch-norm running statistics on the (unlabeled) target
stream, replacing the source statistics that no longer match UAV imagery.

TENT additionally minimises the entropy of the model's predictions by updating
only the normalisation-layer affine parameters, with no labels. It is BatchNorm-
oriented; on a LayerNorm backbone (DeiT) it adapts the LayerNorm affine
parameters instead, which we flag as a variant rather than canonical TENT.

Neither method uses target labels, so both are evaluated transductively on the
target stream and reported through the same harmonisation and calibration
metrics as everything else, to expose whether adaptation trades accuracy for
overconfidence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
_NORM = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)


def bn_layers(model):
    return [m for m in model.modules() if isinstance(m, _BN)]


def has_batchnorm(model) -> bool:
    return len(bn_layers(model)) > 0


def norm_affine_params(model):
    params = []
    for m in model.modules():
        if isinstance(m, _NORM):
            if getattr(m, "weight", None) is not None:
                params.append(m.weight)
            if getattr(m, "bias", None) is not None:
                params.append(m.bias)
    return params


@torch.no_grad()
def bn_adapt(model, loader, device):
    """Recompute BN running statistics on the unlabeled target stream."""
    model.eval()
    bns = bn_layers(model)
    if not bns:
        return model  # no BN (e.g. DeiT) -> nothing to adapt
    for m in bns:
        m.reset_running_stats()
        m.momentum = None     # cumulative moving average over the whole stream
        m.train()             # use and update batch statistics
    for img, _ in loader:
        model(img.to(device, non_blocking=True))
    for m in bns:
        m.eval()
    return model


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def tent_adapt(model, loader, device, steps: int = 1, lr: float = 1e-3):
    """Entropy minimisation over normalisation affine parameters (no labels)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in bn_layers(model):
        m.train()             # BN uses batch statistics during adaptation
    params = norm_affine_params(model)
    if not params:
        return model
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        for img, _ in loader:
            img = img.to(device, non_blocking=True)
            opt.zero_grad()
            loss = softmax_entropy(model(img)).mean()
            loss.backward()
            opt.step()
    model.eval()
    return model
