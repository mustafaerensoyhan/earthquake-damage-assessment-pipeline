"""
Temperature scaling fit directly from dumped logits.

A single scalar T is fit by minimising NLL of softmax(logits / T) against the
labels (LBFGS on log T, so T stays positive). Dividing every logit by the same
T preserves the argmax, so accuracy is unchanged; only calibration moves.

Two fitting modes matter for the cross-domain study:
  - source-fit: fit T on the 4-class xBD validation logits (standard).
  - target-fit: fit T on a small 3-class UAV calibration split, where the
    objective is the HARMONISED 3-class NLL (softmax then 4->3 merge).
Comparing the two shows whether source-domain calibration transfers.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from .harmonize import harmonize_probs_4to3


def _softmax_t(logits: torch.Tensor, log_T: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits / log_T.exp(), dim=1)


def fit_temperature_logits(logits: np.ndarray, labels: np.ndarray,
                           harmonize: bool = False,
                           max_iter: int = 100, lr: float = 0.05) -> float:
    """
    Fit T on dumped logits. If harmonize=True, the 4-class probabilities are
    merged to 3 classes before the NLL is computed (labels must then be 3-class).

    Returns the scalar T (>0). Falls back to 1.0 on a degenerate optimum.
    """
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    labels_t = torch.as_tensor(np.asarray(labels).ravel(), dtype=torch.long)
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        probs = _softmax_t(logits_t, log_T)
        if harmonize:
            p = probs
            probs = torch.stack([p[:, 0], p[:, 1] + p[:, 2], p[:, 3]], dim=1)
        nll = -torch.log(probs[torch.arange(len(labels_t)), labels_t] + 1e-12).mean()
        nll.backward()
        return nll

    opt.step(closure)
    T = float(log_T.exp().item())
    return T if (np.isfinite(T) and T > 0) else 1.0


def apply_temperature(logits: np.ndarray, T: float,
                      harmonize: bool = False) -> np.ndarray:
    """
    Return probabilities from logits at temperature T. If harmonize=True the
    4-class probabilities are merged to a (N, 3) distribution.
    """
    logits = np.asarray(logits, dtype=np.float64) / float(T)
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    return harmonize_probs_4to3(probs) if harmonize else probs
