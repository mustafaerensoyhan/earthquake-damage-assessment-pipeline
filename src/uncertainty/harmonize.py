"""
xBD (4-class) to UAV (3-class) harmonisation, applied at the probability level.

The Stage-2 classifier outputs four xBD classes:
    0 no-damage, 1 minor-damage, 2 major-damage, 3 destroyed
The UAV targets (UAVs-TEBDE, EarthquakeNet) use three classes:
    0 Intact, 1 Damaged, 2 Collapsed

Harmonisation maps no-damage -> Intact, {minor, major} -> Damaged,
destroyed -> Collapsed. We harmonise the SOFTMAX PROBABILITIES (not just the
argmax), so the Damaged probability is the sum of the minor and major
probabilities. This keeps a valid 3-class distribution for calibration metrics.
"""

from __future__ import annotations

import numpy as np

# xBD 4-class index -> UAV 3-class index
XBD_TO_TEBDE = {0: 0, 1: 1, 2: 1, 3: 2}
TEBDE_CLASSES = ["Intact", "Damaged", "Collapsed"]
XBD_DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def harmonize_probs_4to3(probs4: np.ndarray) -> np.ndarray:
    """
    Map (N, 4) xBD softmax probabilities to (N, 3) UAV probabilities by summing
    the minor- and major-damage columns into the single Damaged column.
    """
    probs4 = np.asarray(probs4)
    assert probs4.shape[1] == 4, f"expected 4-class probs, got {probs4.shape}"
    p_intact = probs4[:, 0]
    p_damaged = probs4[:, 1] + probs4[:, 2]
    p_collapsed = probs4[:, 3]
    return np.stack([p_intact, p_damaged, p_collapsed], axis=1)


def harmonize_labels_4to3(labels4: np.ndarray) -> np.ndarray:
    """Map (N,) 4-class labels to 3-class labels via XBD_TO_TEBDE."""
    labels4 = np.asarray(labels4).ravel()
    return np.vectorize(XBD_TO_TEBDE.get)(labels4).astype(np.int64)


def stratified_split(labels: np.ndarray, frac: float, seed: int = 42):
    """
    Seeded, class-stratified split into a small calibration set and a test set.

    Returns (cal_idx, test_idx) as int arrays. `frac` is the fraction routed to
    the calibration set (used to fit temperature and thresholds). Stratifying
    keeps the rare Collapsed class represented in the small calibration split.
    """
    labels = np.asarray(labels).ravel()
    rng = np.random.default_rng(seed)
    cal_idx, test_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * frac)))
        cal_idx.extend(idx[:k].tolist())
        test_idx.extend(idx[k:].tolist())
    return np.array(sorted(cal_idx)), np.array(sorted(test_idx))
