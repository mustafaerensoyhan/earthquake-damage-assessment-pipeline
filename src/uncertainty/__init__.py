"""
Trustworthiness extension for the two-stage damage assessment pipeline.

This package adds calibration, uncertainty-weighted test-time augmentation,
and selective prediction on top of the existing Stage-2 classifier, without
retraining. The metrics, temperature-scaling, and TTA fusion code is adapted
from the author's prior calibration study (uncertainty-tta-medmnist); the
4-to-3 class harmonisation is specific to the xBD -> UAV cross-domain setting.
"""
