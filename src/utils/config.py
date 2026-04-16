"""
Configuration management for the two-stage damage assessment pipeline.

Centralises all paths, hyperparameters, and experiment settings so that
every other module imports from here rather than hard-coding values.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import yaml


# ---------------------------------------------------------------------------
# Path configuration — edit ROOT to match your machine
# ---------------------------------------------------------------------------
ROOT = Path(r"C:\Users\mustafaerensoyhan\Downloads\SoftwareResearchProject")

# Dataset roots
XBD_ROOT = ROOT / "xbd"
TEBDE_ROOT = (
    ROOT / "uavs-tebde" / "UAVs-TEBDE (Original Dataset)"
)

# Derived xBD paths
XBD_TRAIN_IMAGES = XBD_ROOT / "train" / "images"
XBD_TRAIN_LABELS = XBD_ROOT / "train" / "labels"
XBD_TEST_IMAGES = XBD_ROOT / "test" / "images"
XBD_TEST_LABELS = XBD_ROOT / "test" / "labels"

# Output paths
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Intermediate artefacts produced by data preparation
YOLO_DATASET_DIR = ROOT / "yolo_dataset"       # YOLO-format images + labels
CLASSIFIER_PATCHES_DIR = ROOT / "classifier_patches"  # Cropped building patches

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
# Earthquake-related disaster prefixes used to filter xBD for the classifier.
EARTHQUAKE_PREFIXES = ("mexico-earthquake", "palu-tsunami")

# xBD damage labels in canonical order (used as class indices 0-3).
XBD_DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]

# Label harmonisation for cross-domain evaluation (xBD 4-class → TEBDE 3-class).
HARMONISATION_MAP = {
    "no-damage": "Intact",
    "minor-damage": "Damaged",
    "major-damage": "Damaged",
    "destroyed": "Collapsed",
}
TEBDE_CLASSES = ["Intact", "Damaged", "Collapsed"]

# Image dimensions
XBD_IMAGE_SIZE = 1024  # xBD images are 1024×1024

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2  # 80/20 train/val, stratified by disaster type

# Detector training
DETECTOR_EPOCHS = 100
DETECTOR_PATIENCE = 15  # Early-stopping patience (validation mAP)
DETECTOR_BATCH_SIZES = {"yolov8n": 16, "yolov8s": 8, "yolov8m": 4}

# Classifier training
CLASSIFIER_INPUT_SIZE = 224
CLASSIFIER_EPOCHS = 50
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR = 1e-3
CLASSIFIER_WARMUP_EPOCHS = 5
CLASSIFIER_WEIGHT_DECAY = 1e-4

# Inference thresholds
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.5

# Latency benchmarking
WARMUP_PASSES = 10
TIMING_REPEATS = 50


# ---------------------------------------------------------------------------
# Experiment configurations (full sweep)
# ---------------------------------------------------------------------------
@dataclass
class DetectorConfig:
    model_name: str       # e.g. "yolov8n"
    resolution: int       # 640 or 800
    precision: str        # "fp32" or "fp16"

    @property
    def tag(self) -> str:
        return f"{self.model_name}_r{self.resolution}_{self.precision}"


@dataclass
class ClassifierConfig:
    model_name: str       # "efficientnet_b0" or "resnet34"
    precision: str        # "fp32" or "fp16"

    @property
    def tag(self) -> str:
        return f"{self.model_name}_{self.precision}"


def build_detector_configs() -> List[DetectorConfig]:
    """Return all 12 detector configurations for the sweep."""
    configs = []
    for model in ("yolov8n", "yolov8s", "yolov8m"):
        for res in (640, 800):
            for prec in ("fp32", "fp16"):
                configs.append(DetectorConfig(model, res, prec))
    return configs


def build_classifier_configs() -> List[ClassifierConfig]:
    """Return all 4 classifier configurations for the sweep."""
    configs = []
    for model in ("efficientnet_b0", "resnet34"):
        for prec in ("fp32", "fp16"):
            configs.append(ClassifierConfig(model, prec))
    return configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create all output directories if they do not already exist."""
    for d in (
        OUTPUT_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR,
        YOLO_DATASET_DIR, CLASSIFIER_PATCHES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def save_config_snapshot(path: Path) -> None:
    """Dump key settings to a YAML file for reproducibility."""
    snapshot = {
        "random_seed": RANDOM_SEED,
        "validation_split": VALIDATION_SPLIT,
        "detector_epochs": DETECTOR_EPOCHS,
        "classifier_epochs": CLASSIFIER_EPOCHS,
        "classifier_lr": CLASSIFIER_LR,
        "conf_threshold": CONF_THRESHOLD,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "earthquake_prefixes": list(EARTHQUAKE_PREFIXES),
    }
    with open(path, "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False)
