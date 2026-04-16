"""
Convert xBD annotations to YOLO object-detection format.

For each post-disaster image the converter:
  1. Reads all building polygons from the matching JSON label file.
  2. Computes axis-aligned bounding boxes and normalises to [0, 1].
  3. Writes a ``.txt`` label file in YOLO format:
         ``class_id  x_centre  y_centre  width  height``
     where *class_id* is always 0 (single-class: building).
  4. Copies (or symlinks) the corresponding image into the YOLO tree.

The resulting directory layout follows the Ultralytics convention::

    yolo_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── dataset.yaml

Usage
-----
    python -m src.data.xbd_yolo_converter          (or via prepare_yolo_data.py)
"""

from __future__ import annotations

import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from src.data.xbd_parser import parse_label_file, BuildingAnnotation
from src.utils.config import (
    XBD_IMAGE_SIZE,
    XBD_TRAIN_IMAGES,
    XBD_TRAIN_LABELS,
    YOLO_DATASET_DIR,
    RANDOM_SEED,
    VALIDATION_SPLIT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounding-box conversion
# ---------------------------------------------------------------------------
def _bbox_to_yolo(
    bbox: Tuple[float, float, float, float],
    img_w: int = XBD_IMAGE_SIZE,
    img_h: int = XBD_IMAGE_SIZE,
) -> Tuple[float, float, float, float]:
    """
    Convert pixel bbox (minx, miny, maxx, maxy) to normalised YOLO format.

    Returns (x_centre, y_centre, width, height) each in [0, 1].
    Coordinates are clipped to valid image bounds.
    """
    minx, miny, maxx, maxy = bbox

    # Clip to image boundaries
    minx = max(0.0, min(minx, img_w))
    miny = max(0.0, min(miny, img_h))
    maxx = max(0.0, min(maxx, img_w))
    maxy = max(0.0, min(maxy, img_h))

    bw = maxx - minx
    bh = maxy - miny
    if bw <= 0 or bh <= 0:
        return None  # degenerate box

    x_centre = (minx + maxx) / 2.0 / img_w
    y_centre = (miny + maxy) / 2.0 / img_h
    w_norm = bw / img_w
    h_norm = bh / img_h

    return (x_centre, y_centre, w_norm, h_norm)


# ---------------------------------------------------------------------------
# Label-file writer
# ---------------------------------------------------------------------------
def buildings_to_yolo_lines(
    buildings: List[BuildingAnnotation],
    img_w: int = XBD_IMAGE_SIZE,
    img_h: int = XBD_IMAGE_SIZE,
) -> List[str]:
    """
    Convert a list of BuildingAnnotation to YOLO label lines.

    Each line: ``0 x_centre y_centre width height``  (class 0 = building).
    """
    lines = []
    for bld in buildings:
        yolo_box = _bbox_to_yolo(bld.bbox, img_w, img_h)
        if yolo_box is None:
            continue
        xc, yc, w, h = yolo_box
        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


# ---------------------------------------------------------------------------
# Train / val splitting
# ---------------------------------------------------------------------------
def _stratified_split(
    image_stems: List[str],
    val_ratio: float = VALIDATION_SPLIT,
    seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[str]]:
    """
    Split image stems into train and val sets, stratified by disaster type.

    Disaster type is inferred from the filename prefix (everything before
    the last underscore-separated numeric id).
    """
    rng = random.Random(seed)

    # Group by disaster name
    disaster_groups: Dict[str, List[str]] = defaultdict(list)
    for stem in image_stems:
        # stem example: "mexico-earthquake_00000042_post_disaster"
        # disaster name: everything before the numeric id segment
        parts = stem.replace("_post_disaster", "").rsplit("_", 1)
        disaster_name = parts[0] if len(parts) == 2 else stem
        disaster_groups[disaster_name].append(stem)

    train_stems, val_stems = [], []
    for disaster, stems in sorted(disaster_groups.items()):
        rng.shuffle(stems)
        n_val = max(1, int(len(stems) * val_ratio))
        val_stems.extend(stems[:n_val])
        train_stems.extend(stems[n_val:])

    logger.info(
        "Split %d images → %d train, %d val (%.0f%% val, stratified by disaster)",
        len(image_stems), len(train_stems), len(val_stems),
        100 * len(val_stems) / max(len(image_stems), 1),
    )
    return train_stems, val_stems


# ---------------------------------------------------------------------------
# Main conversion routine
# ---------------------------------------------------------------------------
def convert_xbd_to_yolo(
    images_dir: Path = XBD_TRAIN_IMAGES,
    labels_dir: Path = XBD_TRAIN_LABELS,
    output_dir: Path = YOLO_DATASET_DIR,
    val_ratio: float = VALIDATION_SPLIT,
    copy_images: bool = True,
) -> Path:
    """
    Build a YOLO-format dataset from xBD post-disaster annotations.

    Parameters
    ----------
    images_dir : Path
        Directory containing xBD ``*_post_disaster.png`` images.
    labels_dir : Path
        Directory containing xBD ``*_post_disaster.json`` label files.
    output_dir : Path
        Root of the output YOLO dataset tree.
    val_ratio : float
        Fraction of images to hold out for validation.
    copy_images : bool
        If True, copy images into the YOLO tree.  If False, create
        symlinks (saves disk space).

    Returns
    -------
    Path
        Path to the generated ``dataset.yaml``.
    """
    output_dir = Path(output_dir)

    # Prepare output directories
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Discover post-disaster label files
    label_files = sorted(Path(labels_dir).glob("*_post_disaster.json"))
    if not label_files:
        raise FileNotFoundError(
            f"No *_post_disaster.json files found in {labels_dir}"
        )

    # Collect image stems and parse labels
    stem_to_buildings: Dict[str, List[BuildingAnnotation]] = {}
    for lf in label_files:
        try:
            buildings = parse_label_file(lf)
        except Exception as exc:
            logger.warning("Skipping %s: %s", lf.name, exc)
            continue
        if not buildings:
            continue
        stem_to_buildings[lf.stem] = buildings  # stem includes _post_disaster

    logger.info("Parsed %d label files with buildings", len(stem_to_buildings))

    # Train / val split
    all_stems = list(stem_to_buildings.keys())
    train_stems, val_stems = _stratified_split(all_stems, val_ratio)
    split_map = {s: "train" for s in train_stems}
    split_map.update({s: "val" for s in val_stems})

    # Write labels and copy images
    total_boxes = 0
    for stem, split in split_map.items():
        buildings = stem_to_buildings[stem]
        lines = buildings_to_yolo_lines(buildings)
        if not lines:
            continue

        # Write YOLO label .txt
        label_out = output_dir / "labels" / split / (stem + ".txt")
        label_out.write_text("\n".join(lines) + "\n")
        total_boxes += len(lines)

        # Copy or link the image
        img_name = stem + ".png"
        src_img = Path(images_dir) / img_name
        dst_img = output_dir / "images" / split / img_name

        if src_img.exists() and not dst_img.exists():
            if copy_images:
                shutil.copy2(src_img, dst_img)
            else:
                dst_img.symlink_to(src_img.resolve())

    # Write dataset.yaml for Ultralytics
    yaml_path = output_dir / "dataset.yaml"
    dataset_cfg = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["building"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)

    logger.info(
        "YOLO dataset written to %s — %d images, %d bounding boxes",
        output_dir, len(split_map), total_boxes,
    )
    return yaml_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    yaml_path = convert_xbd_to_yolo()
    print(f"Dataset YAML: {yaml_path}")
