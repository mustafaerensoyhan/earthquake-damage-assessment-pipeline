"""
xBD JSON label parser.

Reads the xBD annotation files and extracts per-building information:
  - polygon geometry  (Shapely Polygon)
  - axis-aligned bounding box  (minx, miny, maxx, maxy in pixel coords)
  - damage label  (from post-disaster files only)
  - metadata  (disaster name, disaster type, image id)

Usage
-----
    from src.data.xbd_parser import parse_label_file, iter_xbd_labels

    # Single file
    buildings = parse_label_file("path/to/post_disaster.json")

    # Iterate over a split
    for record in iter_xbd_labels(labels_dir, post_only=True):
        ...
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from shapely import wkt
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BuildingAnnotation:
    """A single building extracted from an xBD label file."""

    uid: str
    polygon: Polygon
    bbox: tuple  # (minx, miny, maxx, maxy) in pixel coordinates
    damage_label: Optional[str]  # None for pre-disaster files
    subtype: Optional[str]       # building subtype, e.g. "residential"
    image_id: str                # e.g. "mexico-earthquake_00000042"
    disaster: str                # e.g. "mexico-earthquake"
    disaster_type: str           # e.g. "earthquake"
    is_post: bool                # True if from a post-disaster file


# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------
def parse_label_file(json_path: Path) -> List[BuildingAnnotation]:
    """
    Parse one xBD JSON label file and return a list of BuildingAnnotation.

    Parameters
    ----------
    json_path : Path
        Path to a ``*_pre_disaster.json`` or ``*_post_disaster.json`` file.

    Returns
    -------
    list[BuildingAnnotation]
        One entry per building polygon found in the file.
    """
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    disaster = metadata.get("disaster", "unknown")
    disaster_type = metadata.get("disaster_type", "unknown")

    # Determine pre/post from the filename convention
    stem = json_path.stem  # e.g. "mexico-earthquake_00000042_post_disaster"
    is_post = "_post_disaster" in stem
    # image_id strips the pre/post suffix
    image_id = stem.replace("_pre_disaster", "").replace("_post_disaster", "")

    features_xy = data.get("features", {}).get("xy", [])
    buildings: List[BuildingAnnotation] = []

    for feat in features_xy:
        properties = feat.get("properties", {})

        # Only process building features
        if properties.get("feature_type") != "building":
            continue

        wkt_str = feat.get("wkt")
        if not wkt_str:
            logger.debug("Skipping feature without WKT in %s", json_path.name)
            continue

        try:
            polygon = wkt.loads(wkt_str)
        except Exception as exc:
            logger.warning(
                "Malformed WKT in %s: %s — skipping", json_path.name, exc
            )
            continue

        if not isinstance(polygon, Polygon) or polygon.is_empty:
            continue

        minx, miny, maxx, maxy = polygon.bounds

        # Damage label is only present in post-disaster files
        damage_label = properties.get("subtype") if is_post else None

        buildings.append(
            BuildingAnnotation(
                uid=properties.get("uid", ""),
                polygon=polygon,
                bbox=(minx, miny, maxx, maxy),
                damage_label=damage_label,
                subtype=properties.get("subtype"),
                image_id=image_id,
                disaster=disaster,
                disaster_type=disaster_type,
                is_post=is_post,
            )
        )

    return buildings


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------
def iter_xbd_labels(
    labels_dir: Path,
    *,
    post_only: bool = False,
    disaster_prefixes: Optional[tuple] = None,
) -> Iterator[BuildingAnnotation]:
    """
    Yield BuildingAnnotation objects from all JSON files in *labels_dir*.

    Parameters
    ----------
    labels_dir : Path
        Directory containing xBD ``*.json`` label files.
    post_only : bool
        If True, skip pre-disaster files (which lack damage labels).
    disaster_prefixes : tuple[str, ...] or None
        If given, only yield buildings whose *disaster* field starts with
        one of the provided prefixes (e.g. earthquake-only filtering).

    Yields
    ------
    BuildingAnnotation
    """
    labels_dir = Path(labels_dir)
    json_files = sorted(labels_dir.glob("*.json"))
    skipped, parsed = 0, 0

    for jf in json_files:
        if post_only and "_post_disaster" not in jf.name:
            continue

        try:
            buildings = parse_label_file(jf)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s — skipping", jf.name, exc)
            skipped += 1
            continue

        parsed += 1
        for bld in buildings:
            if disaster_prefixes is not None:
                if not bld.disaster.startswith(disaster_prefixes):
                    continue
            yield bld

    logger.info(
        "Parsed %d label files (%d skipped) from %s", parsed, skipped, labels_dir
    )


# ---------------------------------------------------------------------------
# Convenience queries
# ---------------------------------------------------------------------------
def get_disaster_names(labels_dir: Path) -> set:
    """Return the set of unique disaster names found in *labels_dir*."""
    names = set()
    for jf in Path(labels_dir).glob("*.json"):
        with open(jf) as f:
            data = json.load(f)
        disaster = data.get("metadata", {}).get("disaster")
        if disaster:
            names.add(disaster)
    return names


def count_buildings_by_damage(labels_dir: Path, disaster_prefixes=None):
    """Return a dict mapping damage_label → count for post-disaster files."""
    from collections import Counter

    counts = Counter()
    for bld in iter_xbd_labels(
        labels_dir, post_only=True, disaster_prefixes=disaster_prefixes
    ):
        label = bld.damage_label or "unlabelled"
        counts[label] += 1
    return dict(counts)
