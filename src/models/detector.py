"""
YOLO detector wrapper for building localization (Stage 1).

Provides a thin abstraction over the Ultralytics YOLOv8 API for:
  - Loading pretrained models (nano, small, medium)
  - Training on the prepared YOLO dataset
  - Running inference with configurable precision (FP32 / FP16)
  - Extracting bounding boxes with confidence filtering and NMS

Usage
-----
    from src.models.detector import YOLODetector

    det = YOLODetector("yolov8s")
    det.train(data_yaml="yolo_dataset/dataset.yaml", epochs=100, imgsz=640)
    boxes = det.predict(image_path, conf=0.25, imgsz=640)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from src.utils.config import (
    CONF_THRESHOLD,
    DETECTOR_BATCH_SIZES,
    DETECTOR_EPOCHS,
    DETECTOR_PATIENCE,
    MODEL_DIR,
    NMS_IOU_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Map short names to Ultralytics model identifiers
_MODEL_MAP = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
}


class YOLODetector:
    """
    Wrapper around Ultralytics YOLOv8 for building detection.

    Parameters
    ----------
    model_name : str
        One of ``"yolov8n"``, ``"yolov8s"``, ``"yolov8m"``.
    weights_path : Path or None
        Path to a trained ``.pt`` file.  If None, loads the COCO-pretrained
        checkpoint (for fine-tuning).
    """

    def __init__(
        self,
        model_name: str = "yolov8s",
        weights_path: Optional[Path] = None,
    ):
        self.model_name = model_name

        if weights_path is not None:
            self.model = YOLO(str(weights_path))
            logger.info("Loaded trained weights from %s", weights_path)
        else:
            pretrained = _MODEL_MAP.get(model_name)
            if pretrained is None:
                raise ValueError(
                    f"Unknown model name '{model_name}'. "
                    f"Choose from {list(_MODEL_MAP)}"
                )
            self.model = YOLO(pretrained)
            logger.info("Loaded pretrained %s", pretrained)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        data_yaml: str | Path,
        epochs: int = DETECTOR_EPOCHS,
        imgsz: int = 640,
        batch: Optional[int] = None,
        patience: int = DETECTOR_PATIENCE,
        project: str | Path = MODEL_DIR,
        name: Optional[str] = None,
        seed: int = 42,
        **extra_args,
    ) -> Path:
        """
        Fine-tune the detector on the YOLO-format dataset.

        Returns the path to the best checkpoint.
        """
        if batch is None:
            batch = DETECTOR_BATCH_SIZES.get(self.model_name, 16)
        if name is None:
            name = f"{self.model_name}_r{imgsz}"

        logger.info(
            "Training %s — imgsz=%d, batch=%d, epochs=%d, patience=%d",
            self.model_name, imgsz, batch, epochs, patience,
        )

        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            project=str(project),
            name=name,
            seed=seed,
            exist_ok=True,
            verbose=True,
            # Standard augmentation (aligned with detector_configs.yaml)
            mosaic=1.0,
            mixup=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            flipud=0.5,
            fliplr=0.5,
            **extra_args,
        )

        best_weights = Path(project) / name / "weights" / "best.pt"
        logger.info("Training complete — best weights: %s", best_weights)
        return best_weights

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(
        self,
        source,
        imgsz: int = 640,
        conf: float = CONF_THRESHOLD,
        iou: float = NMS_IOU_THRESHOLD,
        half: bool = False,
        verbose: bool = False,
    ) -> List[dict]:
        """
        Run detection on a single image or batch.

        Parameters
        ----------
        source : str, Path, or np.ndarray
            Image path or numpy array (H, W, C) in BGR or RGB.
        imgsz : int
            Inference input resolution.
        conf : float
            Confidence threshold.
        iou : float
            NMS IoU threshold.
        half : bool
            If True, run inference in FP16 mode.

        Returns
        -------
        list[dict]
            Each dict has keys: ``bbox`` (x1, y1, x2, y2 in pixel coords),
            ``confidence``, ``class_id``.
        """
        results = self.model.predict(
            source=source,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            half=half,
            verbose=verbose,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                detections.append({
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "confidence": float(boxes.conf[i].cpu()),
                    "class_id": int(boxes.cls[i].cpu()),
                })
        return detections

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def validate(
        self,
        data_yaml: str | Path,
        imgsz: int = 640,
        half: bool = False,
        **extra_args,
    ) -> dict:
        """
        Run validation and return mAP metrics.

        Returns
        -------
        dict
            Keys include ``mAP50``, ``mAP50-95``, ``precision``, ``recall``.
        """
        metrics = self.model.val(
            data=str(data_yaml),
            imgsz=imgsz,
            half=half,
            verbose=True,
            **extra_args,
        )
        return {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
