#!/usr/bin/env python
"""
Latency benchmarking for the two-stage pipeline.

Measures stage-level timing (preprocess, detect, classify) with:
  - GPU warm-up (10 passes)
  - Repeated measurement (50 images × multiple passes)
  - Median and P95 reporting
  - FP32 and FP16 comparison

Usage
-----
    python scripts/latency_benchmark.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import YOLODetector
from src.models.classifier import DamageClassifier
from src.data.xbd_classifier_dataset import get_val_transforms
from src.utils.config import (
    CLASSIFIER_INPUT_SIZE,
    CONF_THRESHOLD,
    MODEL_DIR,
    NMS_IOU_THRESHOLD,
    RESULTS_DIR,
    WARMUP_PASSES,
    TIMING_REPEATS,
    YOLO_DATASET_DIR,
    XBD_DAMAGE_CLASSES,
    ensure_dirs,
)
from src.utils.timing import timer, latency_stats

logger = logging.getLogger(__name__)


def find_trained_detectors():
    """Find all trained YOLO detector weights."""
    detectors = []
    model_dir = Path(MODEL_DIR)
    for run_dir in sorted(model_dir.glob("yolov8*_r*")):
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            tag = run_dir.name
            parts = tag.split("_r")
            model_name = parts[0]
            resolution = int(parts[1])
            detectors.append({
                "tag": tag,
                "model_name": model_name,
                "resolution": resolution,
                "weights": best_pt,
            })
    return detectors


def find_trained_classifiers():
    """Find all trained classifier weights."""
    classifiers = []
    model_dir = Path(MODEL_DIR)
    for pt_file in sorted(model_dir.glob("*_fp32_best.pt")):
        tag = pt_file.stem.replace("_best", "")
        model_name = tag.replace("_fp32", "")
        classifiers.append({
            "tag": tag,
            "model_name": model_name,
            "weights": pt_file,
        })
    return classifiers


def get_test_images(n: int = 50):
    """Get a sample of validation images for benchmarking."""
    val_dir = YOLO_DATASET_DIR / "images" / "val"
    images = sorted(val_dir.glob("*.png"))[:n]
    if not images:
        raise FileNotFoundError(f"No validation images found in {val_dir}")
    return images


def benchmark_detector(
    model_name: str,
    weights_path: Path,
    resolution: int,
    test_images: list,
    half: bool = False,
    warmup: int = WARMUP_PASSES,
    repeats: int = TIMING_REPEATS,
):
    """Benchmark a single detector configuration."""
    detector = YOLODetector(model_name, weights_path=weights_path)

    # Warm-up
    logger.info("  Warm-up (%d passes)...", warmup)
    for _ in range(warmup):
        detector.predict(str(test_images[0]), imgsz=resolution, half=half, verbose=False)

    # Benchmark
    preprocess_times = []
    detect_times = []
    total_times = []
    num_detections = []

    sample_images = test_images[:repeats] if len(test_images) >= repeats else test_images
    logger.info("  Benchmarking %d images...", len(sample_images))

    for img_path in sample_images:
        t0 = time.perf_counter()
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        t1 = time.perf_counter()

        detections = detector.predict(
            img_np, imgsz=resolution, half=half,
            conf=CONF_THRESHOLD, iou=NMS_IOU_THRESHOLD, verbose=False,
        )
        t2 = time.perf_counter()

        preprocess_times.append((t1 - t0) * 1000)
        detect_times.append((t2 - t1) * 1000)
        total_times.append((t2 - t0) * 1000)
        num_detections.append(len(detections))

    return {
        "preprocess": latency_stats(preprocess_times),
        "detection": latency_stats(detect_times),
        "total": latency_stats(total_times),
        "avg_detections": float(np.mean(num_detections)),
    }


def benchmark_classifier(
    model_name: str,
    weights_path: Path,
    test_images: list,
    half: bool = False,
    warmup: int = WARMUP_PASSES,
    repeats: int = TIMING_REPEATS,
):
    """Benchmark a single classifier configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DamageClassifier.load(model_name, weights_path).to(device)
    model.eval()

    if half:
        model = model.half()

    transform = get_val_transforms()

    # Create a dummy batch for warm-up
    dummy = torch.randn(1, 3, CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE).to(device)
    if half:
        dummy = dummy.half()

    # Warm-up
    logger.info("  Warm-up (%d passes)...", warmup)
    for _ in range(warmup):
        with torch.no_grad():
            model(dummy)
    torch.cuda.synchronize()

    # Benchmark single-patch inference
    classify_times = []
    sample_images = test_images[:repeats] if len(test_images) >= repeats else test_images

    logger.info("  Benchmarking %d patches...", len(sample_images))

    for img_path in sample_images:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        if half:
            tensor = tensor.half()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(tensor)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        classify_times.append((t1 - t0) * 1000)

    return {
        "per_patch": latency_stats(classify_times),
    }


def benchmark_pipeline(
    detector_info: dict,
    classifier_info: dict,
    test_images: list,
    det_half: bool = False,
    clf_half: bool = False,
    repeats: int = TIMING_REPEATS,
):
    """Benchmark the full two-stage pipeline end-to-end."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = YOLODetector(detector_info["model_name"], weights_path=detector_info["weights"])
    classifier = DamageClassifier.load(
        classifier_info["model_name"], classifier_info["weights"]
    ).to(device)
    classifier.eval()
    if clf_half:
        classifier = classifier.half()

    transform = get_val_transforms()
    resolution = detector_info["resolution"]

    # Warm-up
    for _ in range(5):
        detector.predict(str(test_images[0]), imgsz=resolution, half=det_half, verbose=False)
    dummy = torch.randn(1, 3, CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE).to(device)
    if clf_half:
        dummy = dummy.half()
    for _ in range(5):
        with torch.no_grad():
            classifier(dummy)

    # Benchmark
    preprocess_times = []
    detect_times = []
    classify_times = []
    total_times = []

    sample_images = test_images[:repeats] if len(test_images) >= repeats else test_images
    logger.info("  Pipeline benchmark: %d images...", len(sample_images))

    for img_path in sample_images:
        # Preprocess
        t0 = time.perf_counter()
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        t1 = time.perf_counter()

        # Detect
        detections = detector.predict(
            img_np, imgsz=resolution, half=det_half,
            conf=CONF_THRESHOLD, iou=NMS_IOU_THRESHOLD, verbose=False,
        )
        t2 = time.perf_counter()

        # Classify all patches
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            patch = img.crop((x1, y1, x2, y2))
            tensor = transform(patch).unsqueeze(0).to(device)
            if clf_half:
                tensor = tensor.half()
            torch.cuda.synchronize()
            with torch.no_grad():
                _ = classifier(tensor)
            torch.cuda.synchronize()

        t3 = time.perf_counter()

        preprocess_times.append((t1 - t0) * 1000)
        detect_times.append((t2 - t1) * 1000)
        classify_times.append((t3 - t2) * 1000)
        total_times.append((t3 - t0) * 1000)

    return {
        "preprocess": latency_stats(preprocess_times),
        "detection": latency_stats(detect_times),
        "classification": latency_stats(classify_times),
        "total": latency_stats(total_times),
        "throughput_fps": 1000.0 / np.median(total_times) if total_times else 0,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    test_images = get_test_images(n=50)
    logger.info("Using %d test images for benchmarking", len(test_images))

    detectors = find_trained_detectors()
    classifiers = find_trained_classifiers()
    logger.info("Found %d detectors, %d classifiers", len(detectors), len(classifiers))

    all_results = []

    # Benchmark each detector in FP32 and FP16
    for det_info in detectors:
        for half in [False, True]:
            prec = "fp16" if half else "fp32"
            tag = f"{det_info['tag']}_{prec}"
            logger.info("Benchmarking detector: %s", tag)

            det_metrics = benchmark_detector(
                det_info["model_name"], det_info["weights"],
                det_info["resolution"], test_images, half=half,
            )

            all_results.append({
                "tag": tag,
                "component": "detector",
                "model": det_info["model_name"],
                "resolution": det_info["resolution"],
                "precision_mode": prec,
                "median_ms": det_metrics["total"]["median_ms"],
                "p95_ms": det_metrics["total"]["p95_ms"],
                "avg_detections": det_metrics["avg_detections"],
                "details": det_metrics,
            })

            logger.info(
                "  %s — median=%.1fms  P95=%.1fms  avg_det=%.1f",
                tag, det_metrics["total"]["median_ms"],
                det_metrics["total"]["p95_ms"],
                det_metrics["avg_detections"],
            )

    # Benchmark each classifier in FP32 and FP16
    # Use classifier patches from val set
    clf_test_dir = Path(RESULTS_DIR).parent / "classifier_patches" / "val" / "no-damage"
    clf_test_images = sorted(clf_test_dir.glob("*.png"))[:50] if clf_test_dir.exists() else []

    if clf_test_images:
        for clf_info in classifiers:
            for half in [False, True]:
                prec = "fp16" if half else "fp32"
                tag = f"{clf_info['model_name']}_{prec}"
                logger.info("Benchmarking classifier: %s", tag)

                clf_metrics = benchmark_classifier(
                    clf_info["model_name"], clf_info["weights"],
                    clf_test_images, half=half,
                )

                all_results.append({
                    "tag": tag,
                    "component": "classifier",
                    "model": clf_info["model_name"],
                    "precision_mode": prec,
                    "median_ms": clf_metrics["per_patch"]["median_ms"],
                    "p95_ms": clf_metrics["per_patch"]["p95_ms"],
                    "details": clf_metrics,
                })

                logger.info(
                    "  %s — median=%.1fms/patch  P95=%.1fms/patch",
                    tag, clf_metrics["per_patch"]["median_ms"],
                    clf_metrics["per_patch"]["p95_ms"],
                )

    # Benchmark full pipeline (best detector + each classifier)
    if detectors and classifiers:
        # Use the detector with best mAP — load from detector results
        det_results_path = RESULTS_DIR / "detector_training_results.json"
        best_det = detectors[0]  # default to first
        if det_results_path.exists():
            with open(det_results_path) as f:
                det_results = json.load(f)
            best_tag = max(det_results, key=lambda x: x.get("mAP50", 0))["tag"]
            for d in detectors:
                if d["tag"] == best_tag:
                    best_det = d
                    break

        logger.info("Pipeline benchmarks using detector: %s", best_det["tag"])

        for clf_info in classifiers:
            for det_half, clf_half in [(False, False), (True, True)]:
                prec = "fp16" if det_half else "fp32"
                tag = f"pipeline_{best_det['tag']}_{clf_info['model_name']}_{prec}"
                logger.info("Benchmarking pipeline: %s", tag)

                pipe_metrics = benchmark_pipeline(
                    best_det, clf_info, test_images,
                    det_half=det_half, clf_half=clf_half,
                )

                # Load mAP for this detector
                det_map50 = 0
                if det_results_path.exists():
                    for dr in det_results:
                        if dr["tag"] == best_det["tag"]:
                            det_map50 = dr.get("mAP50", 0)

                all_results.append({
                    "tag": tag,
                    "component": "pipeline",
                    "detector": best_det["tag"],
                    "classifier": clf_info["model_name"],
                    "model": clf_info["model_name"],
                    "resolution": best_det["resolution"],
                    "precision_mode": prec,
                    "median_ms": pipe_metrics["total"]["median_ms"],
                    "p95_ms": pipe_metrics["total"]["p95_ms"],
                    "throughput_fps": pipe_metrics["throughput_fps"],
                    "mAP50": det_map50,
                    "details": pipe_metrics,
                })

                logger.info(
                    "  %s — median=%.1fms  P95=%.1fms  FPS=%.1f",
                    tag, pipe_metrics["total"]["median_ms"],
                    pipe_metrics["total"]["p95_ms"],
                    pipe_metrics["throughput_fps"],
                )

    # Save results
    results_path = RESULTS_DIR / "latency_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Latency results saved to %s", results_path)

    # Summary table
    logger.info("")
    logger.info("%-45s  %10s  %10s", "Config", "Median (ms)", "P95 (ms)")
    logger.info("-" * 70)
    for r in all_results:
        logger.info(
            "%-45s  %10.1f  %10.1f",
            r["tag"], r["median_ms"], r["p95_ms"],
        )

    logger.info("=== Latency benchmarking complete ===")


if __name__ == "__main__":
    main()
