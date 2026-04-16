#!/usr/bin/env python
"""
Generate all plots, tables, and LaTeX snippets from saved results.

Run this after training and evaluation are complete:
    python scripts/generate_results.py

Outputs go to outputs/figures/ and outputs/results/.
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import FIGURES_DIR, RESULTS_DIR, ensure_dirs
from src.utils.visualization import generate_all_figures


def print_summary():
    """Print a human-readable summary of all results."""
    logger = logging.getLogger(__name__)

    # Detector results
    det_path = RESULTS_DIR / "detector_training_results.json"
    if det_path.exists():
        with open(det_path) as f:
            det = json.load(f)
        print("\n" + "=" * 70)
        print("DETECTOR RESULTS")
        print("=" * 70)
        print(f"{'Config':<25} {'mAP@0.5':>8} {'mAP@0.5:0.95':>13} {'Prec':>8} {'Recall':>8}")
        print("-" * 70)
        for r in sorted(det, key=lambda x: x["tag"]):
            print(
                f"{r['tag']:<25} {r['mAP50']:>8.4f} {r['mAP50-95']:>13.4f} "
                f"{r.get('precision', 0):>8.4f} {r.get('recall', 0):>8.4f}"
            )
    else:
        print("\nNo detector results found.")

    # Classifier results
    clf_path = RESULTS_DIR / "classifier_training_results.json"
    if clf_path.exists():
        with open(clf_path) as f:
            clf = json.load(f)
        print("\n" + "=" * 70)
        print("CLASSIFIER RESULTS")
        print("=" * 70)
        print(f"{'Config':<30} {'Accuracy':>8} {'Macro F1':>9} {'Best Epoch':>11}")
        print("-" * 70)
        for r in sorted(clf, key=lambda x: x["tag"]):
            print(
                f"{r['tag']:<30} {r['val_accuracy']:>8.4f} "
                f"{r['val_macro_f1']:>9.4f} {r.get('best_epoch', '-'):>11}"
            )
    else:
        print("\nNo classifier results found.")

    # Latency results
    lat_path = RESULTS_DIR / "latency_results.json"
    if lat_path.exists():
        with open(lat_path) as f:
            lat = json.load(f)
        print("\n" + "=" * 70)
        print("LATENCY RESULTS")
        print("=" * 70)
        print(f"{'Config':<35} {'Median (ms)':>12} {'P95 (ms)':>10} {'FPS':>8}")
        print("-" * 70)
        for r in sorted(lat, key=lambda x: x.get("tag", "")):
            median = r.get("median_ms", 0)
            p95 = r.get("p95_ms", 0)
            fps = 1000.0 / median if median > 0 else 0
            print(f"{r.get('tag', '?'):<35} {median:>12.1f} {p95:>10.1f} {fps:>8.1f}")
    else:
        print("\nNo latency results found yet.")

    # Cross-domain results
    cd_path = RESULTS_DIR / "cross_domain_results.json"
    if cd_path.exists():
        with open(cd_path) as f:
            cd = json.load(f)
        print("\n" + "=" * 70)
        print("CROSS-DOMAIN RESULTS (xBD → TEBDE)")
        print("=" * 70)
        print(f"{'Config':<30} {'xBD Acc':>8} {'TEBDE Acc':>10} {'Delta':>8}")
        print("-" * 70)
        for r in sorted(cd, key=lambda x: x.get("tag", "")):
            xbd_acc = r.get("xbd_accuracy", 0)
            tebde_acc = r.get("tebde_accuracy", 0)
            delta = xbd_acc - tebde_acc
            print(
                f"{r.get('tag', '?'):<30} {xbd_acc:>8.4f} "
                f"{tebde_acc:>10.4f} {delta:>+8.4f}"
            )
    else:
        print("\nNo cross-domain results found yet.")

    print("\n" + "=" * 70)

    # List generated figures
    figs = sorted(FIGURES_DIR.glob("*.png"))
    if figs:
        print(f"\nGenerated {len(figs)} figure(s) in {FIGURES_DIR}:")
        for f in figs:
            print(f"  {f.name}")

    tex_files = sorted(FIGURES_DIR.glob("*.tex"))
    if tex_files:
        print(f"\nGenerated {len(tex_files)} LaTeX table(s):")
        for f in tex_files:
            print(f"  {f.name}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ensure_dirs()

    # Generate all figures and tables
    generate_all_figures()

    # Print summary
    print_summary()


if __name__ == "__main__":
    main()
