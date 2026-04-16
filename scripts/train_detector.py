#!/usr/bin/env python
"""
Convenience entry point for detector training.

Delegates to src.training.train_detector.  Run from project root:
    python scripts/train_detector.py
    python scripts/train_detector.py --model yolov8s --resolution 640
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_detector import main

if __name__ == "__main__":
    main()
