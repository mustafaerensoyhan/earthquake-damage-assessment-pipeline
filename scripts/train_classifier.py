#!/usr/bin/env python
"""
Convenience entry point for classifier training.

Delegates to src.training.train_classifier.  Run from project root:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --model efficientnet_b0 --fp16
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_classifier import main

if __name__ == "__main__":
    main()
