# Two-Stage Building Damage Assessment Pipeline

A research project evaluating accuracy-latency tradeoffs in post-earthquake
building damage assessment from satellite and UAV imagery.

**Author:** Mustafa Eren Soyhan — Ontario Tech University  
**Course:** Software Research Project (Computer Engineering)

## Architecture

| Stage | Task | Models |
|-------|------|--------|
| 1 — Detector | Building localisation | YOLOv8-n / s / m |
| 2 — Classifier | Damage severity | EfficientNet-B0, ResNet-34 |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Update paths in src/utils/config.py to match your machine

# 3. Prepare YOLO detector dataset (all disasters)
python scripts/prepare_yolo_data.py

# 4. Prepare classifier patches (earthquake events only)
python scripts/prepare_classifier_data.py
```

## Project Structure

```
SoftwareResearchProject/
├── src/
│   ├── data/                 # Dataset parsers and loaders
│   ├── models/               # Detector and classifier wrappers
│   ├── training/             # Training scripts
│   ├── evaluation/           # Metrics and benchmarking
│   ├── pipeline/             # Two-stage inference
│   └── utils/                # Config, timing, visualisation
├── configs/                  # YAML experiment configurations
├── scripts/                  # Entry-point scripts
├── outputs/                  # Models, results, figures
└── requirements.txt
```

## Datasets

- **xBD** — satellite imagery, 19 disaster events, 4-class damage labels
- **UAVs-TEBDE** — UAV imagery from 2023 Türkiye earthquakes, 3-class labels

## Implementation Phases

1. **Data Preparation** — parse xBD labels, convert to YOLO format, extract classifier patches
2. **Model Training** — train detectors and classifiers across configurations
3. **Evaluation** — mAP, accuracy/F1, latency benchmarks, cross-domain transfer
4. **Visualisation** — tradeoff plots, confusion matrices, result tables
