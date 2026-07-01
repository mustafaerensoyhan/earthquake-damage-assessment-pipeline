# Post-Earthquake Building Damage Assessment under Domain Shift

Code for a study on calibration, selective prediction, and few-shot adaptation
for building damage classification when models trained on satellite imagery are
applied to UAV imagery. A two-stage pipeline (YOLO building detection followed by
a CNN damage classifier) is used; this repository focuses on the damage
classification stage and its behavior under satellite-to-UAV domain shift.

Companion code for a paper submitted to UBMK 2026.

## What this repo contains

- `src/` — models, data loaders, and the uncertainty/adaptation code
  (temperature scaling, selective prediction, test-time adaptation, few-shot LP-FT,
  MC-dropout, metrics, label harmonization).
- `scripts/` — runnable steps that dump predictions and produce the result JSONs
  and paper figures (calibration, selective, ensemble, external validity,
  compute cost, few-shot, cross-domain).
- `paper/` — LaTeX source and generated figures.
- `outputs/results/` — curated result JSONs (metrics only).

Datasets, model weights, extracted patches, and virtual environments are not
included (see Data and Licensing below).

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Developed with Python 3.11, PyTorch 2.5.1 (CUDA 12.1), on an RTX 4080 Laptop GPU.

## Datasets

This code uses three datasets, which you must obtain separately from their sources:

- **xBD** (satellite, training and in-domain evaluation) —
  https://xview2.org/  (released under CC BY-NC-SA 4.0).
- **UAVs-TEBDE** (UAV, 2023 Turkiye earthquakes) —
  https://doi.org/10.17632/5m349hfvkb  (Mendeley Data).
- **RescueNet** (UAV, Hurricane Michael) —
  Rahnemoonfar, Chowdhury, Murphy, *Scientific Data* 10:913 (2023),
  https://doi.org/10.1038/s41597-023-02799-4.

Place datasets at the repository root and point the scripts at them via their
path variables. See each script for the expected layout.

## Reproducing results

1. Train / obtain classifier checkpoints (see `src/training/`).
2. Dump predictions and metrics: run the relevant `scripts/*_dump.py` and
   `scripts/*_report.py` (e.g. `python scripts/calib_dump.py`,
   `python scripts/calib_report.py`).
3. Aggregate across seeds: `python scripts/aggregate_seeds.py --unc-dir outputs/uncertainty`.
4. Build figures: `python scripts/make_reliability.py`, `make_selective_figs.py`,
   `make_fewshot_fig.py`, etc.

Results are averaged over three seeds (42, 7, 123).

## Data and licensing

Source code in this repository is provided for research use. The datasets above
are governed by their own licenses and are **not redistributed here**; obtain them
from the original sources and follow their terms. In particular, xBD is
non-commercial (CC BY-NC-SA 4.0), and RescueNet's dataset release is CC BY-NC-ND.
No dataset imagery, derived image patches, or trained weights are included in
this repository.

## Citation

If you use this code, please cite the associated paper (UBMK 2026) once available,
and the dataset papers listed above.
