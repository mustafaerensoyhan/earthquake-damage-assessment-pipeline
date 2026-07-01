r"""
Select 3 RescueNet patches (one Intact, one Damaged, one Collapsed) for the
bottom UAV row of Figure 4, run the seed-42 EfficientNet-B0 SOURCE classifier on
each (same checkpoint / transform / 4->3 harmonization as the rest of the paper),
and write the crops + a manifest you can verify before regenerating the figure.

Mirrors scripts/dump_rescuenet.py exactly (RescueNetDataset, get_val_transforms,
DamageClassifier.load, harmonize_probs_4to3). True class = RescueNet folder name
(0=Intact,1=Damaged,2=Collapsed; raw mask map 2:0,3:1,4:1,5:2).

Run from the repo ROOT with the venv active:
    python pick_rescuenet_row.py
Out: figure_rescuenet_row/{intact,damaged,collapsed}.png + manifest.csv
Nothing is synthesized; only real extracted patches are used.
"""
import os, sys, glob, csv, random
sys.path.insert(0, os.getcwd())
import numpy as np
from PIL import Image
import torch
from src.models.classifier import DamageClassifier
from src.data.xbd_classifier_dataset import get_val_transforms
from src.uncertainty.harmonize import harmonize_probs_4to3

# ----------------------------- config -----------------------------
ROOT  = r"rescuenet_patches"                                   # class folders live here
CKPT  = r"outputs\models\efficientnet_b0_fp32_seed42_best.pt"  # paper's seed-42 source model
MODEL = "efficientnet_b0"
OUTDIR = "figure_rescuenet_row"
CELL_PX = 256
N_SCAN = 250          # patches to score per class (random sample for speed)
SEED   = 0
# ------------------------------------------------------------------

random.seed(SEED)
CLASSES = ["Intact", "Damaged", "Collapsed"]      # true class = folder name
EXTS = ("*.png", "*.jpg", "*.jpeg")
device = "cuda" if torch.cuda.is_available() else "cpu"
_tf = get_val_transforms()
_model = DamageClassifier.load(MODEL, CKPT).to(device).eval()

@torch.no_grad()
def predict(pil):
    x = _tf(pil).unsqueeze(0).to(device)
    p4 = _model.predict_proba(x).detach().cpu().numpy()
    p3 = harmonize_probs_4to3(p4)[0]
    i = int(p3.argmax())
    return CLASSES[i], float(p3[i]), p3

def list_patches(cls):
    d = os.path.join(ROOT, cls)
    files = sorted(sum([glob.glob(os.path.join(d, e)) for e in EXTS], []))
    if not files:
        raise FileNotFoundError(f"no patches in {d} (run extract_rescuenet_patches.py)")
    random.shuffle(files)
    return files[:N_SCAN]

def score(true_cls, pred, conf):
    """Higher = better illustration of the cross-domain point.
    Intact/Collapsed: reward confident MISclassification into Damaged (the bias).
    Damaged: reward confident CORRECT prediction (anchor the row)."""
    if true_cls == "Damaged":
        return conf if pred == "Damaged" else -1
    # Intact or Collapsed: the figure's point is they get absorbed into Damaged
    if pred == "Damaged":
        return 1.0 + conf
    if pred != true_cls:
        return 0.5 + conf      # any confident wrong still shows failure
    return -conf               # correct = least useful for the point

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"device={device}  model={MODEL}  ckpt={CKPT}")
    rows = []
    for cls in CLASSES:
        best = None
        for fp in list_patches(cls):
            try:
                im = Image.open(fp).convert("RGB")
            except Exception:
                continue
            pred, conf, _ = predict(im)
            s = score(cls, pred, conf)
            if best is None or s > best[0]:
                best = (s, fp, pred, conf, im)
        if best is None:
            print(f"  {cls}: NONE"); continue
        _, fp, pred, conf, im = best
        outpng = os.path.join(OUTDIR, f"{cls.lower()}.png")
        im.resize((CELL_PX, CELL_PX), Image.LANCZOS).save(outpng)
        rows.append((os.path.basename(outpng), cls, pred, round(conf, 4), os.path.basename(fp)))
        print(f"  {cls:9s} -> pred {pred:9s} conf {conf:.3f}   src={os.path.basename(fp)}")
    with open(os.path.join(OUTDIR, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure_file", "true_class", "predicted_class", "confidence", "source_patch"])
        w.writerows(rows)
    print("wrote", os.path.join(OUTDIR, "manifest.csv"))
    print("\nReview the three crops + manifest before regenerating Figure 4.")

if __name__ == "__main__":
    main()
