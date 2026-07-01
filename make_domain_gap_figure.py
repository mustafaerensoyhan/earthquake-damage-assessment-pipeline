r"""
Cross-domain qualitative figure for the UBMK paper (prediction + confidence overlay).

2 x 3 grid:
    row 0 = Satellite (xBD)  -- nadir / rooftop view
    row 1 = UAV (UAVs-TEBDE) -- oblique / facade view
columns = Intact / Damaged / Collapsed

Each cell shows a real building crop with the seed-42 EfficientNet-B0 prediction
and confidence overlaid (green = correct, red = wrong), plus a confidence bar.
Predictions come from the SAME model, transform, and 4->3 harmonization used to
produce the paper's numbers (imported from the repo), so the figure cannot drift
from the tables. xBD crops are forced to come from three DISTINCT tiles.

Run from the repo ROOT with the venv active:
    python make_domain_gap_figure.py
Out: qualitative_domain_gap.png (300 dpi). Nothing is synthesized.
"""
import os, re, glob, json, random, sys
sys.path.insert(0, os.getcwd())          # so `from src...` resolves
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch

# repo modules (guarantee parity with the paper pipeline)
from src.models.classifier import DamageClassifier
from src.data.xbd_classifier_dataset import get_val_transforms
from src.uncertainty.harmonize import harmonize_probs_4to3

# ----------------------------- EDIT PATHS -----------------------------
TEBDE_DIR  = r"C:\Users\mustafaerensoyhan\Downloads\SoftwareResearchProject\uavs-tebde\UAVs-TEBDE (Augmented)"
TEBDE_SUB  = {"Intact": "intact", "Damaged": "damaged", "Collapsed": "collapsed"}
XBD_DIR    = r"C:\Users\mustafaerensoyhan\Downloads\SoftwareResearchProject\xbd\test"
CKPT       = r"outputs\models\efficientnet_b0_fp32_seed42_best.pt"  # paper's primary panel
MODEL_NAME = "efficientnet_b0"
PREFER_PREFIXES = ("mexico-earthquake", "palu-tsunami", "earthquake", "tsunami")
CELL_PX, PAD, SEED = 256, 0.40, 0
# which size-rank building to use per class (0 = largest). Bump a class to
# pick a different building if you do not like the auto choice.
CLASS_RANK = {"Intact": 0, "Damaged": 0, "Collapsed": 2}
OUT = "qualitative_domain_gap.png"
# ----------------------------------------------------------------------

random.seed(SEED)
COLS = ["Intact", "Damaged", "Collapsed"]
ROW_TOP = "Satellite (xBD)\nnadir / rooftop view"
ROW_BOT = "UAV (TEBDE)\noblique / facade view"
EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
SUBTYPE_TO_CLASS = {"no-damage": "Intact", "minor-damage": "Damaged",
                    "major-damage": "Damaged", "destroyed": "Collapsed"}
INK = "#222222"   # single neutral color for all overlay text

device = "cuda" if torch.cuda.is_available() else "cpu"
_tf = get_val_transforms()
_model = DamageClassifier.load(MODEL_NAME, CKPT).to(device).eval()

def square(im):
    return im.convert("RGB").resize((CELL_PX, CELL_PX), Image.LANCZOS)

@torch.no_grad()
def predict(pil):
    """Real forward pass -> (pred_class_name, confidence) in the 3-class scheme."""
    x = _tf(pil).unsqueeze(0).to(device)
    p4 = _model.predict_proba(x).detach().cpu().numpy()   # (1,4)
    p3 = harmonize_probs_4to3(p4)[0]                       # (3,)
    i = int(p3.argmax())
    return COLS[i], float(p3[i])

# ---------- TEBDE: one representative patch per class folder ----------
def tebde_patch(cls):
    folder = os.path.join(TEBDE_DIR, TEBDE_SUB[cls])
    files = sorted(sum([glob.glob(os.path.join(folder, e)) for e in EXTS], []))
    if not files:
        raise FileNotFoundError(f"no images in {folder}")
    return square(Image.open(random.choice(files)))

# ---------- xBD: distinct tile per class, cropped from test labels ----------
def find_dirs():
    img = os.path.join(XBD_DIR, "images")
    lab = os.path.join(XBD_DIR, "labels")
    if not os.path.isdir(lab):
        lab = os.path.join(XBD_DIR, "targets")
    return img, lab

def wkt_bbox(wkt):
    n = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", wkt)]
    xs, ys = n[0::2], n[1::2]
    return min(xs), min(ys), max(xs), max(ys)

def collect_xbd():
    """Candidates per class: list of (area, image_path, bbox), largest first."""
    img_dir, lab_dir = find_dirs()
    labels = glob.glob(os.path.join(lab_dir, "*post_disaster*.json"))
    pref = [p for p in labels if any(k in os.path.basename(p).lower() for k in PREFER_PREFIXES)]
    labels = pref or labels
    random.shuffle(labels)
    cand = {c: [] for c in COLS}
    for lp in labels[:400]:
        try:
            data = json.load(open(lp))
        except Exception:
            continue
        stem = os.path.basename(lp).replace(".json", "")
        ip = os.path.join(img_dir, stem + ".png")
        if not os.path.exists(ip):
            continue
        for f in data.get("features", {}).get("xy", []):
            cls = SUBTYPE_TO_CLASS.get(f.get("properties", {}).get("subtype"))
            if cls is None:
                continue
            try:
                x0, y0, x1, y1 = wkt_bbox(f["wkt"])
            except Exception:
                continue
            area = (x1 - x0) * (y1 - y0)
            if 400 <= area <= 90000:
                cand[cls].append((area, ip, (x0, y0, x1, y1)))
    for c in COLS:
        cand[c].sort(key=lambda t: -t[0])
    return cand

def assign_distinct(cand):
    """Each class gets a building from a tile no other class used; CLASS_RANK
    chooses which one by size (0 = largest, 1 = next, ...)."""
    used, pick = set(), {}
    for c in COLS:
        skip = CLASS_RANK.get(c, 0)
        chosen = None
        for area, ip, bb in cand[c]:          # candidates already sorted largest-first
            if ip in used:
                continue
            if skip > 0:
                skip -= 1
                continue
            chosen = (ip, bb); break
        if chosen is None:                    # fallback: first unused of any rank
            for area, ip, bb in cand[c]:
                if ip not in used:
                    chosen = (ip, bb); break
        if chosen is None and cand[c]:        # last resort: allow reuse
            chosen = (cand[c][0][1], cand[c][0][2])
        if chosen:
            pick[c] = chosen; used.add(chosen[0])
    return pick

def xbd_patch(cls, pick):
    if cls not in pick:
        raise FileNotFoundError(f"no xBD building for {cls}")
    ip, (x0, y0, x1, y1) = pick[cls]
    im = Image.open(ip).convert("RGB"); W, H = im.size
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (1 + PAD) / 2
    box = (int(max(0, cx - half)), int(max(0, cy - half)),
           int(min(W, cx + half)), int(min(H, cy + half)))
    return square(im.crop(box))

# ------------------------------- draw -------------------------------
def main():
    print(f"device={device}  model={MODEL_NAME}  ckpt={CKPT}")
    print("Scanning xBD test tiles (forcing distinct tiles per class)...")
    pick = assign_distinct(collect_xbd())
    for c in COLS:
        print(f"  xBD {c}: {os.path.basename(pick[c][0]) if c in pick else 'NOT FOUND'}")

    fig, axes = plt.subplots(2, 3, figsize=(6.4, 5.1))
    row_stats = {0: [], 1: []}
    for r, label in enumerate([ROW_TOP, ROW_BOT]):
        for c, col in enumerate(COLS):
            ax = axes[r, c]
            try:
                im = xbd_patch(col, pick) if r == 0 else tebde_patch(col)
                pred, conf = predict(im)
                ax.imshow(im)
                # prediction + confidence BELOW the image, single neutral colour
                ax.set_xlabel(f"predicted: {pred}\n(confidence {conf:.2f})",
                              fontsize=8.5, color=INK, labelpad=4)
                row_stats[r].append((pred == col, conf))
            except Exception as e:
                ax.text(0.5, 0.5, f"missing:\n{e}", ha="center", va="center",
                        fontsize=5, transform=ax.transAxes, wrap=True)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#777"); s.set_linewidth(0.8)
            if r == 0:
                ax.set_title(col, fontsize=11, pad=6)   # column header = TRUE class
            if c == 0:
                ax.set_ylabel(label, fontsize=9)

    # per-row summary on the right margin, terms spelled out in full
    for r in (0, 1):
        if row_stats[r]:
            acc = np.mean([ok for ok, _ in row_stats[r]])
            cf  = np.mean([c for _, c in row_stats[r]])
            axes[r, 2].text(1.06, 0.5,
                            f"mean confidence {cf:.2f}\nmean accuracy {acc:.2f}",
                            transform=axes[r, 2].transAxes, fontsize=8.5,
                            ha="left", va="center", color=INK)

    plt.tight_layout(h_pad=0.3, w_pad=0.5)   # smaller h_pad -> rows closer
    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    print("wrote", OUT)

if __name__ == "__main__":
    main()
