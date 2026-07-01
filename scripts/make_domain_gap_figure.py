r"""
Qualitative domain-gap figure for the UBMK paper.

Builds a 2 x 3 grid:
    row 0 = Satellite (xBD)   cropped building patches per damage class
    row 1 = UAV (UAVs-TEBDE)  pre-cropped building patches per damage class
columns = Intact / Damaged / Collapsed

TEBDE already has class folders, so it just picks one image from each.
xBD test holds full tiles + label JSONs, so for the satellite row this script
reads the post-disaster labels, finds one building of each damage subtype, and
crops it from the post-disaster image. Harmonization: no-damage -> Intact,
minor/major-damage -> Damaged, destroyed -> Collapsed.

Run:  python make_domain_gap_figure.py
Out:  qualitative_domain_gap.png  (300 dpi)

Only real dataset crops are used. Nothing is synthesized.
"""
import os, re, glob, json, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------- EDIT PATHS -----------------------------
TEBDE_DIR = r"C:\Users\mustafaerensoyhan\Downloads\SoftwareResearchProject\uavs-tebde\UAVs-TEBDE (Augmented)"
TEBDE_SUB = {"Intact": "intact", "Damaged": "damaged", "Collapsed": "collapsed"}

XBD_DIR   = r"C:\Users\mustafaerensoyhan\Downloads\SoftwareResearchProject\xbd\test"
# inside XBD_DIR the script looks for images/  and  labels/ (falls back to targets/)

# Prefer these event prefixes for the satellite row (earthquake-relevant, to
# match the paper). Falls back to any event if none are found.
PREFER_PREFIXES = ("mexico-earthquake", "palu-tsunami", "earthquake", "tsunami")

CELL_PX = 256          # output size of each patch
PAD     = 0.40         # bbox padding fraction for xBD crops
SEED    = 0            # change to reshuffle which examples are picked
OUT     = "qualitative_domain_gap.png"
# ----------------------------------------------------------------------

random.seed(SEED)
ROWS = ["Satellite (xBD)", "UAV (TEBDE)"]
COLS = ["Intact", "Damaged", "Collapsed"]
EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
SUBTYPE_TO_CLASS = {
    "no-damage": "Intact",
    "minor-damage": "Damaged",
    "major-damage": "Damaged",
    "destroyed": "Collapsed",
}

def square(im):
    return im.convert("RGB").resize((CELL_PX, CELL_PX), Image.LANCZOS)

# ---------- TEBDE: one representative patch per class folder ----------
def tebde_patch(cls):
    folder = os.path.join(TEBDE_DIR, TEBDE_SUB[cls])
    files = sorted(sum([glob.glob(os.path.join(folder, e)) for e in EXTS], []))
    if not files:
        raise FileNotFoundError(f"no images in {folder}")
    return square(Image.open(random.choice(files)))

# ---------- xBD: crop one building of each class from test tiles ----------
def find_dirs():
    img = os.path.join(XBD_DIR, "images")
    lab = os.path.join(XBD_DIR, "labels")
    if not os.path.isdir(lab):
        lab = os.path.join(XBD_DIR, "targets")   # some distributions name it this
    return img, lab

def wkt_bbox(wkt):
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", wkt)]
    xs, ys = nums[0::2], nums[1::2]
    return min(xs), min(ys), max(xs), max(ys)

def collect_xbd():
    """Return {class: (image_path, bbox_area, bbox)} best (largest) per class."""
    img_dir, lab_dir = find_dirs()
    labels = glob.glob(os.path.join(lab_dir, "*post_disaster*.json"))
    # prefer earthquake-relevant tiles, fall back to all
    pref = [p for p in labels if any(k in os.path.basename(p).lower() for k in PREFER_PREFIXES)]
    labels = (pref or labels)
    random.shuffle(labels)
    best = {}   # class -> (area, image_path, bbox)
    for lp in labels[:400]:
        try:
            data = json.load(open(lp))
        except Exception:
            continue
        feats = data.get("features", {}).get("xy", [])
        stem = os.path.basename(lp).replace(".json", "")
        ip = os.path.join(img_dir, stem + ".png")
        if not os.path.exists(ip):
            continue
        for f in feats:
            sub = f.get("properties", {}).get("subtype")
            cls = SUBTYPE_TO_CLASS.get(sub)
            if cls is None:
                continue
            try:
                x0, y0, x1, y1 = wkt_bbox(f["wkt"])
            except Exception:
                continue
            area = (x1 - x0) * (y1 - y0)
            # skip tiny specks and full-tile blobs; keep a clean mid-size building
            if area < 400 or area > 90000:
                continue
            if cls not in best or area > best[cls][0]:
                best[cls] = (area, ip, (x0, y0, x1, y1))
        if all(c in best for c in COLS):
            break
    return best

def xbd_patch(cls, best):
    if cls not in best:
        raise FileNotFoundError(f"no xBD building found for class {cls} "
                                f"(check XBD_DIR/images and /labels)")
    _, ip, (x0, y0, x1, y1) = best[cls]
    im = Image.open(ip).convert("RGB")
    W, H = im.size
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(w, h) * (1 + PAD) / 2
    L, T = max(0, cx - half), max(0, cy - half)
    R, B = min(W, cx + half), min(H, cy + half)
    return square(im.crop((int(L), int(T), int(R), int(B))))

# ------------------------------- build -------------------------------
def main():
    print("Scanning xBD test tiles for one building per class...")
    best = collect_xbd()
    for c in COLS:
        print(f"  xBD {c}: {'OK ' + os.path.basename(best[c][1]) if c in best else 'NOT FOUND'}")

    fig, axes = plt.subplots(2, 3, figsize=(3 * 1.9, 2 * 1.9))
    for r, row in enumerate(ROWS):
        for c, col in enumerate(COLS):
            ax = axes[r, c]
            try:
                im = xbd_patch(col, best) if r == 0 else tebde_patch(col)
                ax.imshow(im)
            except Exception as e:
                ax.text(0.5, 0.5, f"missing:\n{e}", ha="center", va="center",
                        fontsize=5, transform=ax.transAxes, wrap=True)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#777"); s.set_linewidth(0.8)
            if r == 0: ax.set_title(col, fontsize=11, pad=4)
            if c == 0: ax.set_ylabel(row, fontsize=10)
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    print("wrote", OUT)

if __name__ == "__main__":
    main()
