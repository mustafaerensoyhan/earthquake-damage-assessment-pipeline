r"""
Rebuild Figure 4 (qualitative cross-domain gap) with the UAV row from RescueNet.

Top row (Satellite, xBD): unchanged -- selected + predicted live, exactly as the
original figure, using the seed-42 source model + get_val_transforms + 4->3 harmonize.
Bottom row (UAV, RescueNet): the three crops chosen by pick_rescuenet_row.py, with
their predictions/confidences taken from the manifest (no re-inference, no drift).

Layout, fonts, single neutral colour, captions-below-image, per-row summary, and
the distinct-tile xBD selection are all identical to the prior version.

Run from the repo ROOT, venv active, AFTER pick_rescuenet_row.py has produced
figure_rescuenet_row/.  Out: qualitative_domain_gap.png (300 dpi).
"""
import os, re, glob, json, random, sys, csv
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
from src.models.classifier import DamageClassifier
from src.data.xbd_classifier_dataset import get_val_transforms
from src.uncertainty.harmonize import harmonize_probs_4to3

# ----------------------------- config (unchanged xBD side) -----------------------------
XBD_DIR    = r"C:\Users\mustafaerensoyhan\Downloads\SoftwareResearchProject\xbd\test"
CKPT       = r"outputs\models\efficientnet_b0_fp32_seed42_best.pt"
MODEL_NAME = "efficientnet_b0"
PREFER_PREFIXES = ("mexico-earthquake", "palu-tsunami", "earthquake", "tsunami")
CLASS_RANK = {"Intact": 0, "Damaged": 0, "Collapsed": 1}
CELL_PX, PAD, SEED = 256, 0.40, 0
# RescueNet row: crops + manifest from pick_rescuenet_row.py
RESCUE_DIR = "figure_rescuenet_row"
OUT = "qualitative_domain_gap.png"
INK = "#222222"
# ---------------------------------------------------------------------------------------

random.seed(SEED)
COLS = ["Intact", "Damaged", "Collapsed"]
ROW_TOP = "Satellite (xBD)\nnadir / rooftop view"
ROW_BOT = "UAV (RescueNet)\noblique / facade view"
EXTS = ("*.png","*.jpg","*.jpeg","*.tif","*.tiff")
SUBTYPE_TO_CLASS = {"no-damage":"Intact","minor-damage":"Damaged","major-damage":"Damaged","destroyed":"Collapsed"}
device = "cuda" if torch.cuda.is_available() else "cpu"
_tf = get_val_transforms()
_model = DamageClassifier.load(MODEL_NAME, CKPT).to(device).eval()

def square(im): return im.convert("RGB").resize((CELL_PX, CELL_PX), Image.LANCZOS)

@torch.no_grad()
def predict(pil):
    x = _tf(pil).unsqueeze(0).to(device)
    p3 = harmonize_probs_4to3(_model.predict_proba(x).detach().cpu().numpy())[0]
    i = int(p3.argmax()); return COLS[i], float(p3[i])

# ---------- xBD top row (unchanged) ----------
def find_dirs():
    img = os.path.join(XBD_DIR,"images"); lab = os.path.join(XBD_DIR,"labels")
    if not os.path.isdir(lab): lab = os.path.join(XBD_DIR,"targets")
    return img, lab
def wkt_bbox(w):
    n=[float(x) for x in re.findall(r"[-+]?\d*\.?\d+", w)]; xs,ys=n[0::2],n[1::2]
    return min(xs),min(ys),max(xs),max(ys)
def collect_xbd():
    img_dir,lab_dir=find_dirs(); labels=glob.glob(os.path.join(lab_dir,"*post_disaster*.json"))
    pref=[p for p in labels if any(k in os.path.basename(p).lower() for k in PREFER_PREFIXES)]
    labels=pref or labels; random.shuffle(labels); cand={c:[] for c in COLS}
    for lp in labels[:400]:
        try: data=json.load(open(lp))
        except Exception: continue
        stem=os.path.basename(lp).replace(".json",""); ip=os.path.join(img_dir,stem+".png")
        if not os.path.exists(ip): continue
        for f in data.get("features",{}).get("xy",[]):
            cls=SUBTYPE_TO_CLASS.get(f.get("properties",{}).get("subtype"))
            if cls is None: continue
            try: x0,y0,x1,y1=wkt_bbox(f["wkt"])
            except Exception: continue
            area=(x1-x0)*(y1-y0)
            if 400<=area<=90000: cand[cls].append((area,ip,(x0,y0,x1,y1)))
    for c in COLS: cand[c].sort(key=lambda t:-t[0])
    return cand
def assign_distinct(cand):
    used,pick=set(),{}
    for c in COLS:
        skip=CLASS_RANK.get(c,0); chosen=None
        for area,ip,bb in cand[c]:
            if ip in used: continue
            if skip>0: skip-=1; continue
            chosen=(ip,bb); break
        if chosen is None:
            for area,ip,bb in cand[c]:
                if ip not in used: chosen=(ip,bb); break
        if chosen is None and cand[c]: chosen=(cand[c][0][1],cand[c][0][2])
        if chosen: pick[c]=chosen; used.add(chosen[0])
    return pick
def xbd_patch(cls,pick):
    ip,(x0,y0,x1,y1)=pick[cls]; im=Image.open(ip).convert("RGB"); W,H=im.size
    cx,cy=(x0+x1)/2,(y0+y1)/2; half=max(x1-x0,y1-y0)*(1+PAD)/2
    box=(int(max(0,cx-half)),int(max(0,cy-half)),int(min(W,cx+half)),int(min(H,cy+half)))
    return square(im.crop(box))

# ---------- RescueNet bottom row (from manifest, no re-inference) ----------
def load_rescuenet():
    man=os.path.join(RESCUE_DIR,"manifest.csv")
    rows={}
    with open(man) as f:
        for r in csv.DictReader(f):
            rows[r["true_class"]]=(os.path.join(RESCUE_DIR,r["figure_file"]),
                                   r["predicted_class"], float(r["confidence"]))
    return rows

def main():
    pick=assign_distinct(collect_xbd())
    rescue=load_rescuenet()
    fig,axes=plt.subplots(2,3,figsize=(6.4,5.1)); stats={0:[],1:[]}
    for r,label in enumerate([ROW_TOP,ROW_BOT]):
        for c,col in enumerate(COLS):
            ax=axes[r,c]
            try:
                if r==0:
                    im=xbd_patch(col,pick); pred,conf=predict(im)
                else:
                    fp,pred,conf=rescue[col]; im=square(Image.open(fp))
                ax.imshow(im)
                ax.set_xlabel(f"predicted: {pred}\n(confidence {conf:.2f})",fontsize=8.5,color=INK,labelpad=4)
                stats[r].append((pred==col,conf))
            except Exception as e:
                ax.text(0.5,0.5,f"missing:\n{e}",ha="center",va="center",fontsize=5,transform=ax.transAxes,wrap=True)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_color("#777"); s.set_linewidth(0.8)
            if r==0: ax.set_title(col,fontsize=11,pad=6)
            if c==0: ax.set_ylabel(label,fontsize=9)
    for r in (0,1):
        if stats[r]:
            acc=np.mean([o for o,_ in stats[r]]); cf=np.mean([c for _,c in stats[r]])
            axes[r,2].text(1.06,0.5,f"mean confidence {cf:.2f}\nmean accuracy {acc:.2f}",
                           transform=axes[r,2].transAxes,fontsize=8.5,ha="left",va="center",color=INK)
    plt.tight_layout(h_pad=0.3,w_pad=0.5); plt.savefig(OUT,dpi=300,bbox_inches="tight")
    print("wrote",OUT)
    print("xBD row predicted live; RescueNet row from manifest:",
          {k:(v[1],round(v[2],2)) for k,v in rescue.items()})

if __name__=="__main__":
    main()
