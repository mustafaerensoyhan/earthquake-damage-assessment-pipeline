#!/usr/bin/env python
"""
Few-shot domain-adaptation data-efficiency curve.

For each (backbone, training seed), start from the source 4-class checkpoint,
warm-start a 3-class head, and fine-tune on a small BALANCED slice of TEBDE drawn
only from the held-out calibration pool (the 30% not used as the test split).
Evaluate on the same fixed 70% test split used by every other report. Repeat over
several sampling draws per budget so the curve has error bars.

The 0-shot point is the harmonised source model (read from the dumped logits, no
fine-tuning), so the curve starts exactly at your cross-domain baseline.

Writes outputs/uncertainty/fewshot_{model}_seed{seed}.json

Usage:
    python scripts/fewshot_adapt.py                       # efficientnet_b0, both seeds
    python scripts/fewshot_adapt.py --models efficientnet_b0 resnet34 deit_tiny
    python scripts/fewshot_adapt.py --budgets 5 10 20 40 80 --draws 3 --epochs 25
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tebde_dataset import TEBDEDataset
from src.data.xbd_classifier_dataset import get_train_transforms, get_val_transforms
from src.models.classifier import DamageClassifier
from src.uncertainty.metrics import compute_all_metrics, per_class_recall
from src.uncertainty.harmonize import (
    harmonize_probs_4to3, harmonize_labels_4to3, stratified_split,
)
from src.uncertainty.fewshot import (
    replace_head_3class, sample_fewshot, set_backbone_trainable,
)
from src.utils.config import MODEL_DIR, OUTPUT_DIR, TEBDE_ROOT

COLLAPSED = 2


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for img, lab in loader:
        p = torch.softmax(model(img.to(device)), dim=1)
        probs.append(p.cpu().numpy()); labels.append(np.asarray(lab).reshape(-1))
    probs = np.concatenate(probs); labels = np.concatenate(labels)
    preds = probs.argmax(1)
    m = compute_all_metrics(probs, labels, n_boot=500)
    metrics = {
        "accuracy": m["accuracy"], "ece": m["ece"], "nll": m["nll"],
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "collapsed_recall": per_class_recall(preds, labels, 3)[COLLAPSED],
    }
    return metrics, probs, labels


def finetune_and_eval(model_name, src_weights, train_idx, test_ds, device,
                      epochs, lr, batch_size, draw_seed, mode="ft", lp_epochs=15,
                      lp_lr=1e-3):
    seed_all(draw_seed)
    model = DamageClassifier.load(model_name, src_weights).to(device)
    replace_head_3class(model, warm=True)
    model.to(device)

    train_full = TEBDEDataset(root=TEBDE_ROOT, transform=get_train_transforms())
    train_loader = DataLoader(Subset(train_full, train_idx.tolist()),
                              batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    crit = nn.CrossEntropyLoss()  # balanced k-per-class sampling -> no class weights

    def run_phase(params, n_epochs, phase_lr):
        opt = torch.optim.AdamW(params, lr=phase_lr, weight_decay=1e-4)
        model.train()
        for _ in range(n_epochs):
            for img, lab in train_loader:
                img, lab = img.to(device), lab.to(device).long()
                opt.zero_grad()
                crit(model(img), lab).backward()
                opt.step()

    if mode == "lpft":
        # Phase 1: linear probe (head only) to align the head before disturbing features
        set_backbone_trainable(model, False)
        run_phase([p for p in model.parameters() if p.requires_grad], lp_epochs, lp_lr)
        # Phase 2: full fine-tune at low rate
        set_backbone_trainable(model, True)
        run_phase(model.parameters(), epochs, lr)
    else:  # plain full fine-tune
        run_phase(model.parameters(), epochs, lr)

    return evaluate(model, test_loader, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["efficientnet_b0"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    ap.add_argument("--budgets", nargs="+", type=int, default=[5, 10, 20, 40, 80])
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--split-frac", type=float, default=0.3)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--mode", choices=["ft", "lpft"], default="ft",
                    help="ft = full fine-tune (baseline); lpft = linear-probe then fine-tune")
    ap.add_argument("--lp-epochs", type=int, default=15, help="linear-probe epochs (lpft only)")
    ap.add_argument("--lp-lr", type=float, default=1e-3, help="linear-probe lr (lpft only)")
    ap.add_argument("--dump-probs", action="store_true", default=True,
                    help="save per-sample test probabilities for ensembling (Lever 2)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unc = Path(OUTPUT_DIR) / "uncertainty"
    test_ds_val = TEBDEDataset(root=TEBDE_ROOT, transform=get_val_transforms())

    for model_name in args.models:
        for seed in args.seeds:
            src = Path(MODEL_DIR) / f"{model_name}_fp32_seed{seed}_best.pt"
            dump = unc / f"logits_{model_name}_seed{seed}_tebde.npz"
            if not src.exists() or not dump.exists():
                print(f"[fewshot] SKIP {model_name} seed{seed} (missing weights or dump)")
                continue
            d = np.load(dump)
            labels = d["labels"].astype(np.int64)
            cal_pool, test_idx = stratified_split(labels, frac=args.split_frac, seed=args.split_seed)
            test_ds = Subset(test_ds_val, test_idx.tolist())

            # 0-shot baseline from dumped source logits (harmonised, on test split)
            tl = d["logits"][test_idx]; ty = labels[test_idx]
            probs3 = harmonize_probs_4to3(_softmax(tl))
            pred3 = harmonize_labels_4to3(_softmax(tl).argmax(1))
            base_m = compute_all_metrics(probs3, ty, n_boot=500)
            baseline = {
                "k": 0, "accuracy": float((pred3 == ty).mean()),
                "macro_f1": float(f1_score(ty, pred3, average="macro")),
                "collapsed_recall": per_class_recall(pred3, ty, 3)[COLLAPSED],
                "ece": base_m["ece"], "nll": base_m["nll"],
            }
            print(f"[fewshot] {model_name} seed{seed}  0-shot acc={baseline['accuracy']:.3f} "
                  f"collapsed_rec={baseline['collapsed_recall']:.3f} ECE={baseline['ece']:.3f}")

            results = []
            prob_store = {"labels": ty.astype(np.int64), "test_idx": test_idx}
            for k in args.budgets:
                for draw in range(args.draws):
                    train_idx = sample_fewshot(labels, cal_pool, k, seed=1000 * k + draw)
                    r, probs, _ = finetune_and_eval(
                        model_name, src, train_idx, test_ds, device,
                        args.epochs, args.lr, args.batch_size, draw_seed=draw,
                        mode=args.mode, lp_epochs=args.lp_epochs, lp_lr=args.lp_lr)
                    r.update({"k": k, "draw": draw, "n_train": int(len(train_idx))})
                    results.append(r)
                    prob_store[f"k{k}_d{draw}"] = probs.astype(np.float32)
                ks = [x for x in results if x["k"] == k]
                acc = np.mean([x["accuracy"] for x in ks])
                cr = np.nanmean([x["collapsed_recall"] for x in ks])
                ece = np.mean([x["ece"] for x in ks])
                print(f"           k={k:>3}/class  acc={acc:.3f}  collapsed_rec={cr:.3f}  ECE={ece:.3f}")

            tag = "" if args.mode == "ft" else f"_{args.mode}"
            out = {"model": model_name, "seed": seed, "mode": args.mode,
                   "baseline_0shot": baseline, "budgets": args.budgets,
                   "draws": args.draws, "epochs": args.epochs, "lr": args.lr,
                   "lp_epochs": args.lp_epochs, "results": results}
            p = unc / f"fewshot_{model_name}_seed{seed}{tag}.json"
            p.write_text(json.dumps(out, indent=2))
            print(f"[fewshot] wrote {p.name}")
            if args.dump_probs:
                pp = unc / f"fewshot_probs_{model_name}_seed{seed}{tag}.npz"
                np.savez_compressed(pp, **prob_store)
                print(f"[fewshot] wrote {pp.name}")


if __name__ == "__main__":
    main()
