#!/usr/bin/env python
"""
Deployment compute-cost report: parameters, per-patch latency, throughput, and
the MARGINAL inference cost of each trustworthiness method. This is what lets the
paper trade three budgets against each other: labels, compute, and human review.

Per backbone (using the trained checkpoints):
    params                total parameter count (millions)
    latency (b=1)         median and P95 single-patch classifier latency (ms)
    throughput (b=32)     patches/sec at batch 32
    +cal/triage overhead  cost of softmax + temperature + harmonise + collapse
                          sort, per patch (microseconds) -> negligible, "free"
    TENT overhead         (forward + backward + step) / forward, same batch
                          -> the runtime price of the label-free adaptation that
                             also fails, so it is expensive AND unhelpful

Few-shot adds a one-time fine-tune but ZERO added inference cost, and is noted as
such rather than timed here.

Latency protocol mirrors the project's existing benchmark: warm-up passes, then
timed passes with torch.cuda.synchronize() for accurate GPU timing.

Usage:
    python scripts/compute_cost_report.py
    python scripts/compute_cost_report.py --models efficientnet_b0 resnet34 deit_tiny --iters 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classifier import DamageClassifier
from src.uncertainty.harmonize import harmonize_probs_4to3, harmonize_labels_4to3
from src.utils.config import MODEL_DIR, OUTPUT_DIR

_NORM = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_latency(model, device, batch_size, n_warmup, n_iter):
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    for _ in range(n_warmup):
        model(x)
    _sync(device)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        model(x)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms for the batch
    times = np.array(times)
    per_patch = times / batch_size
    return {
        "median_ms": float(np.median(per_patch)),
        "p95_ms": float(np.percentile(per_patch, 95)),
        "throughput_per_s": float(batch_size / (np.median(times) / 1000.0)),
    }


@torch.no_grad()
def measure_postproc(model, device, batch_size, n_iter):
    """Cost of the trustworthiness post-processing: softmax, /T, harmonise,
    collapse-priority sort. Per patch, in microseconds."""
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    logits = model(x).cpu().numpy()
    T = 3.0
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z); p4 = e / e.sum(axis=1, keepdims=True)
        p3 = harmonize_probs_4to3(p4)
        pred3 = harmonize_labels_4to3(p4.argmax(1))
        prio = p3[:, 2].copy(); prio[pred3 == 2] = -1.0
        np.argsort(-prio)
    dt = (time.perf_counter() - t0) / n_iter
    return float(dt / batch_size * 1e6)  # microseconds per patch


def measure_tent_overhead(model, device, batch_size, n_iter):
    """(forward + backward + step) / forward, at the same batch size."""
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    # forward-only baseline
    with torch.no_grad():
        for _ in range(5):
            model(x)
        _sync(device); t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        _sync(device); fwd = (time.perf_counter() - t0) / n_iter

    # TENT step: entropy backward over norm affine params
    for p in model.parameters():
        p.requires_grad_(False)
    affine = []
    for m in model.modules():
        if isinstance(m, _NORM):
            if m.weight is not None: m.weight.requires_grad_(True); affine.append(m.weight)
            if m.bias is not None: m.bias.requires_grad_(True); affine.append(m.bias)
    opt = torch.optim.Adam(affine, lr=1e-3)

    def step():
        opt.zero_grad()
        out = model(x)
        loss = -(out.softmax(1) * out.log_softmax(1)).sum(1).mean()
        loss.backward(); opt.step()

    for _ in range(5):
        step()
    _sync(device); t0 = time.perf_counter()
    for _ in range(n_iter):
        step()
    _sync(device); adapt = (time.perf_counter() - t0) / n_iter
    return float(adapt / fwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["efficientnet_b0", "resnet34", "deit_tiny"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--throughput-batch", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compute_cost] device={device}")
    rows = []
    for model_name in args.models:
        w = Path(MODEL_DIR) / f"{model_name}_fp32_seed{args.seed}_best.pt"
        if not w.exists():
            print(f"[compute_cost] SKIP missing {w.name}"); continue
        model = DamageClassifier.load(model_name, w).to(device)
        params_m = count_params(model) / 1e6
        lat1 = measure_latency(model, device, 1, args.warmup, args.iters)
        thr = measure_latency(model, device, args.throughput_batch, args.warmup, args.iters)
        post_us = measure_postproc(model, device, args.throughput_batch, args.iters)
        tent_x = measure_tent_overhead(model, device, args.throughput_batch, max(20, args.iters // 2))
        rows.append({
            "model": model_name, "params_millions": round(params_m, 2),
            "latency_b1_median_ms": round(lat1["median_ms"], 2),
            "latency_b1_p95_ms": round(lat1["p95_ms"], 2),
            "throughput_b{}_per_s".format(args.throughput_batch): round(thr["throughput_per_s"], 1),
            "cal_triage_overhead_us_per_patch": round(post_us, 2),
            "tent_overhead_x": round(tent_x, 2),
        })
        print(f"[compute_cost] {model_name}: {params_m:.2f}M params, "
              f"{lat1['median_ms']:.2f} ms/patch, TENT {tent_x:.1f}x forward")

    if not rows:
        print("[compute_cost] no checkpoints found."); return

    print("\n=== Deployment compute cost (classifier stage) ===")
    hdr = (f"{'model':<16}{'params(M)':>10}{'lat b1 (ms)':>13}{'P95 (ms)':>10}"
           f"{'thr/s':>9}{'cal+triage(us)':>16}{'TENT (x fwd)':>14}")
    print(hdr); print("-" * len(hdr))
    tb = args.throughput_batch
    for r in rows:
        print(f"{r['model']:<16}{r['params_millions']:>10.2f}"
              f"{r['latency_b1_median_ms']:>13.2f}{r['latency_b1_p95_ms']:>10.2f}"
              f"{r[f'throughput_b{tb}_per_s']:>9.1f}"
              f"{r['cal_triage_overhead_us_per_patch']:>16.2f}{r['tent_overhead_x']:>14.2f}")

    out = Path(OUTPUT_DIR) / "uncertainty" / "compute_cost_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "throughput_batch": tb, "rows": rows}, indent=2))
    print(f"\n[compute_cost] wrote {out}")
    print("Notes: calibration (temperature) and triage add only the per-patch microseconds shown")
    print("       (a division and a sort) on top of one forward pass -> effectively free.")
    print("       Few-shot adds a one-time fine-tune but ZERO added inference cost.")
    print("       TENT multiplies per-batch inference cost by the shown factor (forward+backward).")


if __name__ == "__main__":
    main()
