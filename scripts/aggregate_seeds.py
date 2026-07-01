#!/usr/bin/env python3
"""
Aggregate the uncertainty report JSONs over a set of seeds and print paste-ready
LaTeX table bodies (Tables I-IV) plus the seed-dependent in-text numbers.

The report scripts (calib_report, selective_report, adapt_report,
external_validity_report) already write one entry per seed (or a pre-aggregated
block). This script averages those over whatever seeds are present, or over a
subset given with --seeds, and prints the table rows exactly in the paper format.

Usage:
    python scripts/aggregate_seeds.py --unc-dir results/uncertainty
    python scripts/aggregate_seeds.py --unc-dir results/uncertainty --seeds 42 7 123
"""
import argparse, json, statistics as st
from pathlib import Path

MODELS = [("efficientnet_b0", "EfficientNet-B0"),
          ("resnet34", "ResNet-34"),
          ("deit_tiny", "DeiT-Tiny")]

def f3(x): return f"{x:.3f}"
def dot(x):                      # 0.05 -> .05  (leading zero stripped, 2 dp)
    s = f"{x:.2f}"
    return s[1:] if s.startswith("0.") else s

def load(unc, name):
    p = Path(unc) / name
    return json.load(open(p)) if p.exists() else None

def by_model_seed(entries, seeds):
    out = {}
    for e in entries:
        if seeds and e["seed"] not in seeds: continue
        out.setdefault(e["model"], []).append(e)
    return out

def mean(xs): return sum(xs) / len(xs)

# ---------------- Table I: calibration ----------------
def table_calib(unc, seeds):
    data = load(unc, "calibration_report.json")
    if not data: return "% calibration_report.json not found"
    g = by_model_seed(data, seeds)
    lines = []
    for key, disp in MODELS:
        es = g.get(key, [])
        if not es: continue
        def avg(row, field): return mean([e["rows"][row][field] for e in es])
        def avgci(row): return (mean([e["rows"][row]["ece_ci_low"] for e in es]),
                                 mean([e["rows"][row]["ece_ci_high"] for e in es]))
        rows = [("in-domain", "xbd_val_4class", True),
                ("cross raw", "tebde_raw", False),
                ("~+source-$T$", "tebde_source_T", False),
                ("~+target-$T$", "tebde_target_T", False)]
        for i, (label, rk, indom) in enumerate(rows):
            lo, hi = avgci(rk)
            nll = "--" if indom else f3(avg(rk, "nll"))
            mcol = disp if i == 0 else ""
            lines.append(f"{mcol:<15} & {label:<14} & {f3(avg(rk,'accuracy'))} & "
                         f"{f3(avg(rk,'ece'))} & [{dot(lo)},{dot(hi)}] & {nll} \\\\")
        lines.append("\\hline")
    return "\n".join(lines)

# ---------------- Table II: triage curve ----------------
def table_triage(unc, seeds):
    data = load(unc, "selective_report.json")
    if not data: return "% selective_report.json not found"
    g = by_model_seed(data, seeds)
    lines = []
    for key, disp in MODELS:
        es = g.get(key, [])
        if not es: continue
        def avgvec(name):
            mats = [e["collapsed_miss"][name] for e in es]
            return [mean([m[i] for m in mats]) for i in range(len(mats[0]))]
        co, ca = avgvec("confidence_only"), avgvec("collapse_aware")
        lines.append(f"{disp:<15} & conf-only      & " + " & ".join(f3(x) for x in co) + " \\\\")
        lines.append(f"{'':<15} & collapse-aware & " + " & ".join(f3(x) for x in ca) + " \\\\")
        lines.append("\\hline")
    return "\n".join(lines)

# ---------------- Table III: test-time adaptation ----------------
def table_tta(unc, seeds):
    data = load(unc, "adapt_report.json")
    if not data: return "% adapt_report.json not found"
    g = by_model_seed(data, seeds)
    lines = []
    order = [("raw", "raw"), ("~+target-$T$", "target_T"),
             ("BN-adapt", "bn_adapt"), ("TENT", "tent")]
    for key, disp in MODELS:
        es = g.get(key, [])
        if not es: continue
        for i, (label, rk) in enumerate(order):
            if rk not in es[0]["rows"]:        # DeiT has no bn_adapt
                continue
            def avg(field): return mean([e["rows"][rk][field] for e in es])
            mcol = disp if i == 0 else ""
            lines.append(f"{mcol:<15} & {label:<12} & {f3(avg('accuracy'))} & "
                         f"{f3(avg('ece'))} & {f3(avg('nll'))} & {f3(avg('collapsed_recall'))} \\\\")
        lines.append("\\hline")
    return "\n".join(lines)

# ---------------- Table IV: external validity (pre-aggregated) ----------------
def table_external(unc, seeds):
    data = load(unc, "external_validity_report.json")
    if not data: return "% external_validity_report.json not found"
    lines = []
    for key, disp in MODELS:
        if key not in data: continue
        for i, (tgt, tdisp) in enumerate([("tebde", "TEBDE"), ("rescuenet", "RescueNet")]):
            b = data[key][tgt]
            cal, tri = b["calibration"], b["triage"]
            j = tri["budgets"].index(0.5)
            mcol = disp if i == 0 else ""
            lines.append(f"{mcol:<15} & {tdisp:<9} & {f3(cal['raw']['ece'])} & "
                         f"{f3(cal['target_T']['ece'])} & {f3(cal['raw']['accuracy'])} & "
                         f"{f3(tri['conf_only'][j])} & {f3(tri['collapse_aware'][j])} \\\\")
        lines.append("\\hline")
    return "\n".join(lines)

# ---------------- seed-dependent in-text numbers ----------------
def intext(unc, seeds):
    out = []
    sel = load(unc, "selective_report.json")
    if sel:
        es = [e for e in sel if e["model"] == "efficientnet_b0" and (not seeds or e["seed"] in seeds)]
        if es:
            out.append(f"AURC EfficientNet-B0: raw {mean([e['risk_coverage']['aurc_raw'] for e in es]):.2f} "
                       f"vs cal {mean([e['risk_coverage']['aurc_cal'] for e in es]):.2f}")
    ens = load(unc, "ensemble_report.json")
    if ens and "cross_domain" in ens and "efficientnet_b0" in ens["cross_domain"]:
        e = ens["cross_domain"]["efficientnet_b0"]
        out.append(f"Ensemble EffNet: acc {e['single_target_T']['accuracy']:.3f}->{e['ensemble_calibrated']['accuracy']:.3f}, "
                   f"ECE {e['single_target_T']['ece']:.3f}->{e['ensemble_calibrated']['ece']:.3f}")
        out.append(f"Raw collapsed recall (EffNet): {e['single_raw']['collapsed_recall']:.3f}")
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unc-dir", default="results/uncertainty")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="subset of seeds to average; default = all present")
    a = ap.parse_args()
    seeds = set(a.seeds) if a.seeds else None
    tag = ("seeds " + ",".join(map(str, sorted(seeds)))) if seeds else "all seeds present"
    print(f"% ===== aggregated over {tag} =====")
    for title, fn in [("TABLE I  (calibration)", table_calib),
                      ("TABLE II (collapse-aware triage)", table_triage),
                      ("TABLE III(test-time adaptation)", table_tta),
                      ("TABLE IV (external validity)", table_external)]:
        print(f"\n% ---- {title} ----")
        print(fn(a.unc_dir, seeds))
    print("\n% ---- in-text numbers ----")
    print(intext(a.unc_dir, seeds))

if __name__ == "__main__":
    main()
