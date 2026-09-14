"""Regenerate the two R1-era tables the W1 writing pass left behind: the certificate
tightness table (`tab:tightness`, Table 6) and the adaptive-versus-fixed rank table
(`tab:adaptive`, Table 9).

Both tables previously read narrow run sets: the tightness table came from the two-seed
`results/c2sweep` sub-sweep on a single task, and the adaptive table from a five-seed run that
predates the ceiling diagnosis.  This script rebuilds both over the full task set from the R2
trees:

  tab:tightness  <- results/r2/d2_clip   (9 tasks x clip {0.35,0.7,0.9} x 5 seeds = 135 runs)
  tab:adaptive   <- results/r2/d3_diag   (rotation and a^nb^n, n=64, clip 0.5, 3 seeds;
                                          the same-trajectory fixed-rank sweep plus the
                                          controller variants)

The aggregation conventions are the ones make_r2_d4_tables.py already uses for the D4
tables, so the three blocks of Section 6 can be read against each other: the per-run value
is the mean over the records logged in the last 20% of the run, the seed is the unit of
cross-run aggregation, and the printed spread is the population standard deviation.

Tightness in detail.  Theorem 1's certificate accumulates a geometric factor
1/(1 - rhobar) whenever the certified spectral bound rhobar_t stays below one, so the
table's claim is that the *measured* ratio T_t = e_t / ||E_t||_F tracks that factor.  The
comparison is only defined on a logged step that is (i) certified-contractive, rhobar_t < 1,
and (ii) carrying a non-zero true residual: on copy every contractive log point has
E_t exactly zero, because the task's episodic reset zeroes the sketch error at the step the
logger happens to visit, and 0/0 is not a tightness.  Those steps are counted and reported
rather than silently dropped.  We bin the surviving steps by rhobar_t, report the median
measured ratio per task family, and evaluate the prediction at the bin's pooled mean
rhobar.  The median is the statistic of record because the top bin has a heavy right tail
(a handful of steps just below rhobar = 1 dominate any mean); both are printed to the
console so the text can say so.

Usage:  python make_r2_w1b_tables.py [--root .] [--texdir ../paper/secs]
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

FAMILY = {"copy": "diagnostic", "adding": "diagnostic", "rotation": "diagnostic",
          "anbn": "diagnostic", "henon": "chaotic", "lorenz": "chaotic",
          "mackeyglass": "chaotic", "laser": "real", "sunspot": "real"}
TASK_ORDER = ["copy", "adding", "rotation", "anbn", "henon", "lorenz", "mackeyglass",
              "laser", "sunspot"]
FAM_ORDER = ["diagnostic", "chaotic", "real"]
# Bin edges on the certified spectral bound.  The earlier table indexed five nominal
# rhobar values (0.17 ... 0.80) produced by a clip sweep that targeted them directly; the R2
# clip sweep does not target rhobar, so we bin the realised values instead and print the
# bin's mean rhobar, which is what the geometric prediction has to be evaluated at.
RHO_EDGES = [0.0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
# Tasks whose logged `metric` is an accuracy; the tables report 1 - accuracy so that every
# error column in Section 6 is lower-is-better.
CE_TASKS = {"copy", "anbn"}


def tail_mean(recs, field, frac=0.2):
    """Mean of `field` over the last `frac` of the logged checkpoints.

    The rounding here (`round`, not truncation) is deliberately the one
    make_r2_d3_report.py uses, so that the printed cells of `tab:adaptive` reproduce
    D3_SUMMARY.md exactly rather than differing in the fourth decimal.
    """
    v = [r[field] for r in recs
         if r.get(field) is not None and isinstance(r[field], (int, float))
         and math.isfinite(r[field])]
    if not v:
        return None
    k = max(1, int(round(len(v) * frac)))
    return float(np.mean(v[-k:]))


def ms(vals, ddof=1):
    """Mean and sample s.d. over the seeds, None-tolerant.

    `ddof=1` matches make_r2_d3_report.py, whose numbers `tab:adaptive` has to agree
    with; the D4 tables use the population s.d. because they aggregate ten seeds, where
    the two conventions differ in the last printed digit at most.  With three seeds the
    sample estimate is the defensible one and the caption says which is printed.
    """
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    m = float(np.mean(v))
    s = float(np.std(v, ddof=ddof)) if len(v) > 1 else 0.0
    return m, s, len(v)


# --------------------------------------------------------------------------------------
# tab:tightness  --  certificate tightness in certified-contractive phases
# --------------------------------------------------------------------------------------
def collect_tightness(d2dir):
    pts, dropped_zero, dropped_rho, per_run = [], 0, 0, set()
    for p in sorted(glob.glob(os.path.join(d2dir, "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        a = d.get("args", {})
        task = a.get("task")
        if task not in FAMILY:
            continue
        per_run.add((task, a.get("clip"), a.get("seed")))
        for r in d.get("records", []):
            rb, e, tE = r.get("rho_bar"), r.get("e_t"), r.get("true_E")
            if rb is None or e is None or tE is None:
                continue
            if not (math.isfinite(rb) and math.isfinite(e) and math.isfinite(tE)):
                continue
            if rb >= 1.0:
                dropped_rho += 1
                continue
            if tE <= 1e-12 or e <= 0.0:
                dropped_zero += 1
                continue
            pts.append((task, FAMILY[task], a.get("clip"), a.get("seed"), rb, e / tE))
    return pts, dropped_zero, dropped_rho, len(per_run)


def tightness_rows(pts):
    rows = []
    for lo, hi in zip(RHO_EDGES[:-1], RHO_EDGES[1:]):
        sel = [x for x in pts if lo <= x[4] < hi]
        if not sel:
            continue
        rb = float(np.mean([x[4] for x in sel]))
        cell = {}
        for fam in FAM_ORDER + ["all"]:
            s = [x[5] for x in sel if fam == "all" or x[1] == fam]
            cell[fam] = (float(np.median(s)), float(np.mean(s)), len(s)) if s else None
        rows.append({"lo": lo, "hi": hi, "rhobar": rb, "pred": 1.0 / (1.0 - rb),
                     "n": len(sel), "cells": cell,
                     "tasks": sorted({x[0] for x in sel})})
    return rows


def write_tightness_tex(rows, meta, path):
    def fmt(c):
        return "$%.2f$" % c[0] if c else "---"
    L = []
    L.append("% generated by make_r2_w1b_tables.py from results/r2/d2_clip "
              "(certificate tightness)")
    L.append("%% Generated %s by code/make_r2_w1b_tables.py -- do not edit by hand; "
             "re-run the script." % time.strftime("%Y-%m-%d %H:%M:%S"))
    L.append("%% Inputs: results/r2/d2_clip (%d runs, %d contractive log points with a "
             "defined ratio)" % (meta["n_runs"], meta["n_pts"]))
    L.append("% Supersedes the earlier table, which was read from results/c2sweep "
             "(one task, two seeds).")
    L.append("")
    L.append(r"\begin{table}[t]")
    L.append(r"\centering\footnotesize")
    L.append(r"\setlength{\tabcolsep}{3.5pt}")
    L.append(
        r"\caption{\textbf{Certificate tightness in certified-contractive phases}, over the "
        r"full task set (the D2 clip sweep: \skrtrl{} $r{=}16$, nine tasks, "
        r"clip $\in\{0.35,0.7,0.9\}$, five seeds, $20{,}000$ steps, \Cref{tab:protocol}). "
        r"Each row pools the logged steps whose certified spectral bound $\rhobar_t$ lies in "
        r"the stated interval and reports the \emph{median} measured ratio "
        r"$e_t/\fnorm{E_t}$ against the geometric factor $1/(1-\rhobar)$ evaluated at the "
        r"row's pooled mean $\rhobar$; the reading of the top bin and the absent copy row are "
        r"discussed in \Cref{sec:exp-cert}.}")
    L.append(r"\label{tab:tightness}")
    L.append(r"\begin{tabular}{lcccccc}")
    L.append(r"\toprule")
    L.append(r" & & \multicolumn{4}{c}{measured $e_t/\fnorm{E_t}$ (median)} & \\")
    L.append(r"\cmidrule(lr){3-6}")
    L.append(r"$\rhobar_t$ range & $\rhobar$ & diag. & chaotic & real & all & "
             r"$1/(1-\rhobar)$ \\")
    L.append(r"\midrule")
    for r in rows:
        L.append(r"$[%.2f,%.2f)$ & $%.3f$ & %s & %s & %s & %s & $%.2f$ \\"
                 % (r["lo"], r["hi"], r["rhobar"],
                    fmt(r["cells"]["diagnostic"]), fmt(r["cells"]["chaotic"]),
                    fmt(r["cells"]["real"]), fmt(r["cells"]["all"]), r["pred"]))
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print("  wrote", path)


# --------------------------------------------------------------------------------------
# tab:adaptive  --  controller against the same-trajectory fixed-rank sweep
# --------------------------------------------------------------------------------------
FIXED_R = [4, 8, 16, 24, 32, 48, 64]
# (glob suffix, printed policy label) for the controller variants we print.  The remaining
# d3_diag variants (fixed append budget c=8, pre-projection off, n=128) are discussed in the
# text rather than tabulated: the first reproduces the default bitwise and the other two
# change the kernel rather than the rank policy.
ADAPTIVE = [("vA_default", r"\textbf{adaptive} $[4,32]$"),
            ("vB_rmax64_c8fp", r"\textbf{adaptive} $[4,64]$")]


def d3_cell(d3dir, pat, task, is_adaptive=False):
    files = sorted(glob.glob(os.path.join(d3dir, pat)))
    if not files:
        return None
    err, cos, rank, mb, cap, viol, checks, seeds = [], [], [], [], [], 0, 0, []
    for p in files:
        d = json.load(open(p))
        recs = d.get("records", [])
        m = tail_mean(recs, "metric")
        err.append(None if m is None else (1.0 - m if task in CE_TASKS else m))
        cos.append(tail_mean(recs, "grad_cos"))
        rank.append(d.get("avg_rank"))
        mb.append(d.get("peak_MB"))
        cap.append(d.get("frac_at_cap"))
        viol += int(d.get("cert_violations") or 0)
        checks += int(d.get("cert_checks_with_shadow") or 0)
        seeds.append(d.get("args", {}).get("seed"))
    # For a FIXED-rank row `frac_at_cap` is tautological -- a run pinned at r = r_max
    # reports 1.000 and one pinned below it 0.000, neither of which says anything about
    # the controller -- so it is only meaningful on the adaptive rows.
    return {"err": ms(err), "cos": ms(cos), "rank": ms(rank), "mb": ms(mb),
            "cap": ms(cap) if is_adaptive else (None, None, 0),
            "viol": viol, "checks": checks, "n": len(files),
            "seeds": sorted(s for s in seeds if s is not None)}


def collect_adaptive(d3dir):
    out = {}
    for task in ("rotation", "anbn"):
        rows = []
        for r in FIXED_R:
            c = d3_cell(d3dir, "%s_fixed%d_s*_sweep.json" % (task, r), task)
            if c:
                rows.append(("fixed $r{=}%d$" % r, c))
        for suf, lab in ADAPTIVE:
            c = d3_cell(d3dir, "%s_adaptive-eta_s*_%s.json" % (task, suf), task,
                        is_adaptive=True)
            if c:
                rows.append((lab, c))
        out[task] = rows
    return out


def write_adaptive_tex(data, path, printed_ranks=(4, 8, 16, 32, 64)):
    # Every cell of this grid has zero Theorem-1 violations, so a "viol." column would be
    # eighteen zeros; the count and its denominator go into the caption instead.
    tot_v = sum(c["viol"] for rs in data.values() for _, c in rs)
    tot_c = sum(c["checks"] for rs in data.values() for _, c in rs)
    def f(t, nd):
        return "---" if t is None or t[0] is None else (
            ("$%.*f$" % (nd, t[0])) if (t[1] or 0.0) < 10 ** (-nd) / 2
            else ("$%.*f\\pm%.*f$" % (nd, t[0], nd, t[1])))
    L = []
    L.append("% generated by make_r2_w1b_tables.py from results/r2/d3_diag "
              "(adaptive vs fixed rank)")
    L.append("%% Generated %s by code/make_r2_w1b_tables.py -- do not edit by hand; "
             "re-run the script." % time.strftime("%Y-%m-%d %H:%M:%S"))
    L.append("% Inputs: results/r2/d3_diag (n=64, clip 0.5, lr 1e-3, 12,000 steps, seeds 0-2)")
    L.append("% Supersedes the earlier table, which reported two fixed ranks and the "
             "controller only.")
    L.append("")
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering\small")
    L.append(
        r"\caption{\textbf{Certificate-safe adaptive rank against the same-trajectory "
        r"fixed-rank sweep} ($n{=}64$, clip $0.5$, learning rate $10^{-3}$, $12{,}000$ steps, "
        r"seeds $0$--$2$, mean $\pm$ sample s.d.). The fixed rows sweep $r$ along one "
        r"trajectory and the adaptive rows are the controller of \eqref{eq:controller} at "
        r"$[4,32]$ and at $[4,64]$, with %s Theorem-1 violations over the %s certificate "
        r"checks behind the whole sweep; the columns are defined in "
        r"\Cref{sec:exp-adaptive}.}"
        % (tot_v or "zero", "{:,}".format(tot_c).replace(",", "{,}")))
    L.append(r"\label{tab:adaptive}")
    L.append(r"\begin{tabular}{llccccc}")
    L.append(r"\toprule")
    L.append(r"Task & Policy & avg rank & at cap & peak MB & error & grad cosine \\")
    L.append(r"\midrule")
    for ti, (task, label) in enumerate([("rotation", "rotation"), ("anbn", "$a^nb^n$")]):
        rows = [x for x in data[task]
                if (not x[0].startswith("fixed"))
                or any(("$r{=}%d$" % r) in x[0] for r in printed_ranks)]
        if ti:
            L.append(r"\midrule")
        L.append(r"\multirow{%d}{*}{%s}" % (len(rows), label))
        for lab, c in rows:
            L.append(r" & %s & %s & %s & %s & %s & %s \\"
                     % (lab, f(c["rank"], 2), f(c["cap"], 3), f(c["mb"], 1),
                        f(c["err"], 4), f(c["cos"], 4)))
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")
    L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print("  wrote", path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=here)
    ap.add_argument("--texdir", default=os.path.join(here, "..", "paper", "secs"))
    args = ap.parse_args()
    d2 = os.path.join(args.root, "results", "r2", "d2_clip")
    d3 = os.path.join(args.root, "results", "r2", "d3_diag")
    texdir = os.path.abspath(args.texdir)

    print("tab:tightness  <-", d2)
    pts, dz, dr, nruns = collect_tightness(d2)
    rows = tightness_rows(pts)
    meta = {"n_runs": nruns, "n_pts": len(pts)}
    print("  %d runs, %d usable log points; dropped %d with rhobar>=1 and %d with a zero "
          "true residual" % (nruns, len(pts), dr, dz))
    print("  tasks contributing: %s" % ", ".join(sorted({p[0] for p in pts})))
    for r in rows:
        c = r["cells"]["all"]
        print("   [%.2f,%.2f)  n=%5d  rhobar=%.3f  median=%6.2f  mean=%8.2f  pred=%6.2f  "
              "tasks=%d" % (r["lo"], r["hi"], c[2], r["rhobar"], c[0], c[1], r["pred"],
                            len(r["tasks"])))
    write_tightness_tex(rows, meta, os.path.join(texdir, "tab_r2_tightness.tex"))

    print("tab:adaptive   <-", d3)
    data = collect_adaptive(d3)
    for task, rows in data.items():
        for lab, c in rows:
            print("   %-10s %-26s n=%d rank %s cap %s MB %s err %s cos %s viol %d/%d"
                  % (task, lab, c["n"],
                     "%.2f" % c["rank"][0] if c["rank"][0] is not None else "--",
                     "%.3f" % c["cap"][0] if c["cap"][0] is not None else "--",
                     "%.1f" % c["mb"][0] if c["mb"][0] is not None else "--",
                     "%.5f" % c["err"][0] if c["err"][0] is not None else "--",
                     "%.4f" % c["cos"][0] if c["cos"][0] is not None else "--",
                     c["viol"], c["checks"]))
    write_adaptive_tex(data, os.path.join(texdir, "tab_r2_adaptive.tex"))
    tot_v = sum(c["viol"] for rs in data.values() for _, c in rs)
    tot_c = sum(c["checks"] for rs in data.values() for _, c in rs)
    print("  D3 diagnosis totals: %d Theorem-1 violations over %d exact-shadow checks"
          % (tot_v, tot_c))


if __name__ == "__main__":
    main()
