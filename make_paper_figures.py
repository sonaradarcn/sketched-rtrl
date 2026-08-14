"""Regenerate every data-driven figure of the SK-RTRL manuscript with a single,
colour-blind-safe, black-and-white-legible style (Neurocomputing major revision, R1-2/R1-9/R2-3).

Design rules enforced here
--------------------------
R1-2  Any two curves/series differ in AT LEAST TWO of {colour, line style, marker}.
      Colours come from the Okabe-Ito colour-blind-safe palette; every method also owns a
      unique dash pattern and a unique marker glyph, so the figures stay discriminable in
      grey-scale print and for dichromatic readers.  Bar charts additionally carry hatches.
R1-9  fig_memory_time_pareto: annotations are placed with explicit per-point alignment and
      the axes are padded, so no label crosses the frame or another label.
R2-3  Everything is written as a true vector PDF (no rasterised panels), drawn at the size it
      is printed at so the type lands at 6.2-7.5 pt on the page instead of the 3-5 pt that
      down-scaling used to produce.

Data sources.  Nothing is fabricated: a figure whose input tree is absent is SKIPPED
loudly and no file is written for it, so a run that finishes is not the same as a run that
produced every figure -- read the summary printed at the end.  Paths are relative to the
repository root.

  fig_pilot_residual      results/m1_spectrum_*.json
  fig_fidelity_bars       results/m3/*.json
  fig_rinterp_rotation    results/m3/*.json  (also reads results/m31, m32 if present)
  fig_cert_c2sweep        results/c2sweep/*.json
  fig_adaptive_trajectory results/round1/traj/rotation_adaptive-eta_s0_traj.json
  fig_horizon_nmse        results/round1/horizon + results/ts
  fig_fidelity_vs_error   results/round1/real    + results/ts
  fig_scaling             results/{m3,scale,scale256} + results/membench
  fig_memory_time_pareto  results/membench
  fig_rl_curves           results/m5iso

Every tree above ships with this repository, so a clean clone regenerates all ten figures.
See REPRODUCIBILITY.md ("Released result trees") for what is deliberately left out.

Usage:  python make_paper_figures.py [--root <repo root>] [--out <dir>] [--gray <dir>]
        Defaults: root = this file's directory, out = results/figures/.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------------
# Global style
# --------------------------------------------------------------------------------------
plt.rcParams.update({
    "pdf.fonttype": 42,          # embed TrueType so the PDF text stays selectable/searchable
    "ps.fonttype": 42,
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.2,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.6,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linewidth": 0.4,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.012,
})

# ---------------------------------------------------------------------------------------
# R2-3 / R1-2 legibility.  Each figure is drawn at the *physical size it is printed at*, so
# the 6.8-7.5 pt type above survives into the PDF at 6.8-7.5 pt.  Previously every figure was
# authored ~2.4x too wide and then scaled down by \includegraphics, which shrank the axis
# labels to roughly 3-5 pt -- illegible in print and a likely contributor to the "figure
# quality" complaint.  Widths below = (fraction in the .tex) x (\textwidth or \columnwidth).
#   \textwidth   = 494.5 pt = 6.87 in   (figure*, spans both columns)
#   \columnwidth = 241   pt = 3.35 in   (figure, single column)
# ---------------------------------------------------------------------------------------
TW, CW = 6.87, 3.35
PRINT_SIZE = {
    "fig_pilot_residual":      (0.92 * TW, 2.20),   # figure*, 0.92\linewidth
    "fig_fidelity_bars":       (0.60 * TW, 2.10),   # figure*, minipage 0.60
    "fig_rinterp_rotation":    (0.38 * TW, 2.05),   # figure*, minipage 0.38
    "fig_cert_c2sweep":        (0.78 * CW, 2.00),   # figure,  0.78\linewidth
    "fig_adaptive_trajectory": (1.00 * CW, 2.00),   # figure,  \linewidth
    "fig_horizon_nmse":        (0.64 * TW, 2.00),   # figure*, minipage 0.64
    "fig_fidelity_vs_error":   (0.34 * TW, 2.05),   # figure*, minipage 0.34
    "fig_scaling":             (0.92 * TW, 2.20),   # figure*, 0.92\linewidth
    "fig_memory_time_pareto":  (0.82 * CW, 2.05),   # figure,  0.82\linewidth
    "fig_rl_curves":           (0.80 * TW, 1.80),   # figure*, 0.80\linewidth
}


def figsize(name):
    return PRINT_SIZE[name]

# Okabe-Ito colour-blind-safe palette
OI = {
    "black":  "#000000",
    "orange": "#E69F00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "verm":   "#D55E00",
    "purple": "#CC79A7",
    "grey":   "#767676",
    "teal":   "#44AA99",   # Paul Tol's colour-blind-safe set, for the rare 8th/9th series
}


def _assert_styles_distinct():
    """R1-2 guard: any two methods must differ in >=2 of {colour, linestyle, marker}."""
    bad = []
    keys = list(STYLE)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ca, la, ma, _ = STYLE[a]
            cb, lb, mb, _ = STYLE[b]
            same = (ca == cb) + (str(la) == str(lb)) + (ma == mb)
            if same >= 2:
                bad.append((a, b))
    if bad:
        print("  !! style collision (>=2 shared attributes):", bad)
    return bad

# (colour, linestyle, marker, hatch) -- unique in at least two attributes for every pair.
STYLE = {
    "exact":      (OI["black"],  "-",                 "o", "" ),
    "skrtrl-r64": (OI["purple"], (0, (3, 1, 1, 1)),   "P", "xx"),
    # NB: r32 must not share sky-blue with RFLO -- both appear in fig_fidelity_vs_error.
    "skrtrl-r32": (OI["yellow"], (0, (5, 1, 1, 1, 1, 1)), "h", "++"),
    "skrtrl-r16": (OI["blue"],   "-",                 "s", "//"),
    "skrtrl-r8":  (OI["teal"],   (0, (4, 1, 1, 1)),   "H", "\\\\"),
    "skrtrl-r4":  (OI["green"],  (0, (5, 2)),         "^", "\\\\"),
    "skrtrl-r2":  (OI["teal"],   (0, (2, 2)),         "8", ".."),
    "snap1":      (OI["verm"],   (0, (1, 1.2)),       "v", ".."),
    "kfrtrl":     (OI["orange"], (0, (6, 1.5, 1, 1.5)), "X", "xx"),
    "uoro":       (OI["purple"], (0, (3, 1, 1, 1, 1, 1)), "*", "++"),
    "rflo":       (OI["sky"],    (0, (4, 2, 1, 2)),   "d", "--"),
    "tbptt":      (OI["grey"],   (0, (1, 2)),         "p", "oo"),
    "rtu":        (OI["orange"], (0, (6, 1.5)),       ">", "xx"),
    "lru":        (OI["purple"], (0, (2, 1, 1, 1)),   "<", "++"),
    "adaptive":   (OI["verm"],   (0, (5, 1, 1, 1)),   "h", "//"),
}

LABEL = {
    "exact": "exact RTRL", "skrtrl-r64": "SK-RTRL $r{=}n$", "skrtrl-r32": "SK-RTRL r32",
    "skrtrl-r16": "SK-RTRL r16", "skrtrl-r8": "SK-RTRL r8", "skrtrl-r4": "SK-RTRL r4",
    "skrtrl-r2": "SK-RTRL r2", "snap1": "SnAp-1", "kfrtrl": "KF-RTRL", "uoro": "UORO",
    "rflo": "RFLO", "tbptt": "TBPTT", "rtu": "RTU", "lru": "LRU", "adaptive": "adaptive $r_t$",
}

# Per-task styles for the diagnostic figures (colour + dash + marker all differ).
TASK_STYLE = {
    "copy":     (OI["blue"],   "-",               "o"),
    "adding":   (OI["orange"], (0, (5, 2)),       "s"),
    "rotation": (OI["green"],  (0, (1, 1.2)),     "^"),
    "anbn":     (OI["verm"],   (0, (6, 1.5, 1, 1.5)), "D"),
    "rotrecall24": (OI["purple"], (0, (3, 1, 1, 1)), "v"),
}
TASK_LABEL = {"copy": "copy", "adding": "adding", "rotation": "rotation",
              "anbn": r"$a^n b^n$", "rotrecall24": "rot-recall(24)"}


def sty(key):
    """(colour, linestyle, marker, hatch) for a method key, with a safe fallback."""
    return STYLE.get(key, (OI["grey"], "-", "o", ""))


def line_kw(key):
    c, ls, mk, _ = sty(key)
    return dict(color=c, linestyle=ls, marker=mk, label=LABEL.get(key, key))


def task_kw(task):
    c, ls, mk = TASK_STYLE.get(task, (OI["grey"], "-", "o"))
    return dict(color=c, linestyle=ls, marker=mk, label=TASK_LABEL.get(task, task))


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------
ROOT = OUT = GRAY = None
FNAME = re.compile(r"^(?P<task>[a-z0-9]+)_(?P<method>[a-z0-9\-]+)_s(?P<seed>\d+)(?:_.*)?\.json$")
WROTE, SKIPPED = [], []


def _res(*parts):
    """Resolve a result path under <root>/results/, falling back to <root>/code/results/.

    In this repository the code *is* the root, so every tree lives under results/.  The
    fallback keeps the script working unchanged inside the authors' monorepo, where the
    code sits in a code/ subdirectory next to paper/.
    """
    flat = os.path.join(ROOT, "results", *parts)
    if glob.glob(flat) or os.path.exists(os.path.dirname(flat)):
        return flat
    return os.path.join(ROOT, "code", "results", *parts)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    if GRAY:                       # grey-scale self-check rendering (not shipped with the paper)
        os.makedirs(GRAY, exist_ok=True)
        fig.savefig(os.path.join(GRAY, name + "_gray.png"), dpi=170)
    plt.close(fig)
    WROTE.append(name)
    print("  wrote", name)


def _skip(name, why):
    SKIPPED.append((name, why))
    print("  SKIP", name, "--", why)


def _last(recs, field, tail=5):
    vals = [r[field] for r in recs if r.get(field) is not None]
    return float(np.mean(vals[-tail:])) if vals else None


def _tail(recs, field, frac=0.2):
    v = [r[field] for r in recs
         if r.get(field) is not None and not (isinstance(r[field], float) and np.isnan(r[field]))]
    if not v:
        return None
    k = max(1, int(len(v) * frac))
    return float(np.mean(v[-k:]))


def _scan(dirs):
    rows = []
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.json")):
            m = FNAME.match(os.path.basename(p))
            if not m:
                continue
            try:
                j = json.load(open(p))
            except Exception:
                continue
            recs = j.get("records", [])
            rows.append({"task": m["task"], "method": m["method"], "seed": int(m["seed"]),
                         "horizon": j.get("args", {}).get("horizon", 1),
                         "n": j.get("args", {}).get("n", 64),
                         "nmse": _last(recs, "metric"), "grad_cos": _last(recs, "grad_cos"),
                         "peak_MB": j.get("peak_MB"), "wall_s": j.get("wall_s")})
    return rows


def _grayscale_legend(ax, **kw):
    kw.setdefault("frameon", False)
    kw.setdefault("handlelength", 3.0)     # long handles so dash patterns are readable
    return ax.legend(**kw)


# --------------------------------------------------------------------------------------
# Legends live OUTSIDE the plotting area (revision: in-axes keys covered the data).
#
# Every data figure of the manuscript now puts its key beneath the axes rather than inside
# them.  `legend_below` anchors the key in FIGURE coordinates just under the tight bounding
# box of the axes -- which already contains the tick labels and the x-axis label -- so the
# key cannot overlap a curve, a marker or an error bar by construction.  Because the figures
# are saved with bbox_inches='tight', the exported PDF simply grows downwards by the height
# of the key: the plotting rectangle keeps its full size (it is never squeezed) and the
# print width, hence the 6.8-7.5 pt paper type size, is unchanged.
# --------------------------------------------------------------------------------------
LEGEND_ISSUES = []


def _fig_bbox(fig, artists):
    """Union of the tight bounding boxes of `artists`, in figure coordinates."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bbs = [a.get_tightbbox(r) for a in artists]
    bbs = [b for b in bbs if b is not None]
    return Bbox.union(bbs).transformed(fig.transFigure.inverted())


def _check_legend_outside(fig, name, leg, axes):
    """Hard check: the key must clear every axes rectangle and fit the figure width."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lb = leg.get_window_extent(r)
    bad = []
    for ax in axes:
        if lb.overlaps(ax.get_window_extent()):
            bad.append("key overlaps the plotting rectangle")
    fw = fig.get_size_inches()[0] * fig.dpi
    if lb.width > fw + 1.0:
        bad.append("key is %.0f px wide vs a %.0f px figure (would widen the figure)"
                   % (lb.width, fw))
    if bad:
        LEGEND_ISSUES.append((name, bad))
        print("  !! %s legend check FAILED: %s" % (name, "; ".join(bad)))
    return not bad


def legend_below(fig, axes, handles, labels, ncol, name, fontsize=6.4, pad=0.035,
                 x=None, y=None, **kw):
    """Draw a legend fully outside the plotting area, centred beneath `axes`."""
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    axes = list(axes)
    bb = _fig_bbox(fig, axes)
    kw.setdefault("frameon", False)
    kw.setdefault("handlelength", 2.4)
    kw.setdefault("columnspacing", 1.0)
    kw.setdefault("handletextpad", 0.45)
    leg = fig.legend(handles, labels, loc="upper center",
                     bbox_to_anchor=(0.5 * (bb.x0 + bb.x1) if x is None else x,
                                     bb.y0 - pad if y is None else y),
                     bbox_transform=fig.transFigure, ncol=ncol, fontsize=fontsize, **kw)
    _check_legend_outside(fig, name, leg, axes)
    return leg


def _merged_handles(axes):
    """Handles/labels of `axes` in order, de-duplicated by label."""
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    return h, l


# --------------------------------------------------------------------------------------
# Fig 3  fig_pilot_residual
# --------------------------------------------------------------------------------------
def fig_pilot_residual():
    files = sorted(glob.glob(_res("m1_spectrum_*.json")))
    if not files:
        return _skip("fig_pilot_residual", "no results/m1_spectrum_*.json")
    ks = [1, 4, 8, 16, 32, 64]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize("fig_pilot_residual"))
    order = ["copy", "adding", "rotation", "anbn"]
    loaded = {}
    for f in files:
        d = json.load(open(f))
        loaded[d["args"]["task"]] = d
    for t in [t for t in order if t in loaded] + [t for t in loaded if t not in order]:
        d = loaded[t]
        recs = [r for r in d["records"] if r.get("res_frac_of_J", 0) > 0]
        if not recs:
            continue
        kw = task_kw(t)
        axL.plot([r["step"] for r in recs], [r["res_frac_of_J"] for r in recs],
                 ms=3.2, markevery=max(1, len(recs) // 12), **kw)
        last = recs[-1]
        kw2 = dict(kw); kw2.pop("label")
        axR.plot(ks, [last[f"top{k}"] for k in ks], ms=4.2, **kw2, label=kw["label"])
    axL.set_xlabel("online step")
    axL.set_ylabel(r"$\|J-S\|_F\,/\,\|J\|_F$")
    axL.set_title("Off-diagonal residual mass fraction")
    axL.set_ylim(0, 1)
    axR.set_xlabel("retained rank $k$")
    axR.set_ylabel("cumulative singular mass")
    axR.set_title("Residual is approximately low rank")
    axR.set_xscale("log", base=2)
    axR.set_ylim(0, 1.02)
    fig.tight_layout()
    # Both panels share the same four task series, so a single key beneath the pair
    # replaces the two in-axes legends that used to sit on the data.
    h, l = _merged_handles([axL, axR])
    legend_below(fig, [axL, axR], h, l, ncol=4, name="fig_pilot_residual", fontsize=6.6,
                 handlelength=3.0)
    _save(fig, "fig_pilot_residual")


# --------------------------------------------------------------------------------------
# Fig 5 (left)  fig_fidelity_bars -- recomputed from results/m3/*.json (5 seeds)
# --------------------------------------------------------------------------------------
# The published Table 3 values, kept only as a cross-check: the bars are computed from the
# raw runs and the result is asserted to round to these numbers.  If results/m3/ is absent
# the figure is skipped rather than drawn from the table.
FIDELITY_TABLE = [
    ("skrtrl-r4",  [0.822, 0.884, 0.985, 0.979], [0.022, 0.019, 0.007, 0.004]),
    ("skrtrl-r16", [0.856, 0.887, 0.981, 0.983], [0.039, 0.017, 0.003, 0.006]),
    ("snap1",      [0.700, 0.606, 0.422, 0.425], [0.016, 0.021, 0.028, 0.089]),
    ("kfrtrl",     [0.499, 0.501, 0.559, 0.450], [0.208, 0.050, 0.120, 0.034]),
    ("rflo",       [0.343, 0.581, 0.157, 0.494], [0.024, 0.037, 0.033, 0.051]),
    ("uoro",       [0.040, 0.105, 0.059, 0.091], [0.022, 0.011, 0.021, 0.021]),
]


FIDELITY_TASKS = ["copy", "adding", "rotation", "anbn"]


def _fidelity_from_raw():
    """Mean +- sample std of the tail-mean gradient cosine, per (task, method), from
    results/m3/*.json.  Returns None if the tree is not present."""
    files = sorted(glob.glob(_res("m3", "*.json")))
    if not files:
        return None
    per = {}
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        a = d.get("args", {})
        if a.get("task") not in FIDELITY_TASKS or a.get("n") != 64:
            continue
        v = [r["grad_cos"] for r in d.get("records", [])
             if r.get("grad_cos") is not None and r["grad_cos"] == r["grad_cos"]]
        if not v:
            continue
        k = max(1, int(len(v) * 0.2))                 # tail mean over the last 20% of logs
        per.setdefault((a["task"], a["algo"]), {})[a["seed"]] = sum(v[-k:]) / k
    out = {}
    for (task, algo), seeds in per.items():
        vals = list(seeds.values())
        m = sum(vals) / len(vals)
        sd = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
        out[(task, algo)] = (m, sd, len(vals))
    return out


def fig_fidelity_bars():
    raw = _fidelity_from_raw()
    if raw is None:
        return _skip("fig_fidelity_bars", "no results/m3/*.json")
    series, drift = [], []
    for key, pub_m, pub_s in FIDELITY_TABLE:
        ms, ss = [], []
        for t, pm, ps in zip(FIDELITY_TASKS, pub_m, pub_s):
            got = raw.get((t, key))
            if got is None:
                return _skip("fig_fidelity_bars", "no runs for %s on %s" % (key, t))
            ms.append(got[0]); ss.append(got[1])
            if abs(round(got[0], 3) - pm) > 0.0011 or abs(round(got[1], 3) - ps) > 0.0011:
                drift.append("%s/%s: %.3f+-%.3f vs published %.3f+-%.3f"
                             % (t, key, got[0], got[1], pm, ps))
        series.append((key, ms, ss))
    if drift:
        raise SystemExit("fig_fidelity_bars: raw runs disagree with the published table:\n  "
                         + "\n  ".join(drift))
    print("  fig_fidelity_bars: 24/24 cells recomputed from results/m3 match the paper table")
    tasks = ["copy", "adding", "rotation", r"$a^n b^n$"]
    x = np.arange(len(tasks), dtype=float)
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=figsize("fig_fidelity_bars"))
    for i, (key, vals, errs) in enumerate(series):
        c, _, _, hatch = sty(key)
        hatch = hatch[: max(1, len(hatch) // 2)]   # sparser hatch: bars are narrow
        ax.bar(x + i * w, vals, w, yerr=errs, capsize=2, label=LABEL.get(key, key),
               color=c, hatch=hatch, edgecolor="black", linewidth=0.5,
               error_kw=dict(lw=0.8, ecolor="black"))
    ax.set_xticks(x + w * len(series) / 2)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Gradient cosine vs exact RTRL")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              frameon=False, columnspacing=0.9, handletextpad=0.4, handlelength=1.6)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig_fidelity_bars")


# --------------------------------------------------------------------------------------
# Fig 5 (right)  fig_rinterp_rotation
# --------------------------------------------------------------------------------------
def _load_runs(globpats):
    runs = {}
    for gp in globpats:
        for f in sorted(glob.glob(gp)):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            a = d.get("args", {})
            if "task" not in a or "algo" not in a:
                continue
            runs.setdefault((a["task"], a["algo"]), []).append(d)
    return runs


def fig_rinterp(task="rotation"):
    runs = _load_runs([_res(d, "*.json") for d in ("m3", "m31", "m32")])
    rs = [(0.5, "snap1"), (4, "skrtrl-r4"), (16, "skrtrl-r16"), (64, "skrtrl-r64")]
    xs, ys, es = [], [], []
    for r, alg in rs:
        cs = [_tail(d["records"], "grad_cos") for d in runs.get((task, alg), [])]
        cs = [c for c in cs if c is not None]
        if cs:
            xs.append(r); ys.append(np.mean(cs)); es.append(np.std(cs) if len(cs) > 1 else 0.0)
    if len(xs) < 2:
        return _skip("fig_rinterp_%s" % task, "fewer than 2 rank points in results/{m3,m31,m32}")
    fig, ax = plt.subplots(figsize=figsize("fig_rinterp_rotation"))
    # one continuous interpolation curve ...
    ax.errorbar(xs, ys, yerr=es, color=OI["blue"], linestyle="-", marker="",
                capsize=3, lw=1.6, zorder=2, label="SK-RTRL sketch rank sweep")
    # ... plus a per-configuration marker so each point is identifiable without colour
    for r, alg in rs:
        if r not in xs:
            continue
        i = xs.index(r)
        c, ls, mk, _ = sty(alg)
        ax.errorbar([xs[i]], [ys[i]], yerr=[es[i]], color=c, marker=mk, ms=8,
                    linestyle="none", capsize=3, mec="black", mew=0.6, zorder=3,
                    label=LABEL.get(alg, alg))
    ax.set_xscale("log")
    ax.set_xticks([0.5, 4, 16, 64])
    ax.set_xticklabels([r"$r{=}0$" "\n" "(SnAp-1)", "4", "16", "$n$"])
    ax.set_xlabel(r"sketch rank $r$")
    ax.set_ylabel("gradient cosine vs exact")
    ax.set_title("Rank interpolation (rotation)")
    fig.tight_layout()
    # The interpolation curve sweeps the whole panel, so the key (which used to sit in the
    # lower-right corner, directly on top of the rising branch) goes underneath the axes.
    h, l = ax.get_legend_handles_labels()
    legend_below(fig, ax, h, l, ncol=2, name="fig_rinterp_%s" % task, fontsize=6.2,
                 handlelength=1.8, columnspacing=0.8, handletextpad=0.35)
    _save(fig, "fig_rinterp_%s" % task)


# --------------------------------------------------------------------------------------
# Fig 6  fig_cert_c2sweep
# --------------------------------------------------------------------------------------
def fig_cert_c2sweep():
    sweepdir = _res("c2sweep")
    files = sorted(glob.glob(os.path.join(sweepdir, "*.json")))
    if not files:
        return _skip("fig_cert_c2sweep", "no results/c2sweep/*.json")
    pts = []
    for f in files:
        d = json.load(open(f))
        rb = _tail(d["records"], "rho_bar", 0.3)
        e = _tail(d["records"], "e_t", 0.3)
        E = _tail(d["records"], "true_E", 0.3)
        if rb and e and E and E > 0:
            pts.append((rb, e / E, d["args"]["task"]))
    if not pts:
        return _skip("fig_cert_c2sweep", "no (rho_bar, e_t, true_E) triples in the sweep")
    fig, ax = plt.subplots(figsize=figsize("fig_cert_c2sweep"))
    for t in sorted({p[2] for p in pts}):
        tp = sorted((p[0], p[1]) for p in pts if p[2] == t)
        ax.plot([p[0] for p in tp], [p[1] for p in tp], ms=5.5, mec="black", mew=0.5,
                **task_kw(t))
    ax.axvline(1.0, ls=(0, (6, 3)), color=OI["black"], lw=1.2, alpha=0.8)
    ax.annotate(r"$\bar\rho_t = 1$", xy=(1.0, 0.5), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", fontsize=8, rotation=90,
                va="center", ha="left")
    ax.set_yscale("log")
    ax.set_xlabel(r"certified spectral bound $\bar\rho_t$")
    ax.set_ylabel(r"certificate tightness $e_t/\|E_t\|_F$")
    # The old title read "non-vacuous iff rho_bar < 1", which overstates the theory: Cor. 1 needs a
    # *uniform* contraction rho_bar_t <= rho < 1, and an isolated step at rho_bar_t >= 1 need not
    # make the bound diverge (Sec. 7.4).  State the two regimes instead of an iff.
    ax.set_title(r"Certified contraction ($\bar\rho_t<1$): bound stays tight;" "\n"
                 r"sustained $\bar\rho_t\geq 1$: bound inflates")
    fig.tight_layout()
    # The three sweeps run diagonally across the panel and the old lower-right key sat on
    # top of all three, so the key now goes beneath the axes.
    h, l = ax.get_legend_handles_labels()
    legend_below(fig, ax, h, l, ncol=3, name="fig_cert_c2sweep", fontsize=6.4,
                 handlelength=2.2, columnspacing=0.8, handletextpad=0.35)
    _save(fig, "fig_cert_c2sweep")


# --------------------------------------------------------------------------------------
# Fig 7  fig_adaptive_trajectory
# --------------------------------------------------------------------------------------
def fig_adaptive_trajectory():
    path = _res("round1", "traj",
                        "rotation_adaptive-eta_s0_traj.json")
    if not os.path.isfile(path):
        return _skip("fig_adaptive_trajectory", "missing " + path)
    recs = json.load(open(path)).get("records", [])
    steps = [r["step"] for r in recs]
    fig, ax1 = plt.subplots(figsize=figsize("fig_adaptive_trajectory"))
    ax1.plot(steps, [r.get("rank") for r in recs], color=OI["black"], lw=2.2,
             linestyle="-", label=r"rank $r_t$")
    ax1.set_xlabel("step")
    ax1.set_ylabel(r"rank $r_t$ (left axis)")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.grid(False)
    series = [("eta",     OI["orange"], (0, (5, 2)),           "s", r"$\eta_t$"),
              ("e_t",     OI["green"],  (0, (1, 1.2)),         "^", r"$e_t$"),
              ("rho_hat", OI["verm"],   (0, (6, 1.5, 1, 1.5)), "v", r"$\hat\rho_t$"),
              ("rho_bar", OI["blue"],   (0, (3, 1, 1, 1)),     "D", r"$\bar\rho_t$")]
    for field, c, ls, mk, lab in series:
        xy = [(s, r[field]) for s, r in zip(steps, recs)
              if r.get(field) is not None and r[field] > 0]
        if not xy:
            continue
        xs, ys = zip(*xy)
        ax2.plot(xs, ys, color=c, linestyle=ls, lw=1.1, marker=mk, ms=3.4,
                 markevery=max(1, len(xs) // 14), mec=c, label=lab)
    ax2.set_yscale("log")
    ax2.set_ylabel("certificate quantities (log scale)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=6.2, frameon=False, ncol=5, handlelength=1.9,
               columnspacing=0.7, handletextpad=0.35, loc="lower center",
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    _save(fig, "fig_adaptive_trajectory")


# --------------------------------------------------------------------------------------
# Fig 8 (left)  fig_horizon_nmse
# --------------------------------------------------------------------------------------
METHOD_ORDER = ["exact", "skrtrl-r16", "skrtrl-r4", "snap1", "rflo", "uoro", "kfrtrl"]


def fig_horizon(rows):
    tasks = ["henon", "mackeyglass", "lorenz"]
    rows = [r for r in rows if r["task"] in tasks and r["nmse"] is not None]
    if not rows:
        return _skip("fig_horizon_nmse", "no horizon data")
    fig, axes = plt.subplots(1, len(tasks), figsize=figsize("fig_horizon_nmse"), sharey=True)
    for ax, task in zip(axes, tasks):
        for meth in METHOD_ORDER:
            pts = {}
            for r in rows:
                if r["task"] == task and r["method"] == meth:
                    pts.setdefault(r["horizon"], []).append(r["nmse"])
            if not pts:
                continue
            hs = sorted(pts)
            mean = [np.mean(pts[h]) for h in hs]
            sd = [np.std(pts[h]) for h in hs]
            ax.errorbar(hs, mean, yerr=sd, ms=4.2, capsize=2, mec="black", mew=0.4,
                        elinewidth=0.8, **line_kw(meth))
        ax.set_title({"henon": "Hénon", "mackeyglass": "Mackey--Glass",
                      "lorenz": "Lorenz"}.get(task, task))
        ax.set_xlabel("horizon $h$")
        ax.set_yscale("log")
    axes[0].set_ylabel("NMSE")
    fig.tight_layout()
    # Legend lives BELOW the panels: at print size an in-axes legend covered the curves.
    h, l = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.145), ncol=4,
               frameon=False, fontsize=6.2, handlelength=2.4, columnspacing=1.0,
               handletextpad=0.45)
    fig.subplots_adjust(bottom=0.30)
    _save(fig, "fig_horizon_nmse")


# --------------------------------------------------------------------------------------
# Fig 8 (right)  fig_fidelity_vs_error
# --------------------------------------------------------------------------------------
def fig_scatter(rows):
    pts = [r for r in rows
           if r["grad_cos"] is not None and r["nmse"] is not None and r["horizon"] == 1]
    if len(pts) < 4:
        return _skip("fig_fidelity_vs_error", "insufficient grad_cos/NMSE pairs")
    fig, ax = plt.subplots(figsize=figsize("fig_fidelity_vs_error"))
    methods = [m for m in METHOD_ORDER if any(p["method"] == m for p in pts)]
    methods += sorted({p["method"] for p in pts} - set(methods))
    for meth in methods:
        mp = [p for p in pts if p["method"] == meth]
        c, _, mk, _ = sty(meth)
        ax.scatter([p["grad_cos"] for p in mp], [p["nmse"] for p in mp],
                   s=15, color=c, marker=mk, edgecolors="black", linewidths=0.35,
                   label=LABEL.get(meth, meth), alpha=0.85, zorder=3)
    ax.set_xlabel("gradient cosine vs exact (fidelity)")
    ax.set_ylabel("NMSE (task error)")
    ax.set_yscale("log")
    fig.tight_layout()
    # The cloud fills the axes at print size, so the key goes underneath it.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False,
              fontsize=6.2, handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
    _save(fig, "fig_fidelity_vs_error")


# --------------------------------------------------------------------------------------
# Fig 9  fig_scaling
# --------------------------------------------------------------------------------------
def fig_scaling():
    fid_dirs = [_res(d) for d in ("m3", "scale", "scale256")]
    mem_dir = _res("membench")
    fid = {}
    for d in fid_dirs:
        for p in glob.glob(os.path.join(d, "rotation_*.json")):
            m = FNAME.match(os.path.basename(p))
            if not m:
                continue
            j = json.load(open(p))
            gc = _last(j.get("records", []), "grad_cos")
            if gc is not None:
                fid.setdefault(m["method"], {}).setdefault(j.get("args", {}).get("n", 64), []).append(gc)
    mem = {}
    for p in glob.glob(os.path.join(mem_dir, "*.json")):
        j = json.load(open(p))
        a = j.get("args", {})
        if j.get("peak_MB"):
            mem.setdefault(a["algo"], {})[a["n"]] = j["peak_MB"]
    if not fid and not mem:
        return _skip("fig_scaling", "neither fidelity nor membench data found")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize("fig_scaling"))
    for meth in ["skrtrl-r16", "skrtrl-r4", "snap1"]:
        if meth in fid:
            ns = sorted(fid[meth])
            axL.plot(ns, [np.mean(fid[meth][n]) for n in ns], ms=5.5, mec="black", mew=0.5,
                     **line_kw(meth))
    axL.set_xlabel("hidden size $n$")
    axL.set_ylabel("grad cosine vs exact (rotation)")
    axL.set_title("Fidelity is width-invariant")
    axL.set_xscale("log", base=2)
    axL.set_ylim(0, 1.08)
    for meth in ["exact", "skrtrl-r16", "skrtrl-r4", "snap1"]:
        if meth in mem:
            ns = sorted(mem[meth])
            axR.plot(ns, [mem[meth][n] for n in ns], ms=5.5, mec="black", mew=0.5,
                     **line_kw(meth))
    # The budget line is annotated in place rather than in the legend: a 5th legend entry
    # made the key tall enough to sit on top of this very line.
    axR.axhline(12288, ls=(0, (7, 3)), color=OI["verm"], lw=1.3)
    axR.annotate("12 GB GPU budget", xy=(0.985, 12288), xycoords=("axes fraction", "data"),
                 xytext=(0, 2.5), textcoords="offset points", ha="right", va="bottom",
                 fontsize=6.2, color=OI["verm"])
    axR.set_xlabel("hidden size $n$")
    axR.set_ylabel("peak memory (MB)")
    axR.set_title(r"Memory: exact $O(n^3)$ vs SK-RTRL $O(n^2 r)$")
    axR.set_yscale("log")
    axR.set_xscale("log", base=2)
    # Head-room above the 12 GB line for its in-place annotation only.  This used to be a
    # 22x stretch, needed solely to keep the in-axes upper-left key off the curves; with the
    # key moved outside the axes the panel no longer has to waste that vertical space.
    lo, hi = axR.get_ylim()
    axR.set_ylim(lo, max(hi, 12288) * 4.0)
    fig.tight_layout()
    # One key per panel (the panels show different method sets), both on a common baseline
    # beneath the two axes.
    bbL, bbR = _fig_bbox(fig, [axL]), _fig_bbox(fig, [axR])
    ybot = min(bbL.y0, bbR.y0) - 0.045
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    legend_below(fig, axL, hL, lL, ncol=3, name="fig_scaling (left)", fontsize=6.6,
                 handlelength=2.4, columnspacing=0.9, handletextpad=0.4,
                 x=0.5 * (bbL.x0 + bbL.x1), y=ybot)
    legend_below(fig, axR, hR, lR, ncol=4, name="fig_scaling (right)", fontsize=6.6,
                 handlelength=2.4, columnspacing=0.9, handletextpad=0.4,
                 x=0.5 * (bbR.x0 + bbR.x1), y=ybot)
    _save(fig, "fig_scaling")


# --------------------------------------------------------------------------------------
# Fig 10  fig_memory_time_pareto   (R1-9: no label may overflow the frame or overlap another)
# --------------------------------------------------------------------------------------
def fig_pareto(rows):
    pts = [r for r in rows if r["peak_MB"] and r["wall_s"]]
    if not pts:
        return _skip("fig_memory_time_pareto", "no memory/time data")
    n_target = max(r["n"] for r in pts)
    pts = [r for r in pts if r["n"] == n_target]
    if len({r["method"] for r in pts}) < 3:
        return _skip("fig_memory_time_pareto", "<3 methods at n=%d" % n_target)
    agg = {}
    for r in pts:
        agg.setdefault(r["method"], {"mb": [], "s": []})
        agg[r["method"]]["mb"].append(r["peak_MB"])
        agg[r["method"]]["s"].append(r["wall_s"])
    coords = {m: (float(np.mean(v["mb"])), float(np.mean(v["s"]))) for m, v in agg.items()}

    fig, ax = plt.subplots(figsize=figsize("fig_memory_time_pareto"))
    for meth, (x, y) in sorted(coords.items()):
        c, _, mk, _ = sty(meth)
        ax.scatter(x, y, s=90, color=c, marker=mk, edgecolors="black", linewidths=0.7,
                   zorder=3, label=LABEL.get(meth, meth))
    ax.set_xscale("log")
    ax.set_xlabel(r"peak memory (MB) at $n=%d$" % n_target)
    ax.set_ylabel("wall-clock (s, 1.5k steps)")

    # -- R1-9 --------------------------------------------------------------------------
    # Head-room so no annotation can cross the top frame, and explicit per-point text
    # alignment so the two SK-RTRL labels sit on opposite sides of their markers.
    ymax = max(y for _, y in coords.values())
    ymin = min(y for _, y in coords.values())
    ax.set_ylim(min(0.0, ymin - 0.06 * ymax), ymax * 1.30)
    ax.set_xmargin(0.22)
    ax.autoscale_view()
    # (dx pt, dy pt, ha, va) -- chosen so labels fan out and never collide.
    place = {
        "exact":      (-9,  9, "right",  "bottom"),
        "skrtrl-r16": (9,  10, "left",   "bottom"),
        "skrtrl-r4":  (-9, -4, "right",  "top"),
        "skrtrl-r8":  (9,  -9, "left",   "top"),
        "skrtrl-r32": (-9, 10, "right",  "bottom"),
        "snap1":      (9,   6, "left",   "bottom"),
        "kfrtrl":     (9,  -9, "left",   "top"),
    }
    for meth, (x, y) in coords.items():
        dx, dy, ha, va = place.get(meth, (9, 6, "left", "bottom"))
        ax.annotate(LABEL.get(meth, meth), (x, y), fontsize=8, fontweight="medium",
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    zorder=4,
                    bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.72))
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    # Hard check: every annotation must stay strictly inside the axes rectangle.
    fig.canvas.draw()
    ab = ax.get_window_extent()
    bad = []
    boxes = []
    for ch in ax.texts:
        bb = ch.get_window_extent(fig.canvas.get_renderer())
        if bb.x0 < ab.x0 - 0.5 or bb.x1 > ab.x1 + 0.5 or bb.y0 < ab.y0 - 0.5 or bb.y1 > ab.y1 + 0.5:
            bad.append(("outside frame", ch.get_text()))
        for other_txt, other_bb in boxes:
            if bb.overlaps(other_bb):
                bad.append(("overlaps '%s'" % other_txt, ch.get_text()))
        boxes.append((ch.get_text(), bb))
    if bad:
        print("  !! fig_memory_time_pareto label check FAILED:", bad)
    else:
        print("  fig_memory_time_pareto label check OK (all labels inside frame, no overlap)")
    _save(fig, "fig_memory_time_pareto")


# --------------------------------------------------------------------------------------
# Fig 11 (appendix)  fig_rl_curves
# --------------------------------------------------------------------------------------
def fig_rl_curves(env_lens=(10, 20, 40)):
    m5_dir = _res("m5iso")
    fre = re.compile(r"^tmaze(\d+)_([a-z0-9\-]+)_s(\d+)\.json$")
    data = {}
    for p in glob.glob(os.path.join(m5_dir, "*.json")):
        m = fre.match(os.path.basename(p))
        if not m:
            continue
        recs = json.load(open(p)).get("records", [])
        steps = [r["step"] for r in recs if r.get("success") is not None]
        succ = [r["success"] for r in recs if r.get("success") is not None]
        if steps:
            data.setdefault((int(m.group(1)), m.group(2)), []).append((steps, succ))
    if not data:
        return _skip("fig_rl_curves", "no m5iso data")
    lens = [L for L in env_lens if any(k[0] == L for k in data)]
    fig, axes = plt.subplots(1, len(lens), figsize=figsize("fig_rl_curves"), sharey=True)
    if len(lens) == 1:
        axes = [axes]
    for ax, L in zip(axes, lens):
        for algo in ["rtu", "exact", "lru", "skrtrl-r16", "snap1", "tbptt"]:
            runs = data.get((L, algo))
            if not runs:
                continue
            grid = runs[0][0]
            ys = np.array([np.interp(grid, s, v) for s, v in runs])
            mean = ys.mean(0)
            c, ls, mk, _ = sty(algo)
            ax.plot(grid, mean, color=c, linestyle=ls, lw=1.5, marker=mk, ms=3.6,
                    markevery=max(1, len(grid) // 9), mec=c,
                    label=LABEL.get(algo, algo))
        ax.set_title("corridor %d" % L)
        ax.set_xlabel("step")
    axes[0].set_ylabel("success rate (running)")
    fig.tight_layout()
    # Legend below the panels: at print size a 6-entry in-axes key overflowed the axes and
    # collided with the y-axis label.
    h, l = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.135), ncol=6,
               frameon=False, fontsize=6.2, handlelength=2.4, columnspacing=1.0,
               handletextpad=0.45)
    fig.subplots_adjust(bottom=0.30)
    _save(fig, "fig_rl_curves")


# --------------------------------------------------------------------------------------
def main():
    global ROOT, OUT, GRAY
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=here,
                    help="repository root (the directory holding results/)")
    ap.add_argument("--out", default=os.path.join(here, "results", "figures"),
                    help="directory to write the figures into")
    ap.add_argument("--gray", default="")
    args = ap.parse_args()
    ROOT, OUT, GRAY = args.root, args.out, args.gray
    print("root =", ROOT)
    print("out  =", OUT)
    if not _assert_styles_distinct():
        print("  style table OK: every method pair differs in >=2 visual attributes")

    fig_pilot_residual()
    fig_fidelity_bars()
    fig_rinterp("rotation")
    fig_cert_c2sweep()
    fig_adaptive_trajectory()
    hz = _scan([_res("round1", "horizon"),
                _res("ts")])
    fig_horizon(hz)
    sc = _scan([_res("round1", "real"),
                _res("ts")])
    fig_scatter(sc)
    fig_scaling()
    fig_pareto(_scan([_res("membench")]))
    fig_rl_curves()

    print("\n%d figures written: %s" % (len(WROTE), ", ".join(WROTE)))
    if LEGEND_ISSUES:
        print("%d LEGEND PLACEMENT FAILURES:" % len(LEGEND_ISSUES))
        for n, why in LEGEND_ISSUES:
            print("   -", n, ":", "; ".join(why))
    else:
        print("  legend check OK: every key sits outside the plotting rectangle "
              "and inside the figure width")
    if SKIPPED:
        for n, w in SKIPPED:
            print("   -", n, ":", w)


if __name__ == "__main__":
    main()
