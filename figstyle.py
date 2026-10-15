"""Shared figure style for the report scripts (make_r2_d1/d2/d3/rl_spectrum_report.py).

Okabe-Ito palette, one (colour, dash, marker, hatch) tuple per method, print-size figure
dimensions, and a legend placed outside the axes with a check that it does not overlap a panel.
This is the style section of the plotting script used for the manuscript figures, copied here
unchanged so that the figures can be drawn from a clone of this repository.
make_paper_figures.py carries its own copy of the same rcParams and tables.
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
# Each figure is drawn at the size it is printed at, so the 6.8-7.5 pt type above lands in the
# PDF at 6.8-7.5 pt.  Widths below = (fraction in the .tex) x (\textwidth or \columnwidth).
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
    """Any two methods must differ in >=2 of {colour, linestyle, marker}."""
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
# Legends live outside the plotting area.
#
# Every data figure puts its key beneath the axes.  `legend_below` anchors the key in FIGURE coordinates just under the tight bounding
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


def _check_legend_outside(fig, name, leg, axes, host):
    """Hard checks on an externalised key.

    1. it must clear every plotting rectangle (that is the whole point);
    2. it must stay inside the horizontal span of the axes it belongs to.  Because the
       figures are saved with bbox_inches='tight', a key that stuck out sideways would
       widen the exported PDF, and since the .tex fixes the printed width the whole figure
       would then be scaled down, shrinking the 6.8-7.5 pt type.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lb = leg.get_window_extent(r)
    hb = host.transformed(fig.transFigure)          # host span back to display coords
    bad = []
    for ax in axes:
        if lb.overlaps(ax.get_window_extent()):
            bad.append("key overlaps the plotting rectangle")
    if lb.x0 < hb.x0 - 1.0 or lb.x1 > hb.x1 + 1.0:
        bad.append("key spans %.0f-%.0f px outside the %.0f-%.0f px axes span "
                   "(would widen the exported figure)" % (lb.x0, lb.x1, hb.x0, hb.x1))
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
    _check_legend_outside(fig, name, leg, axes, bb)
    return leg


def _merged_handles(axes):
    """Handles/labels of `axes` in order, de-duplicated by label."""
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    return h, l

