"""The R1 figure style of the SK-RTRL manuscript, factored out so the second-round
figures can be drawn in exactly the style Figure 5 established.

Why this module exists
----------------------
The first-round revision fixed the "figure quality" complaint by adopting one rule above
all others (see the header of ``paper/figures/gen/regen_all.py``):

    **A figure is authored at the physical size it is printed at.**

Every R1 figure obeys it -- ``fig_fidelity_bars.pdf`` is 3.94 in wide and is printed at
0.60 x \\textwidth = 4.12 in, so its 7.5 pt axis labels land on the page at 7.5 pt.  The
second-round figures were authored at 0.85--0.98 x \\textwidth (5.7--6.5 in) but dropped
into 0.49\\linewidth minipages (3.37 in), i.e. scaled down by ~0.55, which put their type
back at the 4 pt the revision had just eliminated.  This module makes the printed width
the *input* to ``figsize`` instead of a hand-typed guess, so that regression cannot recur.

Figure 5 is the reference.  It is ``\\label{fig:fidelity}`` in ``secs/6_experiments.tex``:
a ``figure*`` holding two single-panel PDFs side by side, ``fig_fidelity_bars`` in a
0.60\\linewidth minipage and ``fig_rinterp_rotation`` in a 0.38\\linewidth one.  Its style
parameters are listed in ``FIG5`` below and are the ones this module hands out.

What is here
------------
``RC``              the rcParams of the R1 style (identical to regen_all's).
``FIG5``            the documented Figure-5 reference parameter sheet.
``TW`` / ``CW``     \\textwidth / \\columnwidth of cas-dc.cls, in inches.
``TEX_WIDTH``       printed width of every data figure, read off the .tex.
``print_size()``    (width, height) in inches for a figure, derived from TEX_WIDTH.
``grid_for()``      how many panel columns fit at a given printed width.
``legend_below_fit`` / ``legend_above_fit``
                    regen_all's externalised keys, but with the column count reduced
                    until the placement check actually passes.
``distinct_table()`` builds a (colour, linestyle, marker) table for an arbitrary list of
                    series keys that is guaranteed pairwise-distinct in >= 2 attributes
                    (the R1-2 rule), so new figures cannot introduce a collision.
``load_regen()``    imports ``paper/figures/gen/regen_all.py`` by path (it is not a
                    package) and returns it; that module owns the Okabe-Ito palette, the
                    per-method STYLE/LABEL tables, ``legend_below``/``legend_above`` with
                    their hard placement checks, and ``_save`` (vector PDF + 300 dpi PNG).

Nothing here draws anything and nothing here touches data: it is style only.
"""
import importlib.util
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------------
# 1. rcParams -- byte-identical to paper/figures/gen/regen_all.py
# --------------------------------------------------------------------------------------
RC = {
    "pdf.fonttype": 42,          # embed TrueType so the PDF text stays selectable
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
}

# --------------------------------------------------------------------------------------
# 2. The Figure-5 reference sheet (secs/6_experiments.tex, \label{fig:fidelity})
# --------------------------------------------------------------------------------------
FIG5 = {
    "tex": {
        "env": "figure*",                       # spans both columns
        "panel_a": ("fig_fidelity_bars", 0.60),   # minipage 0.60\linewidth, width=\linewidth
        "panel_b": ("fig_rinterp_rotation", 0.38),
        "panel_labels": None,                   # no (a)/(b) glyphs: the caption says
                                                # "\textbf{Left:} ... \textbf{right:} ..."
    },
    "figsize_in": {"fig_fidelity_bars": (0.60 * 6.87, 2.10),
                   "fig_rinterp_rotation": (0.38 * 6.87, 2.05)},
    "printed_in": {"fig_fidelity_bars": 0.60 * 6.87,      # 4.12 in
                   "fig_rinterp_rotation": 0.38 * 6.87},  # 2.61 in
    "font": {"family": "matplotlib default (DejaVu Sans), TrueType-embedded",
             "title": 7.5, "axis_label": 7.5, "tick": 6.8, "legend": 6.0},
    "lines": {"linewidth": 1.1, "sweep_curve_lw": 1.6, "markersize": 3.6,
              "point_markersize": 7.0, "marker_edge": ("black", 0.6)},
    "bars": {"group_width": 0.8, "edgecolor": "black", "edge_lw": 0.5,
             "hatch": "half of the method's STYLE hatch (bars are narrow)",
             "errorbar": {"capsize": 1.6, "lw": 0.8, "ecolor": "black"}},
    "errorbars": {"capsize": 3, "elinewidth": "rcParams lines.linewidth"},
    "palette": {"source": "Okabe-Ito (colour-blind safe) + Paul Tol teal",
                "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
                "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
                "verm": "#D55E00", "purple": "#CC79A7", "grey": "#767676",
                "teal": "#44AA99"},
    "grid": {"on": True, "alpha": 0.30, "linewidth": 0.4, "axisbelow": True},
    "spines": "matplotlib default full box, axes.linewidth 0.7 (no de-spining)",
    "legend": {"placement": "outside the plotting rectangle, checked",
               "panel_a": "above the axes, ncol=4, fontsize 6.6, frameon=False",
               "panel_b": "below the axes, ncol=2, fontsize 6.0, frameon=False"},
    "export": {"pdf": "vector, savefig.bbox=tight, pad_inches=0.012",
               "png": "dpi=300", "fonttype": 42},
}

# --------------------------------------------------------------------------------------
# 3. Printed widths.  cas-dc.cls: \textwidth = 494.5 pt = 6.87 in, \columnwidth = 241 pt.
# --------------------------------------------------------------------------------------
TW, CW = 6.87, 3.35

# name -> (fraction, span, source).  `span` is "TW" for a figure* (two columns) and "CW"
# for a single-column figure.  `source` records where the number comes from, so a figure
# whose .tex slot later changes is easy to find.
TEX_WIDTH = {
    # ---- R1 figures, unchanged (these are the reference) -------------------------------
    "fig_pilot_residual":       (0.92, "TW", "secs/3_prelim.tex:59"),
    "fig_fidelity_bars":        (0.60, "TW", "secs/6_experiments.tex:128 (Fig. 5 left)"),
    "fig_rinterp_rotation":     (0.38, "TW", "secs/6_experiments.tex:130 (Fig. 5 right)"),
    "fig_cert_c2sweep":         (0.78, "CW", "secs/6_experiments.tex:605"),
    "fig_adaptive_trajectory":  (1.00, "CW", "secs/6_experiments.tex:615"),
    "fig_horizon_nmse":         (0.64, "TW", "secs/6_experiments.tex:741"),
    "fig_fidelity_vs_error":    (0.34, "TW", "secs/6_experiments.tex:743"),
    "fig_scaling":              (0.92, "TW", "secs/6_experiments.tex:855"),
    "fig_memory_time_pareto":   (0.82, "CW", "secs/6_experiments.tex:865"),
    "fig_rl_curves":            (0.80, "TW", "secs/D_rlcase.tex:47"),
    # ---- R2 figures ------------------------------------------------------------------
    # Six sit in 0.49\linewidth minipages of a figure*, two per figure environment, so
    # their printed width is 0.49 x \textwidth = 3.37 in -- not the 0.85-0.98 they were
    # authored at.
    "fig_r2_spectrum_stage":    (0.49, "TW", "secs/6_experiments.tex:258"),
    "fig_r2_spectrum_width":    (0.49, "TW", "secs/6_experiments.tex:260"),
    "fig_r2_cert_frac_vs_clip": (0.49, "TW", "secs/6_experiments.tex:492"),
    "fig_r2_cert_cdf":          (0.49, "TW", "secs/6_experiments.tex:494"),
    "fig_r2_adaptive_diag":     (0.49, "TW", "secs/6_experiments.tex:626"),
    "fig_r2_adaptive_oat":      (0.49, "TW", "secs/6_experiments.tex:628"),
    # Two are generated but not yet cited by any \includegraphics; the width below is the
    # one this module recommends for them, and is what they are drawn at.
    "fig_r2_spectrum_age":      (0.49, "TW", "NOT IN .tex -- recommended slot"),
    "fig_r2_cert_stage":        (0.92, "TW", "NOT IN .tex -- recommended own figure*"),
}

# Heights in inches.  Figure 5's panels are 2.05-2.10 in tall, and every R1 figure is
# 1.80-2.20; a multi-row grid is the only reason to exceed that.
DEFAULT_HEIGHT = 2.10


def printed_width(name):
    """Physical width in inches that `name` occupies on the page."""
    frac, span, _ = TEX_WIDTH[name]
    return frac * (TW if span == "TW" else CW)


def tex_recommendation(name):
    """Human-readable `\\includegraphics` width recommendation for `name`."""
    frac, span, src = TEX_WIDTH[name]
    unit = r"\linewidth of a figure*" if span == "TW" else r"\linewidth of a figure"
    return "%.2f x %s (= %.2f in); source: %s" % (frac, unit, printed_width(name), src)


def print_size(name, height=DEFAULT_HEIGHT, rows=1, row_height=None):
    """(width, height) in inches: width is the *printed* width, always.

    `rows`/`row_height` grow the height for a panel grid; the width never changes,
    because the width is what the .tex fixes.
    """
    w = printed_width(name)
    if rows > 1:
        h = rows * (row_height if row_height else 0.95 * height)
    else:
        h = height
    return (w, h)


def grid_for(name, n_panels, min_panel_in=1.45, max_cols=None):
    """(nrows, ncols) for `n_panels` panels at `name`'s printed width.

    A panel narrower than ~1.45 in cannot carry a 7.5 pt axis label plus 6.8 pt ticks, so
    a row that would go below that is wrapped into a grid instead of being squeezed.  This
    is the one liberty taken with layout: the panel *set* and what each panel shows are
    unchanged, only their arrangement on the page.
    """
    w = printed_width(name)
    fit = max(1, int(w // min_panel_in))
    if max_cols:
        fit = min(fit, max_cols)
    ncols = min(n_panels, fit)
    nrows = int(math.ceil(n_panels / float(ncols)))
    return nrows, ncols


# --------------------------------------------------------------------------------------
# 4. regen_all: the module that owns the palette, the method tables and the checked keys
# --------------------------------------------------------------------------------------
_REGEN = None
REGEN_PATH = os.path.normpath(os.path.join(HERE, "..", "paper", "figures", "gen",
                                           "regen_all.py"))


def load_regen(path=None):
    """Import paper/figures/gen/regen_all.py by path and cache it."""
    global _REGEN
    if _REGEN is not None and path is None:
        return _REGEN
    p = path or REGEN_PATH
    if not os.path.isfile(p):
        raise FileNotFoundError("regen_all.py not found at %s -- the manuscript style "
                                "cannot be guaranteed without it" % p)
    spec = importlib.util.spec_from_file_location("regen_all_style", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # module level = rcParams + style tables only
    if path is None:
        _REGEN = mod
    return mod


def bind(ra):
    """Adopt an already-imported regen_all module as the one this module uses.

    The R2 report scripts import regen_all themselves (by sys.path or by file path).
    Binding their instance keeps ``LEGEND_ISSUES``, ``WROTE`` and ``OUT`` on one object,
    instead of silently splitting the run across two copies of the same file.
    """
    global _REGEN
    _REGEN = ra
    return ra


def apply_rc(plt):
    """Push the R1 rcParams (idempotent; regen_all already did it on import)."""
    plt.rcParams.update(RC)


def register(ra, name, height=DEFAULT_HEIGHT, rows=1, row_height=None):
    """Set ra.PRINT_SIZE[name] from the .tex printed width and return the figsize.

    Replaces the `ra.PRINT_SIZE.setdefault(name, (0.92 * ra.TW, 2.2))` guesses that put
    the R2 figures ~1.85x too wide.
    """
    size = print_size(name, height=height, rows=rows, row_height=row_height)
    ra.PRINT_SIZE[name] = size
    return size


# --------------------------------------------------------------------------------------
# 5. Legend placement: the checked, externalised key with an auto-fitted column count
# --------------------------------------------------------------------------------------
def _fit(placer, fig, axes, handles, labels, ncol, name, **kw):
    """Call regen_all's legend_below/legend_above, reducing ncol until the check passes.

    ``_check_legend_outside`` rejects a key that is wider than the axes span, because
    bbox_inches='tight' would then widen the exported PDF and \\includegraphics would scale
    the whole figure -- and the type with it -- back down.  At Figure 5's printed widths
    (2.6-4.1 in) the honest fix is fewer columns and more rows, which costs height only.
    """
    ra = load_regen()
    n = max(1, int(ncol))
    leg = None
    while True:
        before = len(ra.LEGEND_ISSUES)
        leg = placer(fig, axes, handles, labels, ncol=n, name=name, **kw)
        if len(ra.LEGEND_ISSUES) == before or n == 1:
            return leg
        ra.LEGEND_ISSUES.pop()        # retry with fewer columns; drop the failed report
        leg.remove()
        n -= 1


def legend_below_fit(fig, axes, handles, labels, ncol, name, **kw):
    ra = load_regen()
    kw.setdefault("fontsize", FIG5["font"]["legend"])
    return _fit(ra.legend_below, fig, axes, handles, labels, ncol, name, **kw)


def legend_above_fit(fig, axes, handles, labels, ncol, name, **kw):
    ra = load_regen()
    kw.setdefault("fontsize", FIG5["font"]["legend"])
    return _fit(ra.legend_above, fig, axes, handles, labels, ncol, name, **kw)


# --------------------------------------------------------------------------------------
# 6. R1-2: a (colour, linestyle, marker) table that cannot collide
# --------------------------------------------------------------------------------------
# Ordered so that consecutive entries differ in all three attributes and any two entries
# differ in at least two: the colour cycle has length 8, the dash cycle 5 and the marker
# cycle 7, which are pairwise coprime enough that the first 40 slots are safe (asserted).
_C_ORDER = ["blue", "orange", "green", "verm", "purple", "teal", "sky", "grey"]
_LS_ORDER = ["-", (0, (5, 2)), (0, (1, 1.2)), (0, (6, 1.5, 1, 1.5)), (0, (3, 1, 1, 1))]
_MK_ORDER = ["o", "s", "^", "D", "v", "P", "X"]


def distinct_table(keys, oi=None):
    """(colour, linestyle, marker) per key, pairwise distinct in >= 2 attributes.

    Raises if the guarantee fails, so a figure with a new series set cannot silently break
    the R1-2 rule the way ``fig_r2_adaptive_diag`` did (two series shared both colour and
    marker and differed only in the dash pattern).
    """
    oi = oi or load_regen().OI
    tbl = {}
    for i, k in enumerate(keys):
        tbl[k] = (oi[_C_ORDER[i % len(_C_ORDER)]],
                  _LS_ORDER[i % len(_LS_ORDER)],
                  _MK_ORDER[i % len(_MK_ORDER)])
    bad = []
    ks = list(tbl)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            same = sum(str(x) == str(y) for x, y in zip(tbl[a], tbl[b]))
            if same >= 2:
                bad.append((a, b))
    if bad:
        raise ValueError("distinct_table: >=2 shared visual attributes for %r" % (bad,))
    return tbl


# --------------------------------------------------------------------------------------
def describe():
    """The Figure-5 parameter sheet as printable lines (used by make_fig_style_check.py)."""
    L = ["Figure 5 (secs/6_experiments.tex, \\label{fig:fidelity}) -- style of record", ""]
    L.append("  environment      figure* (two columns), two single-panel PDFs side by side")
    L.append("  panel a          fig_fidelity_bars     minipage 0.60\\linewidth -> 4.12 in")
    L.append("  panel b          fig_rinterp_rotation  minipage 0.38\\linewidth -> 2.61 in")
    L.append("  figsize          (4.12, 2.10) and (2.61, 2.05) in -- authored AT print size")
    L.append("  panel labels     none; the caption carries \\textbf{Left:}/\\textbf{right:}")
    L.append("  font             DejaVu Sans, TrueType-embedded (pdf.fonttype 42)")
    L.append("  font sizes       title 7.5, axis label 7.5, tick 6.8, legend 6.0-6.6 pt")
    L.append("  lines            lw 1.1 (sweep curve 1.6), ms 3.6 (point markers 7.0),")
    L.append("                   marker edge black 0.6")
    L.append("  bars             group width 0.8, black edge 0.5, half-density hatch,")
    L.append("                   yerr capsize 1.6 / lw 0.8 / black")
    L.append("  palette          Okabe-Ito: #000000 #E69F00 #56B4E9 #009E73 #F0E442")
    L.append("                   #0072B2 #D55E00 #CC79A7 #767676 + Tol teal #44AA99")
    L.append("  grid             on, alpha 0.30, lw 0.4, axisbelow")
    L.append("  spines           matplotlib default box, axes.linewidth 0.7")
    L.append("  legend           OUTSIDE the plotting rectangle, frameon=False, and")
    L.append("                   verified by _check_legend_outside (clears every panel and")
    L.append("                   stays inside the axes span so the export is not widened)")
    L.append("  export           vector PDF (bbox tight, pad 0.012) + PNG at dpi 300")
    return L


if __name__ == "__main__":
    print("\n".join(describe()))
    print()
    print("printed widths (in):")
    for k in sorted(TEX_WIDTH):
        print("  %-28s %5.2f   %s" % (k, printed_width(k), TEX_WIDTH[k][2]))
