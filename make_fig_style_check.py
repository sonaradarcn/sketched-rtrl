"""Build paper/revise_r2/FIG_STYLE_CHECK.pdf -- the side-by-side style audit page.

Figure 5 (fig_fidelity_bars + fig_rinterp_rotation) is the style of record for the
manuscript's data figures; this page puts it next to every second-round figure so an
external reviewer can check, without reading any code, that they now share it.

For each figure it prints
  * the size the PDF was authored at (read from the PDF MediaBox, not from the source),
  * the width it is printed at (from the \\includegraphics slot in the .tex),
  * the resulting scale factor and the on-page size of the 7.5 pt axis label and the
    6.8 pt tick label -- the number the R1 revision was actually about,
  * the figure's panel grid and whether its key sits outside the plotting rectangle.

Usage:  python make_fig_style_check.py [--figdir <paper/figures>] [--out <pdf path>]
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_style_r1 as fs                                          # noqa: E402

# (name, role, panel grid, legend placement) -- role drives the banner colour.
SHEET = [
    ("fig_fidelity_bars", "REFERENCE -- Figure 5 (left)", "1 x 1", "above axes"),
    ("fig_rinterp_rotation", "REFERENCE -- Figure 5 (right)", "1 x 1", "below axes"),
    ("fig_fidelity_vs_error", "unchanged R1 panel (Fig. 11 right)", "1 x 1", "below axes"),
    ("fig_r2_spectrum_stage", "R2, restyled", "1 x 2", "below axes"),
    ("fig_r2_spectrum_width", "R2, restyled", "1 x 2", "below axes"),
    ("fig_r2_spectrum_age", "R2, restyled (not yet cited)", "1 x 2", "none (1 series)"),
    ("fig_r2_cert_frac_vs_clip", "R2, restyled", "1 x 2", "below axes"),
    ("fig_r2_cert_cdf", "R2, restyled (row -> 2x2 grid)", "2 x 2", "below axes"),
    ("fig_r2_cert_stage", "R2, restyled (not yet cited)", "2 x 4", "below axes"),
    ("fig_r2_adaptive_diag", "R2, restyled", "1 x 2", "below axes"),
    ("fig_r2_adaptive_oat", "R2, restyled (axes transposed)", "1 x 5", "below axes"),
]

ROLE_COLOUR = {"REFERENCE": "#0072B2", "unchanged": "#767676"}


def pdf_size(path):
    """(width, height) in inches from the first /MediaBox of a PDF."""
    with open(path, "rb") as fh:
        blob = fh.read()
    m = re.search(rb"/MediaBox\s*\[([^\]]*)\]", blob)
    if not m:
        return None
    v = [float(x) for x in m.group(1).split()]
    return ((v[2] - v[0]) / 72.0, (v[3] - v[1]) / 72.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figdir", default=os.path.join(HERE, "..", "paper", "figures"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "paper", "revise_r2",
                                                  "FIG_STYLE_CHECK.pdf"))
    a = ap.parse_args()
    figdir, out = os.path.abspath(a.figdir), os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows, missing = [], []
    for name, role, grid, leg in SHEET:
        p_pdf = os.path.join(figdir, name + ".pdf")
        p_png = os.path.join(figdir, name + ".png")
        if not (os.path.isfile(p_pdf) and os.path.isfile(p_png)):
            missing.append(name)
            continue
        dw, dh = pdf_size(p_pdf)
        frac, span, src = fs.TEX_WIDTH[name]
        pw = fs.printed_width(name)
        rows.append(dict(name=name, role=role, grid=grid, leg=leg, png=p_png,
                         dw=dw, dh=dh, frac=frac, span=span, src=src, pw=pw,
                         scale=pw / dw))
    if missing:
        print("  !! missing PDF/PNG for: %s" % ", ".join(missing))
    if not rows:
        raise SystemExit("no figure found under %s" % figdir)

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 7.0,
                         "font.family": "DejaVu Sans"})
    with PdfPages(out) as pdf:
        _page_sheet(pdf, rows)
        _page_thumbs(pdf, rows[:6], 1, 2)
        _page_thumbs(pdf, rows[6:], 2, 2)
    print("wrote %s (%d figures, %d page(s))" % (out, len(rows), 3))


# --------------------------------------------------------------------------------------
def _page_sheet(pdf, rows):
    fig = plt.figure(figsize=(8.27, 11.69))          # A4 portrait
    fig.text(0.06, 0.975, "SK-RTRL figure style audit", fontsize=15, weight="bold",
             va="top")
    fig.text(0.06, 0.955,
             "Reference = Figure 5 (secs/6_experiments.tex, \\label{fig:fidelity}).  "
             "Sizes are read from each PDF's /MediaBox;\nprinted widths from the "
             "\\includegraphics slot in the .tex.  The R1 rule is drawn size == printed "
             "size, so that\nthe 7.5 pt axis label reaches the page at 7.5 pt.",
             fontsize=8, va="top")

    y = 0.905
    for line in fs.describe()[2:]:
        fig.text(0.06, y, line, fontsize=7.2, family="DejaVu Sans Mono", va="top")
        y -= 0.0145

    y -= 0.012
    fig.text(0.06, y, "Per-figure measurements", fontsize=11, weight="bold", va="top")
    y -= 0.022
    hdr = ("%-26s %-11s %-14s %-8s %-5s %-5s %-9s"
           % ("figure", "drawn (in)", "printed (in)", "scale", "label", "tick", "panels"))
    fig.text(0.06, y, hdr, fontsize=7.0, family="DejaVu Sans Mono", va="top",
             weight="bold")
    y -= 0.0135
    fig.text(0.06, y, "-" * 92, fontsize=7.0, family="DejaVu Sans Mono", va="top")
    y -= 0.0135
    for r in rows:
        ok = "OK" if 0.90 <= r["scale"] <= 1.12 else "!!"
        line = ("%-26s %5.2fx%-5.2f %4.2f%-3s= %-5.2f %5.2f %-3s %4.1f %5.1f  %-9s"
                % (r["name"], r["dw"], r["dh"], r["frac"], r["span"], r["pw"],
                   r["scale"], ok, 7.5 * r["scale"], 6.8 * r["scale"], r["grid"]))
        fig.text(0.06, y, line, fontsize=7.0, family="DejaVu Sans Mono", va="top",
                 color=("#000000" if ok == "OK" else "#D55E00"))
        y -= 0.0135

    y -= 0.016
    notes = [
        "label / tick = the on-page size in points of the 7.5 pt axis label and the",
        "6.8 pt tick label, i.e. nominal x scale.  Figure 5's own panels run at",
        "scale 1.05-1.07, so 1.00-1.12 is the band this audit accepts.",
        "",
        "TW = \\textwidth of cas-dc.cls (494.5 pt = 6.87 in, a figure* spanning both",
        "columns); CW = \\columnwidth (241 pt = 3.35 in).",
        "",
        "Every key on this page is drawn outside its plotting rectangle and is verified",
        "by regen_all._check_legend_outside, which rejects a key that overlaps a panel",
        "or that is wider than the axes span (a wider export would be scaled down by",
        "\\includegraphics and would shrink the type again).",
        "",
        "fig_r2_spectrum_age and fig_r2_cert_stage are generated but are not cited by",
        "any \\includegraphics yet; their 'printed' width is the one recommended for them.",
    ]
    for line in notes:
        fig.text(0.06, y, line, fontsize=7.2, va="top")
        y -= 0.0135
    pdf.savefig(fig)
    plt.close(fig)


def _page_thumbs(pdf, rows, page, ncol):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.982, "Figure style audit -- thumbnails, page %d of 2" % page,
             fontsize=12, weight="bold", ha="center", va="top")
    nrow = max(1, -(-len(rows) // ncol))
    for i, r in enumerate(rows):
        row, col = i // ncol, i % ncol
        # each cell: image on top, caption block underneath
        cw, ch = 0.92 / ncol, 0.945 / nrow
        x0 = 0.04 + col * cw
        y0 = 0.955 - (row + 1) * ch
        ax = fig.add_axes([x0 + 0.012, y0 + 0.085 * ch, cw - 0.024, ch * 0.80])
        ax.imshow(mpimg.imread(r["png"]))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6)
            s.set_edgecolor("#767676")
        head = r["role"].split(" ")[0]
        ax.set_title("%s\n%s" % (r["name"], r["role"]), fontsize=7.6,
                     color=ROLE_COLOUR.get(head, "#000000"), pad=3.0)
        cap = ("drawn %.2f x %.2f in | printed %.2f x %s = %.2f in | scale %.2f\n"
               "on page: label %.1f pt, tick %.1f pt | panels %s | key %s"
               % (r["dw"], r["dh"], r["frac"], r["span"], r["pw"], r["scale"],
                  7.5 * r["scale"], 6.8 * r["scale"], r["grid"], r["leg"]))
        fig.text(x0 + 0.012, y0 + 0.070 * ch, cap, fontsize=6.4, va="top",
                 family="DejaVu Sans Mono")
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
