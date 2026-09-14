"""Draw fig_fidelity_bars (Fig. 5, left) on its own.

This script used to hard-code the published Table 3 means and standard deviations, because
the raw diagnostic JSONs were not in the released tree. They are released now, so the bars
are computed from the runs and cross-checked against the table of record: if any cell
disagrees at the printed precision the script fails instead of drawing.

Which runs, and which table.  `--source r2` (the default, matching make_paper_figures.py)
reads results/r2/d4_eval -- the second-round tree in which the learning rate is tuned per
(task, estimator) pair, as R3 asked -- and cross-checks the bars against
paper/secs/tab_r2_fidelity.tex, the R2 table of record, which it only ever reads. `--source
r1` reads the first-round results/m3 tree at the single shared learning rate and checks it
against the hard-coded first-round Table 3, i.e. the panel as the first-round submission
carried it.

The figure shipped with the paper is produced by make_paper_figures.py, which owns the
print sizes and the colour-blind-safe style; this entry point exists to regenerate that one
panel on its own.

Usage:  python make_fidelity_bars.py [--out results/figures] [--source r1|r2]
"""
import argparse
import os
import sys

import make_paper_figures as mpf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)),
                    help="repository root (the directory holding results/)")
    ap.add_argument("--out", default=None, help="directory to write the figure into")
    ap.add_argument("--source", choices=["r1", "r2"], default="r2",
                    help="r2 (default): results/r2/d4_eval, checked against "
                         "paper/secs/tab_r2_fidelity.tex; r1: results/m3, checked against the "
                         "first-round Table 3")
    ap.add_argument("--fidelity-tex", default="",
                    help="override the R2 table to check against (read-only)")
    args = ap.parse_args()
    mpf.ROOT = args.root
    mpf.OUT = args.out or os.path.join(args.root, "results", "figures")
    mpf.GRAY = ""
    if args.source == "r2":
        mpf.fig_fidelity_bars_r2(args.fidelity_tex or None)
    else:
        mpf.fig_fidelity_bars()
    if mpf.SKIPPED:
        for name, why in mpf.SKIPPED:
            print("SKIPPED", name, "--", why)
        sys.exit(1)


if __name__ == "__main__":
    main()
