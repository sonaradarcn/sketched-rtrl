"""Draw fig_fidelity_bars (Fig. 5, left) from the raw diagnostic runs in results/m3/.

This script used to hard-code the published Table 3 means and standard deviations, because
the raw diagnostic JSONs were not in the released tree. They are released now, so the bars
are computed from the runs and cross-checked against the published table: if any of the 24
cells disagrees at the printed precision the script fails instead of drawing.

The figure shipped with the paper is produced by make_paper_figures.py, which owns the
print sizes and the colour-blind-safe style; this entry point exists to regenerate that one
panel on its own.

Usage:  python make_fidelity_bars.py [--out results/figures]
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
    args = ap.parse_args()
    mpf.ROOT = args.root
    mpf.OUT = args.out or os.path.join(args.root, "results", "figures")
    mpf.GRAY = ""
    mpf.fig_fidelity_bars()
    if mpf.SKIPPED:
        for name, why in mpf.SKIPPED:
            print("SKIPPED", name, "--", why)
        sys.exit(1)


if __name__ == "__main__":
    main()
