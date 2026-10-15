"""Table 7 of the paper: what the spectral clip costs in task metric.

SK-RTRL r=16, n=64, seeds 0-4. The unclipped column is read from the estimator comparison
(results/r2/d4_eval, the r=16 runs of seeds 0-4); the clipped columns from the clip sweep
(results/r2/d2_clip). Per run the value is the mean task metric over the last 20% of the
run (make_r2_d4_select.window_score), then the mean over the five seeds. Percentages are the
relative change of the raw metric against the unclipped column; a row whose largest change
reaches 100% is printed without a decimal.

Usage:  python make_r2_clipcost_table.py [--out results/tables/tab_r2_clipcost.tex]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r2_d4_select as sel  # noqa: E402

ROWS = [  # (task, label, metric label), in the order of the printed table
    ("copy", "copy", r"accuracy $\uparrow$"),
    ("anbn", r"$a^nb^n$", r"accuracy $\uparrow$"),
    ("adding", "adding", r"MSE $\downarrow$"),
    ("rotation", "rotation", r"MSE $\downarrow$"),
    None,
    ("henon", r"H\'enon", r"NMSE $\downarrow$"),
    ("mackeyglass", r"Mackey--Glass", r"NMSE $\downarrow$"),
    ("lorenz", "Lorenz", r"NMSE $\downarrow$"),
    None,
    ("sunspot", "Sunspot", r"NMSE $\downarrow$"),
    ("laser", "laser", r"NMSE $\downarrow$"),
]
CLIPS = ["0.35", "0.7", "0.9"]
SEEDS = range(5)

CAPTION = (r"\caption{\textbf{What the spectral clip costs} (\skrtrl{} $r{=}16$, $n{=}64$, five "
           r"seeds $0$--$4$; the arrow gives the direction of the raw metric and percentages "
           r"are its relative change, so a negative entry means a worse accuracy or a lower "
           r"error). Seed matching and the comparison with the ten-seed tables are discussed in "
           r"\Cref{sec:exp-cert}.}")


def run_value(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    v, n, _ = sel.window_score(d.get("records") or [], d["args"]["steps"], sel.WINDOW_FRAC)
    if v is None:
        raise SystemExit("no records in the last 20%% of %s" % path)
    return v


def one(pattern):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise SystemExit("expected one file for %s, found %d" % (pattern, len(hits)))
    return hits[0]


def cells(root):
    out = {}
    for row in ROWS:
        if row is None:
            continue
        task = row[0]
        base = [run_value(one(os.path.join(root, "results", "r2", "d4_eval",
                                           "%s_skrtrl-r16_s%d_lr*.json" % (task, s))))
                for s in SEEDS]
        out[(task, "none")] = sum(base) / len(base)
        for c in CLIPS:
            vals = [run_value(one(os.path.join(root, "results", "r2", "d2_clip",
                                               "%s_skrtrl-r16_s%d_clip%s.json" % (task, s, c))))
                    for s in SEEDS]
            out[(task, c)] = sum(vals) / len(vals)
    return out


def fmt_pct(p, integer):
    if integer:
        return "$%+d\\%%$" % round(p)
    return "$%+.1f\\%%$" % p


def body_lines(cl):
    L = []
    for row in ROWS:
        if row is None:
            L.append(r"\midrule")
            continue
        task, label, metric = row
        b = cl[(task, "none")]
        pcts = [100.0 * (cl[(task, c)] - b) / b for c in CLIPS]
        integer = max(abs(p) for p in pcts) >= 100.0
        parts = ["$%.4f$ (%s)" % (cl[(task, c)], fmt_pct(p, integer)) for c, p in zip(CLIPS, pcts)]
        L.append("%s & %s & $%.4f$ & %s \\\\" % (label, metric, b, " & ".join(parts)))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=HERE, help="repository root")
    ap.add_argument("--out", default=os.path.join("results", "tables", "tab_r2_clipcost.tex"))
    args = ap.parse_args()
    cl = cells(args.root)
    L = ["% Table 7 (clip cost), written by make_r2_clipcost_table.py",
         "% Inputs: results/r2/d4_eval (skrtrl-r16, seeds 0-4), results/r2/d2_clip "
         "(skrtrl-r16, seeds 0-4); per-run last-20% mean, then mean over seeds.",
         "",
         r"\begin{table*}[t]",
         r"\centering\small",
         CAPTION,
         r"\label{tab:r2clipcost}",
         r"\begin{tabular}{llcccc}",
         r"\toprule",
         r"Task & metric & unclipped & clip $0.35$ & clip $0.7$ & clip $0.9$ \\",
         r"\midrule"]
    L += body_lines(cl)
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))
    print("wrote", args.out)
    for line in body_lines(cl):
        print("  " + line)


if __name__ == "__main__":
    main()
