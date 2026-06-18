"""Time-series report: per (task, algo) NMSE and grad-cosine, mean+/-std over seeds.
Tail-mean over last 20% of logged steps. Usage: python make_ts_report.py results/ts"""
import glob, json, math, sys
from collections import defaultdict


def tail(recs, k, frac=0.2):
    v = [r[k] for r in recs if r.get(k) is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
    return sum(v[-max(1, int(len(v) * frac)):]) / max(1, int(len(v) * frac)) if v else None


def ms(vals, fmt="{:.4f}"):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return fmt.format(m)
    s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return f"{fmt.format(m)}$\\pm${fmt.format(s)}"


def main():
    d_ = sys.argv[1] if len(sys.argv) > 1 else "results/ts"
    runs = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{d_}/*.json")):
        d = json.load(open(f))
        a = d["args"]
        runs[a["task"]][a["algo"]].append({
            "nmse": tail(d["records"], "metric"), "cos": tail(d["records"], "grad_cos"),
            "mb": d.get("peak_MB"), "ms": (d.get("wall_s", 0) / max(a.get("steps", 1), 1)) * 1000})

    algos = ["exact", "skrtrl-r32", "skrtrl-r16", "skrtrl-r4", "snap1", "rflo", "uoro", "kfrtrl", "tbptt"]
    print(f"# Time-series report ({d_})\n")
    for t in ("henon", "mackeyglass", "lorenz"):
        if t not in runs:
            continue
        print(f"## {t}")
        print("| algo | NMSE | grad-cos | peak MB | ms/step | n_seeds |")
        print("|---|---|---|---|---|---|")
        for a in algos:
            if a in runs[t]:
                rs = runs[t][a]
                print(f"| {a} | {ms([r['nmse'] for r in rs])} | {ms([r['cos'] for r in rs], '{:.3f}')} | "
                      f"{ms([r['mb'] for r in rs], '{:.0f}')} | {ms([r['ms'] for r in rs], '{:.1f}')} | {len(rs)} |")
        print()


if __name__ == "__main__":
    main()
