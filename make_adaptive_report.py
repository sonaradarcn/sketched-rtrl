"""Adaptive-rank report: per (task, policy) avg-rank, peak MB, grad-cos, NMSE, cert violations.
Usage: python make_adaptive_report.py results/ademo"""
import glob, json, math, sys
from collections import defaultdict


def tail(recs, k, frac=0.2):
    v = [r[k] for r in recs if r.get(k) is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
    return sum(v[-max(1, int(len(v) * frac)):]) / max(1, int(len(v) * frac)) if v else None


def ms(vals, fmt="{:.3f}"):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return fmt.format(m)
    s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return f"{fmt.format(m)}$\\pm${fmt.format(s)}"


def main():
    d_ = sys.argv[1] if len(sys.argv) > 1 else "results/ademo"
    agg = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{d_}/*.json")):
        d = json.load(open(f))
        a = d["args"]
        pol = "adaptive" if a.get("fixed_r", -1) < 0 else f"fixed{a['fixed_r']}"
        agg[a["task"]][pol].append({
            "ar": d.get("avg_rank"), "mb": d.get("peak_MB"),
            "cos": tail(d["records"], "grad_cos"), "nmse": tail(d["records"], "metric"),
            "viol": d.get("cert_violations", 0)})
    for t in agg:
        print(f"### {t}")
        print("| policy | avg rank | peak MB | grad cosine | NMSE | cert viol |")
        print("|---|---|---|---|---|---|")
        for pol in ("fixed4", "fixed16", "fixed32", "adaptive"):
            if pol in agg[t]:
                rs = agg[t][pol]
                print(f"| {pol} | {ms([r['ar'] for r in rs], '{:.1f}')} | {ms([r['mb'] for r in rs], '{:.0f}')} | "
                      f"{ms([r['cos'] for r in rs])} | {ms([r['nmse'] for r in rs], '{:.4f}')} | {sum(r['viol'] for r in rs)} |")
        print()


if __name__ == "__main__":
    main()
