"""E8: paired statistical tests for the Round-1 result tables.

For each task it builds a method x seed matrix of a chosen scalar (default: the
held-out NMSE = mean of the last `--tail` logged `metric` values, lower = better),
then compares a reference method against the others on the common seeds with:
  * paired mean difference + 95% bootstrap CI (paired resampling over seeds),
  * Wilcoxon signed-rank test (scipy, exact for small n),
  * Cohen's d_z paired effect size,
  * Holm step-down correction across each task's comparison family.

With ~5 seeds the Wilcoxon floor is p>=0.0625 (two-sided); this is reported honestly
and the bootstrap CI / effect size carry the quantitative weight. No values are
fabricated: a comparison is emitted only when both methods have >=3 common seeds.

Usage:
  python make_stats.py --dirs results/round1/real results/ts \
      --metric metric --ref skrtrl-r16 --out results/round1/STATS
"""
import argparse, glob, json, os, re
import numpy as np
from scipy import stats

FNAME = re.compile(r"^(?P<task>[a-z0-9]+)_(?P<method>[a-z0-9\-]+)_s(?P<seed>\d+)(?:_.*)?\.json$")


def load(dirs, field, tail):
    """-> {task: {method: {seed: value}}}"""
    table = {}
    for d in dirs:
        for path in glob.glob(os.path.join(d, "*.json")):
            m = FNAME.match(os.path.basename(path))
            if not m:
                continue
            try:
                j = json.load(open(path))
            except Exception:
                continue
            recs = j.get("records", [])
            if field == "final_nmse" and "final_nmse" in j:
                val = j["final_nmse"]
            else:
                vals = [r[field] for r in recs if r.get(field) is not None]
                if not vals:
                    continue
                val = float(np.mean(vals[-tail:]))
            t, meth, s = m["task"], m["method"], int(m["seed"])
            table.setdefault(t, {}).setdefault(meth, {})[s] = val
    return table


def bootstrap_ci(d, n=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        v = (m - rank) * pvals[i]
        run = max(run, v)
        adj[i] = min(run, 1.0)
    return adj


def compare(table, ref, others, higher_better):
    out = {}
    for task, methods in sorted(table.items()):
        if ref not in methods:
            continue
        comps = []
        for o in others:
            if o not in methods:
                continue
            seeds = sorted(set(methods[ref]) & set(methods[o]))
            if len(seeds) < 3:
                continue
            a = np.array([methods[ref][s] for s in seeds])   # reference
            b = np.array([methods[o][s] for s in seeds])      # other
            d = b - a                                          # other - ref
            # sign so that positive "delta" = ref is better
            delta = d if higher_better is False else -d
            md = float(delta.mean())
            lo, hi = bootstrap_ci(delta)
            try:
                # 'method' replaced the deprecated 'mode' kw in scipy>=1.13; fall back for old scipy
                try:
                    w = stats.wilcoxon(a, b, zero_method="wilcox", correction=False,
                                       alternative="two-sided", method="auto")
                except TypeError:
                    w = stats.wilcoxon(a, b, zero_method="wilcox", correction=False,
                                       alternative="two-sided", mode="auto")
                p = float(w.pvalue)
            except ValueError:
                p = float("nan")   # all-zero differences
            dz = float(delta.mean() / (delta.std(ddof=1) + 1e-12))
            comps.append({"other": o, "n": len(seeds),
                          "ref_mean": float(a.mean()), "other_mean": float(b.mean()),
                          "mean_delta_ref_better": md, "ci95": [lo, hi],
                          "wilcoxon_p": p, "cohen_dz": dz})
        ps = np.array([c["wilcoxon_p"] for c in comps], dtype=float)
        if len(ps):
            ph = holm(np.nan_to_num(ps, nan=1.0))
            for c, q in zip(comps, ph):
                c["holm_p"] = float(q)
        out[task] = comps
    return out


def to_markdown(res, ref, metric, higher_better):
    direction = "higher better" if higher_better else "lower better"
    lines = [f"# E8 paired statistics (reference = `{ref}`, metric = `{metric}`, {direction})", "",
             "`delta` = how much better the reference is than the other method (positive favors "
             "reference). CI is a paired 95% bootstrap over seeds; `p` is Wilcoxon signed-rank "
             "(two-sided, exact for small n); `holm_p` is Holm-corrected within each task; "
             "`d_z` is the paired Cohen effect size. With ~5 seeds the Wilcoxon floor is p>=0.0625.",
             ""]
    for task, comps in res.items():
        if not comps:
            continue
        lines.append(f"## {task}")
        lines.append("| other | n | ref mean | other mean | delta (ref better) | 95% CI | p | holm_p | d_z |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in comps:
            lines.append("| {other} | {n} | {ref_mean:.4f} | {other_mean:.4f} | {md:+.4f} | "
                         "[{lo:+.4f}, {hi:+.4f}] | {p:.4f} | {hp:.4f} | {dz:+.2f} |".format(
                             other=c["other"], n=c["n"], ref_mean=c["ref_mean"],
                             other_mean=c["other_mean"], md=c["mean_delta_ref_better"],
                             lo=c["ci95"][0], hi=c["ci95"][1], p=c["wilcoxon_p"],
                             hp=c.get("holm_p", float("nan")), dz=c["cohen_dz"]))
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--metric", default="metric")           # 'metric' (NMSE) or 'grad_cos'
    ap.add_argument("--tail", type=int, default=5)
    ap.add_argument("--ref", default="skrtrl-r16")
    ap.add_argument("--others", nargs="+",
                    default=["exact", "snap1", "rflo", "uoro", "kfrtrl", "skrtrl-r4"])
    ap.add_argument("--higher_better", type=int, default=0)  # 1 for grad_cos, 0 for NMSE
    ap.add_argument("--out", default="results/round1/STATS")
    args = ap.parse_args()

    table = load(args.dirs, args.metric, args.tail)
    res = compare(table, args.ref, args.others, bool(args.higher_better))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"ref": args.ref, "metric": args.metric, "result": res},
              open(args.out + ".json", "w"), indent=1)
    md = to_markdown(res, args.ref, args.metric, bool(args.higher_better))
    open(args.out + ".md", "w").write(md)
    print("wrote", args.out + ".md", "and", args.out + ".json")
    print(f"tasks with comparisons: {sum(1 for v in res.values() if v)}")


if __name__ == "__main__":
    main()
