"""Aggregate results/m3/*.json into a markdown report: final metric & mean grad-cos per
(task, algo) with mean±std over seeds, plus certificate stats for SK-RTRL runs."""
import glob, json, math, os, sys
from collections import defaultdict


def tail_mean(records, key, frac=0.2):
    vals = [r[key] for r in records if r.get(key) is not None and not (
        isinstance(r[key], float) and math.isnan(r[key]))]
    if not vals:
        return None
    k = max(1, int(len(vals) * frac))
    return sum(vals[-k:]) / k


def main():
    d_ = sys.argv[1] if len(sys.argv) > 1 else "results/m3"
    runs = defaultdict(list)
    for f in sorted(glob.glob(f"{d_}/*.json")):
        d = json.load(open(f))
        a = d["args"]
        recs = d["records"]
        runs[(a["task"], a["algo"])].append({
            "metric": tail_mean(recs, "metric"),
            "cos": tail_mean(recs, "grad_cos"),
            "e_t": tail_mean(recs, "e_t"),
            "true_E": tail_mean(recs, "true_E"),
            "rho_bar": tail_mean(recs, "rho_bar"),
            "wall_s": d.get("wall_s"),
        })

    def ms(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        m = sum(vals) / len(vals)
        if len(vals) < 2:
            return f"{m:.4f}"
        s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        return f"{m:.4f}±{s:.4f}"

    lines = ["# M3 First-Pass Report\n",
             "| Task | Algo | Final metric (tail mean±std) | Grad-cos vs exact | e_t | true ‖E‖ | tightness e/E |",
             "|---|---|---|---|---|---|---|"]
    for (task, algo), rs in sorted(runs.items()):
        et, tE = ms([r["e_t"] for r in rs]), ms([r["true_E"] for r in rs])
        tight = "—"
        es = [r["e_t"] for r in rs if r["e_t"]]
        Es = [r["true_E"] for r in rs if r["true_E"]]
        if es and Es and sum(Es):
            tight = f"{(sum(es)/len(es))/(sum(Es)/len(Es)):.2e}"
        lines.append(f"| {task} | {algo} | {ms([r['metric'] for r in rs])} | "
                     f"{ms([r['cos'] for r in rs])} | {et} | {tE} | {tight} |")
    out = "\n".join(lines) + "\n"
    os.makedirs("results", exist_ok=True)
    open(f"results/REPORT_{os.path.basename(d_)}.md", "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
