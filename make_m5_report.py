"""Aggregate results/m5/*.json into a T-maze return table by (corridor_len, algo) over seeds.

Reports tail-mean return, success rate, and steps-to-threshold (succ>=0.8).
Optimal return per length L: +4 (correct turn) - 0.1*L (corridor steps) approx.
"""
import glob, json, math, os, sys
from collections import defaultdict


def tail_mean(records, key, frac=0.2):
    vals = [r[key] for r in records if r.get(key) is not None and not (
        isinstance(r[key], float) and math.isnan(r[key]))]
    if not vals:
        return None
    k = max(1, int(len(vals) * frac))
    return sum(vals[-k:]) / k


def steps_to(records, key, thr=0.8):
    for r in records:
        if r.get(key) is not None and r[key] >= thr:
            return r.get("step")
    return None


def main():
    d_ = sys.argv[1] if len(sys.argv) > 1 else "results/m5"
    runs = defaultdict(list)
    for f in sorted(glob.glob(f"{d_}/*.json")):
        d = json.load(open(f))
        a = d["args"]
        recs = d["records"]
        key_ret = "ret" if any("ret" in r for r in recs) else "return"
        key_succ = "succ" if any("succ" in r for r in recs) else "success"
        runs[(a.get("env_len"), a["algo"])].append({
            "ret": tail_mean(recs, key_ret),
            "succ": tail_mean(recs, key_succ),
            "s2t": steps_to(recs, key_succ),
        })

    def ms(vals, fmt="{:.3f}"):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        m = sum(vals) / len(vals)
        if len(vals) < 2:
            return fmt.format(m)
        s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        return f"{fmt.format(m)}±{fmt.format(s)}"

    lines = ["# M5 T-maze Report (return / success over seeds)\n",
             "| Corridor | Algo | Final return | Success rate | Steps→0.8 succ (mean) |",
             "|---|---|---|---|---|"]
    for (L, algo) in sorted(runs, key=lambda k: (k[0] or 0, k[1])):
        rs = runs[(L, algo)]
        s2t = [r["s2t"] for r in rs if r["s2t"] is not None]
        s2t_str = f"{int(sum(s2t)/len(s2t))}" if s2t else "never"
        lines.append(f"| {L} | {algo} | {ms([r['ret'] for r in rs])} | "
                     f"{ms([r['succ'] for r in rs])} | {s2t_str} ({len(s2t)}/{len(rs)}) |")
    out = "\n".join(lines) + "\n"
    os.makedirs("results", exist_ok=True)
    open(f"results/REPORT_{os.path.basename(d_)}.md", "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
