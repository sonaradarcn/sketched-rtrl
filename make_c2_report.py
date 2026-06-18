"""C2 certificate-tightness report (contractive runs). Decomposes looseness into
norm-surrogate (rho_bar/rho_hat) and compounding, and reports certificate validity."""
import glob, json, math, os, sys
from collections import defaultdict


def tail(recs, k, frac=0.3):
    v = [r[k] for r in recs if r.get(k) is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
    if not v:
        return None
    return sum(v[-max(1, int(len(v) * frac)):]) / max(1, int(len(v) * frac))


def main():
    d_ = sys.argv[1] if len(sys.argv) > 1 else "results/c2"
    runs = defaultdict(list)
    valid_total = valid_ok = 0
    for f in sorted(glob.glob(f"{d_}/*.json")):
        d = json.load(open(f))
        a = d["args"]
        recs = d["records"]
        for r in recs:
            if r.get("e_t") is not None and r.get("true_E") is not None:
                valid_total += 1
                if r["e_t"] + 1e-6 >= r["true_E"]:
                    valid_ok += 1
        runs[(a["task"], a["algo"])].append({
            "cos": tail(recs, "grad_cos"), "e": tail(recs, "e_t"), "E": tail(recs, "true_E"),
            "rb": tail(recs, "rho_bar"), "rh": tail(recs, "rho_hat"), "eta": tail(recs, "eta")})

    def avg(rs, x):
        v = [r[x] for r in rs if r[x] is not None]
        return sum(v) / len(v) if v else float("nan")

    lines = [f"# C2 Certificate Report ({os.path.basename(d_)})\n",
             f"Certificate validity (e_t >= ||E||): {valid_ok}/{valid_total} = {100*valid_ok/max(valid_total,1):.1f}%\n",
             "| task/algo | grad_cos | rho_bar | rho_hat | e_t | true_E | tightness e/E | surrogate rb/rh |",
             "|---|---|---|---|---|---|---|---|"]
    for k in sorted(runs):
        rs = runs[k]
        cos, rb, rh, e, E = avg(rs, "cos"), avg(rs, "rb"), avg(rs, "rh"), avg(rs, "e"), avg(rs, "E")
        t = e / E if (E and not math.isnan(E) and E > 0) else float("nan")
        sg = rb / rh if (rh and not math.isnan(rh) and rh > 0) else float("nan")
        lines.append(f"| {k[0]}/{k[1]} | {cos:.3f} | {rb:.3f} | {rh:.3f} | {e:.2e} | {E:.2e} | {t:.2f} | {sg:.2f} |")
    out = "\n".join(lines) + "\n"
    os.makedirs("results", exist_ok=True)
    open(f"results/REPORT_{os.path.basename(d_)}.md", "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
