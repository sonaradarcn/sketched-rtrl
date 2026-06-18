"""Publication figures from result JSONs. Run on a server with matplotlib.
Produces: fig_fidelity_bars, fig_r_interpolation, fig_certificate_tightness, fig_anchor_curves.
"""
import glob, json, math, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

ALGO_ORDER = ["exact", "skrtrl-r64", "skrtrl-r16", "skrtrl-r4", "snap1", "kfrtrl", "uoro", "rflo", "tbptt"]
ALGO_LABEL = {"exact": "Exact RTRL", "skrtrl-r64": "SK-RTRL r=n", "skrtrl-r16": "SK-RTRL r16",
              "skrtrl-r4": "SK-RTRL r4", "snap1": "SnAp-1", "kfrtrl": "KF-RTRL",
              "uoro": "UORO", "rflo": "RFLO/eProp", "tbptt": "TBPTT-25",
              "skrtrl-rp16": "SK-RTRL randproj16", "skrtrl-rp4": "SK-RTRL randproj4"}


def load(globpat):
    runs = defaultdict(list)
    for f in sorted(glob.glob(globpat)):
        d = json.load(open(f))
        a = d["args"]
        runs[(a["task"], a["algo"])].append(d)
    return runs


def tail(recs, k, frac=0.2):
    v = [r[k] for r in recs if r.get(k) is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
    return sum(v[-max(1, int(len(v) * frac)):]) / max(1, int(len(v) * frac)) if v else None


def fig_fidelity_bars(runs, tasks, fname):
    algos = [a for a in ALGO_ORDER if a != "exact" and a != "tbptt"
             and any((t, a) in runs for t in tasks)]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(tasks))
    w = 0.8 / max(len(algos), 1)
    for i, alg in enumerate(algos):
        vals, errs = [], []
        for t in tasks:
            cs = [tail(d["records"], "grad_cos") for d in runs.get((t, alg), [])]
            cs = [c for c in cs if c is not None]
            vals.append(np.mean(cs) if cs else 0)
            errs.append(np.std(cs) if len(cs) > 1 else 0)
        ax.bar(x + i * w, vals, w, yerr=errs, label=ALGO_LABEL.get(alg, alg), capsize=2)
    ax.set_xticks(x + w * len(algos) / 2)
    TASK_LABEL = {"anbn": "$a^nb^n$", "copy": "copy", "adding": "adding", "rotation": "rotation"}
    ax.set_xticklabels([TASK_LABEL.get(t, t) for t in tasks])
    ax.set_ylabel("Gradient cosine vs exact RTRL")
    ax.set_ylim(0, 1.05)
    # legend above the axes so it never overlaps the bars (caption gives the title)
    ax.legend(fontsize=7, ncol=min(len(algos), 6), loc="lower center", bbox_to_anchor=(0.5, 1.01),
              frameon=False, columnspacing=1.0, handletextpad=0.4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=300); fig.savefig(f"{OUT}/{fname[:-4]}.pdf")
    print("saved", fname)


def fig_r_interpolation(runs, task, fname):
    rs = {0: "snap1", 4: "skrtrl-r4", 16: "skrtrl-r16", 64: "skrtrl-r64"}
    xs, ys, es = [], [], []
    for r, alg in rs.items():
        cs = [tail(d["records"], "grad_cos") for d in runs.get((task, alg), [])]
        cs = [c for c in cs if c is not None]
        if cs:
            xs.append(r if r > 0 else 0.5)
            ys.append(np.mean(cs))
            es.append(np.std(cs) if len(cs) > 1 else 0)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("sketch rank r  (r=0.5 ≙ SnAp-1)")
    ax.set_ylabel("gradient cosine vs exact")
    ax.set_title(f"r-interpolation on {task}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=300); fig.savefig(f"{OUT}/{fname[:-4]}.pdf")
    print("saved", fname)


def fig_certificate(sweepdir, fname):
    pts = []
    for f in sorted(glob.glob(f"{sweepdir}/*.json")):
        d = json.load(open(f))
        rb = tail(d["records"], "rho_bar", 0.3)
        e = tail(d["records"], "e_t", 0.3)
        E = tail(d["records"], "true_E", 0.3)
        if rb and e and E and E > 0:
            pts.append((rb, e / E, d["args"]["task"]))
    if not pts:
        print("no certificate points in", sweepdir)
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    for t in sorted(set(p[2] for p in pts)):
        tp = sorted([(p[0], p[1]) for p in pts if p[2] == t])
        ax.plot([p[0] for p in tp], [p[1] for p in tp], marker="o", label=t)
    ax.axvline(1.0, ls="--", color="k", alpha=0.5, label="ρ̄=1")
    ax.set_yscale("log")
    ax.set_xlabel("certified ρ̄ (spectral upper bound)")
    ax.set_ylabel("certificate tightness e_t / ‖E‖")
    ax.set_title("Certificate non-vacuous iff ρ̄ < 1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=300); fig.savefig(f"{OUT}/{fname[:-4]}.pdf")
    print("saved", fname)


def fig_anchor_curves(runs, task, fname):
    fig, ax = plt.subplots(figsize=(6, 4))
    for alg in ALGO_ORDER:
        ds = runs.get((task, alg), [])
        if not ds:
            continue
        curves = []
        for d in ds:
            steps = [r["step"] for r in d["records"] if r.get("metric") is not None]
            mets = [r["metric"] for r in d["records"] if r.get("metric") is not None]
            if steps:
                curves.append((steps, mets))
        if not curves:
            continue
        L = min(len(c[1]) for c in curves)
        steps = curves[0][0][:L]
        arr = np.array([c[1][:L] for c in curves])
        m = arr.mean(0)
        ax.plot(steps, m, label=ALGO_LABEL.get(alg, alg))
        if arr.shape[0] > 1:
            ax.fill_between(steps, m - arr.std(0), m + arr.std(0), alpha=0.15)
    ax.set_xlabel("online step")
    ax.set_ylabel("accuracy" if task in ("copy", "anbn") else "loss")
    ax.set_title(f"Online learning curve: {task}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=300); fig.savefig(f"{OUT}/{fname[:-4]}.pdf")
    print("saved", fname)


if __name__ == "__main__":
    m3 = load(f"{RES}/m3/*.json")
    m31 = load(f"{RES}/m31/*.json")
    m32 = load(f"{RES}/m32/*.json")
    allr = defaultdict(list)
    for src in (m3, m31, m32):
        for k, v in src.items():
            allr[k].extend(v)
    try:
        fig_fidelity_bars(allr, ["copy", "adding", "rotation", "anbn"], "fig_fidelity_bars.png")
    except Exception as e:
        print("fidelity_bars failed:", e)
    for t in ("copy", "rotation", "rotrecall", "anbn"):
        try:
            fig_r_interpolation(allr, t, f"fig_rinterp_{t}.png")
        except Exception as e:
            print(f"rinterp {t} failed:", e)
    for sd in (f"{RES}/c2", f"{RES}/c2sweep"):
        if glob.glob(f"{sd}/*.json"):
            try:
                fig_certificate(sd, f"fig_cert_{os.path.basename(sd)}.png")
            except Exception as e:
                print("cert failed:", e)
    for t in ("copy", "anbn"):
        try:
            fig_anchor_curves(allr, t, f"fig_curve_{t}.png")
        except Exception as e:
            print(f"curve {t} failed:", e)
    print("done -> results/figures/")


def fig_scaling(res, fname):
    """Fidelity-vs-n (left) and memory-vs-n from m0_profile (right)."""
    import json as _json
    # gather all scale dirs + m3 (n=64); bucket every run by its args['n']
    pool = []
    for pat in (f"{res}/m3/*.json", f"{res}/scale/*.json", f"{res}/scale128/*.json",
                f"{res}/scale256/*.json"):
        for f in glob.glob(pat):
            d = _json.load(open(f))
            pool.append((d["args"].get("n", 64), d["args"]["task"], d["args"]["algo"],
                         tail(d["records"], "grad_cos")))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
    for alg, mk in [("skrtrl-r16", "o"), ("skrtrl-r4", "s"), ("snap1", "^"), ("kfrtrl", "v")]:
        xs, ys = [], []
        for n in (64, 128, 256):
            cs = [c for (nn, t, a, c) in pool if nn == n and t == "rotation" and a == alg and c is not None]
            if cs:
                xs.append(n); ys.append(np.mean(cs))
        if xs:
            axL.plot(xs, ys, marker=mk, label=ALGO_LABEL.get(alg, alg))
    axL.set_xlabel("hidden size n"); axL.set_ylabel("grad cosine vs exact (rotation)")
    axL.set_title("Fidelity is scale-invariant"); axL.set_xscale("log", base=2)
    axL.legend(fontsize=8); axL.grid(alpha=0.3)
    # right: memory vs n from m0_profile.json
    try:
        prof = _json.load(open(f"{res}/m0_profile.json"))
        series = {}
        for rec in prof:
            key = rec["algo"] if rec["algo"] != "skrtrl" else f"skrtrl-r{rec['r']}"
            series.setdefault(key, []).append((rec["n"], rec.get("peak_MB")))
        for key in ("exact", "skrtrl-r16", "skrtrl-r4", "snap1"):
            if key in series:
                pts = sorted([(n, m) for n, m in series[key] if isinstance(m, (int, float))])
                if pts:
                    axR.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                             label=ALGO_LABEL.get(key, key))
        axR.axhline(12288, ls="--", color="r", alpha=0.5, label="12 GB GPU")
        axR.set_xlabel("hidden size n"); axR.set_ylabel("peak memory (MB)")
        axR.set_title("Memory: exact O(n³) vs SK-RTRL O(n²r)")
        axR.set_yscale("log"); axR.set_xscale("log", base=2)
        axR.legend(fontsize=8); axR.grid(alpha=0.3)
    except Exception as e:
        print("memory panel failed:", e)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=300); fig.savefig(f"{OUT}/{fname[:-4]}.pdf")
    print("saved", fname)


try:
    fig_scaling(RES, "fig_scaling.png")
except Exception as e:
    print("scaling fig failed:", e)
