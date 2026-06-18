"""Round-2 figures F1-F4 from merged Round-1 results.

F1  multi-step horizon curves      (NMSE vs horizon, per task, per method)
F2  grad-cosine <-> NMSE scatter    (fidelity is not the same as task error)
F3  adaptive-rank trajectory        (r_t, eta_t, e_t, rho_hat, rho_bar over time)
F4  memory-time Pareto              (peak MB vs wall-clock, per method)

Each figure is written as vector PDF + 300-dpi PNG to paper/figures/ and is skipped
(with a printed note) when its inputs are absent, so no placeholder/fabricated data is
ever drawn. Inputs are read from a local merged tree (collect_round1.sh pulls the
per-server results/round1/* and results/ts/* into these dirs first).
"""
import argparse, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FNAME = re.compile(r"^(?P<task>[a-z0-9]+)_(?P<method>[a-z0-9\-]+)_s(?P<seed>\d+)(?:_.*)?\.json$")
OUT = "../paper/figures"   # run from code/; writes to the paper's figures dir


def _last(recs, field, tail=5):
    vals = [r[field] for r in recs if r.get(field) is not None]
    return float(np.mean(vals[-tail:])) if vals else None


def _scan(dirs):
    rows = []
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.json")):
            m = FNAME.match(os.path.basename(p))
            if not m:
                continue
            try:
                j = json.load(open(p))
            except Exception:
                continue
            recs = j.get("records", [])
            rows.append({"task": m["task"], "method": m["method"], "seed": int(m["seed"]),
                         "horizon": j.get("args", {}).get("horizon", 1),
                         "n": j.get("args", {}).get("n", 64),
                         "nmse": _last(recs, "metric"), "grad_cos": _last(recs, "grad_cos"),
                         "peak_MB": j.get("peak_MB"), "wall_s": j.get("wall_s")})
    return rows


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


METHOD_ORDER = ["exact", "skrtrl-r16", "skrtrl-r4", "snap1", "rflo", "uoro", "kfrtrl"]
LABEL = {"exact": "exact RTRL", "skrtrl-r16": "SK-RTRL r16", "skrtrl-r4": "SK-RTRL r4",
         "skrtrl-r32": "SK-RTRL r32", "skrtrl-r8": "SK-RTRL r8", "skrtrl-r2": "SK-RTRL r2",
         "snap1": "SnAp-1", "rflo": "RFLO", "uoro": "UORO", "kfrtrl": "KF-RTRL"}


def fig_horizon(rows):
    tasks = ["henon", "mackeyglass", "lorenz"]
    rows = [r for r in rows if r["task"] in tasks and r["nmse"] is not None]
    if not rows:
        print("F1 skipped: no horizon data"); return
    fig, axes = plt.subplots(1, len(tasks), figsize=(11, 3.4), sharey=False)
    for ax, task in zip(axes, tasks):
        for meth in METHOD_ORDER:
            pts = {}
            for r in rows:
                if r["task"] == task and r["method"] == meth:
                    pts.setdefault(r["horizon"], []).append(r["nmse"])
            if not pts:
                continue
            hs = sorted(pts)
            mean = [np.mean(pts[h]) for h in hs]
            sd = [np.std(pts[h]) for h in hs]
            ax.errorbar(hs, mean, yerr=sd, marker="o", ms=4, capsize=2, label=LABEL.get(meth, meth))
        ax.set_title(task); ax.set_xlabel("horizon $h$"); ax.set_yscale("log")
    axes[0].set_ylabel("NMSE")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, "fig_horizon_nmse")


def fig_scatter(rows):
    pts = [r for r in rows if r["grad_cos"] is not None and r["nmse"] is not None
           and r["horizon"] == 1]
    if len(pts) < 4:
        print("F2 skipped: insufficient grad_cos/NMSE pairs"); return
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    methods = sorted(set(p["method"] for p in pts))
    cmap = plt.get_cmap("tab10")
    for i, meth in enumerate(methods):
        mp = [p for p in pts if p["method"] == meth]
        ax.scatter([p["grad_cos"] for p in mp], [p["nmse"] for p in mp],
                   s=22, color=cmap(i % 10), label=LABEL.get(meth, meth), alpha=0.8)
    ax.set_xlabel("gradient cosine vs exact (fidelity)")
    ax.set_ylabel("NMSE (task error)"); ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, "fig_fidelity_vs_error")


def fig_trajectory(path):
    if not path or not os.path.isfile(path):
        print("F3 skipped: trajectory json not found:", path); return
    j = json.load(open(path))
    recs = j.get("records", [])
    steps = [r["step"] for r in recs]
    fig, ax1 = plt.subplots(figsize=(6, 3.6))
    ax1.plot(steps, [r.get("rank") for r in recs], color="C0", lw=2, label="rank $r_t$")
    ax1.set_xlabel("step"); ax1.set_ylabel("rank $r_t$", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    # certificate quantities span many decades (e_t blows up in the untrained transient,
    # then the spectral clip makes it contractive); a log axis keeps all of them visible.
    for field, c, lab in [("eta", "C1", r"$\eta_t$"), ("e_t", "C2", "$e_t$"),
                          ("rho_hat", "C3", r"$\hat\rho_t$"), ("rho_bar", "C4", r"$\bar\rho_t$")]:
        xs = [s for s, r in zip(steps, recs) if r.get(field) is not None and r.get(field) > 0]
        ys = [r[field] for r in recs if r.get(field) is not None and r.get(field) > 0]
        if ys:
            ax2.plot(xs, ys, color=c, lw=1.0, label=lab)
    ax2.set_yscale("log"); ax2.set_ylabel("certificate quantities (log)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig_adaptive_trajectory")


def fig_pareto(rows):
    # Memory-time trade-off at the LARGEST benchmarked n (shadow OFF, so peak_MB isolates each
    # method). At large n exact RTRL's O(n^3) influence matrix makes memory the binding
    # constraint while the SK-RTRL sketch is O(n^2 r); exact is faster in wall-clock (no SVD),
    # so the two occupy distinct Pareto corners. At small n exact wins both axes -- the sketch's
    # memory advantage is asymptotic (see report). Uses clean results/membench measurements.
    pts = [r for r in rows if r["peak_MB"] and r["wall_s"]]
    if not pts:
        print("F4 skipped: no memory/time data"); return
    n_target = max(r["n"] for r in pts)            # largest benchmarked width
    pts = [r for r in pts if r["n"] == n_target]
    if len({r["method"] for r in pts}) < 3:
        print(f"F4 skipped: <3 methods at n={n_target}"); return
    agg = {}
    for r in pts:
        agg.setdefault(r["method"], {"mb": [], "s": []})
        agg[r["method"]]["mb"].append(r["peak_MB"])
        agg[r["method"]]["s"].append(r["wall_s"])
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    offs = {"exact": (-6, 6), "snap1": (6, -4), "skrtrl-r16": (6, 2),
            "skrtrl-r4": (6, -10), "kfrtrl": (6, 4)}
    for meth, v in sorted(agg.items()):
        x, y = np.mean(v["mb"]), np.mean(v["s"])
        ax.scatter(x, y, s=55)
        ax.annotate(LABEL.get(meth, meth), (x, y), fontsize=8,
                    xytext=offs.get(meth, (6, 4)), textcoords="offset points")
    ax.set_xlabel(f"peak memory (MB) at $n={n_target}$"); ax.set_ylabel("wall-clock (s, 1.5k steps)")
    ax.set_xscale("log")
    fig.tight_layout()
    _save(fig, "fig_memory_time_pareto")


RL_LABEL = {"rtu": "RTU", "lru": "LRU", "snap1": "SnAp-1", "skrtrl-r16": "SK-RTRL r16",
            "exact": "exact RTRL", "tbptt": "TBPTT"}


def fig_rl_curves(m5_dir, env_lens=(10, 20, 40)):
    """RL learning curves: running success rate vs training step, mean over seeds, per algo,
    one panel per corridor length. Filenames: tmaze<len>_<algo>_s<seed>.json with records
    carrying 'step' and 'succ'."""
    fre = re.compile(r"^tmaze(\d+)_([a-z0-9\-]+)_s(\d+)\.json$")
    data = {}   # (len, algo) -> list of (steps, succ) per seed
    for p in glob.glob(os.path.join(m5_dir, "*.json")):
        m = fre.match(os.path.basename(p))
        if not m:
            continue
        j = json.load(open(p)); recs = j.get("records", [])
        steps = [r["step"] for r in recs if r.get("success") is not None]
        succ = [r["success"] for r in recs if r.get("success") is not None]
        if steps:
            data.setdefault((int(m.group(1)), m.group(2)), []).append((steps, succ))
    if not data:
        print("RL curves skipped: no m5iso data"); return
    lens = [L for L in env_lens if any(k[0] == L for k in data)]
    fig, axes = plt.subplots(1, len(lens), figsize=(4.0 * len(lens), 3.4), sharey=True)
    if len(lens) == 1:
        axes = [axes]
    for ax, L in zip(axes, lens):
        for algo in ["rtu", "exact", "lru", "skrtrl-r16", "snap1", "tbptt"]:
            runs = data.get((L, algo))
            if not runs:
                continue
            grid = runs[0][0]                       # assume shared logging grid
            ys = np.array([np.interp(grid, s, v) for s, v in runs])
            mean = ys.mean(0)
            ax.plot(grid, mean, label=RL_LABEL.get(algo, algo), lw=1.4)
        ax.set_title(f"corridor {L}"); ax.set_xlabel("step")
    axes[0].set_ylabel("success rate (running)")
    axes[0].legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig_rl_curves")


def fig_scaling_clean(fid_dirs, mem_dir):
    """Regenerate fig_scaling from CLEAN data: left = rotation grad-cos vs n (fidelity is
    width-invariant), right = peak memory vs n from shadow-off membench (exact O(n^3) reaches
    the 12 GB budget at n=512 while SK-RTRL stays O(n^2 r))."""
    import matplotlib.pyplot as plt
    fid = {}   # (algo) -> {n: [grad_cos,...]}
    for d in fid_dirs:
        for p in glob.glob(os.path.join(d, "rotation_*.json")):
            m = FNAME.match(os.path.basename(p))
            if not m:
                continue
            j = json.load(open(p)); a = j.get("args", {})
            gc = _last(j.get("records", []), "grad_cos")
            if gc is not None:
                fid.setdefault(m["method"], {}).setdefault(a.get("n", 64), []).append(gc)
    mem = {}   # algo -> {n: peak_MB}
    for p in glob.glob(os.path.join(mem_dir, "*.json")):
        j = json.load(open(p)); a = j.get("args", {})
        if j.get("peak_MB"):
            mem.setdefault(a["algo"], {})[a["n"]] = j["peak_MB"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.6))
    for meth in ["skrtrl-r16", "skrtrl-r4", "snap1"]:
        if meth in fid:
            ns = sorted(fid[meth])
            axL.plot(ns, [np.mean(fid[meth][n]) for n in ns], marker="o", label=LABEL.get(meth, meth))
    axL.set_xlabel("hidden size $n$"); axL.set_ylabel("grad cosine vs exact (rotation)")
    axL.set_title("Fidelity is width-invariant"); axL.set_xscale("log", base=2)
    axL.legend(fontsize=7, frameon=False)
    for meth in ["exact", "skrtrl-r16", "skrtrl-r4", "snap1"]:
        if meth in mem:
            ns = sorted(mem[meth])
            axR.plot(ns, [mem[meth][n] for n in ns], marker="o", label=LABEL.get(meth, meth))
    axR.axhline(12288, ls="--", color="r", lw=1, label="12 GB GPU")
    axR.set_xlabel("hidden size $n$"); axR.set_ylabel("peak memory (MB)")
    axR.set_title(r"Memory: exact $O(n^3)$ vs SK-RTRL $O(n^2 r)$")
    axR.set_yscale("log"); axR.set_xscale("log", base=2); axR.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, "fig_scaling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon_dirs", nargs="+",
                    default=["results/round1/horizon", "results/ts"])
    ap.add_argument("--scatter_dirs", nargs="+",
                    default=["results/round1/real", "results/ts"])
    ap.add_argument("--pareto_dirs", nargs="+",
                    default=["results/membench"])
    ap.add_argument("--traj", default="results/round1/traj/rotation_adaptive-eta_s0_traj.json")
    args = ap.parse_args()
    fig_horizon(_scan(args.horizon_dirs))
    fig_scatter(_scan(args.scatter_dirs))
    fig_pareto(_scan(args.pareto_dirs))
    fig_trajectory(args.traj)
    fig_scaling_clean(["results/m3", "results/scale", "results/scale256"], "results/membench")
    fig_rl_curves("results/m5iso")


if __name__ == "__main__":
    main()
