"""Revision data-provenance report (F-10): compare the 2026-08 local re-runs under
results/revise_repro/ against (a) the original raw JSONs under code/results/ and
(b) the values quoted in the manuscript.

Usage (from the repository root):  python make_revise_repro_report.py
Writes results/revise_repro/COMPARISON.md and COMPARISON.json.
"""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))            # repository root
ORIG = os.path.join(ROOT, "results")                         # original raw JSONs
# In the released repository the re-runs sit in results/revise_repro; in the authors'
# working tree the repository is a `code/` subdirectory and they sit one level up.
_CANDIDATES = [
    os.path.join(ROOT, "results", "revise_repro"),
    os.path.join(os.path.dirname(ROOT), "results", "revise_repro"),
]
REPRO = next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[0])

# Values as printed in the manuscript.
PAPER_E6 = {"clip0": [0.011, 0.009, 0.022], "clip0.9": [0.0054, 0.0057, 0.0053]}
PAPER_MEM = {"exact": [150, 1062, 3520, 8293], "skrtrl-r16": [55, 167, 352, 614]}
NS = [128, 256, 384, 512]


def load(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def final_metric(d):
    return d["records"][-1]["metric"]


def e6_rows():
    """Per-seed final NMSE, lorenz / skrtrl-r8, clip 0 vs 0.9."""
    rows = []
    for clip in ("clip0", "clip0.9"):
        for s in (0, 1, 2):
            fn = f"lorenz_skrtrl-r8_s{s}_{clip}.json"
            o = load(os.path.join(ORIG, "round1", "ablate_rank", fn))
            r = load(os.path.join(REPRO, "e6_clip", fn))
            rows.append({
                "clip": clip, "seed": s,
                "paper": PAPER_E6[clip][s],
                "original": None if o is None else final_metric(o),
                "repro": None if r is None else final_metric(r),
                "orig_peak_MB": None if o is None else o["peak_MB"],
                "repro_peak_MB": None if r is None else r["peak_MB"],
                "orig_wall_s": None if o is None else o["wall_s"],
                "repro_wall_s": None if r is None else r["wall_s"],
            })
    return rows


def mem_rows():
    rows = []
    for algo in ("exact", "snap1", "skrtrl-r4", "skrtrl-r16"):
        for i, n in enumerate(NS):
            fn = f"anbn_{algo}_s0_n{n}.json"
            o = load(os.path.join(ORIG, "membench", fn))
            r = load(os.path.join(REPRO, "membench", fn))
            rows.append({
                "algo": algo, "n": n,
                "paper": PAPER_MEM.get(algo, [None] * 4)[i],
                "orig_peak_MB": None if o is None else o["peak_MB"],
                "repro_peak_MB": None if r is None else r["peak_MB"],
                "orig_wall_s": None if o is None else o["wall_s"],
                "repro_wall_s": None if r is None else r["wall_s"],
            })
    return rows


def fmt(v, nd=4):
    return "--" if v is None else f"{v:.{nd}f}"


def main():
    e6, mem = e6_rows(), mem_rows()
    os.makedirs(REPRO, exist_ok=True)
    json.dump({"e6_clip": e6, "membench": mem}, open(os.path.join(REPRO, "COMPARISON.json"), "w"), indent=1)

    L = []
    L.append("# Revision re-run vs. original raw data vs. manuscript\n")
    L.append("Original JSONs: `results/` (round-1 campaign). Re-runs: `results/revise_repro/` "
             "(local 2 x RTX 3080, Python 3.10.20, torch 2.6.0+cu124).\n")

    L.append("\n## E6 rank x spectral clip -- Lorenz, SK-RTRL r=8, n=64, batch 8, 15000 steps\n")
    L.append("Final NMSE = `metric` of the last logged record (step 14750).\n")
    L.append("\n| clip | seed | paper | original JSON | re-run | re-run - original | peak MB (orig / repro) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in e6:
        d = ("--" if r["repro"] is None or r["original"] is None
             else f"{r['repro'] - r['original']:+.4f}")
        L.append(f"| {r['clip']} | {r['seed']} | {fmt(r['paper'])} | {fmt(r['original'])} | "
                 f"{fmt(r['repro'])} | {d} | {fmt(r['orig_peak_MB'],1)} / {fmt(r['repro_peak_MB'],1)} |")

    L.append("\n## Fig. 9 memory benchmark -- anbn, seed 0, batch 8, 1500 steps, shadow OFF\n")
    L.append("\n| algo | n | paper (MB) | original (MB) | re-run (MB) | identical? | wall s (orig / repro) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in mem:
        same = ("--" if r["repro_peak_MB"] is None or r["orig_peak_MB"] is None
                else ("yes" if r["repro_peak_MB"] == r["orig_peak_MB"] else "NO"))
        L.append(f"| {r['algo']} | {r['n']} | {r['paper'] if r['paper'] is not None else '--'} | "
                 f"{fmt(r['orig_peak_MB'],3)} | {fmt(r['repro_peak_MB'],3)} | {same} | "
                 f"{fmt(r['orig_wall_s'],1)} / {fmt(r['repro_wall_s'],1)} |")

    # --- stability statistics: why the clip-0 finals move but the clip-0.9 finals do not ---
    L.append("\n## E6 stability statistics (post-washout = logged steps >= 1000)\n")
    L.append("The last-step NMSE is a poor summary of an *unstable* run. These aggregate\n"
             "statistics separate the reproducible signal (the clip suppresses excursions)\n"
             "from the irreproducible one (where the last excursion happens to land).\n")
    L.append("\n| clip | seed | source | final | max post-washout | # points > 0.02 | median post-washout |")
    L.append("|---|---|---|---|---|---|---|")
    stab = []
    for clip in ("clip0", "clip0.9"):
        for s in (0, 1, 2):
            fn = f"lorenz_skrtrl-r8_s{s}_{clip}.json"
            for src, d in (("original", load(os.path.join(ORIG, "round1", "ablate_rank", fn))),
                           ("re-run", load(os.path.join(REPRO, "e6_clip", fn)))):
                if d is None:
                    continue
                post = sorted(x["metric"] for x in d["records"] if x["step"] >= 1000)
                last = d["records"][-1]["metric"]
                med = post[len(post) // 2]
                nex = sum(1 for v in post if v > 0.02)
                stab.append({"clip": clip, "seed": s, "source": src, "final": last,
                             "max_post": post[-1], "n_gt_002": nex, "median_post": med})
                L.append(f"| {clip} | {s} | {src} | {last:.4f} | {post[-1]:.4f} | {nex} | {med:.4f} |")
    json.dump({"e6_clip": e6, "membench": mem, "e6_stability": stab},
              open(os.path.join(REPRO, "COMPARISON.json"), "w"), indent=1)

    out = os.path.join(REPRO, "COMPARISON.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
