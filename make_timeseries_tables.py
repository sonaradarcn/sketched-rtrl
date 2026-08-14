"""Rebuild the two time-series tables of the paper from the released raw runs, and check
every cell against the published value.

  Table 6 (tab:timeseries)  chaotic systems, from results/ts/
  Table 7 (tab:realts)      Sunspot + Santa Fe laser, from results/round1/real/

Convention, which is the one the paper's tables use and which make_ts_report.py /
make_m3_report.py do NOT use (those take a 20%-tail mean and a sample standard deviation,
and therefore print numbers a little different from the tables):

  * per run, the reported value is the mean of the LAST FIVE logged records of the field
    (`metric` for NMSE, `grad_cos` for the gradient cosine);
  * across seeds, mean +/- POPULATION standard deviation (ddof = 0);
  * the "grad cosine" column is a single number per method: the same per-run last-five mean,
    averaged over every run of the block, i.e. over all seeds and all systems at once.

Every printed cell is compared with the published table at the printed precision. If any
cell disagrees the script exits non-zero and names it, so this doubles as a check that the
released records still support the paper.

Usage:  python make_timeseries_tables.py [--out results/TIMESERIES_TABLES.md]
"""
import argparse
import glob
import io
import json
import math
import os
from collections import defaultdict

TS_DIR = os.path.join("results", "ts")
REAL_DIR = os.path.join("results", "round1", "real")

TS_TASKS = ["henon", "mackeyglass", "lorenz"]
REAL_TASKS = ["sunspot", "laser"]
ORDER = ["exact", "skrtrl-r32", "skrtrl-r16", "skrtrl-r4", "snap1", "kfrtrl", "rflo", "uoro", "tbptt"]
LABEL = {"exact": "Exact RTRL", "skrtrl-r32": "SK-RTRL r=32", "skrtrl-r16": "SK-RTRL r=16",
         "skrtrl-r4": "SK-RTRL r=4", "snap1": "SnAp-1", "kfrtrl": "KF-RTRL", "rflo": "RFLO",
         "uoro": "UORO", "tbptt": "TBPTT"}

# Published values: method -> ([(mean, std) per task], grad cosine or None).
PUBLISHED_TS = {
    "exact":      ([(0.0052, 0.0010), (0.0180, 0.0218), (0.0051, 0.0040)], None),
    "skrtrl-r32": ([(0.0069, 0.0016), (0.0070, 0.0030), (0.0130, 0.0184)], 0.954),
    "skrtrl-r16": ([(0.0064, 0.0007), (0.0105, 0.0091), (0.0128, 0.0296)], 0.938),
    "skrtrl-r4":  ([(0.0059, 0.0007), (0.0065, 0.0053), (0.0038, 0.0021)], 0.927),
    "snap1":      ([(0.0052, 0.0012), (0.0050, 0.0010), (0.0024, 0.0008)], 0.505),
    "kfrtrl":     ([(0.0384, 0.0709), (0.0286, 0.0181), (0.0126, 0.0079)], 0.512),
    "rflo":       ([(0.0704, 0.0163), (0.0572, 0.0331), (0.0442, 0.0124)], 0.288),
    "uoro":       ([(0.6583, 0.2966), (0.0497, 0.0238), (0.0425, 0.0431)], 0.056),
    "tbptt":      ([(0.0311, 0.0030), (0.0080, 0.0010), (0.0054, 0.0007)], None),
}
PUBLISHED_REAL = {
    "exact":      ([(0.1451, 0.0061), (0.0199, 0.0041)], None),
    "skrtrl-r16": ([(0.1456, 0.0042), (0.0217, 0.0021)], 0.883),
    "skrtrl-r4":  ([(0.1460, 0.0049), (0.0239, 0.0037)], 0.867),
    "snap1":      ([(0.1437, 0.0019), (0.0214, 0.0012)], 0.439),
    "kfrtrl":     ([(0.1613, 0.0153), (0.1788, 0.2339)], 0.431),
    "rflo":       ([(0.1536, 0.0038), (0.2759, 0.2241)], 0.186),
    "uoro":       ([(0.2119, 0.0461), (0.4145, 0.1523)], 0.046),
}


def _tail_mean(records, field, k=5):
    v = [r[field] for r in records
         if r.get(field) is not None and not (isinstance(r[field], float) and math.isnan(r[field]))]
    return sum(v[-k:]) / len(v[-k:]) if v else None


def _load(directory, tasks):
    """(task, algo) -> {seed: (nmse, cosine)} and the raw per-record cosines per (task, algo)."""
    runs = defaultdict(dict)
    cosines = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(directory, "*.json"))):
        d = json.load(open(f))
        a = d.get("args", {})
        if a.get("task") not in tasks:
            continue
        recs = d.get("records", [])
        runs[(a["task"], a["algo"])][a["seed"]] = (_tail_mean(recs, "metric"),
                                                  _tail_mean(recs, "grad_cos"))
        cosines[(a["task"], a["algo"])] += [r["grad_cos"] for r in recs
                                            if r.get("grad_cos") is not None]
    return runs, cosines


def _mean_popstd(vals):
    m = sum(vals) / len(vals)
    return m, (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5


def build(directory, tasks, published, title, headers):
    runs, cosines = _load(directory, tasks)
    lines = [title, "", "| Method | " + " | ".join(headers) + " | grad cosine |",
             "|---" * (len(headers) + 2) + "|"]
    bad = []
    for algo in ORDER:
        if algo not in published:
            continue
        cells, pub_cells, pub_cos = [], published[algo][0], published[algo][1]
        for i, task in enumerate(tasks):
            seeds = runs.get((task, algo), {})
            vals = [v[0] for v in seeds.values() if v[0] is not None]
            if not vals:
                bad.append("%s/%s: no released runs" % (task, algo))
                cells.append("--")
                continue
            m, sd = _mean_popstd(vals)
            cells.append("%.4f +/- %.4f (%d)" % (m, sd, len(vals)))
            pm, ps = pub_cells[i]
            if abs(round(m, 4) - pm) > 1.1e-4 or abs(round(sd, 4) - ps) > 1.1e-4:
                bad.append("%s/%s: %.4f+/-%.4f vs published %.4f+/-%.4f" % (task, algo, m, sd, pm, ps))
        if pub_cos is None:
            cells.append("--")
        else:
            allc = [v[1] for task in tasks for v in runs.get((task, algo), {}).values()
                    if v[1] is not None]
            c = sum(allc) / len(allc) if allc else None
            cells.append("%.3f" % c if c is not None else "--")
            if c is None or abs(round(c, 3) - pub_cos) > 1.1e-3:
                bad.append("%s: grad cosine %s vs published %.3f" % (algo, c, pub_cos))
        lines.append("| %s | %s |" % (LABEL[algo], " | ".join(cells)))
    return "\n".join(lines) + "\n", bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("results", "TIMESERIES_TABLES.md"))
    args = ap.parse_args()
    t6, bad6 = build(TS_DIR, TS_TASKS, PUBLISHED_TS,
                     "## Table 6 - online next-step prediction on chaotic systems",
                     ["Henon NMSE", "Mackey-Glass NMSE", "Lorenz NMSE"])
    t7, bad7 = build(REAL_DIR, REAL_TASKS, PUBLISHED_REAL,
                     "## Table 7 - Sunspot and Santa Fe laser",
                     ["Sunspot NMSE", "Laser NMSE"])
    body = ("# Time-series tables rebuilt from the released raw runs\n\n"
            "Per run: mean of the last five logged records. Across seeds: mean +/- population\n"
            "standard deviation; the seed count is in parentheses. The grad-cosine column applies\n"
            "the same per-run value over all seeds and all systems of the block at once.\n\n"
            + t6 + "\n" + t7)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
    print(body)
    bad = bad6 + bad7
    if bad:
        print("\nMISMATCH against the published tables:")
        for b in bad:
            print("  -", b)
        raise SystemExit(1)
    print("all cells of Tables 6 and 7 match the published values")


if __name__ == "__main__":
    main()
