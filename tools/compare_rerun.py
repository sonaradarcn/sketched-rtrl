"""Compare a GPU re-run of the README worked example with its released record.

Run from the repository root after the worked-example command of README.md has written
results/rerun/d4_eval/adding_skrtrl-r16_s0_lr3e-03.json:

    python tools/compare_rerun.py [--released PATH] [--rerun PATH] [--rtol R] [--atol A]

What is compared, value by value:
  * "records": the same number of logged records, and every field of every record
    (all numeric fields, including inf values);
  * "cert_summary": every certificate counter and fraction;
  * "svd_stats" and "svd_fallbacks": the SVD-driver counts;
  * "checkpoints": every field of every sketch checkpoint;
  * "args": every argument apart from "outdir".
Each numeric pair is classed as exact (equal floats, or both NaN), within tolerance only
(finite, |a - b| <= atol + rtol * |a|), or different; a non-numeric pair must be equal.
A record counts as identical when all its fields are exact. wall_s and peak_MB are printed,
not compared. The final-window means of metric and grad_cos (records in the last 20% of the
steps, as every table uses) are printed beside the spread of the released seeds of the pair.

Exit status 0 when nothing differs beyond tolerance, 1 otherwise.
"""
import argparse
import glob
import json
import math
import statistics
import sys

REL = "results/r2/d4_eval/adding_skrtrl-r16_s0_lr3e-03.json"
NEW = "results/rerun/d4_eval/adding_skrtrl-r16_s0_lr3e-03.json"
SEEDS = "results/r2/d4_eval/adding_skrtrl-r16_s*_lr3e-03.json"
LABEL = {"exact": "identical", "close": "within tolerance only", "diff": "DIFFERENT"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class Tally:
    """Counts exact, within-tolerance and different values over the whole comparison."""

    def __init__(self, rtol, atol):
        self.rtol, self.atol = rtol, atol
        self.exact = self.close = 0
        self.diffs = []

    def value(self, where, a, b):
        if is_num(a) and is_num(b):
            fa, fb = float(a), float(b)
            if fa == fb or (math.isnan(fa) and math.isnan(fb)):
                self.exact += 1
                return "exact"
            if math.isfinite(fa) and math.isfinite(fb) and \
                    abs(fa - fb) <= self.atol + self.rtol * abs(fa):
                self.close += 1
                return "close"
        elif a == b:
            self.exact += 1
            return "exact"
        self.diffs.append("%s: released %r rerun %r" % (where, a, b))
        return "diff"

    def mapping(self, where, a, b, skip=()):
        """Compare two dicts key by key; return the worst class seen."""
        worst = "exact"
        for k in sorted((set(a) | set(b)) - set(skip)):
            if k not in a or k not in b:
                self.diffs.append("%s.%s: present in only one file" % (where, k))
                worst = "diff"
                continue
            c = self.value("%s.%s" % (where, k), a[k], b[k])
            if c == "diff" or (c == "close" and worst == "exact"):
                worst = c
        return worst


def window(rec, field):
    steps = rec["args"]["steps"]
    v = [r[field] for r in rec["records"]
         if r.get("step", -1) >= 0.8 * steps and is_num(r.get(field))]
    return sum(v) / len(v)


def main():
    ap = argparse.ArgumentParser(description="Compare a re-run with its released record.")
    ap.add_argument("--released", default=REL)
    ap.add_argument("--rerun", default=NEW)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-12)
    o = ap.parse_args()
    a, b = load(o.released), load(o.rerun)
    t = Tally(o.rtol, o.atol)
    print("released: %s" % o.released)
    print("rerun:    %s" % o.rerun)
    print("tolerance: |a - b| <= %g + %g * |a| (finite values); exact counted separately"
          % (o.atol, o.rtol))

    # 1. Every logged record.
    ra, rb = a["records"], b["records"]
    if len(ra) != len(rb):
        t.diffs.append("records: released has %d, rerun has %d" % (len(ra), len(rb)))
    n_rec = min(len(ra), len(rb))
    cls = [t.mapping("records[%d](step %s)" % (i, ra[i].get("step")), ra[i], rb[i])
           for i in range(n_rec)]
    n_val = sum(len(set(ra[i]) | set(rb[i])) for i in range(n_rec))
    print("records       %d released, %d rerun; %d records compared (%d values): %d identical, "
          "%d within tolerance only, %d different"
          % (len(ra), len(rb), n_rec, n_val, cls.count("exact"), cls.count("close"),
             cls.count("diff")))

    # 2. Certificate counters, SVD-driver counts, checkpoints.
    for key in ("cert_summary", "svd_stats", "svd_fallbacks"):
        da, db = a.get(key, {}), b.get(key, {})
        print("%-13s %d fields: %s" % (key, len(set(da) | set(db)),
                                        LABEL[t.mapping(key, da, db)]))
    print("svd_stats released %s" % a.get("svd_stats"))
    print("svd_stats rerun    %s" % b.get("svd_stats"))
    ca, cb = a.get("checkpoints", []), b.get("checkpoints", [])
    if len(ca) != len(cb):
        t.diffs.append("checkpoints: released has %d, rerun has %d" % (len(ca), len(cb)))
    ck = [t.mapping("checkpoints[%d]" % i, ca[i], cb[i]) for i in range(min(len(ca), len(cb)))]
    print("checkpoints   %d compared: %d identical, %d within tolerance only, %d different"
          % (len(ck), ck.count("exact"), ck.count("close"), ck.count("diff")))

    # 3. Arguments apart from the output directory.
    keys = (set(a["args"]) | set(b["args"])) - {"outdir"}
    print("args          %d fields apart from outdir: %s"
          % (len(keys), LABEL[t.mapping("args", a["args"], b["args"], skip=("outdir",))]))

    # 4. Final-window means beside the released seed spread (what the tables report).
    seed_files = sorted(glob.glob(SEEDS))
    for field in ("metric", "grad_cos"):
        rel, new = window(a, field), window(b, field)
        s = [window(load(f), field) for f in seed_files]
        print("%-9s released %.6f  rerun %.6f  diff %+.2e  | released seeds: mean %.6f sd %.6f "
              "min %.6f max %.6f (n=%d)" % (field, rel, new, new - rel, statistics.mean(s),
                                            statistics.stdev(s), min(s), max(s), len(s)))
    print("wall_s released %.0f rerun %.0f; peak_MB released %s rerun %s (not compared)"
          % (a.get("wall_s", 0), b.get("wall_s", 0), a.get("peak_MB"), b.get("peak_MB")))

    print("all values compared: %d exact, %d within tolerance only, %d different"
          % (t.exact, t.close, len(t.diffs)))
    for d in t.diffs[:20]:
        print("DIFF", d)
    ok = not t.diffs
    print("compare_rerun: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
