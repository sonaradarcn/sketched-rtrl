"""D4 (unified protocol) result tables and paired statistics for the second-round revision.

Rebuilds, from `results/r2/d4_eval/` alone, the three result tables that the revision
replaces, plus the learning-rate appendix table and the all-pairs statistics demanded by
R3r1-10 ("reporting confidence intervals and effect sizes for all principal comparisons,
rather than only selected comparisons against SnAp-1").

Protocol -- identical to the one `make_r2_d4_select.py` used to pick the learning rates, and
imported from that module so the two cannot disagree:

  per-run value     mean of the field over every logged record with `step >= frac*args.steps`
                    (the last-20 % window, `--agg window20`, the default).  `--agg tail5`
                    switches to the mean of the last five logged records, the convention of
                    `make_timeseries_tables.py` and of published Tables 6-7, for a
                    like-for-like comparison against the earlier numbers.
  across seeds      mean +/- POPULATION standard deviation (`--ddof 0`, the convention of the
                    published tables), the seed being the unit of aggregation -- never the
                    pooled record set.
  direction         `metric` is accuracy for the cross-entropy tasks (copy, anbn) and the MSE
                    value for every other D4 task; `grad_cos` is higher-is-better everywhere.
                    The direction table is `make_r2_d4_select.HIGHER_IS_BETTER` and it is
                    re-verified against `skrtrl/tasks.py` on every run.
  protocol cell     10 seeds (0-9) x 20 000 steps, shadow on, at the stage-2 learning rate of
                    `results/r2/D4_SELECT_stage2.json`.  Any cell that misses this is listed
                    at the top of D4_TABLES.md and carries a dagger in the LaTeX tables.

Statistics, part 1 -- the pre-registered set (`D4_STATS.md`): for every task and both metrics,
the ten comparisons fixed in advance -- SK-RTRL r=16 against each of the eight other estimators,
plus r=4 and r=32 against SnAp-1 -- each with a paired Wilcoxon signed-rank p, a Holm correction
over the family of that (task, metric), a 10 000-resample paired bootstrap 95 % CI, Cohen's d_z
and Cliff's delta.  The seed is the pairing unit; the Holm family size is the number of
*structurally* applicable planned comparisons, fixed before the data are seen, so a comparison
whose runs are missing is reported as n/a while still costing family size (conservative).
180 planned, 162 applicable.

Statistics, part 2 -- the FULL MATRIX (`--full-matrix`, on by default; `D4_STATS.md`,
`D4_STATS.json`, `tab_r2_stats.tex`): all three ranks (r=4/16/32) against all six other
estimators (exact, SnAp-1, KF-RTRL, RFLO, UORO, TBPTT) on all nine tasks and both metrics, which
is 3x6x9x2 = 324 planned cells and 270 structurally applicable ones (the gradient cosine is
undefined for exact RTRL and for TBPTT: 3x2x9 = 54 undefined).  270 is therefore the correct
"planned comparisons" count for the records the body of the paper quotes -- the "108 of 108"
gradient-cosine sweep (3 ranks x 4 approximate estimators x 9 tasks) and the 31-5-9 / 32-6-7 /
35-6-4 win-tie-loss records (per rank, 5 approximate estimators x 9 tasks = 45) -- none of which
the 10-per-family set above can produce.  Two Holm family definitions are computed for every
comparison and both are written out, because the choice changes what is significant:
`--holm-family task-metric` (default, the manuscript's declared convention) gives families of 18
on the task metric and 12 on the cosine, so the smallest attainable corrected p at ten seeds is
18 x 0.001953 = 0.0352 and 12 x 0.001953 = 0.0234; `--holm-family rank-metric` gives families of
54 and 36, whose floors (0.105 and 0.0703) lie above 0.05, so with ten paired seeds NOTHING can
be Holm-significant in a family that size, whatever the effect.  The three rank-vs-rank
comparisons are reported as an auxiliary block and are not counted in the 270.  A "claim audit"
section at the end of `D4_STATS.md` re-derives every count the body quotes, under four
criteria (sign of delta, bootstrap CI, and Holm under each family definition).

Outputs (all written, none read, by this script)
  results/r2/D4_TABLES.md        the three tables + LR table + the assertion report
  results/r2/D4_STATS.md         all-pairs statistics, human-readable
  results/r2/D4_STATS.json       the same numbers, machine-readable
  <texdir>/tab_r2_fidelity.tex   replaces Table 3 (diagnostic grad_cos)
  <texdir>/tab_r2_timeseries.tex replaces Table 6 (chaotic NMSE + grad cosine)
  <texdir>/tab_r2_realts.tex     replaces Table 7 (Sunspot / laser)
  <texdir>/tab_r2_lr.tex         selected learning rate per (task, estimator)
  <texdir>/tab_r2_stats.tex      the statistics appendix tables (three table* blocks, one per
                                 rank under --full-matrix, one per task block otherwise)

Usage
  python make_r2_d4_tables.py                                # the real run, once d4_eval lands
  python make_r2_d4_tables.py --strict                       # non-zero exit on any violation
  python make_r2_d4_tables.py --indir results/m3 results/ts results/round1/real \
      --outdir /tmp/demo --texdir /tmp/demo                  # R1 stand-in dry run

Relative paths are resolved against this script's directory (code/), like the rest of the R2
tooling.  This script never launches anything and never writes anything but the outputs above.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import make_r2_d4_select as sel        # noqa: E402  protocol + direction, single source of truth
import make_r2_jobs as mj              # noqa: E402  filename / tag conventions

_REQUIRED = ["TASKS", "ALGOS", "HIGHER_IS_BETTER", "EVAL_DIR", "REPORT_DIR", "WINDOW_FRAC",
             "window_score", "is_tbptt", "tag_for", "lr_of_tag", "out_path",
             "loss_types_from_source", "rel", "_TAG2IDX"]
_absent = [a for a in _REQUIRED if not hasattr(sel, a)]
if _absent:
    raise SystemExit("make_r2_d4_select.py no longer exposes %s -- the D4 protocol moved; "
                     "fix make_r2_d4_tables.py before trusting any number it prints." % _absent)

# ---------------------------------------------------------------- protocol constants
TASKS = list(sel.TASKS)
ALGOS = list(sel.ALGOS)
HIGHER_IS_BETTER = set(sel.HIGHER_IS_BETTER)        # task-metric direction (accuracy tasks)
EVAL_DIR = sel.EVAL_DIR
REPORT_DIR = sel.REPORT_DIR
SELECT_JSON = os.path.join(REPORT_DIR, "D4_SELECT_stage2.json")
TEX_DIR = os.path.join("..", "paper", "secs")

SEEDS_EVAL = list(range(10))
STEPS_EVAL = 20000

DIAG_TASKS = ["copy", "adding", "rotation", "anbn"]
CHAOS_TASKS = ["henon", "mackeyglass", "lorenz"]
REAL_TASKS = ["sunspot", "laser"]

# display order of the estimators, as in published Tables 3/6/7
ORDER = ["exact", "skrtrl-r32", "skrtrl-r16", "skrtrl-r4", "snap1", "kfrtrl", "rflo",
         "uoro", "tbptt"]
_unknown = sorted(set([a for a in ORDER if a not in ALGOS] + [a for a in ALGOS if a not in ORDER]))
if _unknown:
    raise SystemExit("the D4 estimator set changed (%s); extend ORDER/LABEL in "
                     "make_r2_d4_tables.py" % _unknown)

LABEL = {"exact": "Exact RTRL", "skrtrl-r32": r"\skrtrl{} $r{=}32$",
         "skrtrl-r16": r"\skrtrl{} $r{=}16$", "skrtrl-r4": r"\skrtrl{} $r{=}4$",
         "snap1": "SnAp-1", "kfrtrl": "KF-RTRL", "rflo": "RFLO", "uoro": "UORO",
         "tbptt": "TBPTT"}
PLAIN = {"exact": "exact RTRL", "skrtrl-r32": "SK-RTRL r=32", "skrtrl-r16": "SK-RTRL r=16",
         "skrtrl-r4": "SK-RTRL r=4", "snap1": "SnAp-1", "kfrtrl": "KF-RTRL", "rflo": "RFLO",
         "uoro": "UORO", "tbptt": "TBPTT"}
TASK_TEX = {"copy": "copy", "adding": "adding", "rotation": "rotation", "anbn": "$a^nb^n$",
            "henon": r"H\'enon", "mackeyglass": "Mackey--Glass", "lorenz": "Lorenz",
            "sunspot": "Sunspot", "laser": "Laser"}

# grad_cos is not a defined quantity for these two: run_m3.py logs None for the exact run
# (its cosine to itself is trivially 1) and TBPTT's truncated gradient is not the RTRL
# gradient at all.
NO_COS = {"exact", "tbptt"}

REF = "skrtrl-r16"
# Holm families are fixed HERE, before any data are read (R2 plan, D4: "seed is the unit;
# Holm families fixed in advance").  One family = one (task, metric).
PLANNED = [(REF, o) for o in ORDER if o != REF] + [("skrtrl-r4", "snap1"),
                                                   ("skrtrl-r32", "snap1")]

# ---------------------------------------------------------------------------
# FULL MATRIX (R2 second-round gap fix).  The set above is the pre-registered
# R1-revision set: it covers r=16 against everything and r=4 / r=32 against
# SnAp-1 only, which is 10 comparisons per (task, metric), 180 planned and 162
# structurally applicable.  The body of the paper, however, quotes records that
# range over ALL THREE ranks against ALL SIX non-SK-RTRL estimators -- the
# "108 of 108" gradient-cosine sweep (3 ranks x 4 approximate estimators x 9
# tasks) and the 31-5-9 / 32-6-7 / 35-6-4 win-tie-loss records (per rank, the 5
# approximate estimators x 9 tasks).  Those numbers are NOT readable from the
# 10-per-family set, so the full matrix below is computed as well:
#
#     3 ranks x 6 opponents x 9 tasks x 2 metrics          = 324 planned cells
#     minus the gradient cosine for exact RTRL and TBPTT
#     (3 ranks x 2 opponents x 9 tasks = 54 undefined)     = 270 applicable
#
# so "270 pre-registered paired comparisons" is the count of *structurally
# applicable* full-matrix comparisons, and 324 is the count including the ones
# the protocol leaves undefined.  The 180 quoted elsewhere is the planned count
# of the older 10-per-(task, metric) set (162 applicable) and is not the same
# quantity.
FULL_RANKS = ["skrtrl-r4", "skrtrl-r16", "skrtrl-r32"]
FULL_OPPONENTS = ["exact", "snap1", "kfrtrl", "rflo", "uoro", "tbptt"]
FULL_PAIRS = [(r, o) for r in FULL_RANKS for o in FULL_OPPONENTS]

# The three rank-vs-rank comparisons are *not* part of the 270 (they compare the
# method with itself at another rank, not with a competitor).  They are reported
# in D4_STATS.md as an auxiliary block so that nothing the older set contained
# is lost when the LaTeX appendix switches to the full matrix.
INTER_RANK_PAIRS = [("skrtrl-r16", "skrtrl-r4"), ("skrtrl-r32", "skrtrl-r16"),
                    ("skrtrl-r32", "skrtrl-r4")]

# The two opponent subsets the body's records are defined over.
APPROX_FOUR = ["snap1", "kfrtrl", "rflo", "uoro"]          # cosine sweep (108)
APPROX_FIVE = ["snap1", "kfrtrl", "rflo", "uoro", "tbptt"]  # win-tie-loss (45/rank)

# ---------------------------------------------------------------------------
# HOLM FAMILIES ON THE FULL MATRIX.  Two definitions are computed for every
# comparison, because the choice is not neutral and the paper has to state one:
#
#   "rank-metric"  one family per (rank, metric): every comparison that the one
#                  rank makes on the one metric, pooled over all nine tasks.
#                  Family size = 6 x 9 = 54 on the task metric and 4 x 9 = 36 on
#                  the gradient cosine (exact / TBPTT have no cosine).  This is
#                  the family that matches how the body *states* the claims (one
#                  sweep per rank across tasks), and it is the conservative
#                  reading.  Consequence, and it is a large one: the two-sided
#                  exact Wilcoxon floor at n=10 is p = 2/2^10 = 0.001953, so the
#                  smallest attainable Holm-corrected p is 36 x 0.001953 =
#                  0.0703 on the cosine and 54 x 0.001953 = 0.105 on the metric
#                  -- NOTHING can reach 0.05 with ten seeds in a family this
#                  large, whatever the effect size.
#   "task-metric"  one family per (task, metric), which is the definition the
#                  manuscript declares (Sec. 6.5 and App. B) and the one the
#                  pre-registered set used.  On the full matrix the size grows
#                  from 10/8 to 18 on the task metric (3 ranks x 6 opponents)
#                  and 12 on the cosine (3 ranks x 4 opponents), so the floor
#                  moves from 8 x 0.001953 = 0.0156 to 12 x 0.001953 = 0.0234 on
#                  the cosine and 18 x 0.001953 = 0.0352 on the metric, both
#                  still below 0.05.
#
# `holm_p` carries whichever of the two --holm-family selects (default
# task-metric, the manuscript's declared convention and the one the LaTeX
# appendix prints); `holm_p_taskfam` and `holm_p_rankfam` always carry both, in
# D4_STATS.md and D4_STATS.json, so either can be audited without a re-run.
HOLM_FAMILY_MODES = ("task-metric", "rank-metric")

ND = {"metric": 4, "grad_cos": 3}          # printed decimals, as in the published tables
DAG = r"\,^{\dagger}"


def higher_better(task, field):
    return True if field == "grad_cos" else (task in HIGHER_IS_BETTER)


def metric_name(task):
    if task in HIGHER_IS_BETTER:
        return "accuracy"
    return "NMSE" if task in (CHAOS_TASKS + REAL_TASKS) else "MSE"


def applicable(task, field, a, b):
    """Is this planned comparison defined by the protocol (ignoring data availability)?"""
    if field == "grad_cos" and (a in NO_COS or b in NO_COS):
        return False
    return True


# ---------------------------------------------------------------- run loading
def _clean(vals):
    out = []
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        out.append(float(v))
    return out


def agg_field(records, field, steps, agg, frac, tail):
    """Per-run value of `field` under the requested aggregation.  -> (value, n_used)."""
    if agg == "tail5":
        vals = _clean([r.get(field) for r in records])[-tail:]
    else:
        lo = frac * float(steps or 0)
        vals = _clean([r.get(field) for r in records
                       if isinstance(r.get("step"), (int, float)) and r["step"] >= lo])
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def read_one(path, agg, frac, tail):
    """Parse one run json.  Never raises: a bad file becomes a `problem` record."""
    r = {"path": path, "file": os.path.basename(path), "status": "ok"}
    try:
        with io.open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as e:                                              # noqa: BLE001
        r.update(status="unreadable", error="%s: %s" % (type(e).__name__, e))
        return r
    if not isinstance(d, dict):
        r.update(status="unreadable", error="top level is not an object")
        return r
    a = d.get("args") or {}
    recs = d.get("records") or []
    r.update(task=a.get("task"), algo=a.get("algo"), seed=a.get("seed"), lr=a.get("lr"),
             steps=a.get("steps"), tag=a.get("tag"), clip=a.get("clip"),
             svd_driver=a.get("svd_driver"), shadow=a.get("shadow"), n_hid=a.get("n"),
             n_records=len(recs))
    if None in (r["task"], r["algo"], r["seed"], r["lr"]):
        r.update(status="bad_args", error="args missing task/algo/seed/lr")
        return r
    r["seed"] = int(r["seed"])
    r["lrtag"] = mj.lr_tag(float(r["lr"]))
    if sel.is_tbptt(r["algo"]):
        w = a.get("tbptt_window", a.get("tbptt_k"))
        r["window"] = int(w) if isinstance(w, (int, float)) else None
    else:
        r["window"] = None
    for field in ("metric", "grad_cos"):
        v, n = agg_field(recs, field, r["steps"], agg, frac, tail)
        r[field] = v
        r["n_" + field] = n
    # the metric window mean must be the very number make_r2_d4_select scored, or the tables
    # and the LR selection are not talking about the same runs
    if agg == "window20":
        s, _, _ = sel.window_score(recs, r["steps"], frac)
        r["select_score"] = s
        m = r["metric"]
        if s is None and m is None:
            same = True
        elif s is None or m is None:
            same = False
        elif not (math.isfinite(s) and math.isfinite(m)):
            same = (math.isfinite(s) == math.isfinite(m))
        else:
            same = abs(s - m) <= 1e-12 * max(1.0, abs(s))
        r["score_matches_select"] = bool(same)
    else:
        r["select_score"] = None
        r["score_matches_select"] = None
    if not recs:
        r["status"] = "no_records"
    elif r["metric"] is None:
        r["status"] = "window_empty"
    elif not math.isfinite(r["metric"]):
        r["status"] = "diverged"
    return r


def load_runs(indirs, agg, frac, tail):
    runs, problems = [], []
    for d in indirs:
        p = sel.rel(d)
        if not os.path.isdir(p):
            problems.append({"file": d, "status": "missing_dir",
                             "error": "input directory %s does not exist" % p})
            continue
        for fn in sorted(os.listdir(p)):
            if not fn.endswith(".json"):
                continue
            r = read_one(os.path.join(p, fn), agg, frac, tail)
            r["indir"] = d
            if r["status"] in ("unreadable", "bad_args"):
                problems.append(r)
                continue
            if r["task"] not in TASKS or r["algo"] not in ALGOS:
                continue                       # the R1 stand-in dirs carry extra algos
            runs.append(r)
    return runs, problems


def cfg_of(r):
    return (r["lrtag"], r["window"])


def cfg_str(cfg):
    if cfg is None:
        return "--"
    return cfg[0] if cfg[1] is None else "%s/w%s" % (cfg[0], cfg[1])


def pick_config(runs, selected):
    """Group runs by (task, algo, seed) and keep exactly one run per cell.

    Preference: the stage-2 selected (lrtag, window) when it is known, else the modal
    configuration of that (task, algo) over seeds, ties broken towards the smaller LR.
    Everything dropped is reported, so a directory holding several LRs per cell cannot
    silently contribute a mixture.
    """
    by_pair = defaultdict(lambda: defaultdict(list))
    for r in runs:
        by_pair[(r["task"], r["algo"])][r["seed"]].append(r)
    kept, dropped, ambiguous, unmatched = {}, [], [], []
    for (task, algo), per_seed in sorted(by_pair.items()):
        want = selected.get((task, algo))
        if want is None:
            counts = defaultdict(int)
            for rs in per_seed.values():
                for r in rs:
                    counts[cfg_of(r)] += 1
            if len(counts) > 1:
                order = sorted(counts.items(),
                               key=lambda kv: (-kv[1], sel._TAG2IDX.get(kv[0][0], 10 ** 6),
                                               -1 if kv[0][1] is None else kv[0][1]))
                want = order[0][0]
                ambiguous.append({"task": task, "algo": algo, "used": cfg_str(want),
                                  "configs": dict((cfg_str(k), v) for k, v in
                                                  sorted(counts.items()))})
            elif counts:
                want = list(counts)[0]
        for seed, rs in sorted(per_seed.items()):
            hit = [r for r in rs if want is None or cfg_of(r) == tuple(want)]
            chosen = hit[0] if hit else None
            if chosen is None:
                unmatched.append({"task": task, "algo": algo, "seed": seed,
                                  "want": cfg_str(want),
                                  "have": [cfg_str(cfg_of(r)) for r in rs]})
                continue
            for r in rs:
                if r is not chosen:
                    dropped.append({"file": r["file"], "task": task, "algo": algo,
                                    "seed": seed, "config": cfg_str(cfg_of(r)),
                                    "reason": "duplicate cell; %s is used" % cfg_str(want)})
            kept[(task, algo, seed)] = chosen
    return kept, dropped, ambiguous, unmatched


# ---------------------------------------------------------------- cells
def build_cell(kept, task, algo, field, seeds, require_steps, ddof):
    """One table cell: mean +/- std over seeds, plus every protocol deviation it carries."""
    c = {"task": task, "algo": algo, "field": field, "mean": None, "std": None, "n": 0,
         "values": {}, "flags": [], "seeds_missing": [], "seeds_bad_steps": [],
         "seeds_nonfinite": [], "configs": set(), "defined": True}
    if field == "grad_cos" and algo in NO_COS:
        c["defined"] = False
        return c
    for s in seeds:
        r = kept.get((task, algo, s))
        if r is None:
            c["seeds_missing"].append(s)
            continue
        c["configs"].add(cfg_str(cfg_of(r)))
        if require_steps and r.get("steps") != require_steps:
            c["seeds_bad_steps"].append((s, r.get("steps")))
        v = r.get(field)
        if v is None or not math.isfinite(v):
            c["seeds_nonfinite"].append((s, "none" if v is None else "%g" % v))
            continue
        c["values"][s] = float(v)
    vals = [c["values"][s] for s in sorted(c["values"])]
    c["n"] = len(vals)
    if vals:
        m = sum(vals) / len(vals)
        c["mean"] = m
        if len(vals) > 1:
            var = sum((x - m) ** 2 for x in vals) / max(1, (len(vals) - ddof))
            c["std"] = math.sqrt(var)          # None for a single seed: no spread to report
    c["configs"] = sorted(c["configs"])
    return c


def cell_flags(c, require_seeds, require_steps):
    f = []
    if not c["defined"]:
        return f
    if c["n"] == 0:
        f.append("no usable run")
        return f
    if require_seeds and c["n"] != require_seeds:
        f.append("n_seeds=%d (protocol %d)" % (c["n"], require_seeds))
    if c["seeds_missing"]:
        f.append("seeds absent: %s" % ",".join(str(s) for s in c["seeds_missing"]))
    if c["seeds_nonfinite"]:
        f.append("non-finite: %s" % ",".join("s%s=%s" % t for t in c["seeds_nonfinite"]))
    if c["seeds_bad_steps"]:
        f.append("steps!=%s: %s" % (require_steps,
                                    ",".join("s%s=%s" % t for t in c["seeds_bad_steps"])))
    if len(c["configs"]) > 1:
        f.append("mixed configs: %s" % ",".join(c["configs"]))
    return f


def build_cells(kept, seeds, require_seeds, require_steps, ddof):
    cells = {}
    for task in TASKS:
        for algo in ORDER:
            for field in ("metric", "grad_cos"):
                c = build_cell(kept, task, algo, field, seeds, require_steps, ddof)
                c["flags"] = cell_flags(c, require_seeds, require_steps)
                c["dagger"] = bool(c["flags"]) and c["n"] > 0
                cells[(task, algo, field)] = c
    return cells


def pooled_cos(kept, tasks, algo, seeds, require_steps, ddof):
    """The published Tables 6/7 grad-cosine column: one per-run value, pooled over every
    seed and every system of the block at once."""
    if algo in NO_COS:
        return {"defined": False, "mean": None, "std": None, "n": 0, "flags": []}
    vals, flags, nonfinite, bad_steps = [], [], [], defaultdict(int)
    for task in tasks:
        for s in seeds:
            r = kept.get((task, algo, s))
            if r is None:
                continue
            v = r.get("grad_cos")
            if v is None or not math.isfinite(v):
                nonfinite.append("%s/s%s" % (task, s))
                continue
            if require_steps and r.get("steps") != require_steps:
                bad_steps[r.get("steps")] += 1
            vals.append(float(v))
    want = len(tasks) * len(seeds)
    if len(vals) != want:
        flags.append("n=%d of %d expected runs" % (len(vals), want))
    if bad_steps:
        flags.append("steps!=%s: %s" % (require_steps, ", ".join(
            "%d run(s) at %s" % (c, k) for k, c in sorted(bad_steps.items(),
                                                          key=lambda kv: str(kv[0])))))
    if nonfinite:
        flags.append("non-finite: %s" % ", ".join(nonfinite[:8])
                     + ("" if len(nonfinite) <= 8 else " and %d more" % (len(nonfinite) - 8)))
    if not vals:
        return {"defined": True, "mean": None, "std": None, "n": 0, "flags": flags}
    m = sum(vals) / len(vals)
    sd = None
    if len(vals) > 1:
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / max(1, (len(vals) - ddof)))
    return {"defined": True, "mean": m, "std": sd, "n": len(vals), "flags": flags}


# ---------------------------------------------------------------- formatting
def bold_set(values, higher, nd):
    """Indices of the best (rounded) value in a column; ties share the bold."""
    idx = [i for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    if not idx:
        return set()
    rounded = dict((i, round(values[i], nd)) for i in idx)
    best = max(rounded.values()) if higher else min(rounded.values())
    return set(i for i in idx if rounded[i] == best)


def tex_cell(mean, std, nd, bold=False, dagger=False, show_n=None):
    if mean is None or not math.isfinite(mean):
        return "---"
    body = (r"\mathbf{%.*f}" % (nd, mean)) if bold else ("%.*f" % (nd, mean))
    if std is not None:
        body += r"\pm%.*f" % (nd, std)
    if show_n is not None:
        body += r"\;(%d)" % show_n
    if dagger:
        body += DAG
    return "$" + body + "$"


def md_cell(c):
    if not c["defined"]:
        return "--"
    if c["mean"] is None:
        return "n/a"
    nd = ND[c["field"]]
    if c["std"] is None:
        s = "%.*f (n=%d)" % (nd, c["mean"], c["n"])
    else:
        s = "%.*f +/- %.*f (n=%d)" % (nd, c["mean"], nd, c["std"], c["n"])
    return s + (" †" if c["dagger"] else "")


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return "%.4f" % p if p >= 1e-4 else "$<$0.0001"


def fmt_p_md(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return "%.4f" % p if p >= 1e-4 else "<0.0001"


def fmt_lr(lrtag, window, tex=True):
    if lrtag is None:
        return "---" if tex else "--"
    s = lrtag[2:] if lrtag.startswith("lr") else lrtag
    s = s.replace("e-0", "e-")
    if tex:
        s = r"\texttt{%s}" % s
        if window is not None:
            s += r"/$k{=}%d$" % window
    elif window is not None:
        s += "/k=%d" % window
    return s


# ---------------------------------------------------------------- selection / assertions
def load_selection(path):
    """-> ({(task, algo): (lrtag, window)}, note).  Missing file is not fatal.

    `path` is kept relative (as passed in, normally results/r2/D4_SELECT_stage2.json) for
    every message below -- only `p`, the resolved filesystem path, is used for actual I/O --
    so the comments/captions this feeds (tex_header's "LR selection" line, D4_TABLES.md) never
    embed the author's local absolute checkout path.
    """
    p = sel.rel(path)
    if not os.path.exists(p):
        return {}, ("absent: %s -- the LR cross-check and the LR table are skipped, and the "
                    "configuration of each cell is taken from the runs themselves" % path)
    try:
        with io.open(p, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as e:                                              # noqa: BLE001
        return {}, "unreadable (%s: %s)" % (type(e).__name__, e)
    out = {}
    for e in d.get("pairs", []):
        s = e.get("selected")
        if not s:
            continue
        w = s.get("window")
        out[(e["task"], e["algo"])] = (s["lrtag"], None if w is None else int(w))
    meta = d.get("meta") or {}
    note = "%s (%d of %d pairs have a stage-2 winner, generated %s)" % (
        path, len(out), len(TASKS) * len(ALGOS), meta.get("generated", "?"))
    return out, note


def verify_direction(tasks_py="skrtrl/tasks.py"):
    """Re-check the accuracy/MSE direction table against skrtrl/tasks.py."""
    path = sel.rel(tasks_py)
    rep = {"path": path, "ok": False, "loss_types": {}, "problems": []}
    if not os.path.exists(path):
        rep["problems"].append("%s not found -- direction table not verified" % path)
        return rep
    try:
        got = sel.loss_types_from_source(path)
    except Exception as e:                                              # noqa: BLE001
        rep["problems"].append("could not parse %s (%s: %s)" % (path, type(e).__name__, e))
        return rep
    rep["loss_types"] = dict((t, got.get(t)) for t in TASKS)
    unknown = [t for t in TASKS if got.get(t) is None]
    ce = set(t for t in TASKS if got.get(t) == "ce")
    if unknown:
        rep["problems"].append("loss_type not resolvable for %s" % unknown)
    if ce != HIGHER_IS_BETTER:
        rep["problems"].append("skrtrl/tasks.py says ce=%s, the direction table says "
                               "higher-is-better=%s" % (sorted(ce), sorted(HIGHER_IS_BETTER)))
    rep["ok"] = not rep["problems"]
    return rep


def check_lrs(kept, unmatched, selected):
    """Every cell must be filled by a run at the stage-2 selected (LR, window).

    Two ways that can fail: a kept run whose config differs (defensive -- `pick_config`
    should never produce one), and a cell for which only off-selection runs exist, which
    `pick_config` refuses to use rather than quietly mixing learning rates.
    """
    bad = []
    for (task, algo, seed), r in sorted(kept.items()):
        want = selected.get((task, algo))
        if want is not None and cfg_of(r) != tuple(want):
            bad.append({"task": task, "algo": algo, "seed": seed, "file": r["file"],
                        "want": cfg_str(want), "got": cfg_str(cfg_of(r))})
    for um in unmatched:
        if selected.get((um["task"], um["algo"])) is None:
            continue
        bad.append({"task": um["task"], "algo": um["algo"], "seed": um["seed"],
                    "file": "(no run at the selected config; the seed is left out of the cell)",
                    "want": um["want"], "got": ", ".join(um["have"])})
    return bad


def missing_runs(kept, selected, seeds, indir):
    """The runs the protocol expects and the directories do not hold."""
    miss = []
    for task in TASKS:
        for algo in ORDER:
            want = selected.get((task, algo))
            for s in seeds:
                if (task, algo, s) in kept:
                    continue
                if want is None:
                    miss.append({"task": task, "algo": algo, "seed": s,
                                 "expected_out": "(stage-2 LR unknown) %s/%s_%s_s%d_*.json"
                                                 % (indir, task, algo, s)})
                else:
                    tag = sel.tag_for(algo, want[0], want[1])
                    miss.append({"task": task, "algo": algo, "seed": s,
                                 "expected_out": sel.out_path(indir, task, algo, s, tag)})
    return miss


def check_scores(kept):
    """The metric window mean must equal make_r2_d4_select's own score for that run."""
    return [{"file": r["file"], "task": r["task"], "algo": r["algo"], "seed": r["seed"]}
            for r in kept.values() if r.get("score_matches_select") is False]


# ---------------------------------------------------------------- statistics
def bootstrap_ci(d, n=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def cliffs_delta(a, b, higher_better):
    """Cliff's delta: the dominance of A over B over all n_a x n_b cross pairs.

    +1 means every value of A beats every value of B (no seed of B reaches any
    seed of A), -1 the reverse, 0 no stochastic ordering.  Computed on the raw
    values, not on the paired differences, which is what "no seed of any
    baseline reaches any seed of SK-RTRL" means.
    """
    A = np.asarray(a, dtype=float)[:, None]
    B = np.asarray(b, dtype=float)[None, :]
    gt = int((A > B).sum())
    lt = int((A < B).sum())
    if not higher_better:
        gt, lt = lt, gt
    return float(gt - lt) / float(A.shape[0] * B.shape[1])


def holm_fixed(pvals, m):
    """Holm step-down with a family size fixed in advance (m >= len(pvals))."""
    m = max(int(m), len(pvals))
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, run = [1.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


def wilcoxon_p(a, b):
    try:
        try:
            w = stats.wilcoxon(a, b, zero_method="wilcox", correction=False,
                               alternative="two-sided", method="auto")
        except TypeError:                       # scipy < 1.13 spells it `mode`
            w = stats.wilcoxon(a, b, zero_method="wilcox", correction=False,
                               alternative="two-sided", mode="auto")
        return float(w.pvalue)
    except ValueError:
        return float("nan")                     # every paired difference is zero


def compare_pair(cells, task, field, ref, other, boot, boot_seed, min_pairs):
    rec = {"task": task, "field": field, "ref": ref, "other": other,
           "applicable": applicable(task, field, ref, other), "status": "ok",
           "n": 0, "seeds": [], "ref_mean": None, "other_mean": None, "delta": None,
           "ci95": [None, None], "wilcoxon_p": None, "holm_p": None,
           "holm_p_taskfam": None, "holm_p_rankfam": None, "cohen_dz": None,
           "cliff_delta": None,
           "higher_better": higher_better(task, field), "note": ""}
    if not rec["applicable"]:
        rec.update(status="not_applicable",
                   note="grad_cos is not defined for %s" % (
                       " / ".join(sorted(set([ref, other]) & NO_COS))))
        return rec
    A = cells[(task, ref, field)]["values"]
    B = cells[(task, other, field)]["values"]
    seeds = sorted(set(A) & set(B))
    rec["seeds"] = seeds
    rec["n"] = len(seeds)
    if len(seeds) < min_pairs:
        rec.update(status="insufficient",
                   note="only %d paired seed(s), %d required" % (len(seeds), min_pairs))
        return rec
    a = np.array([A[s] for s in seeds], dtype=float)
    b = np.array([B[s] for s in seeds], dtype=float)
    delta = (a - b) if rec["higher_better"] else (b - a)      # positive = reference better
    lo, hi = bootstrap_ci(delta, n=boot, seed=boot_seed)
    rec.update(ref_mean=float(a.mean()), other_mean=float(b.mean()),
               delta=float(delta.mean()), ci95=[lo, hi], wilcoxon_p=wilcoxon_p(a, b),
               cohen_dz=float(delta.mean() / (delta.std(ddof=1) + 1e-12)),
               cliff_delta=cliffs_delta(a, b, rec["higher_better"]))
    return rec


def run_stats(cells, boot, boot_seed, min_pairs):
    """-> {(task, field): {"family": m, "comparisons": [rec, ...]}}"""
    out = {}
    for task in TASKS:
        for field in ("metric", "grad_cos"):
            fam = [p for p in PLANNED if applicable(task, field, p[0], p[1])]
            recs = [compare_pair(cells, task, field, r, o, boot, boot_seed, min_pairs)
                    for r, o in PLANNED]
            done = [r for r in recs if r["status"] == "ok"]
            if done:
                ps = [1.0 if math.isnan(r["wilcoxon_p"]) else r["wilcoxon_p"] for r in done]
                for r, q in zip(done, holm_fixed(ps, len(fam))):
                    r["holm_p"] = float(q)
            out[(task, field)] = {"family": len(fam), "comparisons": recs}
    return out


# ------------------------------------------------- full matrix: 3 ranks x 6 opponents
def _family_sizes(pairs, keyfn):
    """Structurally applicable comparison count per family, fixed before any data."""
    sizes = {}
    for task in TASKS:
        for field in ("metric", "grad_cos"):
            for ref, other in pairs:
                if not applicable(task, field, ref, other):
                    continue
                k = keyfn({"task": task, "field": field, "ref": ref, "other": other})
                sizes[k] = sizes.get(k, 0) + 1
    return sizes


def _key_taskfam(r):
    return (r["task"], r["field"])


def _key_rankfam(r):
    return (r["ref"], r["field"])


def _apply_holm(recs, keyfn, sizes, out_field):
    groups = {}
    for r in recs:
        if r["status"] == "ok":
            groups.setdefault(keyfn(r), []).append(r)
    for k, done in groups.items():
        m = sizes.get(k, len(done))
        ps = [1.0 if math.isnan(r["wilcoxon_p"]) else r["wilcoxon_p"] for r in done]
        for r, q in zip(done, holm_fixed(ps, m)):
            r[out_field] = float(q)


def run_stats_full(cells, boot, boot_seed, min_pairs, holm_family="task-metric",
                   pairs=None, with_inter_rank=True):
    """The full (rank x opponent x task x metric) matrix with both Holm families.

    -> {"comparisons": [rec, ...],                    # the 324 planned cells
        "inter_rank":  [rec, ...],                    # auxiliary, not in the 324
        "family_taskfam": {(task, field): m},
        "family_rankfam": {(rank, field): m},
        "holm_family": which definition `holm_p` carries}
    """
    pairs = list(FULL_PAIRS if pairs is None else pairs)
    recs = [compare_pair(cells, task, field, ref, other, boot, boot_seed, min_pairs)
            for task in TASKS for field in ("metric", "grad_cos") for ref, other in pairs]
    inter = ([compare_pair(cells, task, field, ref, other, boot, boot_seed, min_pairs)
              for task in TASKS for field in ("metric", "grad_cos")
              for ref, other in INTER_RANK_PAIRS] if with_inter_rank else [])

    size_t = _family_sizes(pairs, _key_taskfam)
    size_r = _family_sizes(pairs, _key_rankfam)
    _apply_holm(recs, _key_taskfam, size_t, "holm_p_taskfam")
    _apply_holm(recs, _key_rankfam, size_r, "holm_p_rankfam")
    # the auxiliary rank-vs-rank block is corrected inside its own family (one per
    # (task, metric)); it is not pooled with the 270, which would change their
    # family size after the fact.
    isize_t = _family_sizes(INTER_RANK_PAIRS, _key_taskfam)
    _apply_holm(inter, _key_taskfam, isize_t, "holm_p_taskfam")
    _apply_holm(inter, _key_taskfam, isize_t, "holm_p_rankfam")

    prim = "holm_p_rankfam" if holm_family == "rank-metric" else "holm_p_taskfam"
    for r in recs + inter:
        r["holm_p"] = r[prim]
    return {"comparisons": recs, "inter_rank": inter, "pairs": pairs,
            "family_taskfam": size_t, "family_rankfam": size_r,
            "holm_family": holm_family,
            "n_planned": len(TASKS) * 2 * len(pairs),
            "n_applicable": sum(size_t.values())}


def full_lookup(full_res):
    return dict(((r["task"], r["field"], r["ref"], r["other"]), r)
                for r in full_res["comparisons"])


# ------------------------------------------------- claim audit on the full matrix
def _verdict(rec, criterion):
    """'win' / 'loss' / 'tie' / None (not computable) for one comparison.

    criterion 'ci'          the paired bootstrap 95 % interval excludes zero in
                            its favour (the manuscript's win-tie-loss rule);
    criterion 'holm_task'   Holm-corrected p < 0.05 in the (task, metric) family;
    criterion 'holm_rank'   the same in the (rank, metric) family;
    criterion 'sign'        the sign of the paired mean advantage alone.
    """
    if rec["status"] != "ok":
        return None
    d = rec["delta"]
    if criterion == "ci":
        lo, hi = rec["ci95"]
        if lo > 0.0:
            return "win"
        if hi < 0.0:
            return "loss"
        return "tie"
    if criterion == "sign":
        return "win" if d > 0 else ("loss" if d < 0 else "tie")
    p = rec.get("holm_p_rankfam" if criterion == "holm_rank" else "holm_p_taskfam")
    if p is None or math.isnan(p) or p >= 0.05:
        return "tie"
    return "win" if d > 0 else "loss"


def _wtl(recs, criterion):
    out = {"win": 0, "tie": 0, "loss": 0, "n/a": 0}
    for r in recs:
        v = _verdict(r, criterion)
        out["n/a" if v is None else v] += 1
    return out


def audit_claims(full_res):
    """Re-derive, from the full matrix alone, every count the body quotes."""
    L = full_lookup(full_res)
    CRIT = ("sign", "ci", "holm_task", "holm_rank")
    audit = {"counts": {"planned_full": full_res["n_planned"],
                        "applicable_full": full_res["n_applicable"],
                        "planned_legacy": len(TASKS) * 2 * len(PLANNED),
                        "applicable_legacy": sum(
                            1 for t in TASKS for f in ("metric", "grad_cos")
                            for a, b in PLANNED if applicable(t, f, a, b)),
                        "computed_ok": sum(1 for r in full_res["comparisons"]
                                           if r["status"] == "ok"),
                        "undefined": sum(1 for r in full_res["comparisons"]
                                         if r["status"] == "not_applicable"),
                        "insufficient": sum(1 for r in full_res["comparisons"]
                                            if r["status"] == "insufficient")}}

    # ---- claim 1: the 108-comparison gradient-cosine sweep
    cos = [L[(t, "grad_cos", r, o)] for r in FULL_RANKS for o in APPROX_FOUR for t in TASKS]
    cl = [r["cliff_delta"] for r in cos if r["status"] == "ok"]
    hp = [r["holm_p_taskfam"] for r in cos
          if r["status"] == "ok" and r["holm_p_taskfam"] is not None]
    hpr = [r["holm_p_rankfam"] for r in cos
           if r["status"] == "ok" and r["holm_p_rankfam"] is not None]
    audit["cosine_sweep"] = {
        "n": len(cos), "n_ok": sum(1 for r in cos if r["status"] == "ok"),
        "per_criterion": dict((c, _wtl(cos, c)) for c in CRIT),
        "cliff_min": min(cl) if cl else None, "cliff_max": max(cl) if cl else None,
        "cliff_eq_1": sum(1 for v in cl if v == 1.0),
        "holm_taskfam_min": min(hp) if hp else None,
        "holm_taskfam_max": max(hp) if hp else None,
        "holm_rankfam_min": min(hpr) if hpr else None,
        "holm_rankfam_max": max(hpr) if hpr else None,
        "per_rank": dict((r, dict((c, _wtl([L[(t, "grad_cos", r, o)]
                                            for o in APPROX_FOUR for t in TASKS], c))
                                  for c in CRIT)) for r in FULL_RANKS)}

    # ---- claim 2: win-tie-loss on the task metric, per rank, vs the five approximate
    audit["wtl_task_metric"] = {}
    for r in FULL_RANKS:
        recs = [L[(t, "metric", r, o)] for o in APPROX_FIVE for t in TASKS]
        audit["wtl_task_metric"][r] = dict(
            [("n", len(recs))] + [(c, _wtl(recs, c)) for c in CRIT])

    # ---- claim 3: the record against truncated BPTT alone, per rank
    audit["vs_tbptt"] = {}
    for r in FULL_RANKS:
        recs = [L[(t, "metric", r, "tbptt")] for t in TASKS]
        audit["vs_tbptt"][r] = dict(
            [("n", len(recs))] + [(c, _wtl(recs, c)) for c in CRIT]
            + [("per_task", dict((t, {"verdict_holm_task":
                                      _verdict(L[(t, "metric", r, "tbptt")], "holm_task"),
                                      "verdict_ci":
                                      _verdict(L[(t, "metric", r, "tbptt")], "ci"),
                                      "holm_p_taskfam":
                                      L[(t, "metric", r, "tbptt")]["holm_p_taskfam"],
                                      "delta": L[(t, "metric", r, "tbptt")]["delta"]})
                                 for t in TASKS))])

    # ---- claim 4: r=32 against exact RTRL, median d_z over the nine tasks
    for r in FULL_RANKS:
        dz = sorted(L[(t, "metric", r, "exact")]["cohen_dz"] for t in TASKS
                    if L[(t, "metric", r, "exact")]["status"] == "ok")
        audit.setdefault("vs_exact_median_dz", {})[r] = (
            float(np.median(dz)) if dz else None)
    dz_tb = sorted(L[(t, "metric", "skrtrl-r16", "tbptt")]["cohen_dz"] for t in TASKS
                   if L[(t, "metric", "skrtrl-r16", "tbptt")]["status"] == "ok")
    audit["vs_tbptt_median_dz_r16"] = float(np.median(dz_tb)) if dz_tb else None
    return audit


# ---------------------------------------------------------------- LaTeX
SHORT = {"exact": "Exact", "skrtrl-r32": "$r{=}32$", "skrtrl-r16": "$r{=}16$",
         "skrtrl-r4": "$r{=}4$", "snap1": "SnAp-1", "kfrtrl": "KF-RTRL", "rflo": "RFLO",
         "uoro": "UORO", "tbptt": "TBPTT"}


# The full protocol paragraph is printed once, in the caption of the first D4 table the
# reader meets; every later table points at that one.  Printing it six times cost about a
# page and a half of the manuscript and told the reader nothing on repetitions two to six.
PROTO_REF = (r"Protocol, aggregation and the per-pair tuned learning rates are exactly those of "
             r"\Cref{tab:r2fidelity}: two-stage learning-rate selection per (task, estimator) "
             r"pair, then %d seeds ($%d$--$%d$) $\times$ %s steps, the per-run value read over "
             r"the last %d\%% of the run and the seed the unit of cross-run aggregation.")


def proto_clause(meta):
    """Parenthetical protocol clause: seeds x steps at the per-pair tuned rates."""
    return (r"%d seeds ($%d$--$%d$) $\times$ %s steps at the per-pair tuned rates of "
            r"\Cref{tab:r2lr}; protocol and table conventions in \Cref{sec:exp-setup}"
            % (meta["require_seeds"], meta["seeds"][0], meta["seeds"][-1],
               "20\\,000" if meta["require_steps"] == 20000 else str(meta["require_steps"])))


def proto_sentence(meta, short=False):
    if short:
        return PROTO_REF % (meta["require_seeds"], meta["seeds"][0], meta["seeds"][-1],
                            "20\\,000" if meta["require_steps"] == 20000
                            else str(meta["require_steps"]),
                            round((1.0 - meta["frac"]) * 100))
    if meta["agg"] == "tail5":
        per_run = ("the mean of the last %d logged records of the run (the convention of "
                   "published Tables~6--7)" % meta["tail"])
    else:
        per_run = ("the mean over every record logged in the last %d\\%% of the run"
                   % round((1.0 - meta["frac"]) * 100))
    std = "population" if meta["ddof"] == 0 else "sample"
    return (r"Unified D4 protocol: the learning rate is tuned per (task, estimator) pair by the "
            r"two-stage D4 procedure (a 6\,000-step screen over four learning rates $\times$ two "
            r"tuning seeds, then the best two re-run at 20\,000 steps), and every entry of this "
            r"table is then %d seed%s (%s) $\times$ %s steps with the exact shadow run on. Per "
            r"run the reported value is %s; across seeds we report mean $\pm$ %s standard "
            r"deviation, the seed being the unit of aggregation and never the pooled record "
            r"set. The learning rate selected for each pair is listed in \Cref{tab:r2lr}."
            % (meta["require_seeds"], "" if meta["require_seeds"] == 1 else "s",
               "$%d$--$%d$" % (meta["seeds"][0], meta["seeds"][-1]),
               "20\\,000" if meta["require_steps"] == 20000 else str(meta["require_steps"]),
               per_run, std))


BOLD_SENTENCE = (r"The best value in each column is set in bold, regardless of method, and "
                 r"values tied at the printed precision share the bold.")
DAG_SENTENCE = (r"A dagger ($\dagger$) marks an entry that does not meet the protocol -- fewer "
                r"usable seeds than required, a step budget other than the protocol one, a "
                r"non-finite run, or a mixture of configurations; every such deviation is "
                r"itemised in \texttt{results/r2/D4\_TABLES.md}.")


def tex_header(title, meta, provisional):
    L = ["% " + title,
         "% Generated " + meta["generated"] + " by code/make_r2_d4_tables.py -- do not edit by "
         "hand; re-run the script.",
         "% Inputs: " + ", ".join(meta["indirs"]),
         "% Aggregation: " + meta["agg"] + " (frac=" + repr(meta["frac"]) + ", tail="
         + str(meta["tail"]) + "), cross-seed ddof=" + str(meta["ddof"])
         + ", protocol " + str(meta["require_seeds"]) + " seeds x "
         + str(meta["require_steps"]) + " steps.",
         "% LR selection: " + meta["selection_note"].replace("\\", "/"),
         "% CAS class note: the columns are plain l/c only (no p{}), as in the other tables of "
         "main_cas; cell text is kept to one line accordingly."]
    if provisional:
        L.insert(1, "% !! PROVISIONAL: at least one entry does not meet the D4 protocol (see "
                    "the dagger list in results/r2/D4_TABLES.md). Do not \\input this file into "
                    "a submitted build until the report is clean.")
    return L


def tex_table(header_lines, caption, label, colspec, head_row, body, wide=True, size="small",
              tabcolsep=None):
    env = "table*" if wide else "table"
    L = list(header_lines)
    L += ["", "\\begin{%s}[t]" % env, "\\centering\\%s" % size]
    if tabcolsep is not None:
        L.append("\\setlength{\\tabcolsep}{%s}" % tabcolsep)
    L += ["\\caption{%s}" % caption, "\\label{%s}" % label,
          "\\begin{tabular}{%s}" % colspec, "\\toprule"]
    L.append(head_row + r" \\")
    L.append(r"\midrule")
    L += body
    L += [r"\bottomrule", r"\end{tabular}", "\\end{%s}" % env, ""]
    return "\n".join(L) + "\n"


def tex_fidelity(cells, meta):
    algos = [a for a in ORDER if a not in NO_COS]
    nd = ND["grad_cos"]
    cols = {}
    for t in DIAG_TASKS:
        vals = [cells[(t, a, "grad_cos")]["mean"] for a in algos]
        cols[t] = bold_set(vals, True, nd)
    body, dagger_any = [], False
    for i, a in enumerate(algos):
        row = [LABEL[a]]
        for t in DIAG_TASKS:
            c = cells[(t, a, "grad_cos")]
            dagger_any = dagger_any or c["dagger"]
            row.append(tex_cell(c["mean"], c["std"], nd, bold=(i in cols[t]),
                                dagger=c["dagger"]))
        body.append(" & ".join(row) + r" \\")
    caption = (r"\textbf{Gradient cosine to the exact RTRL gradient on the four diagnostic "
               r"tasks} ($n{=}64$; " + proto_clause(meta) + r"). "
               r"Paired Wilcoxon tests with Holm correction, paired bootstrap $95\%$ "
               r"confidence intervals and Cohen's $d_z$ for every one of these comparisons "
               r"are in \Cref{tab:r2stats}.")
    if dagger_any:
        caption += " " + DAG_SENTENCE
    head = "Method & " + " & ".join(TASK_TEX[t] for t in DIAG_TASKS)
    return tex_table(tex_header("generated by make_r2_d4_tables.py from results/r2/d4_eval "
                                 "(D4 gradient fidelity)", meta, dagger_any),
                     caption, "tab:r2fidelity", "l" + "c" * len(DIAG_TASKS), head, body)


def tex_series(cells, pooled, tasks, meta, label, tabnum, title, extra_caption="",
               stats_ref="tab:r2stats"):
    ndm, ndc = ND["metric"], ND["grad_cos"]
    cols = {}
    for t in tasks:
        vals = [cells[(t, a, "metric")]["mean"] for a in ORDER]
        cols[t] = bold_set(vals, higher_better(t, "metric"), ndm)
    cos_vals = [pooled[a]["mean"] if pooled[a]["defined"] else None for a in ORDER]
    cos_bold = bold_set(cos_vals, True, ndc)
    body, dagger_any = [], False
    for i, a in enumerate(ORDER):
        row = [LABEL[a]]
        for t in tasks:
            c = cells[(t, a, "metric")]
            dagger_any = dagger_any or c["dagger"]
            row.append(tex_cell(c["mean"], c["std"], ndm, bold=(i in cols[t]),
                                dagger=c["dagger"]))
        p = pooled[a]
        pdag = bool(p["flags"]) and p["mean"] is not None
        dagger_any = dagger_any or pdag
        row.append("---" if not p["defined"] else
                   tex_cell(p["mean"], p["std"], ndc, bold=(i in cos_bold), dagger=pdag))
        body.append(" & ".join(row) + r" \\")
    caption = (title
               + r"The per-system columns are the task metric (NMSE, lower is better) and the "
                 r"last the gradient cosine to exact RTRL, pooled over every seed and every "
                 r"system of the block at once (%d runs per entry, higher is better); "
                 r"per-system cosines and the paired statistics of every comparison "
                 r"are in \Cref{%s}."
                 % (len(tasks) * meta["require_seeds"], stats_ref)
               + extra_caption)
    if dagger_any:
        caption += " " + DAG_SENTENCE
    head = "Method & " + " & ".join(TASK_TEX[t] + " NMSE" for t in tasks) + " & grad cosine"
    return tex_table(tex_header("generated by make_r2_d4_tables.py from results/r2/d4_eval "
                                 "(D4 %s)" % label, meta, dagger_any),
                     caption, "tab:r2" + label, "l" + "c" * (len(tasks) + 1), head, body,
                     tabcolsep="4pt")


def tex_lr(configs, meta, source_note):
    body = []
    for a in ORDER:
        row = [LABEL[a]]
        for t in TASKS:
            cfg = configs.get((t, a))
            row.append("---" if cfg is None else fmt_lr(cfg[0], cfg[1], tex=True))
        body.append(" & ".join(row) + r" \\")
    caption = (r"\textbf{Learning rate selected for every (task, estimator) pair by the "
               r"two-stage procedure}, and therefore used for every entry of "
               r"\Cref{tab:r2fidelity,tab:r2timeseries,tab:r2realts}, with TBPTT's tuned "
               r"truncation window $k$ printed beside the rate. The selection score and the "
               r"search grid are given in \Cref{sec:exp-setup}.")
    head = "Method & " + " & ".join(TASK_TEX[t] for t in TASKS)
    return tex_table(tex_header("D4 selected learning rates (appendix table)", meta, False),
                     caption, "tab:r2lr", "l" + "c" * len(TASKS), head, body,
                     size="scriptsize", tabcolsep="3pt")


def _tex_delta(rec, nd):
    if rec["status"] != "ok":
        return "---"
    return ("$%+.*f$ $[%+.*f, %+.*f]$"
            % (nd, rec["delta"], nd, rec["ci95"][0], nd, rec["ci95"][1]))


def _tex_dz(rec):
    if rec["status"] != "ok" or rec["cohen_dz"] is None:
        return "---"
    v = rec["cohen_dz"]
    if not math.isfinite(v):
        return r"$\infty$"
    return "$%+.2f$" % v


def _tex_holm(rec):
    if rec["status"] != "ok":
        return "---"
    p = rec["holm_p"]
    if p is None or math.isnan(p):
        return "n/a"
    s = "$<$0.0001" if p < 1e-4 else "%.4f" % p
    return (r"\textbf{%s}" % s) if p < 0.05 else s


def _tex_cliff(rec):
    if rec["status"] != "ok" or rec["cliff_delta"] is None:
        return "---"
    return "$%+.2f$" % rec["cliff_delta"]


def stats_label(i):
    """Label of the i-th statistics table; part 1 keeps the plain name."""
    return "tab:r2stats" if i == 0 else "tab:r2stats%d" % (i + 1)


# The full-matrix appendix keeps exactly three table* blocks, and therefore exactly
# the three labels tab:r2stats / tab:r2stats2 / tab:r2stats3 the body already
# \Cref's, but splits them BY RANK rather than by task block: at 6 opponents x 9
# tasks a rank is 54 data rows, which is the largest block that still fits one
# column-spanning float, whereas the old diagnostic/chaotic/real split would have
# put 72 rows in the first one.
FULL_STATS_BLOCKS = [(r, FULL_OPPONENTS) for r in FULL_RANKS]
FULL_STATS_REF = "tab:r2stats,tab:r2stats2,tab:r2stats3"


STATS_BLOCKS = [("diagnostic tasks", DIAG_TASKS), ("chaotic systems", CHAOS_TASKS),
                ("real series", REAL_TASKS)]
STATS_LABEL_OF_TASK = dict((t, stats_label(i))
                           for i, (_lab, ts) in enumerate(STATS_BLOCKS) for t in ts)


def tex_stats(stats_res, meta, blocks):
    """One table* per task block; 7 columns: comparison, then (delta, Holm p, d_z) per metric."""
    out = []
    for bi, (blabel, tasks) in enumerate(blocks):
        body = []
        for t in tasks:
            fam_m = stats_res[(t, "metric")]["family"]
            fam_c = stats_res[(t, "grad_cos")]["family"]
            body.append(r"\multicolumn{7}{l}{\textit{%s} -- task metric: %s (%s); Holm "
                        r"family: %d comparisons on the metric, %d on the cosine} \\"
                        % (TASK_TEX[t], metric_name(t),
                           "higher is better" if t in HIGHER_IS_BETTER else "lower is better",
                           fam_m, fam_c))
            recs_m = dict(((r["ref"], r["other"]), r)
                          for r in stats_res[(t, "metric")]["comparisons"])
            recs_c = dict(((r["ref"], r["other"]), r)
                          for r in stats_res[(t, "grad_cos")]["comparisons"])
            for ref, other in PLANNED:
                rm, rc = recs_m[(ref, other)], recs_c[(ref, other)]
                body.append(" & ".join([
                    "\\quad %s vs.\\ %s" % (SHORT[ref], SHORT[other]),
                    _tex_delta(rm, ND["metric"]), _tex_holm(rm), _tex_dz(rm),
                    _tex_delta(rc, ND["grad_cos"]), _tex_holm(rc), _tex_dz(rc),
                ]) + r" \\")
            body.append(r"\addlinespace")
        # The conventions common to the three parts (what Delta, p_H and d_z are, what a
        # dash means, the pairing unit) are stated once in the prose of the appendix that
        # holds these tables, so each caption says only what is specific to its part.
        caption = (r"\textbf{All paired comparisons, %s} (part %d of %d): the paired "
                   r"bootstrap $95\%%$ confidence interval, the Holm-corrected Wilcoxon "
                   r"$p_{\mathrm{H}}$ and the paired effect size $d_z$ for every planned "
                   r"comparison of each (task, metric) family, at $n{=}%d$ paired seeds. "
                   r"Conventions, the meaning of a dash and the sign convention on $\Delta$ "
                   r"are given in \Cref{app:suppstats}."
                   % (blabel, bi + 1, len(blocks), meta["require_seeds"]))
        head = (r"Comparison & \multicolumn{3}{c}{task metric} & "
                r"\multicolumn{3}{c}{gradient cosine} \\" "\n"
                r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
                r" & $\Delta$ $[95\%$ CI$]$ & $p_{\mathrm{H}}$ & $d_z$ "
                r"& $\Delta$ $[95\%$ CI$]$ & $p_{\mathrm{H}}$ & $d_z$")
        out.append(tex_table(tex_header("D4 all-pairs statistics (appendix table, part %d)"
                                        % (bi + 1), meta, False),
                             caption, stats_label(bi), "l" + "ccc" * 2, head, body,
                             size="scriptsize", tabcolsep="3pt"))
    return "\n".join(out)


def tex_stats_full(full_res, meta):
    """One table* per rank; 9 columns: comparison, then (Delta [CI], p_H, d_z, delta) x 2."""
    L = full_lookup(full_res)
    size_t, size_r = full_res["family_taskfam"], full_res["family_rankfam"]
    famname = ("(rank, metric)" if full_res["holm_family"] == "rank-metric"
               else "(task, metric)")
    out = []
    for bi, (rank, opponents) in enumerate(FULL_STATS_BLOCKS):
        body = []
        for t in TASKS:
            if full_res["holm_family"] == "rank-metric":
                fam_m, fam_c = size_r[(rank, "metric")], size_r[(rank, "grad_cos")]
            else:
                fam_m, fam_c = size_t[(t, "metric")], size_t[(t, "grad_cos")]
            body.append(r"\multicolumn{9}{l}{\textit{%s} -- task metric: %s (%s); Holm "
                        r"family: %d comparisons on the metric, %d on the cosine} \\"
                        % (TASK_TEX[t], metric_name(t),
                           "higher is better" if t in HIGHER_IS_BETTER else "lower is better",
                           fam_m, fam_c))
            for other in opponents:
                rm = L[(t, "metric", rank, other)]
                rc = L[(t, "grad_cos", rank, other)]
                body.append(" & ".join([
                    "\\quad %s vs.\\ %s" % (SHORT[rank], SHORT[other]),
                    _tex_delta(rm, ND["metric"]), _tex_holm(rm), _tex_dz(rm), _tex_cliff(rm),
                    _tex_delta(rc, ND["grad_cos"]), _tex_holm(rc), _tex_dz(rc), _tex_cliff(rc),
                ]) + r" \\")
            body.append(r"\addlinespace")
        caption = (r"\textbf{All %d paired comparisons of the full matrix, \skrtrl{} "
                   r"%s} (part %d of %d, one part per rank): the paired mean advantage $\Delta$ "
                   r"with a $95\%%$ bootstrap interval, the Holm-corrected Wilcoxon "
                   r"$p_{\mathrm{H}}$, the effect size $d_z$ and Cliff's $\delta$, at "
                   r"$n{=}%d$ paired seeds. Column definitions, conventions and the sign "
                   r"convention on $\Delta$ are in \Cref{app:suppstats}."
                   % (full_res["n_applicable"], SHORT[rank], bi + 1,
                      len(FULL_STATS_BLOCKS), meta["require_seeds"]))
        head = (r"Comparison & \multicolumn{4}{c}{task metric} & "
                r"\multicolumn{4}{c}{gradient cosine} \\" "\n"
                r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}" "\n"
                r" & $\Delta$ $[95\%$ CI$]$ & $p_{\mathrm{H}}$ & $d_z$ & $\delta$ "
                r"& $\Delta$ $[95\%$ CI$]$ & $p_{\mathrm{H}}$ & $d_z$ & $\delta$")
        out.append(tex_table(tex_header("D4 full-matrix statistics (appendix table, part %d "
                                        "of %d, rank %s)"
                                        % (bi + 1, len(FULL_STATS_BLOCKS), rank), meta, False),
                             caption, stats_label(bi), "l" + "cccc" * 2, head, body,
                             size="scriptsize", tabcolsep="2.5pt"))
    return "\n".join(out)


# ---------------------------------------------------------------- markdown reports
def md_matrix(cells, tasks, algos, field, title, note=""):
    L = ["### " + title, ""]
    if note:
        L += [note, ""]
    L += ["| Method | " + " | ".join(tasks) + " |", "|" + "---|" * (len(tasks) + 1)]
    nd = ND[field]
    bolds = {}
    for t in tasks:
        vals = [cells[(t, a, field)]["mean"] for a in algos]
        bolds[t] = bold_set(vals, higher_better(t, field), nd)
    for i, a in enumerate(algos):
        row = [PLAIN[a]]
        for t in tasks:
            s = md_cell(cells[(t, a, field)])
            row.append("**%s**" % s if i in bolds[t] and s not in ("--", "n/a") else s)
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    return L


def md_pooled(pooled, algos, title):
    L = ["### " + title, "", "| Method | grad cosine (pooled) | n runs | notes |",
         "|---|---|---|---|"]
    for a in algos:
        p = pooled[a]
        if not p["defined"]:
            L.append("| %s | -- | -- | not defined for this method |" % PLAIN[a])
            continue
        if p["mean"] is None:
            val = "n/a"
        elif p["std"] is None:
            val = "%.3f" % p["mean"]
        else:
            val = "%.3f +/- %.3f" % (p["mean"], p["std"])
        L.append("| %s | %s | %d | %s |" % (PLAIN[a], val, p["n"],
                                            "; ".join(p["flags"]) or "--"))
    L.append("")
    return L


def write_tables_md(path, meta, cells, pooled_ts, pooled_real, configs, checks, limit):
    L = ["# D4 unified-protocol result tables (replacements for Tables 3, 6 and 7)", "",
         "Generated %s by `code/make_r2_d4_tables.py`. Every number here comes from the run "
         "json files alone; nothing is copied from the submitted paper." % meta["generated"], "",
         "- input directories: %s" % ", ".join("`%s`" % d for d in meta["indirs"]),
         "- protocol asserted per cell: %d seeds %s x %d steps, shadow on, the stage-2 D4 "
         "learning rate" % (meta["require_seeds"],
                            "%s-%s" % (meta["seeds"][0], meta["seeds"][-1]),
                            meta["require_steps"]),
         "- per-run value: %s" % ("mean of the last %d logged records (`--agg tail5`)"
                                  % meta["tail"] if meta["agg"] == "tail5" else
                                  "mean over records with `step >= %g * args.steps` "
                                  "(last %d%% window, the same rule that scored the LR "
                                  "selection)" % (meta["frac"],
                                                  round((1 - meta["frac"]) * 100))),
         "- across seeds: mean +/- %s standard deviation (ddof=%d), seed = unit of aggregation"
         % ("population" if meta["ddof"] == 0 else "sample", meta["ddof"]),
         "- metric direction: accuracy (higher better) for %s; MSE/NMSE (lower better) for "
         "every other task; `grad_cos` higher better everywhere"
         % ", ".join(sorted(HIGHER_IS_BETTER)),
         "- LR selection: %s" % meta["selection_note"],
         "- LaTeX written: %s" % (", ".join("`%s`" % os.path.basename(p)
                                            for p in meta["tex_written"]) or "(none, --no-tex)"),
         "",
         "A cell marked `†` fails at least one protocol assertion; the LaTeX tables carry the "
         "same dagger and every deviation is itemised below.", ""]

    L += ["## Assertions", "", "| check | result |", "|---|---|"]
    for name, val in checks["summary"]:
        L.append("| %s | %s |" % (name, val))
    L.append("")

    if checks["direction"]["problems"]:
        L += ["### Metric-direction table vs `skrtrl/tasks.py`", ""]
        L += ["- FAIL: %s" % p for p in checks["direction"]["problems"]] + [""]

    if checks["bad_cells"]:
        L += ["### Cells that do not meet the protocol (`†`), %d of %d"
              % (len(checks["bad_cells"]), checks["n_cells"]), "",
              "| task | method | field | n | deviations |", "|---|---|---|---|---|"]
        for c in checks["bad_cells"][:limit]:
            L.append("| %s | %s | %s | %d | %s |"
                     % (c["task"], PLAIN[c["algo"]], c["field"], c["n"], "; ".join(c["flags"])))
        if len(checks["bad_cells"]) > limit:
            L.append("| ... | | | | %d more |" % (len(checks["bad_cells"]) - limit))
        L.append("")

    if checks["lr_bad"]:
        L += ["### Runs whose learning rate is not the stage-2 selection (%d)"
              % len(checks["lr_bad"]), "", "| task | method | seed | selected | run | file |",
              "|---|---|---|---|---|---|"]
        for b in checks["lr_bad"][:limit]:
            L.append("| %s | %s | %s | %s | %s | `%s` |"
                     % (b["task"], PLAIN[b["algo"]], b["seed"], b["want"], b["got"], b["file"]))
        L.append("")

    if checks["score_bad"]:
        L += ["### Runs whose window mean disagrees with `make_r2_d4_select`'s own score (%d)"
              % len(checks["score_bad"]), "",
              "This must never happen: it means the tables and the LR selection are scoring "
              "the runs differently.", ""]
        L += ["- `%s`" % b["file"] for b in checks["score_bad"][:limit]] + [""]

    if checks["missing"]:
        L += ["### Missing runs (%d of %d expected)"
              % (len(checks["missing"]), len(TASKS) * len(ORDER) * meta["require_seeds"]), ""]
        per_pair = defaultdict(list)
        for m in checks["missing"]:
            per_pair[(m["task"], m["algo"])].append(m["seed"])
        L += ["| task | method | missing seeds | example expected path |",
              "|---|---|---|---|"]
        shown = 0
        for (task, algo), seeds in sorted(per_pair.items()):
            if shown >= limit:
                L.append("| ... | | | %d more (task, method) pair(s) |"
                         % (len(per_pair) - shown))
                break
            ex = next(m["expected_out"] for m in checks["missing"]
                      if m["task"] == task and m["algo"] == algo)
            L.append("| %s | %s | %s | `%s` |"
                     % (task, PLAIN[algo],
                        ",".join(str(s) for s in seeds) if len(seeds) < meta["require_seeds"]
                        else "all %d" % len(seeds), ex))
            shown += 1
        L.append("")

    if checks["dropped"] or checks["ambiguous"] or checks["unmatched"]:
        L += ["### Runs present but not used", ""]
        for am in checks["ambiguous"]:
            L.append("- `%s`/`%s`: several configurations present %s and no stage-2 selection "
                     "to arbitrate; the modal one (`%s`) is used"
                     % (am["task"], am["algo"], am["configs"], am["used"]))
        for um in checks["unmatched"][:limit]:
            L.append("- `%s`/`%s` seed %s: no run at the selected config `%s` (present: %s)"
                     % (um["task"], um["algo"], um["seed"], um["want"], ", ".join(um["have"])))
        for dp in checks["dropped"][:limit]:
            L.append("- `%s`: %s" % (dp["file"], dp["reason"]))
        L.append("")

    if checks["problems"]:
        L += ["### Unreadable / malformed inputs (%d)" % len(checks["problems"]), ""]
        L += ["- `%s`: %s" % (p.get("file"), p.get("error", p.get("status")))
              for p in checks["problems"][:limit]] + [""]

    L += ["## Table 3 replacement -- gradient fidelity on the diagnostic tasks", ""]
    L += md_matrix(cells, DIAG_TASKS, [a for a in ORDER if a not in NO_COS], "grad_cos",
                   "grad_cos, last-window mean per run, mean +/- std over seeds",
                   "LaTeX: `tab_r2_fidelity.tex` (`tab:r2fidelity`). Exact RTRL and TBPTT are "
                   "omitted: the cosine is undefined for them, not missing.")
    L += md_matrix(cells, DIAG_TASKS, ORDER, "metric",
                   "Diagnostic task metric (supplementary, not a paper table)",
                   "copy and $a^nb^n$ are accuracies (higher better); adding and rotation are "
                   "MSE (lower better). Provided because the statistics table reports paired "
                   "tests on this metric as well as on the cosine.")

    L += ["## Table 6 replacement -- chaotic systems", ""]
    L += md_matrix(cells, CHAOS_TASKS, ORDER, "metric",
                   "NMSE (lower better)",
                   "LaTeX: `tab_r2_timeseries.tex` (`tab:r2timeseries`).")
    L += md_matrix(cells, CHAOS_TASKS, [a for a in ORDER if a not in NO_COS], "grad_cos",
                   "grad_cos per system (the LaTeX table pools these into one column)")
    L += md_pooled(pooled_ts, ORDER, "grad cosine pooled over the three systems and all seeds")

    L += ["## Table 7 replacement -- Sunspot and Santa Fe laser", ""]
    L += md_matrix(cells, REAL_TASKS, ORDER, "metric", "NMSE (lower better)",
                   "LaTeX: `tab_r2_realts.tex` (`tab:r2realts`).")
    L += md_matrix(cells, REAL_TASKS, [a for a in ORDER if a not in NO_COS], "grad_cos",
                   "grad_cos per series")
    L += md_pooled(pooled_real, ORDER, "grad cosine pooled over both series and all seeds")

    L += ["## Selected learning rate per (task, estimator)", "",
          "LaTeX: `tab_r2_lr.tex` (`tab:r2lr`). Source: %s" % meta["selection_note"], "",
          "| Method | " + " | ".join(TASKS) + " |", "|" + "---|" * (len(TASKS) + 1)]
    for a in ORDER:
        row = [PLAIN[a]]
        for t in TASKS:
            cfg = configs.get((t, a))
            row.append("--" if cfg is None else fmt_lr(cfg[0], cfg[1], tex=False))
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def _md_num(v, nd=4, signed=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return ("%+.*f" if signed else "%.*f") % (nd, v)


def _md_wtl(d):
    s = "%d-%d-%d" % (d["win"], d["tie"], d["loss"])
    return s + (" (+%d n/a)" % d["n/a"] if d["n/a"] else "")


FULL_MD_ROWS = ("| task | comparison | n | mean A | mean B | delta (A better) | 95% CI | p | "
                "holm_p (task fam) | holm_p (rank fam) | d_z | Cliff delta |")
FULL_MD_SEP = "|" + "---|" * 12


def md_full_rows(recs, field):
    nd = ND[field]
    L = [FULL_MD_ROWS, FULL_MD_SEP]
    for r in recs:
        name = "%s vs %s" % (PLAIN[r["ref"]], PLAIN[r["other"]])
        if r["status"] != "ok":
            L.append("| %s | %s | %d | -- | -- | -- | -- | -- | -- | -- | -- | %s: %s |"
                     % (r["task"], name, r["n"], r["status"], r["note"]))
            continue
        dz = r["cohen_dz"]
        L.append("| %s | %s | %d | %.*f | %.*f | %+.*f | [%+.*f, %+.*f] | %s | %s | %s | %s | "
                 "%+.2f |"
                 % (r["task"], name, r["n"], nd, r["ref_mean"], nd, r["other_mean"],
                    nd, r["delta"], nd, r["ci95"][0], nd, r["ci95"][1],
                    fmt_p_md(r["wilcoxon_p"]), fmt_p_md(r["holm_p_taskfam"]),
                    fmt_p_md(r["holm_p_rankfam"]),
                    ("%+.2f" % dz) if math.isfinite(dz) else "inf", r["cliff_delta"]))
    L.append("")
    return L


def md_full_section(meta, full_res, audit):
    """The full-matrix statistics section and the claim audit read off it."""
    c = audit["counts"]
    st = full_res["family_taskfam"]
    sr = full_res["family_rankfam"]
    L = ["# Full matrix: three ranks x six opponents", "",
         "Every SK-RTRL rank in %s against every one of %s, on all %d tasks and both "
         "metrics. This is the comparison set the *body* of the paper quotes its records "
         "over; the pre-registered set of the previous section (r=16 vs all, r=4 / r=32 vs "
         "SnAp-1 only) cannot produce them."
         % (", ".join(PLAIN[r] for r in FULL_RANKS),
            ", ".join(PLAIN[o] for o in FULL_OPPONENTS), len(TASKS)), "",
         "| count | value | definition |", "|---|---|---|",
         "| planned, full matrix | %d | 3 ranks x 6 opponents x %d tasks x 2 metrics |"
         % (c["planned_full"], len(TASKS)),
         "| structurally applicable, full matrix | **%d** | the above minus the gradient "
         "cosine for exact RTRL and TBPTT (3 x 2 x %d = %d undefined cells) |"
         % (c["applicable_full"], len(TASKS), c["planned_full"] - c["applicable_full"]),
         "| computed (>= %d paired seeds) | %d | |" % (meta["min_pairs"], c["computed_ok"]),
         "| undefined by the protocol | %d | grad_cos for exact / TBPTT |" % c["undefined"],
         "| too few paired seeds | %d | |" % c["insufficient"],
         "| planned, pre-registered set | %d | 10 per (task, metric) x %d tasks x 2 metrics |"
         % (c["planned_legacy"], len(TASKS)),
         "| applicable, pre-registered set | %d | |" % c["applicable_legacy"], "",
         "**Holm families.** Two definitions are reported for every comparison and both are "
         "fixed before the data are seen; `holm_p` in D4_STATS.json carries the one "
         "`--holm-family` selected (this run: **%s**)." % full_res["holm_family"], "",
         "| family definition | size on the task metric | size on the cosine | smallest "
         "attainable holm_p at n=10 (Wilcoxon floor p=0.001953) |", "|---|---|---|---|",
         "| per (task, metric) | %d (3 ranks x 6 opponents) | %d (3 ranks x 4 opponents) | "
         "%.4f on the metric, %.4f on the cosine |"
         % (st[(TASKS[0], "metric")], st[(TASKS[0], "grad_cos")],
            st[(TASKS[0], "metric")] * 0.001953125,
            st[(TASKS[0], "grad_cos")] * 0.001953125),
         "| per (rank, metric) | %d (6 opponents x %d tasks) | %d (4 opponents x %d tasks) | "
         "%.4f on the metric, %.4f on the cosine |"
         % (sr[(FULL_RANKS[0], "metric")], len(TASKS),
            sr[(FULL_RANKS[0], "grad_cos")], len(TASKS),
            sr[(FULL_RANKS[0], "metric")] * 0.001953125,
            sr[(FULL_RANKS[0], "grad_cos")] * 0.001953125), "",
         "With ten seeds the exact two-sided Wilcoxon floor is p = 2/2^10 = 0.001953, so in "
         "a (rank, metric) family NO comparison can reach holm_p < 0.05 however large the "
         "effect: the floor is %.4f on the cosine and %.4f on the task metric. The "
         "(task, metric) family keeps every floor below 0.05 (%.4f and %.4f). This is a "
         "reporting choice, not a data problem, and the manuscript must state which one it "
         "uses."
         % (sr[(FULL_RANKS[0], "grad_cos")] * 0.001953125,
            sr[(FULL_RANKS[0], "metric")] * 0.001953125,
            st[(TASKS[0], "grad_cos")] * 0.001953125,
            st[(TASKS[0], "metric")] * 0.001953125), ""]

    L += ["`delta` is signed so a positive value favours the first-named method; `Cliff "
          "delta` is the dominance of the first over the second across all %d x %d cross "
          "pairs of seed values (+1.00 = no seed of the second reaches any seed of the "
          "first)." % (meta["require_seeds"], meta["require_seeds"]), ""]

    byk = full_lookup(full_res)
    for rank in FULL_RANKS:
        L += ["## %s -- full opponent sweep" % PLAIN[rank], ""]
        for field, fname in (("metric", "task metric"), ("grad_cos", "gradient cosine")):
            L += ["### %s -- %s (Holm family: %d per (task, metric), %d per (rank, metric))"
                  % (PLAIN[rank], fname, st[(TASKS[0], field)], sr[(rank, field)]), ""]
            L += md_full_rows([byk[(t, field, rank, o)] for t in TASKS
                               for o in FULL_OPPONENTS], field)

    if full_res["inter_rank"]:
        L += ["## Auxiliary: rank against rank", "",
              "Not part of the %d (these compare the method with itself at another rank, not "
              "with a competitor); Holm-corrected inside their own (task, metric) family of "
              "%d." % (c["applicable_full"], len(INTER_RANK_PAIRS)), ""]
        ir = dict(((r["task"], r["field"], r["ref"], r["other"]), r)
                  for r in full_res["inter_rank"])
        for field, fname in (("metric", "task metric"), ("grad_cos", "gradient cosine")):
            L += ["### rank vs rank -- %s" % fname, ""]
            L += md_full_rows([ir[(t, field, a, b)] for t in TASKS
                               for a, b in INTER_RANK_PAIRS], field)

    # ------------------------------------------------------------ claim audit
    CRITLAB = [("sign", "sign of delta"), ("ci", "bootstrap 95% CI excludes 0"),
               ("holm_task", "holm_p < 0.05, (task, metric) family"),
               ("holm_rank", "holm_p < 0.05, (rank, metric) family")]
    cs = audit["cosine_sweep"]
    L += ["# Claim audit", "",
          "Every count the body of the paper quotes, re-derived from the full matrix above. "
          "W-T-L is written win-tie-loss.", "",
          "## Claim 1 -- the gradient-cosine sweep (paper: \"SK-RTRL wins all 108, Cliff's "
          "delta exactly 1.00, p_H = 0.0156 throughout\")", "",
          "Set: 3 ranks x 4 approximate estimators (%s) x %d tasks = %d comparisons on the "
          "gradient cosine; %d computed."
          % (", ".join(PLAIN[o] for o in APPROX_FOUR), len(TASKS), cs["n"], cs["n_ok"]), "",
          "| criterion | W-T-L over all %d | %s | %s | %s |"
          % tuple([cs["n"]] + [PLAIN[r] for r in FULL_RANKS]),
          "|---|---|---|---|---|"]
    for key, lab in CRITLAB:
        L.append("| %s | %s | %s | %s | %s |"
                 % (lab, _md_wtl(cs["per_criterion"][key]),
                    _md_wtl(cs["per_rank"][FULL_RANKS[0]][key]),
                    _md_wtl(cs["per_rank"][FULL_RANKS[1]][key]),
                    _md_wtl(cs["per_rank"][FULL_RANKS[2]][key])))
    L += ["",
          "| quantity | value |", "|---|---|",
          "| comparisons with Cliff delta exactly +1.00 | %d of %d |"
          % (cs["cliff_eq_1"], cs["n_ok"]),
          "| Cliff delta range | %s to %s |"
          % (_md_num(cs["cliff_min"], 2, True), _md_num(cs["cliff_max"], 2, True)),
          "| holm_p range, (task, metric) family | %s to %s |"
          % (_md_num(cs["holm_taskfam_min"]), _md_num(cs["holm_taskfam_max"])),
          "| holm_p range, (rank, metric) family | %s to %s |"
          % (_md_num(cs["holm_rankfam_min"]), _md_num(cs["holm_rankfam_max"])), ""]

    L += ["## Claim 2 -- win-tie-loss on the task metric (paper: 31-5-9 at r=4, 32-6-7 at "
          "r=16, 35-6-4 at r=32)", "",
          "Set, per rank: the %d approximate estimators (%s) x %d tasks = %d comparisons on "
          "the task metric. The paper's stated rule is the bootstrap CI one."
          % (len(APPROX_FIVE), ", ".join(PLAIN[o] for o in APPROX_FIVE), len(TASKS),
             audit["wtl_task_metric"][FULL_RANKS[0]]["n"]), "",
          "| criterion | %s | %s | %s |" % tuple(PLAIN[r] for r in FULL_RANKS),
          "|---|---|---|---|"]
    for key, lab in CRITLAB:
        L.append("| %s | %s | %s | %s |"
                 % (lab, _md_wtl(audit["wtl_task_metric"][FULL_RANKS[0]][key]),
                    _md_wtl(audit["wtl_task_metric"][FULL_RANKS[1]][key]),
                    _md_wtl(audit["wtl_task_metric"][FULL_RANKS[2]][key])))
    L.append("")

    L += ["## Claim 3 -- against truncated BPTT alone (paper: 2 wins, 5 ties, 2 losses for "
          "r=16, Holm criterion)", "",
          "| criterion | %s | %s | %s |" % tuple(PLAIN[r] for r in FULL_RANKS),
          "|---|---|---|---|"]
    for key, lab in CRITLAB:
        L.append("| %s | %s | %s | %s |"
                 % (lab, _md_wtl(audit["vs_tbptt"][FULL_RANKS[0]][key]),
                    _md_wtl(audit["vs_tbptt"][FULL_RANKS[1]][key]),
                    _md_wtl(audit["vs_tbptt"][FULL_RANKS[2]][key])))
    L += ["", "Per task, SK-RTRL r=16 vs TBPTT on the task metric:", "",
          "| task | delta (r=16 better) | holm_p (task fam) | verdict (Holm) | verdict (CI) |",
          "|---|---|---|---|---|"]
    for t in TASKS:
        d = audit["vs_tbptt"]["skrtrl-r16"]["per_task"][t]
        L.append("| %s | %s | %s | %s | %s |"
                 % (t, _md_num(d["delta"], 4, True), fmt_p_md(d["holm_p_taskfam"]),
                    d["verdict_holm_task"], d["verdict_ci"]))
    L += ["", "Median d_z of r=16 vs TBPTT over the nine tasks: %s."
          % _md_num(audit["vs_tbptt_median_dz_r16"], 2, True), "",
          "## Claim 4 -- against exact RTRL, median d_z over the nine tasks (task metric)", "",
          "| rank | median d_z |", "|---|---|"]
    for r in FULL_RANKS:
        L.append("| %s | %s |" % (PLAIN[r], _md_num(audit["vs_exact_median_dz"][r], 2, True)))
    L.append("")
    return L


def write_stats_md(path, meta, stats_res, cells, full_res=None, audit=None):
    L = ["# D4 all-pairs paired statistics", "",
         "Generated %s by `code/make_r2_d4_tables.py`. Answers R3r1-10: confidence intervals "
         "and effect sizes for **all** principal comparisons, not only those against SnAp-1."
         % meta["generated"], "",
         "- comparisons, fixed in advance: `%s` against each of the eight other estimators, "
         "plus `skrtrl-r4` and `skrtrl-r32` against `snap1` -- %d per (task, metric)"
         % (REF, len(PLANNED)),
         "- metrics: the task metric (`metric`) and the gradient cosine (`grad_cos`); "
         "`grad_cos` is undefined for %s, so those comparisons are structurally absent and do "
         "not enter the family" % ", ".join(sorted(NO_COS)),
         "- pairing unit: the seed. `delta` is signed so that a positive value means the "
         "first-named method is better, using the direction of that metric",
         "- `p`: two-sided Wilcoxon signed-rank (exact for small n); `holm_p`: Holm step-down "
         "inside the family of that (task, metric), family size fixed in advance, so a "
         "comparison whose runs are missing still costs family size (conservative)",
         "- `CI95`: %d-resample paired bootstrap over seeds (rng seed %d); `d_z`: paired Cohen "
         "effect size" % (meta["boot"], meta["boot_seed"]),
         "- a comparison needs at least %d paired seeds; with 10 the two-sided Wilcoxon floor "
         "is p=0.002, with 5 it is p=0.0625" % meta["min_pairs"], ""]

    n_ok = n_sig = 0
    for (task, field), blk in stats_res.items():
        for r in blk["comparisons"]:
            if r["status"] != "ok":
                continue
            n_ok += 1
            if r["holm_p"] is not None and not math.isnan(r["holm_p"]) and r["holm_p"] < 0.05:
                n_sig += 1
    L += ["Computed %d of %d planned comparisons; %d are Holm-significant at 0.05."
          % (n_ok, len(TASKS) * 2 * len(PLANNED), n_sig), ""]

    for task in TASKS:
        L += ["## %s (task metric: %s, %s)"
              % (task, metric_name(task),
                 "higher better" if task in HIGHER_IS_BETTER else "lower better"), ""]
        for field, fname in (("metric", "task metric"), ("grad_cos", "gradient cosine")):
            blk = stats_res[(task, field)]
            L += ["### %s -- %s (Holm family size %d)" % (task, fname, blk["family"]), "",
                  "| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | "
                  "holm_p | d_z | note |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            nd = ND[field]
            for r in blk["comparisons"]:
                name = "%s vs %s" % (PLAIN[r["ref"]], PLAIN[r["other"]])
                if r["status"] != "ok":
                    L.append("| %s | %d | -- | -- | -- | -- | -- | -- | -- | %s: %s |"
                             % (name, r["n"], r["status"], r["note"]))
                    continue
                dz = r["cohen_dz"]
                L.append("| %s | %d | %.*f | %.*f | %+.*f | [%+.*f, %+.*f] | %s | %s | %s | %s |"
                         % (name, r["n"], nd, r["ref_mean"], nd, r["other_mean"],
                            nd, r["delta"], nd, r["ci95"][0], nd, r["ci95"][1],
                            fmt_p_md(r["wilcoxon_p"]), fmt_p_md(r["holm_p"]),
                            ("%+.2f" % dz) if math.isfinite(dz) else "inf",
                            "seeds %s" % ",".join(str(s) for s in r["seeds"])))
            L.append("")

    if full_res is not None and audit is not None:
        L += ["", "---", ""] + md_full_section(meta, full_res, audit)

    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def write_stats_json(path, meta, stats_res, full_res=None, audit=None):
    payload = {"meta": meta,
               "families": [{"task": t, "field": f, "family_size": blk["family"],
                             "comparisons": blk["comparisons"]}
                            for (t, f), blk in stats_res.items()]}
    if full_res is not None:
        payload["full_matrix"] = {
            "definition": {"ranks": FULL_RANKS, "opponents": FULL_OPPONENTS,
                           "tasks": list(TASKS), "fields": ["metric", "grad_cos"],
                           "approx_four": APPROX_FOUR, "approx_five": APPROX_FIVE,
                           "holm_family_primary": full_res["holm_family"],
                           "n_planned": full_res["n_planned"],
                           "n_applicable": full_res["n_applicable"]},
            "family_taskfam": dict(("%s|%s" % k, v)
                                   for k, v in full_res["family_taskfam"].items()),
            "family_rankfam": dict(("%s|%s" % k, v)
                                   for k, v in full_res["family_rankfam"].items()),
            "comparisons": full_res["comparisons"],
            "inter_rank": full_res["inter_rank"]}
    if audit is not None:
        payload["claim_audit"] = audit

    def clean(o):
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        if isinstance(o, dict):
            return dict((k, clean(v)) for k, v in o.items())
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, set):
            return sorted(o)
        return o

    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(clean(payload), fh, indent=1)


# ---------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--indir", nargs="+", default=[EVAL_DIR],
                    help="run directories (default: %s)" % EVAL_DIR)
    ap.add_argument("--select-json", default=SELECT_JSON,
                    help="stage-2 selection report, for the LR cross-check and the LR table")
    ap.add_argument("--outdir", default=REPORT_DIR, help="where D4_TABLES.md / D4_STATS.md go")
    ap.add_argument("--texdir", default=TEX_DIR, help="where the .tex fragments go")
    ap.add_argument("--agg", choices=["window20", "tail5"], default="window20",
                    help="per-run aggregation (default: the last-20%% window of the D4 protocol)")
    ap.add_argument("--frac", type=float, default=sel.WINDOW_FRAC)
    ap.add_argument("--tail", type=int, default=5)
    ap.add_argument("--ddof", type=int, default=0,
                    help="cross-seed std: 0 = population (published convention), 1 = sample")
    ap.add_argument("--require-seeds", type=int, default=len(SEEDS_EVAL))
    ap.add_argument("--require-steps", type=int, default=STEPS_EVAL)
    ap.add_argument("--seeds", default="", help="comma-separated seed list (default 0-9)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--min-pairs", type=int, default=3)
    ap.add_argument("--list-limit", type=int, default=40)
    ap.add_argument("--tasks-py", default="skrtrl/tasks.py")
    ap.add_argument("--no-tex", action="store_true")
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument("--full-matrix", dest="full_matrix", action="store_true", default=True,
                    help="also compute the full 3-rank x 6-opponent matrix and let "
                         "tab_r2_stats.tex cover it (default)")
    ap.add_argument("--no-full-matrix", dest="full_matrix", action="store_false",
                    help="revert to the pre-registered 10-per-(task, metric) set only")
    ap.add_argument("--holm-family", choices=list(HOLM_FAMILY_MODES), default="task-metric",
                    help="which Holm family `holm_p` and the LaTeX tables use; both are "
                         "always reported in D4_STATS.md/.json (default: task-metric, the "
                         "convention the manuscript declares)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any assertion fails")
    args = ap.parse_args(argv)

    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds
             else list(SEEDS_EVAL))

    direction = verify_direction(args.tasks_py)
    selected, sel_note = load_selection(args.select_json)
    runs, problems = load_runs(args.indir, args.agg, args.frac, args.tail)
    kept, dropped, ambiguous, unmatched = pick_config(runs, selected)
    cells = build_cells(kept, seeds, args.require_seeds, args.require_steps, args.ddof)
    pooled_ts = dict((a, pooled_cos(kept, CHAOS_TASKS, a, seeds, args.require_steps,
                                    args.ddof)) for a in ORDER)
    pooled_real = dict((a, pooled_cos(kept, REAL_TASKS, a, seeds, args.require_steps,
                                      args.ddof)) for a in ORDER)

    configs = {}
    for task in TASKS:
        for algo in ORDER:
            if (task, algo) in selected:
                configs[(task, algo)] = selected[(task, algo)]
            else:
                obs = sorted(set(cfg_of(kept[(task, algo, s)]) for s in seeds
                                 if (task, algo, s) in kept))
                if len(obs) == 1:
                    configs[(task, algo)] = obs[0]
                elif obs:
                    configs[(task, algo)] = obs[0]
    lr_source = ("Taken from the stage-2 selection report." if selected else
                 "**No stage-2 selection report was available**, so the configuration shown is "
                 "the one the runs themselves carry.")

    lr_bad = check_lrs(kept, unmatched, selected)
    score_bad = check_scores(kept)
    missing = missing_runs(kept, selected, seeds, args.indir[0])
    bad_cells = [c for c in (cells[k] for k in sorted(cells))
                 if c["defined"] and c["flags"]]
    empty_cells = [c for c in bad_cells if c["n"] == 0]
    n_cells = sum(1 for c in cells.values() if c["defined"])

    meta = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "indirs": [d.replace("\\", "/") for d in args.indir],
            "agg": args.agg, "frac": args.frac, "tail": args.tail, "ddof": args.ddof,
            "seeds": seeds, "require_seeds": args.require_seeds,
            "require_steps": args.require_steps, "boot": args.boot,
            "boot_seed": args.boot_seed, "min_pairs": args.min_pairs,
            "selection_note": sel_note, "select_json": args.select_json,
            "n_runs": len(runs), "n_cells_used": len(kept), "tex_written": []}

    provisional = bool(bad_cells or missing or lr_bad or score_bad or not direction["ok"])
    stats_res = ({} if args.no_stats else
                 run_stats(cells, args.boot, args.boot_seed, args.min_pairs))
    full_res = audit = None
    if stats_res and args.full_matrix:
        full_res = run_stats_full(cells, args.boot, args.boot_seed, args.min_pairs,
                                  holm_family=args.holm_family)
        audit = audit_claims(full_res)
    meta["holm_family"] = args.holm_family
    meta["full_matrix"] = bool(full_res)

    texdir = sel.rel(args.texdir)
    written = []
    if not args.no_tex:
        os.makedirs(texdir, exist_ok=True)
        files = [("tab_r2_fidelity.tex", tex_fidelity(cells, meta)),
                 ("tab_r2_timeseries.tex",
                  tex_series(cells, pooled_ts, CHAOS_TASKS, meta, "timeseries", "6",
                             r"\textbf{Online next-step prediction on the three chaotic "
                             r"systems} ($n{=}64$; " + proto_clause(meta) + r"). ",
                             stats_ref=(FULL_STATS_REF if full_res else
                                        STATS_LABEL_OF_TASK[CHAOS_TASKS[0]]))),
                 ("tab_r2_realts.tex",
                  tex_series(cells, pooled_real, REAL_TASKS, meta, "realts", "7",
                             r"\textbf{Online next-step prediction on the two real series} "
                             r"(monthly Sunspot and Santa~Fe laser, one-off non-causal "
                             r"fixed-window normalisation, $n{=}64$; "
                             + proto_clause(meta) + r"). ",
                             stats_ref=(FULL_STATS_REF if full_res else
                                        STATS_LABEL_OF_TASK[REAL_TASKS[0]]))),
                 ("tab_r2_lr.tex", tex_lr(configs, meta, lr_source))]
        if stats_res:
            files.append(("tab_r2_stats.tex",
                          tex_stats_full(full_res, meta) if full_res else
                          tex_stats(stats_res, meta, STATS_BLOCKS)))
        for name, body in files:
            p = os.path.join(texdir, name)
            with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(body)
            written.append(p)
    meta["tex_written"] = [p.replace("\\", "/") for p in written]

    checks = {"direction": direction, "bad_cells": bad_cells, "n_cells": n_cells,
              "lr_bad": lr_bad, "score_bad": score_bad, "missing": missing,
              "dropped": dropped, "ambiguous": ambiguous, "unmatched": unmatched,
              "problems": problems}
    checks["summary"] = [
        ("metric-direction table vs `%s`" % args.tasks_py,
         "OK" if direction["ok"] else "**FAIL** (%d problem(s))" % len(direction["problems"])),
        ("runs parsed", str(len(runs))),
        ("cells used (task, method, seed)", str(len(kept))),
        ("cells with n_seeds = %d and steps = %d" % (args.require_seeds, args.require_steps),
         "%d of %d" % (n_cells - len(bad_cells), n_cells)),
        ("cells flagged `†` (present but off-protocol)",
         "%d" % (len(bad_cells) - len(empty_cells)) if bad_cells else "0 (all compliant)"),
        ("cells with no usable run at all",
         "%d" % len(empty_cells) if empty_cells else "0"),
        ("missing runs", "%d of %d expected"
         % (len(missing), len(TASKS) * len(ORDER) * args.require_seeds)),
        ("LR equals the stage-2 selection",
         "not checked (%s)" % sel_note.split(" -- ")[0] if not selected else
         ("OK for all %d runs" % len(kept) if not lr_bad else
          "**FAIL** for %d run(s)" % len(lr_bad))),
        ("window mean equals `make_r2_d4_select`'s score",
         "not applicable (--agg %s)" % args.agg if args.agg != "window20" else
         ("OK for all %d runs" % len(kept) if not score_bad else
          "**FAIL** for %d run(s)" % len(score_bad))),
        ("runs present but unused (duplicate / off-config)",
         str(len(dropped) + len(unmatched))),
        ("unreadable / malformed inputs", str(len(problems))),
    ]

    outdir = sel.rel(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    tables_md = os.path.join(outdir, "D4_TABLES.md")
    write_tables_md(tables_md, meta, cells, pooled_ts, pooled_real, configs, checks,
                    args.list_limit)
    print("wrote %s" % tables_md)
    if stats_res:
        stats_md = os.path.join(outdir, "D4_STATS.md")
        write_stats_md(stats_md, meta, stats_res, cells, full_res, audit)
        write_stats_json(os.path.join(outdir, "D4_STATS.json"), meta, stats_res,
                         full_res, audit)
        print("wrote %s" % stats_md)
        print("wrote %s" % os.path.join(outdir, "D4_STATS.json"))
    for p in written:
        print("wrote %s" % p)

    print("runs parsed %d | cells used %d | cells flagged %d of %d | missing runs %d"
          % (len(runs), len(kept), len(bad_cells), n_cells, len(missing)))
    if audit is not None:
        c, cs = audit["counts"], audit["cosine_sweep"]
        print("  full matrix: %d planned, %d applicable, %d computed (Holm family: %s)"
              % (c["planned_full"], c["applicable_full"], c["computed_ok"],
                 args.holm_family))
        print("  cosine sweep (%d): %s by sign, %s by bootstrap CI, Cliff delta == +1.00 in "
              "%d of %d" % (cs["n"], _md_wtl(cs["per_criterion"]["sign"]),
                            _md_wtl(cs["per_criterion"]["ci"]), cs["cliff_eq_1"],
                            cs["n_ok"]))
        for r in FULL_RANKS:
            print("  task-metric W-T-L vs the five approximate, %s: %s by CI, %s by Holm "
                  "(task fam)" % (PLAIN[r], _md_wtl(audit["wtl_task_metric"][r]["ci"]),
                                  _md_wtl(audit["wtl_task_metric"][r]["holm_task"])))
        print("  vs TBPTT, r=16: %s by Holm (task fam), %s by CI"
              % (_md_wtl(audit["vs_tbptt"]["skrtrl-r16"]["holm_task"]),
                 _md_wtl(audit["vs_tbptt"]["skrtrl-r16"]["ci"])))
    if not direction["ok"]:
        print("  direction table: FAIL -- %s" % "; ".join(direction["problems"]))
    if lr_bad:
        print("  learning-rate mismatches against the stage-2 selection: %d" % len(lr_bad))
    if score_bad:
        print("  window means disagreeing with make_r2_d4_select: %d" % len(score_bad))
    if problems:
        print("  unreadable/malformed inputs: %d" % len(problems))
    if provisional:
        print("  PROVISIONAL: the LaTeX tables carry daggers / the report lists violations")
    if args.strict and provisional:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
