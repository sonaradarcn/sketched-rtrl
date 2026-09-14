"""A11 temporal hold-out report for the second-round revision -- sunspot & laser.

Reads `results/r2/d4_holdout/` for the hold-out numbers and `results/r2/d4_eval/` for the
published full-sequence reference, and writes `results/r2/HOLDOUT_SUMMARY.md`.  It launches
nothing, touches no GPU, and never writes into an experiment directory.

What the hold-out check has to show (R2 plan A11 / reply-letter R3-4: "the real series are
streamed cyclically, so the reported numbers are in-sample")
-------------------------------------------------------------------------------------------
  1. Whether the estimator ORDERING on data the optimiser never saw is the same ordering the
     published full-sequence tables report.  If SK-RTRL only looks competitive because the
     sketch memorises a cycled stream, the hold-out ordering has to break.
  2. Whether SK-RTRL's gap to exact RTRL WIDENS out of sample.  The paired gap is computed
     twice on the very same five seeds -- once on the training tail, once on the hold-out
     segment -- so "the gap widened" is a statement about one pair of numbers, not about two
     differently-powered experiments.
  3. Whether any estimator COLLAPSES out of sample.  The collapse criterion is fixed here,
     before any number is read: hold-out / training-tail ratio > 2 on an error task (the
     estimator at least doubled its error off the training stream).
  4. Every anomaly: missing json, NaN/non-finite records, hold-out passes that ran for a
     different number of steps than their siblings, learning-rate or TBPTT-window mismatch
     against `jobs/d4_eval.txt`.

Protocol
  hold-out value    `metric_holdout` from the run json: the mean one-step-ahead metric over a
                    SINGLE online pass through the held-out tail (`--holdout_frac 0.2`), no
                    parameter updates, hidden state from zero, the same `--washout` reset
                    period as training (`run_m3.holdout_pass`).  `metric_holdout_last200` is
                    printed beside it as a sensitivity column, not as the headline.
  training value    the D4 window: arithmetic mean of `metric` over every logged record with
                    `step >= 0.8 * steps`, taken from the SAME json via
                    `make_r2_d4_tables.agg_field` -- so the two columns of a row come from one
                    run and differ only in which data the metric was measured on.
  full-sequence     the same D4 window over `results/r2/d4_eval/`, 10 seeds, at the D4
                    stage-2 selected learning rate / TBPTT window.  Those runs cycled the
                    WHOLE series, so their number is the in-sample one the paper publishes.
  across seeds      mean +/- POPULATION std (`--ddof 0`, the published convention), the seed
                    being the unit of aggregation -- never the pooled record set.
  95 % CI           Student-t on the seed mean: mean +/- t_{.975,n-1} * s/sqrt(n) with the
                    SAMPLE std (ddof=1).  With 5 seeds t_{.975,4} = 2.776, so the hold-out
                    intervals are wider than the 10-seed D4 ones and are labelled with their n
                    everywhere; they are NOT a substitute for the D4 intervals.
  paired stats      SK-RTRL r=16 / r=32 against exact / SnAp-1 / TBPTT on the five shared
                    seeds: Wilcoxon signed-rank (two-sided, `zero_method="wilcox"`), a
                    bootstrap 95 % CI on the mean paired difference (10 000 resamples,
                    `make_r2_d4_tables.bootstrap_ci`), and Cohen's d_z.  The sign convention is
                    the D4 one: POSITIVE = SK-RTRL is better under the task's own direction.
                    At n=5 the signed-rank test's minimum two-sided p is 0.0625, so no
                    comparison here can reach p<0.05 -- that floor is printed next to every p
                    rather than left for the reader to rediscover, and no Holm correction is
                    applied because correcting a family whose smallest attainable p is already
                    above 0.05 would make every cell vacuous.
  ratio             hold-out / training-tail.  Reported twice: as a ratio of the across-seed
                    means, and as the across-seed summary of the PER-SEED ratio (the paired
                    quantity).  Both are printed because they answer different questions and
                    a divergence between them is itself a finding.

Direction is `make_r2_d4_select.HIGHER_IS_BETTER` (both tasks here are error tasks, so lower
is better), imported rather than restated.

Confounds this script prints rather than hides
  * a `--holdout_frac 0.2` run TRAINS ON 80 % OF THE SERIES; the `d4_eval` runs trained on
    100 % of it.  The training-tail column of a hold-out run is therefore not identical to the
    `d4_eval` column even though the protocol, learning rate, seed and step budget match -- it
    is the same optimiser on a shorter stream.  Every full-sequence comparison is labelled
    accordingly, and the load-bearing hold-out/training comparison is the WITHIN-run one.
  * the hold-out pass is a single pass over ~`holdout_steps` samples, an order of magnitude
    fewer than the 20 000-step training stream, so its per-seed value is noisier by
    construction; `holdout_steps` is printed for every cell.
  * `metric_holdout` starts the hidden state from zero inside the tail, so the first few
    samples of the pass carry a state-estimation transient that training-stream records do not.

Usage
  python make_r2_holdout_report.py
  python make_r2_holdout_report.py --field metric_holdout_last200
  python make_r2_holdout_report.py --outdir /tmp/demo --strict
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import make_r2_d4_select as sel          # noqa: E402  direction + window protocol
import make_r2_d4_tables as d4t          # noqa: E402  aggregation helpers, reused verbatim

_REQUIRED_D4T = ["agg_field", "_clean", "bootstrap_ci", "wilcoxon_p", "cliffs_delta", "PLAIN"]
_absent = [a for a in _REQUIRED_D4T if not hasattr(d4t, a)]
if _absent:
    raise SystemExit("make_r2_d4_tables.py no longer exposes %s -- the D4 aggregation moved; "
                     "fix make_r2_holdout_report.py before trusting any number it prints."
                     % _absent)
_REQUIRED_SEL = ["HIGHER_IS_BETTER", "WINDOW_FRAC", "EVAL_DIR", "REPORT_DIR", "rel",
                 "is_tbptt"]
_absent = [a for a in _REQUIRED_SEL if not hasattr(sel, a)]
if _absent:
    raise SystemExit("make_r2_d4_select.py no longer exposes %s -- the D4 protocol moved; "
                     "fix make_r2_holdout_report.py." % _absent)

# ---------------------------------------------------------------- protocol constants
HOLDOUT_DIR = "results/r2/d4_holdout"
EVAL_DIR = sel.EVAL_DIR
REPORT_DIR = sel.REPORT_DIR
JOBS_HOLDOUT = "jobs/d4_holdout.txt"
JOBS_EVAL = "jobs/d4_eval.txt"

TASKS = ["sunspot", "laser"]
# display order, matching the published tables (Table 3/6/7 order, restricted to the A11 grid)
ORDER = ["exact", "skrtrl-r32", "skrtrl-r16", "snap1", "tbptt"]
SEEDS = [0, 1, 2, 3, 4]
SEEDS_EVAL = list(range(10))
STEPS = 20000
HOLDOUT_FRAC = 0.2

HIGHER_IS_BETTER = set(sel.HIGHER_IS_BETTER)      # neither sunspot nor laser is in it
WINDOW_FRAC = sel.WINDOW_FRAC
PLAIN = dict(d4t.PLAIN)

# the hold-out field and the within-run training-stream field
HO_FIELD = "metric_holdout"
HO_ALT = "metric_holdout_last200"
TRAIN_FIELD = "metric_train_tail"                 # derived in read_one via d4t.agg_field

# SK-RTRL rows and the baselines they are paired against (fixed before any data are read)
SK_ROWS = ["skrtrl-r16", "skrtrl-r32"]
BASELINES = ["exact", "snap1", "tbptt"]
PLANNED = [(sk, b) for sk in SK_ROWS for b in BASELINES]

# the collapse criterion, fixed in advance
COLLAPSE_RATIO = 2.0
# Wilcoxon's attainable floor at n=5 (two-sided, no ties): 2 / 2**5
WILCOXON_FLOOR_N5 = 2.0 / 2 ** 5

ND = {"metric_holdout": 4, "metric_holdout_last200": 4, "metric_train_tail": 4,
      "metric": 4, "ratio": 3}
BOOT = 10000
BOOT_SEED = 0


def rel(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def higher_better(task):
    return task in HIGHER_IS_BETTER


def metric_name(task):
    return "accuracy" if task in HIGHER_IS_BETTER else "NMSE"


# ---------------------------------------------------------------- run loading
def read_one(path, frac):
    """Parse one run json.  Never raises: a bad file becomes a `problem` record.

    Produces, for the same run: the hold-out scalars straight off the json, and the D4
    training window recomputed from `records` with the D4 helper.
    """
    r = {"path": path, "file": os.path.basename(path), "status": "ok", "nonfinite": []}
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
             steps=a.get("steps"), tag=a.get("tag"), n_hid=a.get("n"), batch=a.get("batch"),
             cell=a.get("cell") or "tanh", washout=a.get("washout"), clip=a.get("clip"),
             horizon=a.get("horizon"), causal=a.get("causal"), shadow=a.get("shadow"),
             holdout_frac=a.get("holdout_frac") or 0.0, n_records=len(recs))
    if None in (r["task"], r["algo"], r["seed"], r["lr"]):
        r.update(status="bad_args", error="args missing task/algo/seed/lr")
        return r
    r["seed"] = int(r["seed"])
    if sel.is_tbptt(r["algo"]):
        w = a.get("tbptt_window", a.get("tbptt_k"))
        r["window"] = int(w) if isinstance(w, (int, float)) else None
    else:
        r["window"] = None

    # --- the D4 training window, recomputed with the D4 helper on this run's own records ---
    v, n = d4t.agg_field(recs, "metric", r["steps"], "window20", frac, 5)
    r[TRAIN_FIELD] = v
    r[TRAIN_FIELD + "__n"] = n
    # the same number under its D4 name, so the full-sequence reference cells can be built
    # with the published field name and the two sides cannot drift apart
    r["metric"] = v
    v, n = d4t.agg_field(recs, "grad_cos", r["steps"], "window20", frac, 5)
    r["grad_cos"] = v

    # --- the hold-out scalars, straight off the json (no re-derivation possible) -----------
    for f in (HO_FIELD, HO_ALT):
        x = d.get(f)
        r[f] = float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None
    for f in ("holdout_steps", "holdout_split_step"):
        x = d.get(f)
        r[f] = int(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None
    r["holdout_note"] = d.get("holdout_note")
    r["holdout_frac_json"] = d.get("holdout_frac")

    # --- per-seed ratio: the paired hold-out / training quantity --------------------------
    ho, tr = r.get(HO_FIELD), r.get(TRAIN_FIELD)
    if (ho is not None and tr is not None and math.isfinite(ho) and math.isfinite(tr)
            and tr != 0):
        r["ratio"] = ho / tr
    else:
        r["ratio"] = None

    # --- NaN / non-finite audit over the RAW record stream --------------------------------
    raw_metric = [x.get("metric") for x in recs]
    r["n_nan_metric"] = sum(1 for v in raw_metric
                            if isinstance(v, float) and not math.isfinite(v))
    r["n_none_metric"] = sum(1 for v in raw_metric if v is None)
    fin = d4t._clean(raw_metric)
    r["first_metric"] = fin[0] if fin else None
    r["last_metric"] = fin[-1] if fin else None
    after0 = d4t._clean([x.get("metric") for x in recs
                         if isinstance(x.get("step"), (int, float)) and x["step"] > 0])
    if after0:
        r["best_after0"] = max(after0) if higher_better(r["task"]) else min(after0)
    else:
        r["best_after0"] = None
    for f in (HO_FIELD, HO_ALT, TRAIN_FIELD):
        x = r.get(f)
        if x is not None and not math.isfinite(x):
            r["nonfinite"].append(f)

    sf = d.get("svd_fallbacks") or {}
    r["svd_fallbacks"] = sum(v for v in sf.values()
                             if isinstance(v, (int, float))) if sf else 0
    if not recs:
        r["status"] = "no_records"
    return r


def scan_dir(indir, frac, tasks, algos, need_holdout):
    """-> (kept {(task, algo, seed): run}, problems [run])."""
    p = rel(indir)
    kept, problems = {}, []
    if not os.path.isdir(p):
        return kept, [{"file": indir, "status": "missing_dir",
                       "error": "directory %s does not exist" % p}]
    for name in sorted(os.listdir(p)):
        if not name.endswith(".json"):
            continue
        r = read_one(os.path.join(p, name), frac)
        r["indir"] = indir
        if r["status"] in ("unreadable", "bad_args"):
            problems.append(r)
            continue
        if r["task"] not in tasks or r["algo"] not in algos:
            continue
        if need_holdout and r.get(HO_FIELD) is None:
            problems.append(dict(r, status="no_holdout",
                                 error="json carries no %s (was it run without "
                                       "--holdout_frac?)" % HO_FIELD))
            continue
        key = (r["task"], r["algo"], r["seed"])
        if key in kept:
            problems.append(dict(r, status="duplicate",
                                 error="second file for %s; kept %s"
                                       % (str(key), kept[key]["file"])))
            continue
        kept[key] = r
    return kept, problems


def cfg_str(r):
    lr = r.get("lr")
    s = "lr%g" % lr if lr is not None else "lr?"
    if r.get("window") is not None:
        s += "/w%d" % r["window"]
    return s


# ---------------------------------------------------------------- cells
def t_ci95(vals):
    """Student-t 95 % CI on the mean of `vals` (SAMPLE std).  -> (lo, hi) or (None, None)."""
    n = len(vals)
    if n < 2:
        return None, None
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1))
    if not math.isfinite(s):
        return None, None
    h = float(stats.t.ppf(0.975, n - 1)) * s / math.sqrt(n)
    return m - h, m + h


def build_cell(kept, task, algo, field, seeds, ddof, require_steps):
    c = {"task": task, "algo": algo, "field": field, "mean": None, "std": None,
         "ci": (None, None), "n": 0, "n_expected": len(seeds), "values": {},
         "seeds_missing": [], "seeds_nonfinite": [], "seeds_bad_steps": [],
         "cfgs": set(), "flags": []}
    for s in seeds:
        r = kept.get((task, algo, s))
        if r is None:
            c["seeds_missing"].append(s)
            continue
        c["cfgs"].add(cfg_str(r))
        if require_steps and r.get("steps") != require_steps:
            c["seeds_bad_steps"].append((s, r.get("steps")))
        v = r.get(field)
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v):
            c["seeds_nonfinite"].append((s, "none" if v is None else "%g" % v))
            continue
        c["values"][s] = float(v)
    vals = [c["values"][s] for s in sorted(c["values"])]
    c["n"] = len(vals)
    if vals:
        c["mean"] = float(np.mean(vals))
        if len(vals) > 1:
            c["std"] = float(np.std(vals, ddof=ddof))
            c["ci"] = t_ci95(vals)
    c["cfgs"] = sorted(c["cfgs"])
    if c["n"] == 0:
        c["flags"].append("no usable run")
    elif c["n"] != len(seeds):
        c["flags"].append("n=%d/%d" % (c["n"], len(seeds)))
    if c["seeds_nonfinite"]:
        c["flags"].append("non-finite: "
                          + ",".join("s%s=%s" % t for t in c["seeds_nonfinite"]))
    if c["seeds_bad_steps"]:
        c["flags"].append("steps!=%s: " % require_steps
                          + ",".join("s%s=%s" % t for t in c["seeds_bad_steps"]))
    if len(c["cfgs"]) > 1:
        c["flags"].append("mixed configs: " + ",".join(c["cfgs"]))
    return c


HO_FIELDS = [HO_FIELD, HO_ALT, TRAIN_FIELD, "ratio", "holdout_steps", "grad_cos"]
EVAL_FIELDS = ["metric", "grad_cos"]


def build_cells(kept, fields, seeds, ddof, require_steps):
    return {(t, a, f): build_cell(kept, t, a, f, seeds, ddof, require_steps)
            for t in TASKS for a in ORDER for f in fields}


def ratio_of_means(ho_cells, task, algo):
    """Hold-out mean / training-tail mean, with each side's n.  A ratio of means -- NOT the
    mean of the per-seed ratios, which is reported separately."""
    cn = ho_cells.get((task, algo, HO_FIELD))
    cd = ho_cells.get((task, algo, TRAIN_FIELD))
    if cn is None or cd is None or cn["mean"] is None or cd["mean"] is None or cd["mean"] == 0:
        return None
    return {"ratio": cn["mean"] / cd["mean"], "num": cn["mean"], "den": cd["mean"],
            "n_num": cn["n"], "n_den": cd["n"]}


# ---------------------------------------------------------------- statistics
def compare_pair(cells, task, field, sk, base):
    """Paired SK-RTRL vs baseline on the shared seeds.  D4 sign convention: positive delta
    means SK-RTRL (the `ref` of the pair) is better under the task's own direction."""
    rec = {"task": task, "field": field, "ref": sk, "other": base, "status": "ok",
           "n": 0, "seeds": [], "ref_mean": None, "other_mean": None, "delta": None,
           "ci95": (None, None), "ci95_t": (None, None), "wilcoxon_p": None,
           "cohen_dz": None, "cliff_delta": None, "higher_better": higher_better(task),
           "note": ""}
    A = cells[(task, sk, field)]["values"]
    B = cells[(task, base, field)]["values"]
    seeds = sorted(set(A) & set(B))
    rec["seeds"] = seeds
    rec["n"] = len(seeds)
    if len(seeds) < 2:
        rec.update(status="insufficient", note="%d paired seed(s)" % len(seeds))
        return rec
    a = np.array([A[s] for s in seeds], dtype=float)
    b = np.array([B[s] for s in seeds], dtype=float)
    delta = (a - b) if rec["higher_better"] else (b - a)
    lo, hi = d4t.bootstrap_ci(delta, n=BOOT, seed=BOOT_SEED)
    sd = float(np.std(delta, ddof=1))
    rec.update(ref_mean=float(a.mean()), other_mean=float(b.mean()),
               delta=float(delta.mean()), ci95=(lo, hi), ci95_t=t_ci95(delta),
               wilcoxon_p=d4t.wilcoxon_p(a, b),
               cohen_dz=(float(delta.mean() / sd) if sd > 0 else None),
               cliff_delta=d4t.cliffs_delta(a, b, rec["higher_better"]))
    if sd == 0:
        rec["note"] = "zero paired spread: d_z undefined"
    return rec


def run_stats(cells, field):
    return {(t, sk, b): compare_pair(cells, t, field, sk, b)
            for t in TASKS for sk, b in PLANNED}


def ordering(cells, task, algos, field):
    """-> [(algo, mean)] best first under the task's own direction, plus the algos with no
    usable mean (they cannot be ranked and are listed separately)."""
    have = [(a, cells[(task, a, field)]["mean"]) for a in algos
            if cells.get((task, a, field)) and cells[(task, a, field)]["mean"] is not None]
    miss = [a for a in algos
            if not cells.get((task, a, field))
            or cells[(task, a, field)]["mean"] is None]
    have.sort(key=lambda kv: (-kv[1] if higher_better(task) else kv[1]))
    return have, miss


def resolvability(cells, task, field, algos):
    """How much of the ordering is actually RESOLVED by the data.

    An ordering can "change" between two lenses simply because none of its gaps is
    distinguishable from zero.  For every unordered pair of estimators, this runs the same
    paired bootstrap as section 4 and counts the pairs whose 95 % CI excludes zero.  A task
    where 0/10 pairs separate has no ordering to disagree with, and the report says so instead
    of reading a Spearman rho off noise.
    """
    pairs, sep = [], 0
    for i, a in enumerate(algos):
        for b in algos[i + 1:]:
            rec = compare_pair(cells, task, field, a, b)
            if rec["status"] != "ok" or rec["ci95"][0] is None:
                pairs.append((a, b, rec, None))
                continue
            excl = rec["ci95"][0] * rec["ci95"][1] > 0
            sep += int(excl)
            pairs.append((a, b, rec, excl))
    return {"n_pairs": len(pairs), "n_separated": sep, "pairs": pairs}


def rank_agreement(a_order, b_order):
    """Spearman rho + Kendall tau between two orderings over the SAME algo set.
    -> dict or None when fewer than 3 algos are shared."""
    a_rank = {k: i for i, (k, _) in enumerate(a_order)}
    b_rank = {k: i for i, (k, _) in enumerate(b_order)}
    shared = [k for k in a_rank if k in b_rank]
    if len(shared) < 3:
        return None
    x = [a_rank[k] for k in shared]
    y = [b_rank[k] for k in shared]
    try:
        rho = float(stats.spearmanr(x, y).correlation)
    except Exception:                                                   # noqa: BLE001
        rho = float("nan")
    try:
        tau = float(stats.kendalltau(x, y).correlation)
    except Exception:                                                   # noqa: BLE001
        tau = float("nan")
    inversions = sum(1 for i in range(len(shared)) for j in range(i + 1, len(shared))
                     if (x[i] - x[j]) * (y[i] - y[j]) < 0)
    return {"n": len(shared), "spearman": rho, "kendall": tau,
            "inversions": inversions, "pairs": len(shared) * (len(shared) - 1) // 2,
            "identical": x == y, "shared": shared}


# ---------------------------------------------------------------- inventory
def job_outs(jobs_file):
    """-> ({path: (file, line)}, found?) from every `# out=` annotation."""
    p = rel(jobs_file)
    outs = {}
    if not os.path.isfile(p):
        return outs, False
    with io.open(p, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if line.lstrip().startswith("#"):
                continue
            k = line.rfind("# out=")
            if k < 0:
                continue
            outs.setdefault(line[k + len("# out="):].strip(), (jobs_file, i))
    return outs, True


def inventory(outs, indir):
    present = [p for p in sorted(outs) if os.path.isfile(rel(p))]
    missing = [p for p in sorted(outs) if not os.path.isfile(rel(p))]
    d = rel(indir)
    expected = {os.path.normcase(os.path.basename(p)) for p in outs}
    extra = [n for n in sorted(os.listdir(d))
             if n.endswith(".json") and os.path.normcase(n) not in expected] \
        if os.path.isdir(d) else []
    return {"outs": outs, "present": present, "missing": missing, "extra": extra}


def eval_config(eval_kept, task, algo):
    """The (lr, window) config the d4_eval reference actually used, and whether it is
    unique across its seeds."""
    cfgs = sorted({cfg_str(r) for (t, a, _), r in eval_kept.items()
                   if t == task and a == algo})
    return cfgs


# ---------------------------------------------------------------- formatting
def fmt(x, nd=3):
    if x is None:
        return "--"
    if isinstance(x, float) and not math.isfinite(x):
        return "inf" if x > 0 else "-inf"
    if isinstance(x, float) and x != 0 and (abs(x) < 10 ** -nd or abs(x) >= 1e6):
        return "%.2e" % x
    return ("%." + str(nd) + "f") % x


def md_cell(c, nd=None, with_ci=True):
    """`mean +/- std [lo, hi]` with the cell's n appended whenever it is short."""
    if c is None or c["mean"] is None:
        return "--"
    nd = ND.get(c["field"], 3) if nd is None else nd
    s = fmt(c["mean"], nd)
    if c["std"] is not None:
        s += " ± " + fmt(c["std"], nd)
    if with_ci and c["ci"][0] is not None:
        s += " [%s, %s]" % (fmt(c["ci"][0], nd), fmt(c["ci"][1], nd))
    if c["n"] != c["n_expected"]:
        s += " (n=%d)" % c["n"]
    return s


def fmt_p(p):
    if p is None:
        return "--"
    if isinstance(p, float) and math.isnan(p):
        return "n/a"
    return "%.4f" % p if p >= 0.0001 else "<0.0001"


def fmt_delta(rec, nd=4):
    if rec["status"] != "ok" or rec["delta"] is None:
        return "--"
    s = "%s" % fmt(rec["delta"], nd)
    if rec["delta"] > 0:
        s = "+" + s
    if rec["ci95"][0] is not None:
        s += " [%s, %s]" % (fmt(rec["ci95"][0], nd), fmt(rec["ci95"][1], nd))
    return s


def fmt_dz(rec):
    if rec["status"] != "ok" or rec["cohen_dz"] is None:
        return "--"
    return "%+.2f" % rec["cohen_dz"]


def table(rows, header):
    w = [len(h) for h in header]
    for r in rows:
        for i, v in enumerate(r):
            w[i] = max(w[i], len(str(v)))
    out = ["| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(header)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(header))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v).ljust(w[i]) for i, v in enumerate(r)) + " |")
    return out


def order_str(task, have, miss):
    s = "  <  ".join("%s %s" % (PLAIN.get(a, a), fmt(m, 4)) for a, m in have)
    if miss:
        s += "   (unranked: %s)" % ", ".join(PLAIN.get(a, a) for a in miss)
    return s


# ---------------------------------------------------------------- checks
def run_checks(ho_cells, ev_cells, ho_kept, ev_kept, inv, meta):
    """Everything the headline asserts is computed here, once, from the cells."""
    head, conf, collapse, anom = [], [], [], []

    # the collapse criterion below is written for error tasks (ratio > 1 means "worse out of
    # sample").  Refuse rather than silently invert it if an accuracy task ever enters.
    acc = [t for t in TASKS if higher_better(t)]
    if acc:
        raise SystemExit("make_r2_holdout_report.py's collapse criterion assumes error "
                         "tasks; %s is an accuracy task -- fix the criterion before "
                         "trusting section 5." % acc)

    # modal number of records the D4 training window used, for the anomaly audit
    counts = [r[TRAIN_FIELD + "__n"] for r in ho_kept.values()
              if isinstance(r.get(TRAIN_FIELD + "__n"), int)]
    meta["train_window_n"] = (max(set(counts), key=counts.count) if counts else 0)

    # ---- ordering agreement: hold-out vs full-sequence, and hold-out vs its own train tail
    orders = {}
    for task in TASKS:
        orders[(task, "holdout")] = ordering(ho_cells, task, ORDER, HO_FIELD)
        orders[(task, "train")] = ordering(ho_cells, task, ORDER, TRAIN_FIELD)
        orders[(task, "eval")] = ordering(ev_cells, task, ORDER, "metric")
    agree = {}
    for task in TASKS:
        agree[(task, "eval")] = rank_agreement(orders[(task, "holdout")][0],
                                               orders[(task, "eval")][0])
        agree[(task, "train")] = rank_agreement(orders[(task, "holdout")][0],
                                                orders[(task, "train")][0])

    # how much of each ordering the data actually resolve, on each lens
    res = {}
    for task in TASKS:
        res[(task, "holdout")] = resolvability(ho_cells, task, HO_FIELD, ORDER)
        res[(task, "train")] = resolvability(ho_cells, task, TRAIN_FIELD, ORDER)

    for task in TASKS:
        a = agree[(task, "eval")]
        rh = res[(task, "holdout")]
        if a is None:
            head.append("**%s**: too few ranked estimators to compare orderings." % task)
        else:
            head.append("**%s**: hold-out ordering vs published full-sequence ordering -- "
                        "Spearman rho = %.3f, Kendall tau = %.3f, %d/%d discordant pairs; "
                        "%s. On the hold-out segment **%d of %d** pairwise gaps have a "
                        "bootstrap CI that excludes zero, so %s."
                        % (task, a["spearman"], a["kendall"], a["inversions"], a["pairs"],
                           "the ordering is IDENTICAL" if a["identical"]
                           else "the ordering CHANGES",
                           rh["n_separated"], rh["n_pairs"],
                           "the reordering is within hold-out noise and NO ordering claim is "
                           "supportable on this task" if rh["n_separated"] == 0
                           else "part of the ordering is resolved"))

    # ---- does the SK-RTRL gap to exact widen out of sample?
    st_ho = run_stats(ho_cells, HO_FIELD)
    st_tr = run_stats(ho_cells, TRAIN_FIELD)
    widen = []
    for task in TASKS:
        for sk in SK_ROWS:
            h = st_ho[(task, sk, "exact")]
            t = st_tr[(task, sk, "exact")]
            if h["status"] != "ok" or t["status"] != "ok":
                continue
            # the gap is |delta|; a NEGATIVE delta means SK-RTRL is worse than exact.
            # the gap as a fraction of exact RTRL's own level on the same segment: an
            # absolute delta of 2e-4 on a 0.13 NMSE is a 0.15 % effect and must not be
            # readable as "a gap" without its scale
            widen.append({"task": task, "sk": sk, "d_train": t["delta"],
                          "d_hold": h["delta"],
                          "gap_train": abs(t["delta"]), "gap_hold": abs(h["delta"]),
                          "rel_train": (abs(t["delta"]) / t["other_mean"]
                                        if t["other_mean"] else None),
                          "rel_hold": (abs(h["delta"]) / h["other_mean"]
                                       if h["other_mean"] else None),
                          "grew": abs(h["delta"]) > abs(t["delta"]),
                          "sign_flip": (t["delta"] > 0) != (h["delta"] > 0),
                          "ci_excludes_0": (h["ci95"][0] is not None
                                            and h["ci95"][0] * h["ci95"][1] > 0)})
    if widen:
        grew = [w for w in widen if w["grew"]]
        worse = [w for w in widen if w["d_hold"] < 0]
        rels = [w["rel_hold"] for w in widen if w["rel_hold"] is not None]
        head.append("SK-RTRL vs exact RTRL, paired on the same 5 seeds: the absolute gap is "
                    "larger on the hold-out segment in **%d of %d** (task, rank) cells; "
                    "SK-RTRL is behind exact on the hold-out segment in **%d of %d**; the "
                    "hold-out bootstrap CI excludes zero in **%d of %d**. The largest "
                    "hold-out gap to exact RTRL is **%.2f %%** of exact's own level."
                    % (len(grew), len(widen), len(worse), len(widen),
                       len([w for w in widen if w["ci_excludes_0"]]), len(widen),
                       100 * max(rels) if rels else float("nan")))

    # ---- collapse screen: hold-out / training-tail ratio > COLLAPSE_RATIO
    for task in TASKS:
        for algo in ORDER:
            rm = ratio_of_means(ho_cells, task, algo)
            c = ho_cells[(task, algo, "ratio")]
            per_seed_max = max(c["values"].values()) if c["values"] else None
            bad_seeds = sorted(s for s, v in c["values"].items() if v > COLLAPSE_RATIO)
            row = {"task": task, "algo": algo,
                   "ratio_means": rm["ratio"] if rm else None,
                   "ratio_mean_of_seeds": c["mean"], "ratio_max_seed": per_seed_max,
                   "bad_seeds": bad_seeds,
                   "collapsed": bool(rm and rm["ratio"] > COLLAPSE_RATIO)}
            collapse.append(row)
    n_coll = len([r for r in collapse if r["collapsed"]])
    n_seedy = len([r for r in collapse if r["bad_seeds"]])
    head.append("collapse screen (hold-out / training-tail ratio > %.1f on an error task): "
                "**%d of %d** (task, estimator) cells collapse on the ratio of means; "
                "**%d** cells contain at least one individual seed above the threshold."
                % (COLLAPSE_RATIO, n_coll, len(collapse), n_seedy))

    # ---- anomalies
    for key in sorted(ho_kept):
        r = ho_kept[key]
        bad = []
        if r["n_nan_metric"]:
            bad.append("metric NaN/inf x%d" % r["n_nan_metric"])
        if r["n_none_metric"]:
            bad.append("metric None x%d" % r["n_none_metric"])
        if r["nonfinite"]:
            bad.append("non-finite: " + ",".join(r["nonfinite"]))
        if r["steps"] != STEPS:
            bad.append("steps=%s" % r["steps"])
        if r.get("holdout_frac_json") != HOLDOUT_FRAC:
            bad.append("holdout_frac=%s" % r.get("holdout_frac_json"))
        if r["svd_fallbacks"]:
            bad.append("svd_fallbacks=%s" % r["svd_fallbacks"])
        if r[TRAIN_FIELD + "__n"] != meta["train_window_n"]:
            # the modal record count of the D4 training window across this grid; a run that
            # disagrees was logged on a different grid and its column is not comparable
            bad.append("train window used %d record(s), grid uses %d"
                       % (r[TRAIN_FIELD + "__n"], meta["train_window_n"]))
        if bad:
            anom.append([r["task"], PLAIN.get(r["algo"], r["algo"]), r["seed"],
                         "; ".join(bad)])

    # hold-out pass length must be identical within a task (same series, same split)
    for task in TASKS:
        hs = sorted({r["holdout_steps"] for (t, _, _), r in ho_kept.items() if t == task
                     and r["holdout_steps"] is not None})
        sp = sorted({r["holdout_split_step"] for (t, _, _), r in ho_kept.items() if t == task
                     and r["holdout_split_step"] is not None})
        if len(hs) > 1 or len(sp) > 1:
            anom.append([task, "(all)", "-", "hold-out pass not identical across runs: "
                                              "holdout_steps=%s split=%s" % (hs, sp)])

    # config match against d4_eval
    for task in TASKS:
        for algo in ORDER:
            ho_cfg = ho_cells[(task, algo, HO_FIELD)]["cfgs"]
            ev_cfg = eval_config(ev_kept, task, algo)
            if ho_cfg and ev_cfg and set(ho_cfg) != set(ev_cfg):
                anom.append([task, PLAIN.get(algo, algo), "-",
                             "config differs from d4_eval: hold-out %s vs eval %s"
                             % (",".join(ho_cfg), ",".join(ev_cfg))])

    if inv["missing"]:
        head.append("**%d expected hold-out run(s) missing** -- see section 1."
                    % len(inv["missing"]))
    else:
        head.append("inventory complete: all %d hold-out runs in `%s` are present and parsed."
                    % (len(inv["present"]), JOBS_HOLDOUT))
    head.append("anomalies (NaN, non-finite, wrong step budget, config drift): **%d**."
                % len(anom))

    # ---- confounds
    conf.append("a `--holdout_frac %.1f` run trains on the first %.0f %% of the series only; "
                "the `%s` reference trained on 100 %% of it. The training-tail column of a "
                "hold-out run is therefore the same optimiser on a SHORTER stream, not a "
                "replica of the published column -- the load-bearing comparison in this "
                "report is the WITHIN-run hold-out/training one."
                % (HOLDOUT_FRAC, 100 * (1 - HOLDOUT_FRAC), EVAL_DIR))
    hs = {t: sorted({r["holdout_steps"] for (tt, _, _), r in ho_kept.items() if tt == t
                     and r["holdout_steps"] is not None}) for t in TASKS}
    conf.append("the hold-out pass is a single online pass of %s samples (sunspot / laser) "
                "against a %d-step training stream, so a per-seed hold-out value is noisier "
                "by construction."
                % (" / ".join("%s" % hs[t] for t in TASKS), STEPS))
    conf.append("`%s` starts the hidden state from zero inside the tail, so the first samples "
                "of the pass carry a state-estimation transient; `%s` is printed beside it as "
                "the sensitivity column." % (HO_FIELD, HO_ALT))
    degen = [t for t in TASKS if hs[t] and max(hs[t]) <= 200]
    if degen:
        conf.append("`%s` is DEGENERATE on %s: the whole hold-out pass is %s samples, so the "
                    "last-200 column is the same number as `%s` there and provides no "
                    "sensitivity at all. It is informative on %s only."
                    % (HO_ALT, ", ".join(degen),
                       "/".join(str(x) for t in degen for x in hs[t]), HO_FIELD,
                       ", ".join(t for t in TASKS if t not in degen) or "no task"))
    short = [t for t in TASKS if hs[t] and max(hs[t]) < 1000]
    if short:
        conf.append("the hold-out segments are SHORT (%s). The resulting per-seed spread is "
                    "what makes the hold-out CIs several times wider than the training-tail "
                    "CIs in section 2, and it is why section 3 reports how many pairwise "
                    "gaps actually separate rather than only a rank correlation."
                    % ", ".join("%s %s" % (t, hs[t]) for t in short))
    conf.append("seeds: the hold-out grid has %d, the `%s` reference up to %d. Every cell "
                "prints its own n; the paired statistics use only the %d shared hold-out "
                "seeds." % (len(SEEDS), EVAL_DIR, len(SEEDS_EVAL), len(SEEDS)))
    conf.append("at n=5 the two-sided Wilcoxon signed-rank p cannot go below %.4f, so NO "
                "comparison in section 4 can reach p<0.05. The p column is reported for "
                "completeness; the bootstrap CI and d_z carry the evidence, and no Holm "
                "correction is applied to a family whose attainable floor already exceeds "
                "0.05." % WILCOXON_FLOOR_N5)
    ns = sorted({r["n_hid"] for r in list(ho_kept.values()) + list(ev_kept.values())})
    bs = sorted({r["batch"] for r in list(ho_kept.values()) + list(ev_kept.values())})
    wo = sorted({r["washout"] for r in list(ho_kept.values()) + list(ev_kept.values())})
    conf.append("width / batch / washout across both stages: n in %s, batch in %s, washout "
                "in %s%s." % (ns, bs, wo,
                              " -- matched" if len(ns) == len(bs) == len(wo) == 1
                              else " -- **NOT matched**"))

    return {"headline": head, "confounds": conf, "collapse": collapse, "anomalies": anom,
            "orders": orders, "agree": agree, "stats_holdout": st_ho,
            "stats_train": st_tr, "widen": widen, "resolve": res}


# ---------------------------------------------------------------- report
def write_report(path, meta, ho_cells, ev_cells, ho_kept, ev_kept, problems, inv, checks):
    L = []
    A = L.append
    A("# A11 -- temporal hold-out on the measured series (sunspot, laser)")
    A("")
    A("Generated %s by `make_r2_holdout_report.py` (reads `%s` and `%s`; writes nothing "
      "else)." % (meta["when"], HOLDOUT_DIR, EVAL_DIR))
    A("")
    A("- hold-out value: `%s` from the run json -- the mean one-step-ahead %s over a SINGLE "
      "online pass through the held-out tail (`--holdout_frac %.1f`), no parameter updates, "
      "hidden state from zero, same `--washout` reset period as training "
      "(`run_m3.holdout_pass`)." % (HO_FIELD, "error", HOLDOUT_FRAC))
    A("- training value: the D4 window on the SAME run -- arithmetic mean of `metric` over "
      "every logged record with `step >= %.0f%% of %d`, via "
      "`make_r2_d4_tables.agg_field`." % (100 * meta["frac"], STEPS))
    A("- full-sequence reference: the same D4 window over `%s`, %d seeds, at the D4 stage-2 "
      "selected learning rate / TBPTT window; those runs cycled the whole series."
      % (EVAL_DIR, len(SEEDS_EVAL)))
    A("- across seeds: mean ± population std (ddof=%d), 95%% CI = Student-t on the seed mean "
      "(sample std, t_{.975,n-1}); hold-out seeds %s." % (meta["ddof"], SEEDS))
    A("- paired statistics: Wilcoxon signed-rank (two-sided) + bootstrap 95%% CI on the mean "
      "paired difference (%d resamples) + Cohen's d_z, on the shared hold-out seeds. Sign "
      "convention is the D4 one: **positive = SK-RTRL better**." % BOOT)
    A("- collapse criterion, fixed before any number was read: hold-out / training-tail "
      "ratio > **%.1f** on an error task." % COLLAPSE_RATIO)
    A("- both tasks are error tasks (`%s`), so lower is better throughout."
      % ", ".join(sorted(TASKS)))
    A("")

    # ---------------------------------------------------------------- 0. headline
    A("## 0. Headline")
    A("")
    for line in checks["headline"]:
        A("- " + line)
    A("")

    # ---------------------------------------------------------------- 1. inventory
    A("## 1. Inventory and anomalies")
    A("")
    A("- expected (`# out=` annotations in `%s`): **%d**" % (JOBS_HOLDOUT, len(inv["outs"])))
    A("- present: **%d**" % len(inv["present"]))
    A("- missing: **%d**" % len(inv["missing"]))
    A("- unexpected extra json in `%s`: **%d**" % (HOLDOUT_DIR, len(inv["extra"])))
    A("- hold-out runs aggregated: **%d** ; full-sequence reference runs aggregated: **%d**"
      % (len(ho_kept), len(ev_kept)))
    A("")
    if inv["missing"]:
        A("Missing:")
        A("")
        A("```")
        for p in inv["missing"]:
            A(os.path.basename(p))
        A("```")
        A("")
    else:
        A("No missing runs against the job-file expectation.")
        A("")
    if inv["extra"]:
        A("Extra files (not in the `# out=` list): "
          + ", ".join("`%s`" % e for e in inv["extra"]))
        A("")
    if problems:
        A("Unreadable / malformed / duplicate / hold-out-less files:")
        A("")
        for p in problems:
            A("- `%s`: %s -- %s" % (p.get("file"), p.get("status"), p.get("error", "")))
        A("")
    else:
        A("Every json parsed cleanly and every one carries `%s`." % HO_FIELD)
        A("")

    A("### Hold-out split, as recorded by `run_m3.holdout_pass`")
    A("")
    rows = []
    for task in TASKS:
        hs = sorted({r["holdout_steps"] for (t, _, _), r in ho_kept.items() if t == task
                     and r["holdout_steps"] is not None})
        sp = sorted({r["holdout_split_step"] for (t, _, _), r in ho_kept.items()
                     if t == task and r["holdout_split_step"] is not None})
        note = sorted({r["holdout_note"] for (t, _, _), r in ho_kept.items() if t == task})
        rows.append([task, ",".join(str(x) for x in hs), ",".join(str(x) for x in sp),
                     note[0] if len(note) == 1 else "MIXED: %s" % note])
    L.extend(table(rows, ["task", "holdout_steps", "split_step", "note"]))
    A("")

    A("### Anomaly audit (raw record stream + args, not the window)")
    A("")
    if checks["anomalies"]:
        L.extend(table(checks["anomalies"], ["task", "estimator", "seed", "anomaly"]))
    else:
        A("No NaN, no non-finite window value, no wrong step budget, no config drift "
          "against `%s`, no SVD fallback." % JOBS_EVAL)
    A("")

    # ---------------------------------------------------------------- 2. main table
    A("## 2. Hold-out vs training tail vs published full sequence")
    A("")
    A("One row = one (task, estimator). Columns 2-3 come from the SAME five hold-out runs "
      "and differ only in which data the metric was measured on; column 5 is the published "
      "10-seed number from `%s`, whose runs cycled the whole series (see the confounds)."
      % EVAL_DIR)
    A("")
    for task in TASKS:
        A("**%s** (%s, lower is better)" % (task, metric_name(task)))
        A("")
        rows = []
        for algo in ORDER:
            ho = ho_cells[(task, algo, HO_FIELD)]
            alt = ho_cells[(task, algo, HO_ALT)]
            tr = ho_cells[(task, algo, TRAIN_FIELD)]
            rc = ho_cells[(task, algo, "ratio")]
            rm = ratio_of_means(ho_cells, task, algo)
            ev = ev_cells[(task, algo, "metric")]
            flags = sorted(set(ho["flags"]) | set(tr["flags"]))
            rows.append([PLAIN.get(algo, algo),
                         md_cell(ho), md_cell(tr),
                         fmt(rm["ratio"], 3) if rm else "--",
                         md_cell(rc, nd=3, with_ci=False),
                         md_cell(ev, with_ci=True),
                         md_cell(alt, with_ci=False),
                         "; ".join(flags) if flags else ""])
        L.extend(table(rows, ["estimator", "hold-out (mean±sd [CI95])",
                              "training tail (same runs)", "ratio of means",
                              "per-seed ratio (mean±sd)",
                              "full sequence, 10 seeds (`%s`)" % EVAL_DIR,
                              "hold-out last200", "flags"]))
        A("")

    A("### Per-seed hold-out values")
    A("")
    for task in TASKS:
        rows = []
        for algo in ORDER:
            c = ho_cells[(task, algo, HO_FIELD)]
            t = ho_cells[(task, algo, TRAIN_FIELD)]
            rows.append([PLAIN.get(algo, algo)]
                        + [fmt(c["values"].get(s), 4) for s in SEEDS]
                        + [fmt(t["values"].get(s), 4) for s in SEEDS])
        L.extend(table(rows, ["%s: estimator" % task]
                       + ["ho s%d" % s for s in SEEDS]
                       + ["tr s%d" % s for s in SEEDS]))
        A("")

    # ---------------------------------------------------------------- 3. ordering
    A("## 3. Does the ordering survive out of sample?")
    A("")
    for task in TASKS:
        A("**%s**" % task)
        A("")
        A("- hold-out:        %s" % order_str(task, *checks["orders"][(task, "holdout")]))
        A("- training tail:   %s" % order_str(task, *checks["orders"][(task, "train")]))
        A("- full sequence:   %s" % order_str(task, *checks["orders"][(task, "eval")]))
        for lens, label in (("eval", "hold-out vs full sequence"),
                            ("train", "hold-out vs its own training tail")):
            a = checks["agree"][(task, lens)]
            if a is None:
                A("- %s: too few ranked estimators." % label)
            else:
                A("- %s: Spearman rho = %.3f, Kendall tau = %.3f, discordant pairs %d/%d, "
                  "%s." % (label, a["spearman"], a["kendall"], a["inversions"], a["pairs"],
                           "identical ordering" if a["identical"] else "ordering changes"))
        for lens, label in (("holdout", "hold-out"), ("train", "training tail")):
            r = checks["resolve"][(task, lens)]
            A("- %s: **%d of %d** pairwise gaps have a bootstrap 95%% CI excluding zero."
              % (label, r["n_separated"], r["n_pairs"]))
        rh = checks["resolve"][(task, "holdout")]
        if rh["n_separated"] == 0:
            A("- **the hold-out ordering on this task is not resolved by the data**: not one "
              "of the %d pairwise gaps separates, so any reordering against the "
              "full-sequence table is noise and no ordering claim may be made from it."
              % rh["n_pairs"])
        A("")

    A("### Which hold-out gaps actually separate")
    A("")
    A("Same paired bootstrap as section 4, run over every unordered pair of estimators. "
      "`sep` marks a pair whose 95 % CI excludes zero; those are the only gaps the hold-out "
      "segment licenses a statement about.")
    A("")
    for task in TASKS:
        rows = []
        for a1, b1, rec, excl in checks["resolve"][(task, "holdout")]["pairs"]:
            rows.append([task, PLAIN.get(a1, a1), PLAIN.get(b1, b1),
                         fmt_delta(rec), fmt_dz(rec),
                         "sep" if excl else ("--" if excl is None else "")])
        L.extend(table(rows, ["task", "A", "B", "delta A-B [CI95 boot]", "d_z", "sep"]))
        A("")

    # ---------------------------------------------------------------- 4. paired stats
    A("## 4. SK-RTRL against the baselines, paired on the same seeds")
    A("")
    A("Positive delta = SK-RTRL better (the task's own direction). `delta [CI95]` is the mean "
      "paired difference with a %d-resample bootstrap CI; `p` is the two-sided Wilcoxon "
      "signed-rank p, whose attainable floor at n=5 is %.4f -- **no cell here can reach "
      "p<0.05**, by construction and not by outcome." % (BOOT, WILCOXON_FLOOR_N5))
    A("")
    for field, label in ((HO_FIELD, "hold-out segment"), (TRAIN_FIELD, "training tail")):
        A("**%s** (`%s`)" % (label, field))
        A("")
        st = checks["stats_holdout"] if field == HO_FIELD else checks["stats_train"]
        rows = []
        for task in TASKS:
            for sk in SK_ROWS:
                for b in BASELINES:
                    rec = st[(task, sk, b)]
                    rows.append([task, PLAIN.get(sk, sk), PLAIN.get(b, b), rec["n"],
                                 fmt(rec["ref_mean"], 4), fmt(rec["other_mean"], 4),
                                 fmt_delta(rec), fmt_dz(rec), fmt_p(rec["wilcoxon_p"]),
                                 fmt(rec["cliff_delta"], 2),
                                 rec["note"] or ("" if rec["status"] == "ok"
                                                 else rec["status"])])
        L.extend(table(rows, ["task", "SK-RTRL", "baseline", "n", "SK mean", "base mean",
                              "delta [CI95 boot]", "d_z", "Wilcoxon p", "Cliff", "note"]))
        A("")

    A("### Does the gap to exact RTRL widen out of sample?")
    A("")
    A("Both columns are the same paired statistic on the same five seeds, differing only in "
      "which data the metric was measured on. `gap` is |delta|; a negative delta means "
      "SK-RTRL is behind exact RTRL.")
    A("")
    rows = []
    for w in checks["widen"]:
        rows.append([w["task"], PLAIN.get(w["sk"], w["sk"]),
                     ("%+.4g" % w["d_train"]), ("%+.4g" % w["d_hold"]),
                     fmt(w["gap_train"], 4), fmt(w["gap_hold"], 4),
                     ("%.2f %%" % (100 * w["rel_train"])) if w["rel_train"] else "--",
                     ("%.2f %%" % (100 * w["rel_hold"])) if w["rel_hold"] else "--",
                     fmt(w["gap_hold"] / w["gap_train"], 2) if w["gap_train"] else "--",
                     "gap grows" if w["grew"] else "gap shrinks",
                     "SIGN FLIP" if w["sign_flip"] else "",
                     "CI excludes 0" if w["ci_excludes_0"] else "CI spans 0"])
    if rows:
        L.extend(table(rows, ["task", "SK-RTRL", "delta train", "delta hold-out",
                              "gap train", "gap hold-out", "gap/exact train",
                              "gap/exact hold-out", "gap ratio", "direction",
                              "flip", "hold-out CI"]))
    else:
        A("Not computable: no (task, rank) cell has both columns.")
    A("")

    # ---------------------------------------------------------------- 5. collapse
    A("## 5. Collapse screen")
    A("")
    A("Criterion fixed in advance: an estimator collapses out of sample if its hold-out "
      "metric is more than **%.1f x** its own training-tail metric. The ratio of means is "
      "the headline; the per-seed maximum is printed so a single bad seed cannot hide behind "
      "an average." % COLLAPSE_RATIO)
    A("")
    rows = []
    for r in checks["collapse"]:
        rows.append([r["task"], PLAIN.get(r["algo"], r["algo"]),
                     fmt(r["ratio_means"], 3), fmt(r["ratio_mean_of_seeds"], 3),
                     fmt(r["ratio_max_seed"], 3),
                     ",".join(str(s) for s in r["bad_seeds"]) or "--",
                     "**COLLAPSE**" if r["collapsed"] else "ok"])
    L.extend(table(rows, ["task", "estimator", "ratio of means", "mean per-seed ratio",
                          "max per-seed ratio", "seeds > %.1f" % COLLAPSE_RATIO, "verdict"]))
    A("")

    # ---------------------------------------------------------------- 6. confounds
    A("## 6. Confounds and what this check does NOT show")
    A("")
    for c in checks["confounds"]:
        A("- " + c)
    A("")
    A("- this is a hold-out in TIME on two measured series at n=64, 20 000 steps. It says "
      "nothing about the diagnostic or chaotic tasks, which are generated on the fly and "
      "have no in-sample problem to begin with.")
    A("- the hold-out pass uses no estimator at all: it is a forward pass of the trained "
      "cell and readout. It therefore measures what the estimator's gradients LEARNED, not "
      "how well the estimator tracks a Jacobian.")
    A("")

    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default=HOLDOUT_DIR, help="hold-out results")
    ap.add_argument("--evaldir", default=EVAL_DIR, help="full-sequence reference (D4 eval)")
    ap.add_argument("--outdir", default=REPORT_DIR)
    ap.add_argument("--name", default="HOLDOUT_SUMMARY.md")
    ap.add_argument("--field", default=HO_FIELD, choices=[HO_FIELD, HO_ALT],
                    help="which hold-out scalar is the headline (default %s)" % HO_FIELD)
    ap.add_argument("--frac", type=float, default=WINDOW_FRAC)
    ap.add_argument("--ddof", type=int, default=0)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--eval-seeds", default=",".join(str(s) for s in SEEDS_EVAL))
    ap.add_argument("--jobs", default=JOBS_HOLDOUT)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any run is missing, unreadable, non-finite, or "
                         "any cell collapses")
    args = ap.parse_args(argv)

    if args.field != HO_FIELD:
        globals()["HO_FIELD"] = args.field
        globals()["HO_ALT"] = HO_FIELD if args.field == HO_ALT else HO_ALT
        globals()["HO_FIELDS"] = [args.field, globals()["HO_ALT"], TRAIN_FIELD, "ratio",
                                  "holdout_steps", "grad_cos"]

    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    eval_seeds = [int(x) for x in args.eval_seeds.split(",") if x.strip() != ""]
    globals()["SEEDS"] = seeds
    globals()["SEEDS_EVAL"] = eval_seeds

    algos = set(ORDER)
    ho_kept, ho_prob = scan_dir(args.indir, args.frac, set(TASKS), algos, True)
    ev_kept, ev_prob = scan_dir(args.evaldir, args.frac, set(TASKS), algos, False)

    ho_cells = build_cells(ho_kept, HO_FIELDS, seeds, args.ddof, STEPS)
    ev_cells = build_cells(ev_kept, EVAL_FIELDS, eval_seeds, args.ddof, STEPS)

    outs, found = job_outs(args.jobs)
    if not found:
        ho_prob.append({"file": args.jobs, "status": "missing_jobs",
                        "error": "job file absent; the inventory check is vacuous"})
    inv = inventory(outs, args.indir)

    meta = {"when": datetime.now().strftime("%Y-%m-%d %H:%M"), "ddof": args.ddof,
            "frac": args.frac}
    checks = run_checks(ho_cells, ev_cells, ho_kept, ev_kept, inv, meta)

    outdir = rel(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, args.name)
    write_report(path, meta, ho_cells, ev_cells, ho_kept, ev_kept, ho_prob, inv, checks)
    print("wrote %s" % path)
    print("  hold-out runs: %d ; full-sequence reference runs: %d ; problems: %d"
          % (len(ho_kept), len(ev_kept), len(ho_prob)))
    print("  expected %d / present %d / missing %d / extra %d"
          % (len(outs), len(inv["present"]), len(inv["missing"]), len(inv["extra"])))
    for line in checks["headline"]:
        print("  " + line.replace("**", ""))

    if args.strict:
        bad = len(inv["missing"]) + len(ho_prob) + len(checks["anomalies"])
        bad += len([r for r in checks["collapse"] if r["collapsed"]])
        if bad:
            print("STRICT: %d problem(s)" % bad)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
