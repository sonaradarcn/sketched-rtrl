"""D5 (gated cells) result tables for the second-round revision -- GRU, and LSTM when it lands.

Reads `results/r2/d5_gated/` alone for the gated numbers and `results/r2/d4_eval/` for the
tanh reference, and writes `results/r2/D5_SUMMARY.md`.  It launches nothing, touches no GPU,
and never writes into the experiment directories.

What D5 has to show (R2 plan D5 / R3-5: "applicability of the method to gated cells")
------------------------------------------------------------------------------------
  1. SK-RTRL at r in {4, 16, 32} against exact / SnAp-1 / UORO / RFLO on a gated cell, on the
     same 4 tasks (2 diagnostic + 1 chaotic + 1 real) as the tanh protocol.
  2. KF-RTRL is structurally absent, not missing: its Kronecker factorisation needs
     A_t = D_t W with a rank-1, block-diagonal immediate Jacobian, and a GRU has neither
     (the reset gate makes I_t non-block-diagonal).  `run_m3.py` refuses the combination.
     TBPTT is absent for a different reason: it is a D4 fairness baseline and
     `--algo tbptt_nnrnn` is tanh-only by construction.  Both are reported as such.
  3. Whether the gradient-bias certificate still holds once the cross-unit immediate term
     I^perp_t is absorbed into the low-rank append (Algorithm 1 lines 2-3): violations must
     be zero, and the bound's magnitude is compared against the tanh cell.
  4. Whether the r -> fidelity relation is the same shape as on the tanh cell, and whether the
     absorbed I^perp_t raises the effective rank the sketch has to carry (`stable_rank`,
     `sketch_eff`, `mass_top{4,8,16,32}` at the residual checkpoints).
  5. The memory and wall-clock price of the gated cell relative to tanh.

Protocol (kept deliberately identical to `make_r2_d4_tables.py` so the two can be read side
by side; the aggregation helpers are imported from it, not re-implemented)
  per-run value     mean of the field over every logged record with `step >= 0.8*steps`
                    (the last-20 % window).  `--agg tail5` switches to the last five records.
  across seeds      mean +/- POPULATION std (`--ddof 0`, the published convention), the seed
                    being the unit of aggregation -- never the pooled record set.
  95 % CI           Student-t on the seed mean: mean +/- t_{.975,n-1} * s/sqrt(n) with the
                    SAMPLE std (ddof=1).  With 3 seeds t_{.975,2} = 4.303, so these intervals
                    are wide by construction and are reported as such; they are NOT a
                    substitute for the 10-seed D4 intervals.
  effect size       paired against exact RTRL on the same seed: Cohen's d_z on the signed
                    difference (positive = the estimator is better than exact under the task's
                    own direction), plus a paired-t p.  No Wilcoxon and no Holm correction:
                    at n=3 the signed-rank test has a minimum two-sided p of 0.25 and a
                    multiplicity correction over 4 tasks x 6 estimators would make every cell
                    n/a, which would be less honest than reporting d_z with its n.
  heavy-tailed      `e_t` (the certificate bound) spans >20 orders of magnitude within one
                    run, so its per-run value is the MEDIAN over the window and its across-seed
                    summary is reported in log10.  Means of `e_t` are meaningless and are not
                    printed anywhere.

Direction is `make_r2_d4_select.HIGHER_IS_BETTER` (accuracy for anbn/copy, error elsewhere),
imported rather than restated.

Confounds this script prints rather than hides
  * the tanh reference in `d4_eval` was run at the D4 stage-2 learning rate and 10 seeds; the
    D5 grid ran at the `--default-lr` 1e-3 for every row with 3 seeds (the D5 job file says so
    in its header: `D4_SELECT_stage2.json` did not exist when the grid was generated).  Task
    metrics are therefore NOT a like-for-like tanh/GRU comparison and the tanh/GRU metric
    column is labelled accordingly; the certificate, sketch-spectrum, memory and per-step-cost
    comparisons do not depend on the learning rate in the same way.
  * `wall_s` was collected on a shared GPU.  The ratio column is flagged whenever the two
    sides could not have been contended alike.

Usage
  python make_r2_d5_report.py
  python make_r2_d5_report.py --agg tail5
  python make_r2_d5_report.py --outdir /tmp/demo --strict
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

import make_r2_d4_select as sel          # noqa: E402  direction + window protocol
import make_r2_d4_tables as d4t          # noqa: E402  aggregation helpers, reused verbatim
import make_r2_d5_jobs as d5j            # noqa: E402  the D5 grid definition

_REQUIRED_D4T = ["agg_field", "_clean", "bootstrap_ci", "NO_COS"]
_absent = [a for a in _REQUIRED_D4T if not hasattr(d4t, a)]
if _absent:
    raise SystemExit("make_r2_d4_tables.py no longer exposes %s -- the D4 aggregation moved; "
                     "fix make_r2_d5_report.py before trusting any number it prints." % _absent)
_REQUIRED_D5J = ["TASKS_D5", "ALGOS_D5", "SEEDS_D5", "STEPS_D5", "OUTDIR_D5", "CELLS_D5"]
_absent = [a for a in _REQUIRED_D5J if not hasattr(d5j, a)]
if _absent:
    raise SystemExit("make_r2_d5_jobs.py no longer exposes %s -- the D5 grid moved; fix "
                     "make_r2_d5_report.py." % _absent)

# ---------------------------------------------------------------- protocol constants
CELLS = list(d5j.CELLS_D5)                  # gru first, then lstm
TASKS = list(d5j.TASKS_D5)                  # rotation, anbn, henon, sunspot
ALGOS = list(d5j.ALGOS_D5)
SEEDS = list(d5j.SEEDS_D5)
STEPS = d5j.STEPS_D5
D5_DIR = d5j.OUTDIR_D5
TANH_DIR = sel.EVAL_DIR
REPORT_DIR = sel.REPORT_DIR
JOBS_FILES = ["jobs/d5_gru.txt", "jobs/d5_gated.txt", "jobs/d5_lstm.txt"]

HIGHER_IS_BETTER = set(sel.HIGHER_IS_BETTER)
WINDOW_FRAC = sel.WINDOW_FRAC
NO_COS = set(d4t.NO_COS)                    # grad_cos undefined for exact / tbptt

# display order, matching the published tables
ORDER = ["exact", "skrtrl-r32", "skrtrl-r16", "skrtrl-r4", "snap1", "rflo", "uoro"]
_unknown = sorted(set(ORDER) ^ set(ALGOS))
if _unknown:
    raise SystemExit("the D5 estimator set changed (%s); extend ORDER/PLAIN in "
                     "make_r2_d5_report.py" % _unknown)

# structurally inapplicable on a gated cell -- absent by derivation, not by omission
NOT_APPLICABLE = {
    "kfrtrl": "Kronecker factorisation needs A_t = D_t W and a rank-1, block-diagonal "
              "immediate Jacobian; a GRU's I_t is not block diagonal (reset-gate path "
              "through U_h). run_m3.py refuses --cell gru|lstm --algo kfrtrl.",
    "tbptt": "D4 fairness baseline, not part of the D5 estimator comparison; "
             "--algo tbptt_nnrnn is tanh-only by construction.",
}

PLAIN = {"exact": "exact RTRL", "skrtrl-r32": "SK-RTRL r=32", "skrtrl-r16": "SK-RTRL r=16",
         "skrtrl-r4": "SK-RTRL r=4", "snap1": "SnAp-1", "rflo": "RFLO", "uoro": "UORO",
         "kfrtrl": "KF-RTRL", "tbptt": "TBPTT"}
CELL_LABEL = {"tanh": "tanh RNN", "gru": "GRU", "lstm": "LSTM"}
REF_ALGO = "exact"
# `grad_cos` has no value for exact RTRL (its cosine to itself is trivially 1), so the
# fidelity effect size is paired against SnAp-1 -- the estimator the published tables compare
# against and the only other block-diagonal method that survives on a gated cell.
REF_COS = "snap1"

CHAOS_TASKS = {"henon", "mackeyglass", "lorenz"}
REAL_TASKS = {"sunspot", "laser"}

# fields aggregated as an arithmetic mean over the window
MEAN_FIELDS = ["metric", "grad_cos", "rho_bar", "true_E", "gerr_norm", "g_norm"]
# fields aggregated as a median over the window (heavy tailed over >20 decades)
MEDIAN_FIELDS = ["e_t"]
# derived per-run scalars (computed in read_one, aggregated like any other field)
DERIVED_FIELDS = ["e_t_inf_frac", "r_eps_over_n"]
R_EPS = 0.1                 # Table 11 reports r_eps/n at eps = 0.1
# per-run scalars taken straight from the json
SCALAR_FIELDS = ["wall_s", "peak_MB"]
# residual-spectrum fields read off the LAST checkpoint
CKPT_FIELDS = ["stable_rank", "sketch_eff", "res_norm", "E_norm", "best_rank_err",
               "mass_top4", "mass_top8", "mass_top16", "mass_top32"]

ND = {"metric": 4, "grad_cos": 3, "rho_bar": 2, "r_eps_over_n": 3,
      "stable_rank": 2, "sketch_eff": 2,
      "mass_top4": 3, "mass_top8": 3, "mass_top16": 3, "mass_top32": 3,
      "peak_MB": 1, "wall_s": 0, "res_norm": 2, "E_norm": 2, "best_rank_err": 2}


def higher_better(task, field):
    return True if field == "grad_cos" else (task in HIGHER_IS_BETTER)


def metric_name(task):
    if task in HIGHER_IS_BETTER:
        return "accuracy"
    return "NMSE" if task in (CHAOS_TASKS | REAL_TASKS) else "MSE"


def rel(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


# ---------------------------------------------------------------- run loading
def window_values(records, field, steps, agg, frac, tail):
    """Raw window values of `field`, NaN dropped but +/-inf KEPT."""
    if agg == "tail5":
        vals = [r.get(field) for r in records][-tail:]
    else:
        lo = frac * float(steps or 0)
        vals = [r.get(field) for r in records
                if isinstance(r.get("step"), (int, float)) and r["step"] >= lo]
    out = []
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        out.append(float(v))
    return out


def median_field(records, field, steps, agg, frac, tail):
    """Per-run MEDIAN of the FINITE window values of `field`, plus the overflow fraction.

    `e_t` reaches float64 +inf on the long-memory tasks (the bound multiplies per-step
    contraction factors > 1 over a 20 000-step trajectory), and that happens on the tanh cell
    too -- see section 3.  A median over a window half of which is +inf would just be +inf and
    would hide the magnitude of the half that is representable, so the overflow is reported as
    its own number and the median is taken over the finite part.
    -> (median_finite, n_finite, inf_frac)
    """
    vals = window_values(records, field, steps, agg, frac, tail)
    if not vals:
        return None, 0, None
    fin = [v for v in vals if math.isfinite(v)]
    inf_frac = 1.0 - len(fin) / float(len(vals))
    if not fin:
        return None, 0, inf_frac
    return float(np.median(fin)), len(fin), inf_frac


def read_one(path, agg, frac, tail):
    """Parse one run json.  Never raises: a bad file becomes a `problem` record."""
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
             shadow=a.get("shadow"), svd_driver=a.get("svd_driver"),
             cell=a.get("cell") or "tanh", n_records=len(recs))
    if None in (r["task"], r["algo"], r["seed"], r["lr"]):
        r.update(status="bad_args", error="args missing task/algo/seed/lr")
        return r
    r["seed"] = int(r["seed"])
    if not recs:
        r.update(status="no_records")
        return r

    for f in MEAN_FIELDS:
        v, n = d4t.agg_field(recs, f, r["steps"], agg, frac, tail)
        r[f] = v
        r[f + "__n"] = n
    for f in MEDIAN_FIELDS:
        v, n, inf_frac = median_field(recs, f, r["steps"], agg, frac, tail)
        r[f] = v
        r[f + "__n"] = n
        r[f + "_inf_frac"] = inf_frac
    for f in SCALAR_FIELDS:
        v = d.get(f)
        r[f] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    # --- NaN / non-finite / divergence audit over the RAW record stream -------------
    raw_metric = [x.get("metric") for x in recs]
    raw_cos = [x.get("grad_cos") for x in recs]
    r["n_nan_metric"] = sum(1 for v in raw_metric
                            if isinstance(v, float) and not math.isfinite(v))
    r["n_nan_cos"] = sum(1 for v in raw_cos
                         if isinstance(v, float) and not math.isfinite(v))
    fin = d4t._clean(raw_metric)
    r["last_metric"] = fin[-1] if fin else None
    r["first_metric"] = fin[0] if fin else None
    r["max_metric"] = max(fin) if fin else None
    # best value reached AFTER step 0 -- the step-0 record is degenerate on the accuracy
    # tasks (untrained net, majority symbol) and would make any "worse than start" test fire
    # on every run; see run_checks.
    after0 = d4t._clean([x.get("metric") for x in recs
                         if isinstance(x.get("step"), (int, float)) and x["step"] > 0])
    if after0:
        r["best_after0"] = max(after0) if r["task"] in HIGHER_IS_BETTER else min(after0)
    else:
        r["best_after0"] = None
    for f in MEAN_FIELDS:
        if r.get(f) is not None and not math.isfinite(r[f]):
            r["nonfinite"].append(f)
    for f in MEDIAN_FIELDS:          # median is finite-only by construction; flag "all inf"
        if r.get(f + "_inf_frac") == 1.0:
            r["nonfinite"].append(f + " (+inf over the whole window)")

    # --- certificate summary -------------------------------------------------------
    cs = d.get("cert_summary") or {}
    r["cert"] = cs
    r["cert_violations"] = cs.get("n_bound_violation")
    r["cert_violations_fp"] = cs.get("n_bound_violation_fp")
    r["cert_invalid"] = cs.get("n_cert_invalid")
    r["cert_n_steps"] = cs.get("n_steps")
    r["frac_rhobar_lt1"] = cs.get("frac_rhobar_lt1")
    r["frac_rel_lt1"] = cs.get("frac_rel_lt1")
    r["frac_tight_le10"] = cs.get("frac_tight_le10")
    r["n_bound_finite"] = cs.get("n_bound_finite")
    r["has_shadow"] = cs.get("has_shadow")

    # --- residual spectrum at the last checkpoint ----------------------------------
    ck = d.get("checkpoints") or []
    last_ck = ck[-1] if ck else {}
    for f in CKPT_FIELDS:
        v = last_ck.get(f)
        r["ck_" + f] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) \
            else None
    r["n_ckpt"] = len(ck)
    r["ck_r"] = last_ck.get("r")

    # --- r_eps / n : the rank at which the residual retains 1-eps of its Frobenius mass ----
    # Table 11's second lower-block row.  `mass_topk` is the fraction of the residual's
    # SQUARED spectrum in its top k singular values, logged on the grid k in {4,8,16,32}, so
    # r_eps is found by linear interpolation in k between the two bracketing grid points and
    # reported as a fraction of the state width n.  Values are bracketed, not exact: <=4/n
    # when the top-4 mass already clears the threshold, >32/n when even the top 32 do not.
    r["r_eps_over_n"], r["r_eps_bracket"] = None, ""
    ks = [4, 8, 16, 32]
    mass = [r.get("ck_mass_top%d" % k) for k in ks]
    if r["n_hid"] and all(m is not None for m in mass):
        thr = 1.0 - R_EPS
        if mass[0] >= thr:
            r["r_eps_over_n"] = ks[0] / float(r["n_hid"])
            r["r_eps_bracket"] = "<="
        elif mass[-1] < thr:
            r["r_eps_over_n"] = ks[-1] / float(r["n_hid"])
            r["r_eps_bracket"] = ">"
        else:
            for i in range(len(ks) - 1):
                if mass[i] < thr <= mass[i + 1]:
                    span = mass[i + 1] - mass[i]
                    f = (thr - mass[i]) / span if span > 0 else 1.0
                    k = ks[i] + f * (ks[i + 1] - ks[i])
                    r["r_eps_over_n"] = k / float(r["n_hid"])
                    break

    sf = d.get("svd_fallbacks") or {}
    r["svd_fallbacks"] = sum(v for v in sf.values()
                             if isinstance(v, (int, float))) if sf else 0
    return r


def scan_dir(indir, agg, frac, tail, cells=None, tasks=None):
    """-> (kept {(cell, task, algo, seed): run}, problems [run])."""
    indir = rel(indir)
    kept, problems = {}, []
    if not os.path.isdir(indir):
        return kept, [{"file": indir, "status": "missing_dir",
                       "error": "directory does not exist"}]
    for name in sorted(os.listdir(indir)):
        if not name.endswith(".json"):
            continue
        r = read_one(os.path.join(indir, name), agg, frac, tail)
        if r["status"] != "ok":
            problems.append(r)
            continue
        if cells and r["cell"] not in cells:
            continue
        if tasks and r["task"] not in tasks:
            continue
        key = (r["cell"], r["task"], r["algo"], r["seed"])
        if key in kept:
            problems.append(dict(r, status="duplicate",
                                 error="second file for %s; kept %s"
                                       % (str(key), kept[key]["file"])))
            continue
        kept[key] = r
    return kept, problems


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


def build_cell(kept, cell, task, algo, field, seeds, ddof, require_steps):
    c = {"cell": cell, "task": task, "algo": algo, "field": field,
         "mean": None, "std": None, "ci": (None, None), "n": 0, "values": {},
         "seeds_missing": [], "seeds_nonfinite": [], "seeds_bad_steps": [],
         "lrs": set(), "defined": True, "flags": []}
    if field == "grad_cos" and algo in NO_COS:
        c["defined"] = False
        return c
    for s in seeds:
        r = kept.get((cell, task, algo, s))
        if r is None:
            c["seeds_missing"].append(s)
            continue
        if r.get("lr") is not None:
            c["lrs"].add(float(r["lr"]))
        if require_steps and r.get("steps") != require_steps:
            c["seeds_bad_steps"].append((s, r.get("steps")))
        v = r.get(field)
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
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
    c["lrs"] = sorted(c["lrs"])
    if c["n"] == 0:
        c["flags"].append("no usable run")
    elif c["n"] != len(seeds):
        c["flags"].append("n=%d/%d" % (c["n"], len(seeds)))
    if c["seeds_nonfinite"]:
        c["flags"].append("non-finite: " + ",".join("s%s=%s" % t for t in c["seeds_nonfinite"]))
    if c["seeds_bad_steps"]:
        c["flags"].append("steps!=%s: " % require_steps
                          + ",".join("s%s=%s" % t for t in c["seeds_bad_steps"]))
    return c


ALL_FIELDS = (MEAN_FIELDS + MEDIAN_FIELDS + DERIVED_FIELDS + SCALAR_FIELDS
              + ["ck_" + f for f in CKPT_FIELDS]
              + ["frac_rhobar_lt1", "frac_rel_lt1", "frac_tight_le10",
                 "cert_violations", "cert_invalid", "svd_fallbacks"])


def build_cells(kept, cells, tasks, algos, seeds, ddof, require_steps):
    out = {}
    for cell in cells:
        for task in tasks:
            for algo in algos:
                for f in ALL_FIELDS:
                    out[(cell, task, algo, f)] = build_cell(
                        kept, cell, task, algo, f, seeds, ddof, require_steps)
    return out


# ---------------------------------------------------------------- statistics
def paired_vs_ref(cells, cell, task, field, algo, ref=REF_ALGO, boot=10000, boot_seed=0):
    """Paired comparison of `algo` against `ref` on the same seeds.  d_z + paired-t + CI."""
    rec = {"cell": cell, "task": task, "field": field, "algo": algo, "ref": ref,
           "status": "ok", "n": 0, "delta": None, "ci95": (None, None),
           "cohen_dz": None, "ttest_p": None, "note": "",
           "higher_better": higher_better(task, field)}
    if algo == ref:
        rec.update(status="is_reference")
        return rec
    if field == "grad_cos" and (algo in NO_COS or ref in NO_COS):
        rec.update(status="not_applicable",
                   note="grad_cos undefined for " + "/".join(sorted({algo, ref} & NO_COS)))
        return rec
    A = cells[(cell, task, algo, field)]["values"]
    B = cells[(cell, task, ref, field)]["values"]
    seeds = sorted(set(A) & set(B))
    rec["n"] = len(seeds)
    if len(seeds) < 2:
        rec.update(status="insufficient",
                   note="%d paired seed(s)" % len(seeds))
        return rec
    a = np.array([A[s] for s in seeds], dtype=float)
    b = np.array([B[s] for s in seeds], dtype=float)
    # positive delta = `algo` better than `ref` under the task's own direction
    delta = (a - b) if rec["higher_better"] else (b - a)
    sd = float(np.std(delta, ddof=1))
    rec["delta"] = float(np.mean(delta))
    rec["cohen_dz"] = float(np.mean(delta) / sd) if sd > 0 else None
    rec["ci95"] = t_ci95(delta)
    try:
        rec["ttest_p"] = float(stats.ttest_rel(a, b).pvalue)
    except Exception:                                                   # noqa: BLE001
        rec["ttest_p"] = float("nan")
    if sd == 0:
        rec["note"] = "zero paired spread: d_z undefined"
    return rec


def cross_cell_ratio(cells, task, algo, field, num="gru", den="tanh"):
    """Ratio of the across-seed means, num/den, with the per-cell n.  Seeds do NOT pair
    across cells (different grids), so this is a ratio of means, not a paired statistic."""
    cn = cells.get((num, task, algo, field))
    cd = cells.get((den, task, algo, field))
    if cn is None or cd is None or cn["mean"] is None or cd["mean"] is None:
        return None
    if cd["mean"] == 0:
        return None
    return {"ratio": cn["mean"] / cd["mean"], "num": cn["mean"], "den": cd["mean"],
            "n_num": cn["n"], "n_den": cd["n"]}


# ---------------------------------------------------------------- inventory
def job_outs(jobs_files):
    """-> {path: line_no} from every `# out=` annotation, the authoritative expectation."""
    outs = {}
    found = []
    for jf in jobs_files:
        p = rel(jf)
        if not os.path.isfile(p):
            continue
        found.append(jf)
        with io.open(p, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if line.startswith("#"):
                    continue
                k = line.rfind("# out=")
                if k < 0:
                    continue
                outs.setdefault(line[k + len("# out="):].strip(), (jf, i))
    return outs, found


def inventory(outs):
    present, missing = [], []
    for p in sorted(outs):
        (present if os.path.isfile(rel(p)) else missing).append(p)
    return present, missing


def extra_files(indir, outs):
    indir = rel(indir)
    if not os.path.isdir(indir):
        return []
    expected = {os.path.normcase(os.path.basename(p)) for p in outs}
    return [n for n in sorted(os.listdir(indir))
            if n.endswith(".json") and os.path.normcase(n) not in expected]


# ---------------------------------------------------------------- formatting
def fmt(x, nd=3):
    if x is None:
        return "--"
    if isinstance(x, float) and not math.isfinite(x):
        return "inf" if x > 0 else "-inf"
    if isinstance(x, float) and x != 0 and (abs(x) < 10 ** -(nd) or abs(x) >= 1e6):
        return "%.2e" % x
    return ("%." + str(nd) + "f") % x


def md_cell(c, nd=None, with_ci=True):
    """`mean +/- std [lo, hi]` with the cell's protocol flags appended."""
    if not c["defined"]:
        return "n/a"
    if c["mean"] is None:
        return "--"
    nd = ND.get(c["field"], 3) if nd is None else nd
    s = fmt(c["mean"], nd)
    if c["std"] is not None:
        s += " ± " + fmt(c["std"], nd)
    if with_ci and c["ci"][0] is not None:
        s += " [%s, %s]" % (fmt(c["ci"][0], nd), fmt(c["ci"][1], nd))
    if c["n"] != len(SEEDS):
        s += " (n=%d)" % c["n"]
    return s


def log10_summary(c):
    """Across-seed summary of a heavy-tailed positive field, in log10."""
    vals = [v for v in c["values"].values() if v is not None and math.isfinite(v) and v > 0]
    nz = [v for v in c["values"].values() if v == 0]
    if not vals:
        # every finite per-run value is exactly 0.  That is not "1e0": e_t == 0 occurs only
        # at the zero-signal diagnostic steps where true_E == 0 too (so the bound holds
        # trivially), and it happens on all three cells at the same rate -- it is a property
        # of the diagnostic schedule, not of the cell.  Label it so it cannot read as 10^0.
        return "0 (all finite = 0)" if nz else "--"
    lg = np.log10(vals)
    s = "1e%+.1f" % float(np.median(lg))
    if len(lg) > 1:
        s += " [1e%+.1f, 1e%+.1f]" % (float(lg.min()), float(lg.max()))
    if len(vals) != len(SEEDS):
        s += " (n=%d)" % len(vals)
    return s


def fmt_p(p):
    if p is None:
        return "--"
    if isinstance(p, float) and math.isnan(p):
        return "n/a"
    return "%.3f" % p if p >= 0.001 else "<0.001"


def fmt_dz(rec):
    if rec["status"] == "is_reference":
        return "ref"
    if rec["status"] == "not_applicable":
        return "n/a"
    if rec["cohen_dz"] is None:
        return "--"
    return "%+.2f" % rec["cohen_dz"]


def fmt_delta(rec, nd):
    if rec["status"] in ("is_reference", "not_applicable") or rec["delta"] is None:
        return {"is_reference": "ref", "not_applicable": "n/a"}.get(rec["status"], "--")
    s = "%+s" % fmt(rec["delta"], nd)
    if rec["ci95"][0] is not None:
        s += " [%s, %s]" % (fmt(rec["ci95"][0], nd), fmt(rec["ci95"][1], nd))
    return s


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


# ---------------------------------------------------------------- report
def write_report(path, meta, cells, kept, problems, inv, checks, present_cells):
    L = []
    A = L.append
    A("# D5 -- gated cells (GRU / LSTM): SK-RTRL applicability")
    A("")
    A("Generated %s by `make_r2_d5_report.py` (reads `%s` and `%s`; writes nothing else)."
      % (meta["when"], D5_DIR, TANH_DIR))
    A("")
    A("- aggregation: %s (window from step >= %.0f%% of %d)"
      % (meta["agg"], 100 * WINDOW_FRAC, STEPS))
    A("- across seeds: mean ± population std (ddof=%d), 95%% CI = Student-t on the seed mean "
      "(sample std, t_{.975,n-1}); seeds %s" % (meta["ddof"], SEEDS))
    A("- effect size: paired on the same seed, Cohen's d_z (+ = the row is better under the "
      "task's own direction) and a paired-t p. The task metric is paired against **%s**; "
      "`grad_cos` is paired against **%s**, because the cosine is not a defined quantity for "
      "exact RTRL." % (PLAIN[REF_ALGO], PLAIN[REF_COS]))
    A("- `e_t` is reported as a log10 median over the window (the bound spans >20 decades "
      "inside a single run; an arithmetic mean of it is meaningless)")
    A("- cells present in `%s`: %s" % (D5_DIR, ", ".join(present_cells) or "none"))
    A("")

    # ---------------------------------------------------------------- 0. verdict
    A("## 0. Headline")
    A("")
    for line in checks["headline"]:
        A("- " + line)
    A("")

    # ---------------------------------------------------------------- 1. inventory
    A("## 1. Inventory and anomalies")
    A("")
    A("Job files read: %s" % (", ".join("`%s`" % j for j in inv["jobs_found"]) or "none"))
    A("")
    A("- expected (`# out=` annotations): **%d**" % len(inv["outs"]))
    A("- present: **%d**" % len(inv["present"]))
    A("- missing: **%d**" % len(inv["missing"]))
    A("- unexpected extra json in `%s`: **%d**" % (D5_DIR, len(inv["extra"])))
    A("")
    if inv["missing"]:
        by_cell = defaultdict(list)
        for p in inv["missing"]:
            b = os.path.basename(p)
            by_cell["lstm" if "_lstm_" in b else ("gru" if "_gru_" in b else "?")].append(b)
        for cell in sorted(by_cell):
            A("Missing, cell `%s` (%d):" % (cell, len(by_cell[cell])))
            A("")
            A("```")
            for b in by_cell[cell]:
                A(b)
            A("```")
            A("")
    else:
        A("No missing runs against the job-file expectation.")
        A("")
    if inv["extra"]:
        A("Extra files (not in any `# out=` list): " + ", ".join("`%s`" % e
                                                                for e in inv["extra"]))
        A("")
    if problems:
        A("Unreadable / malformed / duplicate files:")
        A("")
        for p in problems:
            A("- `%s`: %s -- %s" % (p.get("file"), p.get("status"), p.get("error", "")))
        A("")
    else:
        A("Every json parsed cleanly.")
        A("")

    A("### NaN / divergence audit (raw record stream, not the window)")
    A("")
    rows = []
    for (cell, task, algo, seed), r in sorted(kept.items()):
        bad = []
        if r["n_nan_metric"]:
            bad.append("metric NaN/inf x%d" % r["n_nan_metric"])
        if r["n_nan_cos"]:
            bad.append("grad_cos NaN/inf x%d" % r["n_nan_cos"])
        if r["nonfinite"]:
            bad.append("window non-finite: " + ",".join(r["nonfinite"]))
        if r["cert_violations"]:
            bad.append("cert_violations=%s" % r["cert_violations"])
        if r["cert_invalid"]:
            bad.append("cert_invalid=%s" % r["cert_invalid"])
        if r["svd_fallbacks"]:
            bad.append("svd_fallbacks=%s" % r["svd_fallbacks"])
        if r["steps"] != STEPS:
            bad.append("steps=%s" % r["steps"])
        if bad:
            rows.append([cell, task, PLAIN.get(algo, algo), seed, "; ".join(bad)])
    if rows:
        L.extend(table(rows, ["cell", "task", "estimator", "seed", "anomaly"]))
    else:
        A("No NaN, no non-finite window value, no certificate violation, no certificate "
          "invalidation, no SVD fallback, correct step count -- in every run present.")
    A("")
    A("### Divergence screen")
    A("")
    A("Criterion: the last-20 % window mean against the best value the run reached **after "
      "step 0** -- accuracy tasks fire below 0.80x their own peak, error tasks above 2.0x "
      "their own best. (The step-0 record is degenerate on the accuracy tasks: an untrained "
      "net scores 1.000 by emitting the majority symbol, so a naive \"worse than step 0\" "
      "test fires on all 84 runs and on the tanh reference too, and says nothing.)")
    A("")
    if checks["divergence"]:
        L.extend(table(checks["divergence"],
                       ["cell", "task", "estimator", "seed", "step-0", "window", "note"]))
    else:
        A("No run in the gated grid walked away from its own best by that margin, and no run "
          "logged a non-finite task metric.")
    A("")
    A("Structurally absent estimators (absent by derivation, NOT missing data):")
    A("")
    for a, why in NOT_APPLICABLE.items():
        A("- **%s**: %s" % (PLAIN.get(a, a), why))
    A("")

    # ---------------------------------------------------------------- 2. main table
    for cell in present_cells:
        A("## 2%s. %s -- task metric and gradient fidelity"
          % ("" if cell == present_cells[0] else "'", CELL_LABEL.get(cell, cell)))
        A("")
        for task in TASKS:
            lrs = sorted({x for a in ALGOS
                          for x in cells[(cell, task, a, "metric")]["lrs"]})
            A("**%s** (metric = %s, %s is better; lr %s)"
              % (task, metric_name(task),
                 "higher" if task in HIGHER_IS_BETTER else "lower",
                 ", ".join("%g" % x for x in lrs) or "--"))
            A("")
            rows = []
            for algo in ORDER:
                cm = cells[(cell, task, algo, "metric")]
                cc = cells[(cell, task, algo, "grad_cos")]
                pm = paired_vs_ref(cells, cell, task, "metric", algo)
                pc = paired_vs_ref(cells, cell, task, "grad_cos", algo, ref=REF_COS)
                ce = cells[(cell, task, algo, "e_t")]
                cv = cells[(cell, task, algo, "cert_violations")]
                cp = cells[(cell, task, algo, "peak_MB")]
                cw = cells[(cell, task, algo, "wall_s")]
                viol = "--" if cv["mean"] is None else "%d" % int(round(cv["mean"] * cv["n"]))
                rows.append([
                    PLAIN.get(algo, algo),
                    md_cell(cm),
                    fmt_delta(pm, ND["metric"]),
                    fmt_dz(pm),
                    md_cell(cc),
                    fmt_dz(pc),
                    log10_summary(ce),
                    fmt(cells[(cell, task, algo, "e_t_inf_frac")]["mean"], 2),
                    viol,
                    md_cell(cp, with_ci=False),
                    md_cell(cw, with_ci=False),
                ])
            L.extend(table(rows, ["estimator", "%s (mean ± sd [95%% CI])" % metric_name(task),
                                  "Δ vs exact [95% CI]", "d_z", "grad_cos (mean ± sd [CI])",
                                  "d_z(cos vs SnAp-1)", "median e_t (log10, finite part)",
                                  "e_t +inf frac", "cert viol.", "peak MB", "wall s"]))
            A("")
        A("Paired-t p values against exact (metric / grad_cos), n=%d seeds:" % len(SEEDS))
        A("")
        rows = []
        for task in TASKS:
            for algo in ORDER:
                if algo == REF_ALGO:
                    continue
                pm = paired_vs_ref(cells, cell, task, "metric", algo)
                pc = paired_vs_ref(cells, cell, task, "grad_cos", algo, ref=REF_COS)
                rows.append([task, PLAIN.get(algo, algo), pm["n"], fmt_p(pm["ttest_p"]),
                             fmt_p(pc["ttest_p"]) if pc["status"] == "ok" else "n/a"])
        L.extend(table(rows, ["task", "estimator", "n", "p(metric vs exact)",
                              "p(grad_cos vs SnAp-1)"]))
        A("")
        A("> With 3 seeds the 95 % CI half-width is 4.30 s/sqrt(3) = 2.48 s and the paired-t "
          "has 2 df: read these intervals as a spread indicator, not as the 10-seed D4 "
          "inference. The D5 grid was sized for a structural applicability claim, not for a "
          "powered head-to-head.")
        A("")

    # ---------------------------------------------------------------- 3. certificate
    A("## 3. Certificate on a gated cell vs the tanh cell")
    A("")
    A("`cert_violations` = `cert_summary.n_bound_violation`: steps where the certified bound "
      "fell below the measured gradient error, i.e. Theorem 1 failed. `frac rho_bar<1` is the "
      "fraction of steps where the contraction condition of Corollary 1 held (the plateau "
      "regime). `frac rel<1` is the fraction where the bound was below the gradient norm, "
      "i.e. where the certificate was non-vacuous as a relative statement. `e_t +inf frac` is "
      "the fraction of window steps where the bound overflowed float64: the bound accumulates "
      "per-step factors rho_bar > 1 over a 20 000-step trajectory, so on the long-memory "
      "tasks it leaves the representable range. **This is a property of the bound, not of the "
      "cell** -- the tanh rows overflow at the same rate, which is why both cells are printed "
      "in the same table. An overflowed bound cannot be violated, so it is counted as "
      "vacuous-but-valid, never as a pass.")
    A("")
    for task in TASKS:
        A("**%s**" % task)
        A("")
        rows = []
        for algo in ORDER:
            if algo not in ("skrtrl-r4", "skrtrl-r16", "skrtrl-r32", "snap1"):
                continue
            for cell in ["tanh"] + present_cells:
                cv = cells.get((cell, task, algo, "cert_violations"))
                if cv is None or cv["n"] == 0:
                    continue
                ce = cells[(cell, task, algo, "e_t")]
                ct = cells[(cell, task, algo, "true_E")]
                crb = cells[(cell, task, algo, "frac_rhobar_lt1")]
                crl = cells[(cell, task, algo, "frac_rel_lt1")]
                ctt = cells[(cell, task, algo, "frac_tight_le10")]
                crho = cells[(cell, task, algo, "rho_bar")]
                rows.append([
                    PLAIN.get(algo, algo), CELL_LABEL.get(cell, cell), cv["n"],
                    "%d" % int(round((cv["mean"] or 0) * cv["n"])),
                    log10_summary(ce),
                    fmt(cells[(cell, task, algo, "e_t_inf_frac")]["mean"], 2),
                    fmt(ct["mean"], 3),
                    fmt(crho["mean"], 2),
                    fmt(crb["mean"], 3), fmt(crl["mean"], 3), fmt(ctt["mean"], 3),
                ])
        if rows:
            L.extend(table(rows, ["estimator", "cell", "n", "cert viol.",
                                  "median e_t (log10, finite part)", "e_t +inf frac",
                                  "mean true_E", "mean rho_bar",
                                  "frac rho_bar<1", "frac rel<1", "frac tight<=10"]))
        A("")

    # ---------------------------------------------------------------- 4. rank / spectrum
    A("## 4. Sketch spectrum: does absorbing I^perp_t raise the effective rank?")
    A("")
    A("Read at the LAST residual checkpoint (`frac`=1.0) of each run. `stable_rank` = "
      "(||E||_F/||E||_2)^2 of the residual, `sketch_eff` = ||E||_F normalised by the "
      "sketch's own factor norm, `mass_topk` = fraction of the residual's squared spectrum "
      "in its top k singular values (so a SMALLER mass_top4 means a flatter, harder "
      "spectrum), `best_rank_err` = the rank-r truncation error the sketch is chasing.")
    A("")
    for algo in ["skrtrl-r4", "skrtrl-r16", "skrtrl-r32"]:
        A("**%s**" % PLAIN[algo])
        A("")
        rows = []
        for task in TASKS:
            for cell in ["tanh"] + present_cells:
                c_sr = cells.get((cell, task, algo, "ck_stable_rank"))
                if c_sr is None or c_sr["n"] == 0:
                    continue
                rows.append([
                    task, CELL_LABEL.get(cell, cell), c_sr["n"],
                    md_cell(c_sr, with_ci=False),
                    md_cell(cells[(cell, task, algo, "ck_sketch_eff")], with_ci=False),
                    md_cell(cells[(cell, task, algo, "ck_mass_top4")], with_ci=False),
                    md_cell(cells[(cell, task, algo, "ck_mass_top8")], with_ci=False),
                    md_cell(cells[(cell, task, algo, "ck_mass_top16")], with_ci=False),
                    md_cell(cells[(cell, task, algo, "ck_mass_top32")], with_ci=False),
                    md_cell(cells[(cell, task, algo, "ck_best_rank_err")], with_ci=False),
                    md_cell(cells[(cell, task, algo, "ck_E_norm")], with_ci=False),
                ])
        if rows:
            L.extend(table(rows, ["task", "cell", "n", "stable_rank", "sketch_eff",
                                  "mass_top4", "mass_top8", "mass_top16", "mass_top32",
                                  "best_rank_err", "||E||_F"]))
        A("")
    A("### r -> fidelity, both cells side by side")
    A("")
    A("Mean window `grad_cos` at each r, and the gain from r=4 to r=32.")
    A("")
    rows = []
    for task in TASKS:
        for cell in ["tanh"] + present_cells:
            vals = {}
            for algo in ["skrtrl-r4", "skrtrl-r16", "skrtrl-r32", "snap1", "uoro", "rflo"]:
                c = cells.get((cell, task, algo, "grad_cos"))
                vals[algo] = c["mean"] if c else None
            if vals["skrtrl-r16"] is None:
                continue
            g = (vals["skrtrl-r32"] - vals["skrtrl-r4"]) \
                if (vals["skrtrl-r32"] is not None and vals["skrtrl-r4"] is not None) else None
            mono = "yes" if (vals["skrtrl-r4"] is not None and vals["skrtrl-r32"] is not None
                             and vals["skrtrl-r4"] <= vals["skrtrl-r16"] <= vals["skrtrl-r32"]) \
                else "no"
            rows.append([task, CELL_LABEL.get(cell, cell)]
                        + [fmt(vals[a], 3) for a in ["skrtrl-r4", "skrtrl-r16", "skrtrl-r32",
                                                     "snap1", "uoro", "rflo"]]
                        + [fmt(g, 3), mono])
    L.extend(table(rows, ["task", "cell", "r=4", "r=16", "r=32", "SnAp-1", "UORO", "RFLO",
                          "r32-r4", "monotone in r"]))
    A("")

    # ---------------------------------------------------------------- 4b. Table 11 feed
    A("### Table 11 feed: the two lower-block rows, verbatim")
    A("")
    A("`fraction rho^g_t<1` is `cert_summary.frac_rel_lt1` at SK-RTRL r=16 -- the online "
      "direction-certifiable criterion ||delta_t||_2 e_t / ||ghat_t||_2 < 1. "
      "`r_eps/n` (eps=0.1) is interpolated in k on the logged `mass_topk` grid "
      "k in {4,8,16,32} and divided by n; a value pinned at 0.062 = 4/64 means the top-4 "
      "mass already cleared 1-eps, so the true value sits at or below that bound.")
    A("")
    rows = []
    for cell in ["tanh"] + present_cells:
        crl = [cells[(cell, t, "skrtrl-r16", "frac_rel_lt1")] for t in TASKS]
        cre = [cells[(cell, t, "skrtrl-r16", "r_eps_over_n")] for t in TASKS]
        if all(c["mean"] is None for c in crl):
            continue
        rows.append([CELL_LABEL.get(cell, cell), "fraction rho^g_t<1 (r=16)"]
                    + [fmt(c["mean"], 4) for c in crl])
        rows.append([CELL_LABEL.get(cell, cell), "r_eps/n (eps=0.1, r=16 residual)"]
                    + [md_cell(c, with_ci=False) for c in cre])
    L.extend(table(rows, ["cell", "row"] + TASKS))
    A("")
    A("Same row at the other ranks, as a robustness check (the residual rank is a property "
      "of the task and the cell, so it should not move much with r):")
    A("")
    rows = []
    for algo in ["skrtrl-r4", "skrtrl-r16", "skrtrl-r32"]:
        for cell in ["tanh"] + present_cells:
            cre = [cells[(cell, t, algo, "r_eps_over_n")] for t in TASKS]
            if all(c["mean"] is None for c in cre):
                continue
            rows.append([PLAIN[algo], CELL_LABEL.get(cell, cell)]
                        + [md_cell(c, with_ci=False) for c in cre])
    L.extend(table(rows, ["estimator", "cell"] + TASKS))
    A("")

    # ---------------------------------------------------------------- 5. cost
    A("## 5. Cost of the gated cell relative to tanh")
    A("")
    A("Ratio of across-seed means (seeds do not pair across cells -- different grids -- so "
      "this is a ratio of means, not a paired statistic).")
    A("")
    for cell in present_cells:
        A("**%s / tanh**" % CELL_LABEL.get(cell, cell))
        A("")
        rows = []
        for task in TASKS:
            for algo in ORDER:
                rm = cross_cell_ratio(cells, task, algo, "peak_MB", cell, "tanh")
                rw = cross_cell_ratio(cells, task, algo, "wall_s", cell, "tanh")
                if rm is None and rw is None:
                    continue
                rows.append([
                    task, PLAIN.get(algo, algo),
                    fmt(rm["den"], 1) if rm else "--", fmt(rm["num"], 1) if rm else "--",
                    "%.2fx" % rm["ratio"] if rm else "--",
                    fmt(rw["den"], 0) if rw else "--", fmt(rw["num"], 0) if rw else "--",
                    "%.2fx" % rw["ratio"] if rw else "--",
                ])
        L.extend(table(rows, ["task", "estimator", "tanh peak MB", "%s peak MB" % cell,
                              "MB ratio", "tanh wall s", "%s wall s" % cell, "wall ratio"]))
        A("")
    A("> `wall_s` was measured on a shared GPU in both stages and the two stages were not "
      "contended alike, so the wall-clock ratio is an upper bound on the real per-step cost "
      "of the gated cell, not a clean measurement. `peak_MB` is allocator high-water mark and "
      "is not contention-sensitive.")
    A("")

    # ---------------------------------------------------------------- 6. confounds
    A("## 6. Protocol deviations and confounds")
    A("")
    for line in checks["confounds"]:
        A("- " + line)
    A("")
    flagged = [(k, c) for k, c in sorted(cells.items())
               if c["defined"] and c["flags"] and c["cell"] in present_cells
               and c["field"] == "metric"]
    if flagged:
        A("Cells with a protocol flag (metric field shown; the same seeds drive every field):")
        A("")
        for (cell, task, algo, _f), c in flagged:
            A("- `%s / %s / %s`: %s" % (cell, task, PLAIN.get(algo, algo),
                                         "; ".join(c["flags"])))
        A("")
    else:
        A("No cell carries a seed-count, step-count or non-finite flag.")
        A("")

    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    return L


# ---------------------------------------------------------------- checks
def run_checks(cells, kept, inv, present_cells, meta):
    head, conf, diverge = [], [], []

    # ---- divergence screen (d5 cells only; the tanh reference is D4's business)
    #
    # NOT "ends worse than step 0": on the accuracy tasks the step-0 record is a degenerate
    # 1.000 (the untrained net emits the majority symbol before any `b` is due), so every run
    # ever logged "ends worse than it starts" and the screen would fire 84/84.  The real
    # question is whether a run walked away from its own best, so the criterion is the
    # last-20 % window mean against the best value the run reached AFTER step 0:
    #   accuracy task   window mean < 0.80 * best        (lost a fifth of its peak accuracy)
    #   error task      window mean > 2.0 * best         (doubled its best error)
    for (cell, task, algo, seed), r in sorted(kept.items()):
        if cell not in present_cells:
            continue
        win, best = r.get("metric"), r.get("best_after0")
        if win is None or best is None or not math.isfinite(win) or not math.isfinite(best):
            continue
        hb = task in HIGHER_IS_BETTER
        if hb:
            bad, thr = win < 0.80 * best, "window %.3f < 0.80 x best %.3f" % (win, best)
        else:
            bad, thr = win > 2.0 * best, "window %.4g > 2.0 x best %.4g" % (win, best)
        note = []
        if bad:
            note.append(thr)
        if r.get("n_nan_metric"):
            note.append("%d non-finite metric record(s)" % r["n_nan_metric"])
        if note:
            diverge.append([cell, task, PLAIN.get(algo, algo), seed,
                            fmt(r.get("first_metric"), 4), fmt(win, 4), "; ".join(note)])

    # ---- headline
    for cell in present_cells:
        tot_viol, n_cert = 0, 0
        for task in TASKS:
            for algo in ALGOS:
                c = cells[(cell, task, algo, "cert_violations")]
                n_cert += c["n"]
                if c["mean"] is not None:
                    tot_viol += int(round(c["mean"] * c["n"]))
        n_runs = sum(1 for k in kept if k[0] == cell)
        head.append("**%s**: %d runs aggregated, of which %d carry a certificate summary "
                    "(the certificate is only defined for the sketched/SnAp estimators); "
                    "total certificate violations across every such run and every step: "
                    "**%d**." % (CELL_LABEL.get(cell, cell), n_runs, n_cert, tot_viol))
        # plateau applicability
        fr = []
        for task in TASKS:
            c = cells[(cell, task, "skrtrl-r16", "frac_rhobar_lt1")]
            if c["mean"] is not None:
                fr.append((task, c["mean"]))
        if fr:
            head.append("**%s**: fraction of steps with rho_bar<1 (Corollary 1 plateau "
                        "regime), SK-RTRL r=16: %s -- so the plateau corollary is %s."
                        % (CELL_LABEL.get(cell, cell),
                           ", ".join("%s %.3f" % t for t in fr),
                           "vacuous on this grid" if max(x for _, x in fr) < 0.01
                           else "partly applicable"))
    absent_cells = [c for c in CELLS if c not in present_cells]
    if absent_cells:
        head.append("Cells NOT run: %s -- the corresponding rows of Table 11 stay `\\PH{}` "
                    "and the paper must say the derivation is given but not run."
                    % ", ".join(absent_cells))
    n_kf = sum(1 for k in kept if k[0] in present_cells and k[2] == "kfrtrl")
    head.append("KF-RTRL is absent by derivation on every gated row (see section 1); "
                "KF-RTRL json files in `%s` for the cell(s) present: **%d** -- %s."
                % (D5_DIR, n_kf,
                   "confirmed absent" if n_kf == 0 else
                   "UNEXPECTED: the grid is not supposed to contain any"))
    if inv["missing"]:
        head.append("**%d expected run(s) missing** -- see section 1." % len(inv["missing"]))
    else:
        head.append("Inventory complete against the job files for the cell(s) present.")

    # ---- confounds
    lr_d5 = sorted({r["lr"] for k, r in kept.items() if r["cell"] in present_cells})
    lr_tanh = sorted({r["lr"] for k, r in kept.items() if r["cell"] == "tanh"})
    conf.append("learning rate: D5 used %s for every row (the job-file header says "
                "`D4_SELECT_stage2.json` was absent at generation time, so `--default-lr` "
                "applied); the tanh reference in `%s` used the D4 stage-2 selected rates %s. "
                "**Task-metric comparisons across cells are therefore not like-for-like** -- "
                "the certificate, spectrum and memory comparisons are the load-bearing ones."
                % (", ".join("%g" % x for x in lr_d5), TANH_DIR,
                   ", ".join("%g" % x for x in lr_tanh)))
    conf.append("seeds: D5 has %d seeds per cell, the tanh reference has up to 10. Every "
                "across-cell row prints its own n." % len(SEEDS))
    conf.append("`wall_s` came off a shared GPU in both stages; treat the ratio as an upper "
                "bound.")
    ns = sorted({r["n_hid"] for r in kept.values()})
    bs = sorted({r["batch"] for r in kept.values()})
    conf.append("width/batch: n in %s, batch in %s across both stages%s."
                % (ns, bs, " -- matched" if len(ns) == 1 and len(bs) == 1
                   else " -- **NOT matched, the cost ratio is confounded**"))
    sh = sorted({r["shadow"] for r in kept.values()})
    conf.append("shadow: %s (the exact-gradient reference that `grad_cos`, `true_E` and the "
                "certificate check need)." % sh)
    noshadow = [r["file"] for r in kept.values() if r.get("has_shadow") is False]
    if noshadow:
        conf.append("**%d run(s) report `has_shadow=false`**: their certificate columns are "
                    "unverified -- %s" % (len(noshadow), ", ".join(noshadow[:6])))
    return {"headline": head, "confounds": conf, "divergence": diverge}


# ---------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default=D5_DIR, help="gated-cell results")
    ap.add_argument("--tanhdir", default=TANH_DIR, help="tanh reference results (D4 eval)")
    ap.add_argument("--outdir", default=REPORT_DIR)
    ap.add_argument("--name", default="D5_SUMMARY.md")
    ap.add_argument("--agg", choices=["window20", "tail5"], default="window20")
    ap.add_argument("--frac", type=float, default=WINDOW_FRAC)
    ap.add_argument("--tail", type=int, default=5)
    ap.add_argument("--ddof", type=int, default=0)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--jobs", default=",".join(JOBS_FILES))
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any run is missing, unreadable, non-finite, or "
                         "violates the certificate")
    args = ap.parse_args(argv)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    globals()["SEEDS"] = seeds          # md_cell / log10_summary read it for the "(n=)" note

    d5_kept, d5_prob = scan_dir(args.indir, args.agg, args.frac, args.tail,
                                cells=set(CELLS), tasks=set(TASKS))
    tanh_kept, _ = scan_dir(args.tanhdir, args.agg, args.frac, args.tail,
                            cells={"tanh"}, tasks=set(TASKS))
    kept = dict(d5_kept)
    kept.update(tanh_kept)

    present_cells = [c for c in CELLS if any(k[0] == c for k in d5_kept)]
    tanh_seeds = sorted({k[3] for k in tanh_kept})

    cells = build_cells(kept, present_cells, TASKS, ALGOS, seeds, args.ddof, STEPS)
    cells.update(build_cells(kept, ["tanh"], TASKS, ALGOS, tanh_seeds, args.ddof, STEPS))

    outs, jobs_found = job_outs([j.strip() for j in args.jobs.split(",") if j.strip()])
    # only hold the cells we actually intended to run to account
    outs = {p: v for p, v in outs.items()
            if any(("_%s_" % c) in os.path.basename(p) for c in present_cells)} or outs
    present, missing = inventory(outs)
    inv = {"outs": outs, "present": present, "missing": missing, "jobs_found": jobs_found,
           "extra": extra_files(args.indir, outs)}

    meta = {"when": datetime.now().strftime("%Y-%m-%d %H:%M"), "agg": args.agg,
            "ddof": args.ddof, "frac": args.frac}
    checks = run_checks(cells, kept, inv, present_cells, meta)

    outdir = rel(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, args.name)
    write_report(path, meta, cells, d5_kept, d5_prob, inv, checks, present_cells)
    print("wrote %s" % path)
    print("  cells present: %s ; d5 runs: %d ; tanh reference runs: %d"
          % (", ".join(present_cells), len(d5_kept), len(tanh_kept)))
    print("  expected %d / present %d / missing %d / extra %d"
          % (len(outs), len(present), len(missing), len(inv["extra"])))

    bad = 0
    for line in checks["headline"]:
        print("  " + line.replace("**", ""))
    if args.strict:
        bad += len(missing) + len(d5_prob)
        for (cell, task, algo, seed), r in d5_kept.items():
            if r["cert_violations"] or r["n_nan_metric"] or r["nonfinite"]:
                bad += 1
        if bad:
            print("STRICT: %d problem(s)" % bad)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
