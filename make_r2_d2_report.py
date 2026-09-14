"""D2 (certificate informativeness, R3-2) aggregation + figures for the R2 revision.

Answers plan question D2 directly: *how informative is the certificate at each
spectral-clip level, and what does that clip cost the task?*

Definitions (R2_EXPERIMENT_PLAN.md D2 + section 4, renamed after the plan review)
---------------------------------------------------------------------------------
  rho^g_t = ||delta_t|| e_t / ||ghat_t||   < 1  =>  "direction-certifiable"
            (then g . ghat >= ||ghat||^2 (1 - rho^g) > 0; rho^g < 0.5 is the
            strengthened bin, which also gives ||g-ghat||/||g|| <= rho/(1-rho))
  T_t     = e_t / ||E_t||_F                <= 10 = the tightness bin
  rho_bar_t < 1                            = the step does not inflate e_t
Steps with no gradient signal (||delta_t|| = 0 or ||ghat_t|| = 0) are counted in
`n_no_signal` and are NEVER in the denominator of a fraction -- exactly the
counting rule of skrtrl.rl.CertCounters, which both runners share.

Aggregation rule (plan section 4, D2): *the seed is the unit*.  Every fraction is
computed inside a run by the runner (`cert_summary`), and this script only ever
averages those per-run fractions across seeds (mean +- s.d., n_seeds stated), then
across seeds within a task family.  Steps are never pooled across runs.

Inputs (all configurable; defaults follow the plan)
---------------------------------------------------
  results/r2/d4_eval     "no clip" level, taken from the D4 evaluation logs
                         (filtered to --d4-algo / --d4-clip so the level is one
                         estimator at one clip, not a mixture)
  results/r2/d2_clip     supervised clip sweep, clip in {0.35, 0.7, 0.9}
  results/r2/d2_noreset  washout = 0 with the exact shadow on
  results/r2/rl          RL (run_m5.py), clip 0.0
  results/r2/rl_clip05   RL, clip 0.5
The clip value is ALWAYS read from `args.clip` in the json, never guessed from a
file name.  run_m3 files are named <task>_<algo>_s<seed>[_tag].json and run_m5
files tmaze<len>_<algo>_s<seed>.json; the runner is detected from the args keys
(`env_len` = m5, `task` = m3), so a naming change cannot silently mislabel a run.

Outputs
-------
  results/r2/D2_SUMMARY.md   tables A-E (+ F when present) and auto-generated
                             numeric bullet points
  paper/figures/fig_r2_cert_frac_vs_clip.{pdf,png}
  paper/figures/fig_r2_cert_cdf.{pdf,png}
  paper/figures/fig_r2_cert_stage.{pdf,png}
Figure style, the >=2-visual-attributes guard and the legend-placement checks are
imported from paper/figures/gen/regen_all.py, so the D2 figures cannot drift from
the rest of the manuscript.

Missing data is never invented: every absent directory, unreadable file, run
without a certificate and missing `cert_summary` field is listed in the
"Data coverage" section at the top of the report, and a figure with no input is
skipped loudly.

Usage
-----
  D:/Anaconda/python.exe make_r2_d2_report.py                     # defaults
  D:/Anaconda/python.exe make_r2_d2_report.py --no-figs           # tables only
  D:/Anaconda/python.exe make_r2_d2_report.py --extra-dir tag=DIR # e.g. a smoke dir
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------- protocol constants
# The last-20%-window score and the metric direction are the D4 selection rule; they
# are imported from make_r2_d4_select.py when it is importable so the two cannot
# disagree, with an identical local fallback (that module asserts against
# make_r2_jobs.py and exits if the job-line format moved).
WINDOW_FRAC = 0.8
HIGHER_IS_BETTER = {"copy", "anbn"}          # run_m3 metric = accuracy for the ce tasks
_D4_SOURCE = "local fallback"


def _window_score_local(records, steps, frac):
    lo = frac * float(steps or 0)
    vals = [r.get("metric") for r in records
            if isinstance(r.get("step"), (int, float)) and r["step"] >= lo]
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None, 0, lo
    return sum(vals) / len(vals), len(vals), lo


window_score = _window_score_local
_D4_IMPORT_ERROR = None
try:                                          # single source of truth when available
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import make_r2_d4_select as _d4            # noqa: E402
    WINDOW_FRAC = float(_d4.WINDOW_FRAC)
    HIGHER_IS_BETTER = set(_d4.HIGHER_IS_BETTER)
    window_score = _d4.window_score
    _D4_SOURCE = "make_r2_d4_select.py"
except Exception as _e:                        # noqa: BLE001  (report, never fail)
    _D4_IMPORT_ERROR = "%s: %s" % (type(_e).__name__, _e)

FAMILIES = ["diagnostic", "chaotic", "real", "RL"]
FAMILY_OF_TASK = {
    "adding": "diagnostic", "copy": "diagnostic", "anbn": "diagnostic",
    "rotation": "diagnostic", "rotrecall24": "diagnostic",
    "henon": "chaotic", "mackeyglass": "chaotic", "lorenz": "chaotic",
    "sunspot": "real", "laser": "real",
}
FAMILY_LABEL = {"diagnostic": "diagnostic", "chaotic": "chaotic",
                "real": "real", "RL": "RL (T-maze)"}
STAGE_LABEL = ["early", "mid", "late"]

M3_FNAME = re.compile(r"^(?P<task>[a-z0-9]+)_(?P<algo>[a-z0-9\-_]+)_s(?P<seed>\d+)"
                      r"(?:_(?P<tag>.+))?\.json$")
M5_FNAME = re.compile(r"^tmaze(?P<len>\d+)_(?P<algo>[a-z0-9\-_]+)_s(?P<seed>\d+)"
                      r"(?:_(?P<tag>.+))?\.json$")

# cert_summary keys this report reads.  Everything outside REQUIRED is optional: it
# was added to skrtrl.rl.CertCounters during R2, so archived smoke runs lack it and
# the corresponding cell is reported as "n/a" instead of being filled with a guess.
REQUIRED_CERT = ("n_steps", "n_rel_lt1", "n_rel_lt05", "n_tight_checked",
                 "n_tight_le10", "n_bound_violation", "n_cert_invalid")
OPTIONAL_CERT = ("n_no_signal", "n_rhobar_lt1", "frac_rhobar_lt1",
                 "n_bound_violation_fp", "stages")
# Written by run_m3 only, by design -- absence is NOT a gap and is not reported as
# one (tables E and F say so instead).
M3_ONLY_CERT = ("n_bound_finite", "age_buckets", "age_edges", "age_span")

# Mid-run snapshots written by the local runner alongside the finished file; they
# carry the same (task, algo, seed, clip) key with fewer records and are never runs.
PARTIAL_SUFFIX = ".localpartial.json"


# ---------------------------------------------------------------- small helpers
def rel(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(HERE, path))


def isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def finite(v):
    return isnum(v) and math.isfinite(v)


def clip_key(clip):
    """Canonical label of a clip level.  0 / absent = 'none' (no spectral clip)."""
    if clip is None:
        return "unknown"
    if float(clip) <= 0:
        return "none"
    return "%g" % float(clip)


def clip_sort_key(key):
    if key == "none":
        return (0, 0.0)
    if key == "unknown":
        return (2, 0.0)
    return (1, float(key))


def fmt(v, nd=3):
    if v is None:
        return "n/a"
    if not isnum(v):
        return str(v)
    if not math.isfinite(v):
        return "inf" if v > 0 else "-inf"
    if abs(v) >= 1e5 or (v != 0 and abs(v) < 1e-4):
        return "%.2e" % v
    return ("%." + str(nd) + "f") % v


def agg(values, nd=3):
    """mean +- s.d. over the per-run values (the seed is the unit).

    Non-finite per-run values (an infinite certificate makes mean_rel_g / tightness
    infinite) are excluded from the moments and reported as a '+k inf' suffix, so an
    inf can never be silently averaged away nor silently swallow the whole cell.
    """
    vals = [v for v in values if isnum(v)]
    fin = [v for v in vals if math.isfinite(v)]
    n_inf = len(vals) - len(fin)
    if not fin:
        return ("%d inf" % n_inf) if n_inf else "n/a"
    m = statistics.fmean(fin)
    s = statistics.stdev(fin) if len(fin) > 1 else 0.0
    out = "%s+-%s" % (fmt(m, nd), fmt(s, nd))
    if n_inf:
        out += " +%dinf" % n_inf
    return out


def agg_mean(values):
    """Numeric (mean, sd, n) over the finite per-run values -- for the figures."""
    fin = [v for v in values if finite(v)]
    if not fin:
        return None, None, 0
    return (statistics.fmean(fin),
            statistics.stdev(fin) if len(fin) > 1 else 0.0,
            len(fin))


def quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    i = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (i - lo)


def md_table(head, rows):
    if not rows:
        return ["_no runs_", ""]
    w = [len(h) for h in head]
    srows = [[("" if c is None else str(c)) for c in r] for r in rows]
    for r in srows:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(c))

    def line(cells):
        return "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cells)) + " |"

    return ([line(head),
             "|" + "|".join("-" * (w[i] + 2) for i in range(len(head))) + "|"]
            + [line(r) for r in srows] + [""])


# ---------------------------------------------------------------- run loading
class Problems:
    def __init__(self):
        self.items = []

    def add(self, kind, what, why):
        self.items.append((kind, what, why))

    def by_kind(self):
        out = {}
        for kind, what, why in self.items:
            out.setdefault(kind, []).append((what, why))
        return out


def read_run(path, source_tag, problems):
    """Parse one run json into a flat record.  Returns None if it is unusable."""
    name = os.path.basename(path)
    where = "%s/%s" % (source_tag, name)
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as e:                                          # noqa: BLE001
        problems.add("unreadable", where, "%s: %s" % (type(e).__name__, e))
        return None
    if not isinstance(d, dict) or not isinstance(d.get("args"), dict):
        problems.add("unreadable", where, "top level is not an object with `args`")
        return None
    a = d["args"]
    records = d.get("records") or []

    # --- which runner wrote this?  args keys are authoritative and the file name is
    # only cross-checked (run_m3.py's job naming is still being edited elsewhere).
    if "env_len" in a:
        runner, task = "m5", "tmaze%s" % a.get("env_len")
        metric_key, higher = "success", True
        m = M5_FNAME.match(name)
    elif "task" in a:
        runner, task = "m3", a.get("task")
        metric_key, higher = "metric", (task in HIGHER_IS_BETTER)
        m = M3_FNAME.match(name)
    else:
        problems.add("unreadable", where,
                     "args carry neither `task` (run_m3) nor `env_len` (run_m5)")
        return None
    if m is None:
        problems.add("name_mismatch", where,
                     "file name does not parse as %s output" % runner)
    elif runner == "m3" and m["task"] != str(task):
        problems.add("name_mismatch", where,
                     "file name says task=%s, args say %s" % (m["task"], task))

    r = {
        "path": path, "file": name, "source": source_tag, "runner": runner,
        "task": task, "algo": a.get("algo"), "seed": a.get("seed"),
        "steps": a.get("steps"), "clip": a.get("clip"),
        "clip_key": clip_key(a.get("clip")),
        "washout": a.get("washout"), "shadow": a.get("shadow"),
        "tag": a.get("tag"), "lr": a.get("lr"),
        "family": "RL" if runner == "m5" else FAMILY_OF_TASK.get(task, "other"),
        "metric_key": metric_key, "higher_is_better": higher,
        "n_records": len(records), "missing": [],
    }
    if r["family"] == "other":
        problems.add("unknown_task", where, "task %r is in no family table" % task)
    if "clip" not in a:
        r["missing"].append("args.clip")
        problems.add("field_missing", where,
                     "args.clip absent -- clip level recorded as 'unknown'")

    # --- task performance: mean metric over the last (1-WINDOW_FRAC) of the run,
    # the D4 selection rule.  For RL the metric is the success rate (higher better).
    recs_m = [{"step": x.get("step"), "metric": x.get(metric_key)} for x in records]
    score, n_win, lo = window_score(recs_m, r["steps"], WINDOW_FRAC)
    r.update(score=score, n_window=n_win, window_lo=lo, score_fallback=False)
    if score is None and records:
        # Same fallback as make_r2_d4_select (`allow_fallback`): a run whose logging
        # stride left no record inside the window is scored on its LAST record, and
        # the cell is marked with `*` so the weaker number is never read as a
        # window average.
        last = records[-1].get(metric_key)
        if isnum(last):
            r.update(score=float(last), score_fallback=True)
        problems.add("window_empty", where,
                     "no record with step >= %g carries %r (%s)"
                     % (lo, metric_key,
                        "scored on the last record instead" if r["score_fallback"]
                        else "no score"))

    # --- certificate summary
    cs = d.get("cert_summary")
    if not isinstance(cs, dict):
        r["cert"] = None
        r["no_cert_reason"] = ("cert_summary is null (estimator exposes no e_t, or "
                               "--shadow 0)" if cs is None
                               else "cert_summary is not an object")
        problems.add("no_certificate", where, r["no_cert_reason"])
        return r
    r["cert"] = cs
    for k in REQUIRED_CERT:
        if not isnum(cs.get(k)):
            r["missing"].append("cert_summary.%s" % k)
    for k in OPTIONAL_CERT:
        if k not in cs:
            r["missing"].append("cert_summary.%s (optional)" % k)
    if runner == "m3":
        for k in M3_ONLY_CERT:
            if k not in cs:
                r["missing"].append("cert_summary.%s (run_m3 only)" % k)
    if r["missing"]:
        problems.add("field_missing", where, ", ".join(r["missing"]))

    ns = cs.get("n_steps")
    nns = cs.get("n_no_signal")
    r["n_steps"] = ns if isnum(ns) else None
    r["n_no_signal"] = nns if isnum(nns) else None
    r["frac_no_signal"] = ((nns / (ns + nns))
                           if (isnum(ns) and isnum(nns) and ns + nns > 0) else None)
    # frac_* are recomputed from the counters ONLY when the runner did not write them
    # (archived smoke runs predate frac_rhobar_lt1); the runner's value always wins.
    for frac, num in (("frac_rel_lt1", "n_rel_lt1"), ("frac_rel_lt05", "n_rel_lt05")):
        v = cs.get(frac)
        if not isnum(v) and isnum(cs.get(num)) and isnum(ns) and ns > 0:
            v = cs[num] / ns
        r[frac] = v if isnum(v) else None
    nt = cs.get("n_tight_checked")
    v = cs.get("frac_tight_le10")
    if not isnum(v) and isnum(cs.get("n_tight_le10")) and isnum(nt) and nt > 0:
        v = cs["n_tight_le10"] / nt
    r["frac_tight_le10"] = v if isnum(v) else None
    v = cs.get("frac_rhobar_lt1")
    if not isnum(v) and isnum(cs.get("n_rhobar_lt1")) and isnum(ns) and ns > 0:
        v = cs["n_rhobar_lt1"] / ns
    r["frac_rhobar_lt1"] = v if isnum(v) else None
    r["n_tight_checked"] = nt if isnum(nt) else None
    for k in ("mean_tightness", "mean_rel_g", "max_tightness", "max_rel_g"):
        r[k] = cs[k] if isnum(cs.get(k)) else None
    for k in ("n_bound_violation", "n_bound_violation_fp", "n_cert_invalid",
              "n_bound_finite"):
        r[k] = cs[k] if isnum(cs.get(k)) else None

    # --- three-stage buckets
    stages = cs.get("stages")
    r["stages"] = None
    if isinstance(stages, list) and stages:
        st = []
        for s in stages:
            if not isinstance(s, dict):
                st = None
                break
            sns, snt = s.get("n_steps"), s.get("n_tight_checked")
            f1 = s.get("frac_rel_lt1")
            if not isnum(f1) and isnum(s.get("n_rel_lt1")) and isnum(sns) and sns > 0:
                f1 = s["n_rel_lt1"] / sns
            f2 = s.get("frac_tight_le10")
            if not isnum(f2) and isnum(s.get("n_tight_le10")) and isnum(snt) and snt > 0:
                f2 = s["n_tight_le10"] / snt
            st.append({"n_steps": sns if isnum(sns) else None,
                       "frac_rel_lt1": f1 if isnum(f1) else None,
                       "frac_tight_le10": f2 if isnum(f2) else None,
                       "step_range": s.get("step_range")})
        if st is not None and len(st) == len(STAGE_LABEL):
            r["stages"] = st
        elif st is not None:
            problems.add("stage_shape", where,
                         "cert_summary.stages has %d buckets, expected %d"
                         % (len(st), len(STAGE_LABEL)))
        else:
            problems.add("stage_shape", where, "cert_summary.stages holds a non-object")
    elif "stages" in cs:
        problems.add("stage_shape", where, "cert_summary.stages is not a non-empty list")

    # --- age-since-reset buckets (plan section 4, D2: "stratify ... by age since
    # reset").  run_m3 writes them; run_m5 does not (RL lanes reset asynchronously,
    # so its counters run at every step and there is no age span to bucket by).
    ab = cs.get("age_buckets")
    r["age_buckets"], r["age_edges"] = None, cs.get("age_edges")
    if isinstance(ab, list) and ab and all(isinstance(s, dict) for s in ab):
        got = []
        for s in ab:
            sns, snt = s.get("n_steps"), s.get("n_tight_checked")
            # An EMPTY bin has no fraction.  The runner still writes
            # frac_rel_lt1 = 0.0 when n_steps = 0 (tasks such as `adding` /
            # `copy` emit a target only at the end of the episode, so the two
            # young-age bins hold no counted step at all), and averaging that
            # 0.0 in would report "not certifiable" for a bin that was never
            # measured.  A zero denominator is n/a here, exactly as everywhere
            # else in this script.
            f1 = s.get("frac_rel_lt1")
            if isnum(sns) and sns <= 0:
                f1 = None
            elif not isnum(f1) and isnum(s.get("n_rel_lt1")) and isnum(sns) and sns > 0:
                f1 = s["n_rel_lt1"] / sns
            f2 = s.get("frac_tight_le10")
            if isnum(snt) and snt <= 0:
                f2 = None
            elif not isnum(f2) and isnum(s.get("n_tight_le10")) and isnum(snt) and snt > 0:
                f2 = s["n_tight_le10"] / snt
            got.append({"n_steps": sns if isnum(sns) else None,
                        "frac_rel_lt1": f1 if isnum(f1) else None,
                        "frac_tight_le10": f2 if isnum(f2) else None})
        r["age_buckets"] = got
    elif "age_buckets" in cs:
        problems.add("stage_shape", where,
                     "cert_summary.age_buckets is not a non-empty list of objects")

    # --- per-record series for the CDF and the Theorem-1 ratio.
    # run_m5 writes rho_g / tight per record; run_m3 does not, so they are rebuilt
    # here from the logged norms with exactly the same definitions.
    rho, tig, ratio = [], [], []
    for x in records:
        rg = x.get("rho_g")
        if not isnum(rg):
            b, gh = x.get("bound_norm"), x.get("ghat_norm")
            rg = (b / gh) if isnum(b) and isnum(gh) and gh > 0 else None
        if isnum(rg):
            rho.append(float(rg))
        tt = x.get("tight")
        if not isnum(tt):
            e, te = x.get("e_t"), x.get("true_E")
            tt = (e / te) if isnum(e) and isnum(te) and te > 1e-12 else None
        if isnum(tt):
            tig.append(float(tt))
        ge, b = x.get("gerr_norm"), x.get("bound_norm")
        if isnum(ge) and isnum(b) and b > 0:
            ratio.append(float(ge) / float(b))
    r["rho_g_series"] = rho
    r["tight_series"] = tig
    r["ratio_series"] = ratio           # ||g-ghat|| / (||delta|| e_t); Theorem 1 => <= 1
    if not rho:
        problems.add("no_series", where,
                     "no record carries rho_g (nor bound_norm/ghat_norm) -- this run "
                     "contributes nothing to the CDF figure")
    return r


def scan_dirs(dirs, problems, d4_algo=None, d4_clip=None):
    """dirs: [(tag, path, role)].  role 'none' = the no-clip level from the D4 logs."""
    runs = []
    for tag, path, role in dirs:
        p = rel(path)
        if not os.path.isdir(p):
            problems.add("dir_missing", "%s (%s)" % (path, tag),
                         "directory does not exist")
            continue
        files = sorted(glob.glob(os.path.join(p, "*.json")))
        # `*.localpartial.json` is a mid-run snapshot the local runner leaves next to
        # the finished file (same task/algo/seed, fewer records).  It is NOT a run and
        # would double-count a seed, so it is dropped here and only reported.
        partial = [f for f in files if f.endswith(PARTIAL_SUFFIX)]
        files = [f for f in files if not f.endswith(PARTIAL_SUFFIX)]
        for f in partial:
            problems.add("partial_snapshot", "%s/%s" % (tag, os.path.basename(f)),
                         "mid-run snapshot (*%s) -- skipped, the finished run is used"
                         % PARTIAL_SUFFIX)
        if not files:
            problems.add("dir_empty", "%s (%s)" % (path, tag),
                         "no *.json in the directory")
            continue
        kept = 0
        for f in files:
            r = read_run(f, tag, problems)
            if r is None:
                continue
            if role == "none":
                # The "no clip" D2 level is ONE estimator at ONE clip taken out of the
                # D4 evaluation campaign; everything else in that directory belongs to
                # D4 and is not a D2 clip level.
                if d4_algo and r["algo"] != d4_algo:
                    continue
                if d4_clip is not None and clip_key(r["clip"]) != clip_key(d4_clip):
                    continue
            r["role"] = role
            r["noreset"] = (tag == "d2_noreset") or (r["washout"] == 0)
            runs.append(r)
            kept += 1
        if kept == 0:
            problems.add("dir_empty", "%s (%s)" % (path, tag),
                         "%d json file(s), none selected%s"
                         % (len(files),
                            (" (filter algo=%s clip=%s)" % (d4_algo, d4_clip))
                            if role == "none" else ""))
    return runs


# ---------------------------------------------------------------- grouping
def group(runs, keyfn):
    out = {}
    for r in runs:
        out.setdefault(keyfn(r), []).append(r)
    return out


def cert_runs(runs):
    return [r for r in runs if r.get("cert")]


def perf_cell(rs):
    """mean +- s.d. of the last-20% window score, with the direction marked."""
    use = [r for r in rs if finite(r.get("score"))]
    if not use:
        return "n/a"
    arrow = "^" if rs[0]["higher_is_better"] else "v"
    star = "*" if any(r.get("score_fallback") for r in use) else ""
    return "%s %s%s" % (agg([r["score"] for r in use], 4), arrow, star)


# ---------------------------------------------------------------- report sections
def section_coverage(runs, problems, dirs, args):
    L = ["## Data coverage and gaps", "",
         "Report generated from %d run file(s).  Nothing below is imputed: an absent "
         "directory, an unreadable file, a run without a certificate and every missing "
         "`cert_summary` field is listed here, and the corresponding table cell reads "
         "`n/a`." % len(runs), ""]
    rows = []
    by_src = group(runs, lambda r: r["source"])
    for tag, path, role in dirs:
        rs = by_src.get(tag, [])
        rows.append([tag, path, "yes" if os.path.isdir(rel(path)) else "MISSING",
                     len(rs), len(cert_runs(rs)),
                     ",".join(sorted({r["clip_key"] for r in rs}, key=clip_sort_key))
                     or "-",
                     ",".join(sorted({r["family"] for r in rs})) or "-",
                     len({(r["task"], r["algo"], r["seed"], r["clip_key"]) for r in rs})])
    L += md_table(["source", "path", "exists", "runs", "with cert", "clip level(s)",
                   "family(ies)", "distinct (task,algo,seed,clip)"], rows)

    kinds = problems.by_kind()
    if not kinds:
        L += ["No problems found: every file parsed and every run carries a certificate "
              "summary with all required counters.", ""]
        return L
    order = ["dir_missing", "dir_empty", "unreadable", "no_certificate", "field_missing",
             "stage_shape", "window_empty", "no_series", "name_mismatch",
             "unknown_task", "figures"]
    titles = {
        "dir_missing": "Input directories that do not exist (data not produced yet)",
        "dir_empty": "Input directories with no usable run",
        "unreadable": "Files that could not be parsed",
        "no_certificate": "Runs without a certificate summary (excluded from A-E)",
        "field_missing": "Runs with missing `cert_summary` fields (cell -> n/a)",
        "stage_shape": "Runs whose stage buckets have an unexpected shape (table C)",
        "window_empty": "Runs with no record inside the last-20% performance window",
        "no_series": "Runs contributing no per-record rho^g (figure 2 only)",
        "name_mismatch": "File names that do not match the runner that wrote them",
        "unknown_task": "Tasks not in the family table",
        "partial_snapshot": "Mid-run snapshots skipped (not runs)",
        "figures": "Figure-side problems",
    }
    for k in order + [k for k in kinds if k not in order]:
        items = kinds.get(k)
        if not items:
            continue
        L += ["### %s (%d)" % (titles.get(k, k), len(items)), ""]
        shown = items[: args.max_problems]
        for what, why in shown:
            L.append("- `%s`: %s" % (what, why))
        if len(items) > len(shown):
            L.append("- ... and %d more (raise `--max-problems`)"
                     % (len(items) - len(shown)))
        L.append("")
    return L


def section_a(runs):
    L = ["## Table A -- certificate informativeness and task performance, per task x clip",
         "",
         "The direct answer to D2: one row per (task, clip) cell.  Every fraction is a "
         "per-run fraction averaged over seeds (mean +- s.d.).  `perf` is the mean "
         "`metric` (run_m3) / `success` (run_m5) over the last %d%% of the run -- the D4 "
         "selection window -- with `^` = higher is better and `v` = lower is better.  "
         "`no-sig` is `n_no_signal / (n_steps + n_no_signal)`: those steps sit outside "
         "every other denominator.  A `*` on `perf` means that run logged nothing "
         "inside the window and was scored on its last record instead.  The estimator "
         "is part of the row key: the certificate depends on it, so SK-RTRL and the "
         "SnAp-1 reference are never averaged into one cell."
         % round((1 - WINDOW_FRAC) * 100), ""]
    rs_all = cert_runs(runs)
    if not rs_all:
        return L + ["_no run with a certificate summary_", ""]
    g = group(rs_all, lambda r: (r["family"], r["task"], str(r["algo"]), r["clip_key"]))
    rows = []
    for fam in FAMILIES + ["other"]:
        for k in sorted([k for k in g if k[0] == fam],
                        key=lambda k: (k[1], k[2], clip_sort_key(k[3]))):
            rs = g[k]
            rows.append([
                fam, k[1], k[3], k[2],
                len({r["seed"] for r in rs}),
                agg([r["frac_rel_lt1"] for r in rs]),
                agg([r["frac_rel_lt05"] for r in rs]),
                agg([r["frac_tight_le10"] for r in rs]),
                agg([r["frac_rhobar_lt1"] for r in rs]),
                agg([r["mean_rel_g"] for r in rs], 2),
                agg([r["mean_tightness"] for r in rs], 2),
                agg([r["frac_no_signal"] for r in rs], 4),
                perf_cell(rs),
            ])
    L += md_table(["family", "task", "clip", "algo", "n_seeds", "frac_rel_lt1",
                   "frac_rel_lt05", "frac_tight_le10", "frac_rhobar_lt1",
                   "mean_rel_g", "mean_tightness", "no-sig", "perf (last 20%)"], rows)
    return L


def section_b(runs):
    L = ["## Table B -- per task family (the seed is the unit)", "",
         "The same per-run fractions as table A, averaged over every seed of every task "
         "in the family, the estimator kept in the key.  `n_runs` is the number of "
         "(task, seed) runs behind the cell and `n_seeds` the number of distinct "
         "seeds.", ""]
    rs_all = cert_runs(runs)
    if not rs_all:
        return L + ["_no run with a certificate summary_", ""]
    g = group(rs_all, lambda r: (r["family"], str(r["algo"]), r["clip_key"]))
    rows = []
    for fam in FAMILIES + ["other"]:
        for k in sorted([k for k in g if k[0] == fam],
                        key=lambda k: (k[1], clip_sort_key(k[2]))):
            rs = g[k]
            rows.append([FAMILY_LABEL.get(fam, fam), k[2], k[1], len(rs),
                         len({r["seed"] for r in rs}), len({r["task"] for r in rs}),
                         agg([r["frac_rel_lt1"] for r in rs]),
                         agg([r["frac_rel_lt05"] for r in rs]),
                         agg([r["frac_tight_le10"] for r in rs]),
                         agg([r["frac_rhobar_lt1"] for r in rs]),
                         agg([r["mean_rel_g"] for r in rs], 2),
                         agg([r["mean_tightness"] for r in rs], 2)])
    L += md_table(["family", "clip", "algo", "n_runs", "n_seeds", "n_tasks",
                   "frac_rel_lt1", "frac_rel_lt05", "frac_tight_le10",
                   "frac_rhobar_lt1", "mean_rel_g", "mean_tightness"], rows)
    return L


def section_c(runs):
    L = ["## Table C -- three training stages (early / mid / late)", "",
         "`cert_summary.stages` buckets the same counters into equal thirds of the run, "
         "so a certificate that is informative only at the start is not hidden by the "
         "run average.  Runs written before the stage buckets existed are listed in the "
         "coverage section and excluded here.", ""]
    rs_all = [r for r in cert_runs(runs) if r.get("stages")]
    if not rs_all:
        return L + ["_no run carries `cert_summary.stages`_", ""]
    g = group(rs_all, lambda r: (r["family"], str(r["algo"]), r["clip_key"]))
    rows = []
    for fam in FAMILIES + ["other"]:
        for k in sorted([k for k in g if k[0] == fam],
                        key=lambda k: (k[1], clip_sort_key(k[2]))):
            rs = g[k]
            row = [FAMILY_LABEL.get(fam, fam), k[2], k[1], len(rs)]
            for i in range(len(STAGE_LABEL)):
                row.append(agg([r["stages"][i]["frac_rel_lt1"] for r in rs]))
            for i in range(len(STAGE_LABEL)):
                row.append(agg([r["stages"][i]["frac_tight_le10"] for r in rs]))
            rows.append(row)
    head = (["family", "clip", "algo", "n_runs"]
            + ["rel_lt1 %s" % s for s in STAGE_LABEL]
            + ["tight_le10 %s" % s for s in STAGE_LABEL])
    L += md_table(head, rows)
    return L


def section_d(runs):
    L = ["## Table D -- no-reset regime (washout = 0, exact shadow on)", "",
         "Validity, tightness and informativeness in the uninterrupted stream, next to "
         "the reset-bearing control at the same (task, clip) from the other input "
         "directories.  A `reset control` row with `n_seeds = 0` means no control run "
         "for that cell exists yet.", ""]
    nr = [r for r in cert_runs(runs) if r.get("noreset")]
    if not nr:
        return L + ["_no washout=0 run found_", ""]
    keyfn = lambda r: (r["task"], str(r["algo"]), r["clip_key"])      # noqa: E731
    ctl = group([r for r in cert_runs(runs) if not r.get("noreset")], keyfn)
    rows = []
    for k, rs in sorted(group(nr, keyfn).items(),
                        key=lambda kv: (kv[0][0], kv[0][1], clip_sort_key(kv[0][2]))):
        for label, grs in (("washout=0", rs), ("reset control", ctl.get(k, []))):
            if not grs:
                rows.append([k[0], k[2], k[1], label, 0] + ["n/a"] * 6)
                continue
            rows.append([k[0], k[2], k[1], label, len({r["seed"] for r in grs}),
                         agg([r["frac_rel_lt1"] for r in grs]),
                         agg([r["frac_tight_le10"] for r in grs]),
                         agg([r["mean_tightness"] for r in grs], 2),
                         sum(r["n_cert_invalid"] or 0 for r in grs),
                         sum(r["n_bound_violation"] or 0 for r in grs),
                         perf_cell(grs)])
    L += md_table(["task", "clip", "algo", "regime", "n_seeds", "frac_rel_lt1",
                   "frac_tight_le10", "mean_tightness", "n_cert_invalid",
                   "n_bound_violation", "perf (last 20%)"], rows)
    return L


def section_e(runs):
    L = ["## Table E -- direct verification of Theorem 1 and Lemma 1", "",
         "`n_bound_violation` counts steps with "
         "`||g - ghat||_F > (||delta|| e)(1 + 1e-4)` (Theorem 1), `_fp` the same test "
         "with an additional absolute fp32 floor, and `n_cert_invalid` steps with "
         "`||E||_F > e (1 + 1e-4)` (Lemma 1).  All three must be 0.  `n_bound_finite` "
         "counts the signal-bearing steps whose bound is finite, so a `0 violations` "
         "cannot be read off an infinite certificate (run_m3 writes it; run_m5 does "
         "not, hence the `n/a` on the RL row).", ""]
    rs_all = cert_runs(runs)
    if not rs_all:
        return L + ["_no run with a certificate summary_", ""]
    g = group(rs_all, lambda r: r["family"])
    rows = []
    tot = dict(runs=0, n_steps=0, viol=0, violfp=0, inval=0, fin=0)
    for fam in FAMILIES + ["other"]:
        rs = g.get(fam)
        if not rs:
            continue
        cells = []
        for key, acc in (("n_bound_violation", "viol"),
                         ("n_bound_violation_fp", "violfp"),
                         ("n_cert_invalid", "inval"),
                         ("n_bound_finite", "fin")):
            have = [r for r in rs if r[key] is not None]
            s = sum(r[key] for r in have)
            cells.append("%d (%d runs)" % (s, len(have)) if have else "n/a")
            tot[acc] += s
        n_steps = sum(r["n_steps"] or 0 for r in rs)
        rows.append([FAMILY_LABEL.get(fam, fam), len(rs), n_steps] + cells)
        tot["runs"] += len(rs)
        tot["n_steps"] += n_steps
    rows.append(["**all**", tot["runs"], tot["n_steps"], tot["viol"], tot["violfp"],
                 tot["inval"], tot["fin"]])
    L += md_table(["family", "n_runs", "counted steps", "n_bound_violation",
                   "n_bound_violation_fp", "n_cert_invalid", "n_bound_finite"], rows)

    L += ["### Distribution of `||g - ghat||_F / (||delta_t|| e_t)`", "",
          "The Theorem-1 slack itself, read off the logged records (therefore sampled "
          "every `--log_every` steps, not at every step as in the counters above).  "
          "Theorem 1 says the ratio is <= 1; a very small ratio means the bound holds "
          "with room to spare and is loose at that step.", ""]
    rows = []
    for fam in FAMILIES + ["other", "all"]:
        vals = sorted(v for r in rs_all if fam in (r["family"], "all")
                      for v in r["ratio_series"] if finite(v))
        if not vals:
            continue
        rows.append([FAMILY_LABEL.get(fam, fam), len(vals),
                     fmt(quantile(vals, 0.5)), fmt(quantile(vals, 0.9)),
                     fmt(quantile(vals, 0.99)), fmt(vals[-1]),
                     sum(1 for v in vals if v > 1.0)])
    L += md_table(["family", "n log points", "median", "p90", "p99", "max", "n > 1"],
                  rows)

    L += ["### Distribution of the two D2 quantities over the logged records", "",
          "The same log points seen as distributions, so the bins of tables A-C can be "
          "read against the shape they come from (the fractions themselves are counted "
          "at every step by the runner, these quantiles only at the log points).  "
          "`T_t < 1` would mean `e_t` under-states `||E_t||_F`, i.e. an invalid "
          "certificate, and must be empty.", ""]
    rows = []
    for series, label, extra in (("rho_g_series", "rho^g_t", "n < 1"),
                                 ("tight_series", "T_t", "n <= 10")):
        for fam in FAMILIES + ["other", "all"]:
            vals = sorted(v for r in rs_all if fam in (r["family"], "all")
                          for v in r[series] if finite(v))
            if not vals:
                continue
            n_hit = (sum(1 for v in vals if v < 1.0) if series == "rho_g_series"
                     else sum(1 for v in vals if v <= 10.0))
            rows.append([label, FAMILY_LABEL.get(fam, fam), len(vals),
                         fmt(quantile(vals, 0.1)), fmt(quantile(vals, 0.5)),
                         fmt(quantile(vals, 0.9)), fmt(vals[-1]),
                         "%s = %d" % (extra, n_hit)])
    L += md_table(["quantity", "family", "n log points", "p10", "median", "p90",
                   "max", "in the bin"], rows)
    return L


def section_f(runs):
    """Optional appendix table: age since the last reset (plan section 4, D2)."""
    rs_all = [r for r in cert_runs(runs) if r.get("age_buckets")]
    L = ["## Table F -- age since the last reset (run_m3 only)", "",
         "`cert_summary.age_buckets` splits the same counters by how many steps have "
         "passed since the last episode / washout reset, with edges "
         "`[0, T/4), [T/4, T/2), [T/2, inf)` and `T` the reset period.  It separates "
         "the post-reset transient from the deep-history regime instead of averaging "
         "them.  run_m5 does not write it: RL lanes reset asynchronously, so its "
         "counters already run at every step with no age span to bucket by.", ""]
    if not rs_all:
        return L + ["_no run carries `cert_summary.age_buckets`_", ""]
    nb = max(len(r["age_buckets"]) for r in rs_all)
    g = group(rs_all, lambda r: (r["family"], str(r["algo"]), r["clip_key"]))
    rows = []
    for fam in FAMILIES + ["other"]:
        for k in sorted([k for k in g if k[0] == fam],
                        key=lambda k: (k[1], clip_sort_key(k[2]))):
            rs = [r for r in g[k] if len(r["age_buckets"]) == nb]
            if not rs:
                continue
            edges = sorted({str(r["age_edges"]) for r in rs})
            row = [FAMILY_LABEL.get(fam, fam), k[2], k[1], len(rs), ",".join(edges)]
            for field in ("frac_rel_lt1", "frac_tight_le10"):
                for i in range(nb):
                    row.append(agg([r["age_buckets"][i][field] for r in rs]))
            rows.append(row)
    head = (["family", "clip", "algo", "n_runs", "age edges"]
            + ["rel_lt1 bin%d" % i for i in range(nb)]
            + ["tight_le10 bin%d" % i for i in range(nb)])
    L += md_table(head, rows)
    return L


def section_bullets(runs, figs):
    """Auto-generated bullet points.  Numbers only -- no interpretation."""
    L = ["## Automatic numeric summary", ""]
    rs_all = cert_runs(runs)
    if not rs_all:
        return L + ["- No run carries a certificate summary, so no number can be "
                    "stated.", ""]
    L.append("- %d run(s) with a certificate summary over %d task(s), %d clip level(s) "
             "{%s} and %d distinct seed(s); %d counted step(s) in total."
             % (len(rs_all), len({r["task"] for r in rs_all}),
                len({r["clip_key"] for r in rs_all}),
                ", ".join(sorted({r["clip_key"] for r in rs_all}, key=clip_sort_key)),
                len({r["seed"] for r in rs_all}),
                sum(r["n_steps"] or 0 for r in rs_all)))
    by_clip = group(rs_all, lambda r: r["clip_key"])
    for ck, rs in sorted(by_clip.items(), key=lambda kv: clip_sort_key(kv[0])):
        cells = []
        for field in ("frac_rel_lt1", "frac_tight_le10", "frac_rhobar_lt1"):
            m, s, n = agg_mean([r[field] for r in rs])
            cells.append("%s = %s (n=%d)"
                         % (field, "%s+-%s" % (fmt(m), fmt(s)) if n else "n/a", n))
        L.append("- clip %s: %s." % (ck, ", ".join(cells)))
    per_clip = {ck: agg_mean([r["frac_rel_lt1"] for r in rs])[0]
                for ck, rs in by_clip.items()}
    per_clip = {k: v for k, v in per_clip.items() if v is not None}
    if len(per_clip) > 1:
        hi = max(per_clip, key=per_clip.get)
        lo = min(per_clip, key=per_clip.get)
        L.append("- highest mean frac_rel_lt1 at clip %s (%s), lowest at clip %s (%s)."
                 % (hi, fmt(per_clip[hi]), lo, fmt(per_clip[lo])))
    hv = [r for r in rs_all if r["n_bound_violation"] is not None]
    hi_ = [r for r in rs_all if r["n_cert_invalid"] is not None]
    L.append("- Theorem-1 violations: %s; Lemma-1 invalid certificates: %s."
             % ("%d over %d run(s)" % (sum(r["n_bound_violation"] for r in hv), len(hv))
                if hv else "not logged",
                "%d over %d run(s)" % (sum(r["n_cert_invalid"] for r in hi_), len(hi_))
                if hi_ else "not logged"))
    ratio = sorted(v for r in rs_all for v in r["ratio_series"] if finite(v))
    if ratio:
        L.append("- `||g-ghat|| / (||delta|| e)` over %d log point(s): median %s, "
                 "p90 %s, max %s, %d above 1."
                 % (len(ratio), fmt(quantile(ratio, 0.5)), fmt(quantile(ratio, 0.9)),
                    fmt(ratio[-1]), sum(1 for v in ratio if v > 1.0)))
    ns = [r for r in rs_all if r["frac_no_signal"] is not None]
    if ns:
        m, s, n = agg_mean([r["frac_no_signal"] for r in ns])
        L.append("- no-signal steps: %s+-%s of all attempted steps over %d run(s) that "
                 "log `n_no_signal` (%d run(s) predate the counter)."
                 % (fmt(m, 4), fmt(s, 4), n, len(rs_all) - len(ns)))
    nr = [r for r in rs_all if r.get("noreset")]
    if nr:
        m, s, n = agg_mean([r["frac_rel_lt1"] for r in nr])
        L.append("- washout=0 runs (%d): frac_rel_lt1 = %s+-%s (n=%d)."
                 % (len(nr), fmt(m), fmt(s), n))
    if figs is not None:
        L.append("- figures written: %s; skipped: %s."
                 % (", ".join(figs[0]) or "none",
                    "; ".join("%s (%s)" % t for t in figs[1]) or "none"))
    L.append("")
    return L


# ---------------------------------------------------------------- figures
def _fit_notes_above_data(fig, noted, top_data=1.0, cap=12.0):
    """Raise each panel's y limit until its in-panel note clears `top_data`.

    `noted` is a list of ``(ax, text)`` pairs whose text is anchored in AXES
    coordinates near the top of the panel.  Raising the y limit therefore does not
    move the box -- it moves the data down underneath it -- so the condition
    "the bottom of the box sits above y = top_data" can be solved directly:

        frac(top_data) = (top_data - y0) / (y1 - y0)  <  bbox.y0

    The box height in axes coordinates depends on the axes height in inches, which
    is only final after ``tight_layout()``; measuring inside the drawing loop (as
    this used to) over-estimated the available room by the amount tight_layout
    later took away, and the note landed on the curves.  Iterating re-measures
    after every limit change, because the rounded patch around the text is drawn
    by the renderer and its extent is not an exact function of the font size.

    `top_data` is 1.0 for an empirical CDF -- the largest value it can take -- so a
    box clearing it provably covers no data point.  `cap` bounds the growth so a
    pathological measurement cannot blow the axis up without bound.
    """
    for _ in range(6):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        moved = False
        for ax, t in noted:
            y0, y1 = ax.get_ylim()
            # The PATCH, not the text: get_window_extent() measures the glyphs, and the
            # rounded box is drawn `pad` outside them (0.2 x 6 pt here, ~5 px at 300 dpi,
            # which is the whole margin at this scale).  Measuring the text and guessing
            # the padding is what left the first version of this fit with the box edge
            # touching the y = 1 line.  get_bbox_patch() is the box actually drawn.
            p = t.get_bbox_patch()
            box = p if p is not None else t
            bb = box.get_window_extent(r).transformed(ax.transAxes.inverted())
            # gap: a visible sliver below the box, in axes fraction, so the box edge and
            # a 1.1 pt curve sitting exactly at top_data cannot share pixels.
            h_pt = ax.get_window_extent(r).height / fig.dpi * 72.0
            gap = min(0.06, 3.0 / max(h_pt, 1.0))
            want = (top_data - y0) / max(bb.y0 - gap, 1e-3) + y0
            if want > y1 * 1.002 and want <= cap:
                ax.set_ylim(y0, want)
                moved = True
        if not moved:
            break
    return noted


def load_regen():
    """Import paper/figures/gen/regen_all.py by path (it is not a package)."""
    p = os.path.join(ROOT, "paper", "figures", "gen", "regen_all.py")
    if not os.path.isfile(p):
        return None, "regen_all.py not found at %s" % p
    try:
        spec = importlib.util.spec_from_file_location("regen_all_d2", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)        # module level = rcParams + style tables only
    except Exception as e:                                          # noqa: BLE001
        return None, "%s: %s" % (type(e).__name__, e)
    return mod, None


def make_figures(runs, outdir, gray, problems, stage_clip="all", fig_algo=None,
                 cdf_xlim=(1e-3, 1e6)):
    """The three D2 figures, in the manuscript's style.  Returns (written, skipped).

    `fig_algo` restricts every panel to one estimator (default SK-RTRL r16, the one
    the D2 clip sweep is run with).  A family line that mixed SK-RTRL with the
    SnAp-1 reference would not be a clip response at all, so mixing is opt-in only.
    """
    sys.path.insert(0, HERE)                 # fig_style_r1.py lives next to this file
    R, err = load_regen()
    if R is None:
        problems.add("figures", "regen_all.py",
                     err + " -- figures skipped, the manuscript style cannot be "
                           "guaranteed without it")
        return [], [("all figures", err)]
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import fig_style_r1 as fs
        fs.bind(R)
        fs.apply_rc(plt)
    except Exception as e:                                          # noqa: BLE001
        why = "%s: %s" % (type(e).__name__, e)
        problems.add("figures", "matplotlib / numpy", why + " -- figures skipped")
        return [], [("all figures", why)]

    R.OUT, R.GRAY = outdir, gray
    R.WROTE, R.SKIPPED, R.LEGEND_ISSUES = [], [], []
    # R1-2 guard on the shared method table (imported unchanged) ...
    if not R._assert_styles_distinct():
        print("  style table OK: every method pair differs in >=2 visual attributes")
    OI = R.OI
    TW = R.TW

    # ... and the same rule applied to the family / clip tables introduced here.
    FAM_STYLE = {
        "diagnostic": (OI["blue"], "-", "o", "//"),
        "chaotic": (OI["orange"], (0, (5, 2)), "s", "xx"),
        "real": (OI["green"], (0, (1, 1.2)), "^", "\\\\"),
        "RL": (OI["verm"], (0, (6, 1.5, 1, 1.5)), "D", ".."),
        "other": (OI["grey"], (0, (2, 1, 1, 1)), "v", "++"),
    }
    CLIP_STYLE = {
        "none": (OI["black"], "-", "o"),
        "0.2": (OI["sky"], (0, (4, 2, 1, 2)), "d"),
        "0.35": (OI["blue"], (0, (5, 2)), "s"),
        "0.5": (OI["teal"], (0, (4, 1, 1, 1)), "H"),
        "0.7": (OI["orange"], (0, (1, 1.2)), "^"),
        "0.9": (OI["purple"], (0, (3, 1, 1, 1)), "P"),
        "unknown": (OI["grey"], (0, (2, 2)), "x"),
    }

    def _pairs_distinct(table, what):
        """Same >=2-attribute rule as regen_all._assert_styles_distinct (R1-2)."""
        bad = []
        keys = list(table)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                same = sum(str(x) == str(y)
                           for x, y in zip(table[a][:3], table[b][:3]))
                if same >= 2:
                    bad.append((a, b))
        if bad:
            print("  !! %s style collision (>=2 shared attributes): %s" % (what, bad))
            problems.add("figures", "%s style table" % what,
                         "style collision: %s" % (bad,))
        else:
            print("  %s style table OK (>=2 attributes differ for every pair)" % what)
        return not bad

    _pairs_distinct(FAM_STYLE, "family")
    _pairs_distinct(CLIP_STYLE, "clip")

    # Authored at the size they are PRINTED at (R2-3 / the Figure-5 rule).  The widths
    # come from fig_style_r1.TEX_WIDTH, i.e. from the \includegraphics slots in
    # secs/6_experiments.tex, not from a hand-typed fraction: frac_vs_clip and cert_cdf
    # sit in 0.49\linewidth minipages of a figure* (3.37 in), and the 0.92 x \textwidth
    # they used to be drawn at was then scaled down by 0.55 on the page, which put their
    # type back at ~4 pt.  cert_stage is not cited by any \includegraphics yet; 0.92 is
    # the width recommended for it (eight panels need the full text width).
    PRINT_SIZE = {
        # Two stacked panels, so this is a 2-row figure: essentially the 0.78-per-extra-
        # row rule cert_cdf uses, which lands it level with its figure* neighbour
        # instead of 1.2 in short of it (2.31 in against 3.59 in, which left it
        # floating in ~0.6 in of white space above and below in a centred minipage).
        #
        # 0.75 rather than 0.78, and the number is measured, not guessed.  The two
        # share a figure* whose height is the TALLER minipage, so this one must land
        # just UNDER cert_cdf or it starts driving the float: at 0.78 it printed 3 pt
        # taller than cert_cdf, and that was enough to push the float past a page
        # boundary and add a 54th page to the manuscript.  At 0.75 it prints 254.4 pt
        # against cert_cdf's 258.2 pt -- 1.5% apart, and free.
        #
        # Printed height is not proportional to this factor, because _save() exports
        # with bbox_inches='tight' and \includegraphics then scales the trimmed box up
        # to the 0.49\linewidth slot: a shorter figure can trim NARROWER (the legend
        # stops setting the width), which raises the scale factor and can leave the
        # printed height higher than before.  Changing it means re-measuring
        # h_pdf * 242.3 / w_pdf, not re-deriving it on paper.
        "fig_r2_cert_frac_vs_clip": fs.print_size("fig_r2_cert_frac_vs_clip",
                                                  height=2.10, rows=2,
                                                  row_height=0.75 * 2.10),
        "fig_r2_cert_cdf": fs.print_size("fig_r2_cert_cdf", height=2.10),
        "fig_r2_cert_stage": fs.print_size("fig_r2_cert_stage", height=2.30),
    }

    rs_all = cert_runs(runs)
    if fig_algo and rs_all:
        keep = [r for r in rs_all if r["algo"] == fig_algo]
        if not keep:
            problems.add("figures", "--fig-algo %s" % fig_algo,
                         "no run with a certificate uses this estimator; the figures "
                         "fall back to every estimator, so a family line may mix them")
            print("  !! no run with algo %r -- figures keep every estimator" % fig_algo)
        else:
            if len(keep) != len(rs_all):
                print("  figures restricted to algo %r (%d of %d certified run(s))"
                      % (fig_algo, len(keep), len(rs_all)))
            rs_all = keep
    fams = [f for f in FAMILIES + ["other"] if any(r["family"] == f for r in rs_all)]
    clips = sorted({r["clip_key"] for r in rs_all}, key=clip_sort_key)

    def famkw(f):
        c, ls, mk, _h = FAM_STYLE.get(f, FAM_STYLE["other"])
        return dict(color=c, linestyle=ls, marker=mk, label=FAMILY_LABEL.get(f, f))

    def clipkw(ck):
        c, ls, mk = CLIP_STYLE.get(ck, CLIP_STYLE["unknown"])
        return dict(color=c, linestyle=ls, marker=mk, label="clip %s" % ck)

    def _fig_frac_vs_clip():
        # ---------------- Fig 1: informative fraction vs clip level
        name = "fig_r2_cert_frac_vs_clip"
        if not any(finite(r["frac_rel_lt1"]) or finite(r["frac_tight_le10"]) for r in rs_all):
            R._skip(name, "no run carries frac_rel_lt1 / frac_tight_le10")
        else:
            # STACKED, not side by side, for two reasons.  (i) The caption of
            # \label{fig:r2cert} already reads "the direction-certifiable fraction
            # (upper panel) and the tightness bin (lower)", so a 1 x 2 layout
            # contradicted the text describing it.  (ii) This figure shares a
            # figure* with fig_r2_cert_cdf, a 2 x 2 grid ~3.4 in tall; at 1 x 2 this
            # one was 2.16 in, and the centred minipages left it floating in ~0.6 in
            # of white space top and bottom.  Stacked it is the same height as its
            # neighbour and each panel gets the full 3.37 in of width instead of
            # 1.45 in, which is also the healthier aspect ratio (Figure 5's panels
            # are 1.1-1.8 wide-to-tall; 1 x 2 put these at 0.7).  The x axis is
            # shared, so "spectral clip" is drawn once.
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=PRINT_SIZE[name])
            x = np.arange(len(clips), dtype=float)
            # Two lines, and the same wording fig_r2_cert_stage uses for the same two
            # quantities.  A rotated y label runs along the panel HEIGHT, and stacking
            # halved that height: the one-line "fraction of steps with ..." was 1.6 in
            # of text in a 1.35 in panel, so the two labels ran into each other and the
            # upper one lost its "1".  Breaking the line halves the length instead of
            # shrinking the type below the 7.5 pt the style fixes.
            panels = [("frac_rel_lt1", "fraction with\n" + r"$\rho^g_t<1$"),
                      ("frac_tight_le10", "fraction with\n" + r"$T_t\leq 10$")]
            for ax, (field, ylab) in zip(axs, panels):
                for j, f in enumerate(fams):
                    xs, ys, es = [], [], []
                    for i, ck in enumerate(clips):
                        rs = [r for r in rs_all
                              if r["family"] == f and r["clip_key"] == ck]
                        m, s, n = agg_mean([r[field] for r in rs])
                        if m is None:
                            continue
                        xs.append(x[i] + 0.06 * (j - (len(fams) - 1) / 2.0))
                        ys.append(m)
                        es.append(s if n > 1 else 0.0)
                    if xs:
                        ax.errorbar(xs, ys, yerr=es, capsize=1.8, elinewidth=0.7,
                                    **famkw(f))
                ax.set_xticks(x)
                ax.set_xticklabels(clips)
                ax.set_ylabel(ylab)
                ax.set_ylim(-0.03, 1.03)
            axs[-1].set_xlabel("spectral clip")      # shared x: labelled once, at the foot
            h, l = R._merged_handles(list(axs))
            fig.tight_layout()
            fs.legend_below_fit(fig, list(axs), h, l, ncol=min(len(l), 4), name=name,
                                handlelength=1.9, columnspacing=0.7, handletextpad=0.35)
            R._save(fig, name)


    def _fig_cdf():
        # ---------------- Fig 2: empirical CDF of rho^g, one panel per family
        #
        # rho^g spans ~35 decades once the certificate goes vacuous, which would squash the
        # only region the figure is about (the neighbourhood of 1 and 0.5) into a single
        # pixel.  The x-axis is therefore held to a fixed window and the probability mass
        # OUTSIDE it is printed in the panel, per clip: the CDF is monotone, so a truncated
        # view plus the tail mass is complete information rather than a cropped curve.
        name = "fig_r2_cert_cdf"
        # Only NON-FINITE values are dropped (an infinite certificate has no place on the
        # axis).  rho^g = 0 -- a step certified exact, the most informative outcome there
        # is -- stays in the sample so the curve heights are right; the log axis simply
        # cannot draw it, and it is counted in the panel's "off-scale (low)" mass.
        series, dropped = {}, 0
        for r in rs_all:
            vals = [v for v in r["rho_g_series"] if finite(v) and v >= 0]
            dropped += len(r["rho_g_series"]) - len(vals)
            if vals:
                series.setdefault((r["family"], r["clip_key"]), []).extend(vals)
        if not series:
            R._skip(name, "no finite non-negative rho^g in any record")
        else:
            lo_x, hi_x = cdf_xlim
            pf = [f for f in fams if any(k[0] == f for k in series)]
            # sharey: every panel is the same empirical-CDF scale, so the inner
            # tick labels are redundant -- dropping them is what gives the
            # off-scale note room to stay inside its own panel.
            #
            # The panels are laid out on a grid rather than in one row: at the
            # 3.37 in this figure is printed at, four panels in a row would be
            # 0.8 in wide, which cannot carry a 7.5 pt axis label.  grid_for()
            # wraps the row at ~1.45 in per panel instead.  The panel set and
            # what each panel shows are unchanged -- one family per panel.
            nrows, ncols = fs.grid_for(name, len(pf))
            w_in, h_in = PRINT_SIZE[name]
            # 0.78 per extra row: the shared x/y axes mean only the outer panels carry
            # tick labels, so a row costs less than a standalone panel's full height.
            fig, axg = plt.subplots(nrows, ncols, squeeze=False, sharey=True, sharex=True,
                                    figsize=(w_in, h_in if nrows == 1 else 0.78 * h_in * nrows))
            axs = [axg[i // ncols][i % ncols] for i in range(len(pf))]
            for i in range(len(pf), nrows * ncols):          # unused cells: no empty frames
                axg[i // ncols][i % ncols].set_visible(False)
            bottom = {axs[i] for i in range(len(pf))
                      if i + ncols >= len(pf)}               # lowest drawn panel per column
            left = {axs[i] for i in range(len(pf)) if i % ncols == 0}
            noted = []                       # (ax, text) pairs fitted after tight_layout
            for ax, f in zip(axs, pf):
                notes = []
                for ck in clips:
                    v = sorted(series.get((f, ck), []))
                    if not v:
                        continue
                    y = np.arange(1, len(v) + 1) / len(v)
                    kw = clipkw(ck)
                    kw.pop("marker")
                    ax.step(v, y, where="post", linewidth=1.1, **kw)
                    above = sum(1 for x_ in v if x_ > hi_x) / len(v)
                    below = sum(1 for x_ in v if x_ < lo_x) / len(v)
                    if above > 0 or below > 0:
                        # Kept short ON PURPOSE: the panel is ~1.5 in wide, and
                        # the long form ("... % off-scale (... high, ... low)")
                        # overflowed into the neighbouring panel.  The legend of
                        # the notation is in the report text and the caption.
                        notes.append("%s: %.0f%% off (%.0f/%.0f)"
                                     % (("none" if ck == "none" else ck),
                                        100 * (above + below), 100 * above,
                                        100 * below))
                ax.axvline(1.0, color=OI["grey"], linewidth=0.7, linestyle=(0, (4, 2)))
                ax.axvline(0.5, color=OI["grey"], linewidth=0.7, linestyle=(0, (1, 1.5)))
                ax.set_xscale("log")
                ax.set_xlim(lo_x, hi_x)
                if ax in bottom:
                    ax.set_xlabel(r"$\rho^g_t$")
                ax.set_title(FAMILY_LABEL.get(f, f))
                # Head-room ABOVE the CDF (which cannot exceed 1) so the
                # off-scale note sits in provably empty space instead of on
                # top of the curves; the ticks still stop at 1.0, so the axis
                # reads exactly as before.
                ax.set_ylim(-0.03, 1.38)
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                if notes:
                    # 6.0 pt is the floor the R1 style uses for in-panel type; the
                    # 5.2 pt this used to be only looked acceptable because the figure
                    # was about to be scaled DOWN, which would have printed it at 2.9 pt.
                    t = ax.text(0.03, 0.99, "\n".join(notes), transform=ax.transAxes,
                                va="top", ha="left", fontsize=6.0,
                                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                          edgecolor=OI["grey"], linewidth=0.4, alpha=0.85))
                    noted.append((ax, t))
            for ax in left:
                ax.set_ylabel("empirical CDF")
            h, l = _clip_legend(axs)
            fig.tight_layout()
            # Head-room for the off-scale notes is measured HERE, not inside the panel
            # loop, because tight_layout() is what fixes the axes height: a note measured
            # against the pre-layout axes came out ~25% shorter in axes coordinates than
            # it ends up, which is why the box was landing on the curves it was supposed
            # to clear.  The measurement is iterated because raising ylim does not move
            # the box (it is anchored in axes coordinates) but does move y = 1 down, and
            # one pass is enough only when the first estimate is already right.
            _fit_notes_above_data(fig, noted)
            fs.legend_below_fit(fig, axs, h, l, ncol=min(len(l), 4), name=name,
                                handlelength=1.9, columnspacing=0.7, handletextpad=0.35)
            R._save(fig, name)
            if dropped:
                print("  note: %d non-finite / non-positive rho^g log point(s) left out of "
                      "the CDF" % dropped)


    def _clip_legend(axs):
        """Merged handles, reordered by clip value.

        `_merged_handles` returns the handles in panel-traversal order, so a clip level
        that appears in no early panel (clip 0.5 is RL-only, and RL is the last panel)
        landed after 0.9 in the legend even though its bars/curves are drawn in clip
        order.  The legend is re-sorted with the same `clip_sort_key` the axes use, so
        legend and drawing order agree.
        """
        h, l = R._merged_handles(list(axs))
        pairs = list(zip(h, l))
        pairs.sort(key=lambda hl: clip_sort_key(hl[1].replace("clip ", "").strip()))
        return [a for a, _ in pairs], [b for _, b in pairs]

    def _fig_stage():
        # ---------------- Fig 3: three training stages.
        #
        # Grid = 2 criteria x task families, bars grouped by CLIP inside each panel.  Bars
        # are never pooled across clip levels: the clip is the dominant factor of D2, so a
        # bar mixing "none" with 0.9 would not be a stage effect at all.
        name = "fig_r2_cert_stage"
        st_runs = [r for r in rs_all if r.get("stages")]
        if stage_clip not in (None, "all"):
            st_runs = [r for r in st_runs if r["clip_key"] == stage_clip]
        st_fams = [f for f in fams if any(r["family"] == f for r in st_runs)]
        st_clips = [c for c in clips if any(r["clip_key"] == c for r in st_runs)]
        if not st_runs or not st_fams:
            R._skip(name, "no run carries cert_summary.stages%s"
                    % ("" if stage_clip in (None, "all") else " at clip %s" % stage_clip))
        else:
            w_in, h_in = PRINT_SIZE[name]
            fig, axs = plt.subplots(2, len(st_fams), squeeze=False,
                                    figsize=(w_in, h_in * 1.55))
            x = np.arange(len(STAGE_LABEL), dtype=float)
            w = 0.8 / max(len(st_clips), 1)
            panels = [("frac_rel_lt1", r"fraction with $\rho^g_t<1$"),
                      ("frac_tight_le10", r"fraction with $T_t\leq 10$")]
            # The upper end of an error bar on a FRACTION is not bounded by 1: a family
            # whose seeds disagree (mean 0.75, s.d. 0.41 on `diagnostic`) has a +1 s.d.
            # whisker above 1.16.  With the fixed ylim(0, 1.03) this used to be, that
            # whisker left the panel through the top edge with no cap on it, which reads
            # as a drawing artefact rather than as a wide spread.  The top is therefore
            # taken from the data, and the ticks are still pinned to 0.0-1.0 so the axis
            # is read the same way as before.
            top = 1.03
            for row, (field, ylab) in enumerate(panels):
                for col, f in enumerate(st_fams):
                    ax = axs[row][col]
                    for j, ck in enumerate(st_clips):
                        rs = [r for r in st_runs
                              if r["family"] == f and r["clip_key"] == ck]
                        if not rs:
                            continue
                        ys, es = [], []
                        for i in range(len(STAGE_LABEL)):
                            m, s, n = agg_mean([r["stages"][i][field] for r in rs])
                            ys.append(0.0 if m is None else m)
                            es.append(0.0 if (m is None or n < 2) else s)
                        top = max([top] + [a + b for a, b in zip(ys, es)])
                        c, _ls, _mk = CLIP_STYLE.get(ck, CLIP_STYLE["unknown"])
                        _fc, _fls, _fmk, hatch = FAM_STYLE.get(f, FAM_STYLE["other"])
                        ax.bar(x + (j - (len(st_clips) - 1) / 2.0) * w, ys, width=w * 0.9,
                               yerr=es, capsize=1.8, color=c, edgecolor="black",
                               linewidth=0.5, hatch=hatch, label="clip %s" % ck,
                               error_kw=dict(elinewidth=0.7))
                    ax.set_xticks(x)
                    ax.set_xticklabels(STAGE_LABEL)
                    if row == 0:
                        ax.set_title(FAMILY_LABEL.get(f, f))
                    else:
                        ax.set_xlabel("training stage")
                    if col == 0:
                        ax.set_ylabel(ylab)
            flat = [a for row in axs for a in row]
            # One limit for all eight panels (they are read against each other), taken
            # from the tallest whisker anywhere in the figure; the ticks stop at 1.0 so
            # the head-room above it does not invite reading a fraction past 1.
            for ax in flat:
                ax.set_ylim(0, 1.04 * top)
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            h, l = _clip_legend(flat)
            fig.tight_layout()
            fs.legend_below_fit(fig, flat, h, l, ncol=min(len(l), 5), name=name,
                                handlelength=1.9, columnspacing=0.7, handletextpad=0.35)
            R._save(fig, name)

    # One bad figure must not cost the report the other two, nor the tables: each
    # is called behind its own guard and an exception is recorded as a skip (the
    # convention of make_r2_d1_report.make_figures).
    for fn in (_fig_frac_vs_clip, _fig_cdf, _fig_stage):
        try:
            fn()
        except Exception as e:                                      # noqa: BLE001
            why = "%s: %s" % (type(e).__name__, e)
            R.SKIPPED.append((fn.__name__.lstrip("_"), why))
            problems.add("figures", fn.__name__.lstrip("_"), "exception: " + why)
            print("  SKIP %s -- exception: %s" % (fn.__name__.lstrip("_"), why))

    for n_, why in R.LEGEND_ISSUES:
        problems.add("figures", n_, "legend placement: %s" % "; ".join(why))
    return list(R.WROTE), list(R.SKIPPED)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d4-eval", default="results/r2/d4_eval",
                    help="D4 evaluation logs = the 'no clip' D2 level")
    ap.add_argument("--d2-clip", default="results/r2/d2_clip")
    ap.add_argument("--d2-noreset", default="results/r2/d2_noreset")
    ap.add_argument("--rl", default="results/r2/rl")
    ap.add_argument("--rl-clip05", default="results/r2/rl_clip05")
    ap.add_argument("--extra-dir", action="append", default=[],
                    help="extra input directory (repeatable); TAG=PATH to name it")
    ap.add_argument("--d4-algo", default="skrtrl-r16",
                    help="estimator taken out of --d4-eval for the no-clip level "
                         "('' = keep every algo)")
    ap.add_argument("--d4-clip", type=float, default=0.0,
                    help="clip value required of a --d4-eval run (default 0 = no clip)")
    ap.add_argument("--out", "--outmd", dest="out",
                    default="results/r2/D2_SUMMARY.md",
                    help="report path (`--outmd` is the make_r2_d1_report.py spelling)")
    ap.add_argument("--figdir", default=os.path.join(HERE, "..", "paper", "figures"),
                    help="same default as make_r2_d1_report.py")
    ap.add_argument("--gray", default="", help="extra grey-scale self-check renderings")
    ap.add_argument("--stage-clip", default="all",
                    help="restrict fig_r2_cert_stage to one clip level ('all' = pool "
                         "every level, which the report states explicitly)")
    ap.add_argument("--fig-algo", default="skrtrl-r16",
                    help="estimator drawn in the figures ('' = every estimator, which "
                         "can mix SK-RTRL with the SnAp-1 reference in one line)")
    ap.add_argument("--cdf-xlim", default="1e-3,1e6",
                    help="x window of fig_r2_cert_cdf as LO,HI; rho^g spans ~35 decades "
                         "when the certificate is vacuous, and the mass outside the "
                         "window is printed inside each panel")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--max-problems", type=int, default=40)
    args = ap.parse_args()

    dirs = [("d4_eval", args.d4_eval, "none"),
            ("d2_clip", args.d2_clip, "sweep"),
            ("d2_noreset", args.d2_noreset, "sweep"),
            ("rl", args.rl, "sweep"),
            ("rl_clip05", args.rl_clip05, "sweep")]
    for i, spec in enumerate(args.extra_dir):
        tag, sep, path = spec.partition("=")
        if not sep:
            tag, path = "extra%d" % (i + 1), spec
        dirs.append((tag, path, "sweep"))

    problems = Problems()
    print("D2 report: scanning %d input directory(ies)" % len(dirs))
    runs = scan_dirs(dirs, problems, d4_algo=(args.d4_algo or None),
                     d4_clip=args.d4_clip)
    print("  %d run(s) loaded, %d with a certificate summary"
          % (len(runs), len(cert_runs(runs))))
    if _D4_IMPORT_ERROR:
        problems.add("field_missing", "make_r2_d4_select.py",
                     "not importable (%s) -- the performance window and metric "
                     "direction come from the local fallback" % _D4_IMPORT_ERROR)

    try:
        _lo, _hi = (float(t) for t in str(args.cdf_xlim).split(","))
        if not (0 < _lo < _hi):
            raise ValueError("need 0 < LO < HI")
        cdf_xlim = (_lo, _hi)
    except Exception as e:                                          # noqa: BLE001
        ap.error("--cdf-xlim %r: %s" % (args.cdf_xlim, e))

    figs = None
    if not args.no_figs:
        figdir = rel(args.figdir)
        print("figures -> %s" % figdir)
        figs = make_figures(runs, figdir, rel(args.gray) if args.gray else "",
                            problems, stage_clip=args.stage_clip,
                            fig_algo=(args.fig_algo or None), cdf_xlim=cdf_xlim)

    out = rel(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    L = ["# D2 -- certificate informativeness (R2 revision, plan D2 / R3-2)", "",
         "Generated by `code/make_r2_d2_report.py`.  Criteria: `rho^g_t < 1` = "
         "*direction-certifiable* (`< 0.5` = strengthened bin), "
         "`T_t = e_t/||E_t||_F <= 10` = tightness bin, `rho_bar_t < 1` = the step does "
         "not inflate `e_t`.  Per-run fractions come from `cert_summary`, which the "
         "runners accumulate at EVERY counted step; this script only averages them "
         "across seeds (the seed is the unit).  Performance window and metric "
         "direction: %s (last %d%% of the run)."
         % (_D4_SOURCE, round((1 - WINDOW_FRAC) * 100)), "",
         "Per-record series (figure 2 and the ratio table in E) are sampled every "
         "`--log_every` steps, so they are a subsample of the counted steps and are "
         "labelled as such wherever they appear.", ""]
    # Same guard as make_r2_d1_report.py: a report built on smoke output must say so
    # on its first screen, so its tables are never quoted as results.
    smoke = sorted({t for t, p, _role in dirs
                    for r in runs if r["source"] == t and "smoke" in p.lower()})
    if smoke:
        L += ["**NOTE: this report includes SMOKE output (source(s): %s).  It "
              "validates the aggregation and plotting path only and must not be "
              "quoted as a result.**" % ", ".join(smoke), ""]
    L += section_coverage(runs, problems, dirs, args)
    L += section_a(runs)
    L += section_b(runs)
    L += section_c(runs)
    L += section_d(runs)
    L += section_e(runs)
    L += section_f(runs)
    L += section_bullets(runs, figs)
    if figs is not None:
        L += ["## Figures", "",
              "Estimator drawn: %s.  Written to `%s` as PDF + PNG."
              % (args.fig_algo or "every estimator (lines may mix them)", args.figdir),
              "",
              "- `fig_r2_cert_frac_vs_clip` -- informative fraction vs spectral clip "
              "(x includes `none`), one line per task family, error bars = s.d. over "
              "seeds; the second panel is the tightness bin.",
              "- `fig_r2_cert_cdf` -- empirical CDF of `rho^g_t` per family, coloured "
              "by clip, with vertical marks at 1 and 0.5 (log x-axis held to %s; the "
              "probability mass outside that window is printed inside the panel as "
              "`<clip>: P%% off (high/low)`, i.e. the total off-window mass followed "
              "by the share above and below it; non-finite / non-positive log points "
              "are left out and counted on stderr).  The y-axis is shared across "
              "panels." % args.cdf_xlim,
              "- `fig_r2_cert_stage` -- early / mid / late bars, a 2 x (task family) "
              "grid with the bars grouped by clip so no bar pools clip levels; clip "
              "level(s) included: %s." % args.stage_clip, "",
              "Written: %s.  Skipped: %s."
              % (", ".join(figs[0]) or "none",
                 "; ".join("%s (%s)" % t for t in figs[1]) or "none"), ""]
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", out)
    print("%d coverage note(s) recorded in the report" % len(problems.items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
