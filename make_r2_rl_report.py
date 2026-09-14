"""RL (T-maze) certificate + task aggregation for the R2 revision (plan D2, RL part).

`make_r2_d2_report.py` already reads both RL directories and produces the family and
per-task rows that Table 7 of the manuscript needs.  This script adds the granularity
that report deliberately does not carry: the per-seed rows, the paired clip cost on the
task metric, the SnAp-1 control next to SK-RTRL at the same corridor, and the anomaly
scan over the logged records.

Arms (the three that were run; the clip value is always read from `args.clip`)
-------------------------------------------------------------------------------
  skrtrl clip 0     results/r2/rl          skrtrl-r16, clip 0.0
  skrtrl clip 0.5   results/r2/rl_clip05   skrtrl-r16, clip 0.5
  snap1 clip 0      results/r2/rl          snap1,      clip 0.0   (reference)

Definitions (identical to make_r2_d2_report.py, which imports them from the runner's
own counters -- `skrtrl.rl.CertCounters`)
-----------------------------------------------------------------------------------
  rho^g_t = ||delta_t|| e_t / ||ghat_t|| < 1  =>  "direction-certifiable"
  T_t     = e_t / ||E_t||_F               <= 10 = the tightness bin
  rho_bar_t < 1                                 = the step does not inflate e_t
Steps with no gradient signal are in `n_no_signal` and never in a denominator;
`frac_tight_le10` is over `n_tight_checked`, not over `n_steps`.

The seed is the unit: every fraction is computed inside a run by the runner, and this
script only averages those per-run numbers over the three seeds.  Steps are never
pooled across runs.

Finite-bound fraction
---------------------
`cert_summary.n_bound_finite` is written by run_m3 only (see make_r2_d2_report.py
table E).  For the RL runs the only available evidence is the logged records, so this
script reports `frac_bound_finite (log)` = the fraction of LOG POINTS with a finite
`bound_norm` among the signal-bearing ones, sampled every `--log_every` steps.  It is
labelled `(log)` everywhere and must not be quoted as a step-level counter.

Task performance
----------------
`success` and `ret` are averaged over the last 20% of the run (`WINDOW_FRAC`, the D4
selection rule, imported from make_r2_d4_select.py when importable).  The clip cost is
computed PAIRED by (corridor, seed) and reported as an absolute difference plus a
percentage of |clip-0 value|, since the T-maze return is negative and a ratio of
negative numbers is not readable on its own.

Inputs / outputs
----------------
  results/r2/rl, results/r2/rl_clip05   ->   results/r2/RL_SUMMARY.md
`*.localpartial.json` is a mid-run snapshot left next to the finished file and is
skipped (same rule as make_r2_d2_report.py).  No figure is written: the RL runs are
already in fig_r2_cert_* through the D2 report.

Usage
  D:/Anaconda/envs/multilingual_lora/python.exe make_r2_rl_report.py
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

PARTIAL_SUFFIX = ".localpartial.json"

# ------------------------------------------------------------------ protocol constants
WINDOW_FRAC = 0.8
_D4_SOURCE = "local fallback"


def _window_mean(records, steps, key, frac=None):
    """Mean of `key` over the records in the last (1-frac) of the run."""
    frac = WINDOW_FRAC if frac is None else frac
    lo = frac * float(steps or 0)
    vals = [r.get(key) for r in records
            if isinstance(r.get("step"), (int, float)) and r["step"] >= lo]
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return None, 0, lo
    return statistics.fmean(vals), len(vals), lo


try:                                        # keep the window rule in one place
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import make_r2_d4_select as _d4         # noqa: E402
    WINDOW_FRAC = float(_d4.WINDOW_FRAC)
    _D4_SOURCE = "make_r2_d4_select.py"
except Exception as _e:                     # noqa: BLE001  (report, never fail)
    _D4_SOURCE = "local fallback (%s: %s)" % (type(_e).__name__, _e)

ARM_ORDER = ["skrtrl clip 0", "skrtrl clip 0.5", "snap1 clip 0"]


# ------------------------------------------------------------------ small helpers
def rel_path(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def finite(v):
    return isnum(v) and math.isfinite(v)


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
    """mean +- s.d. over per-run values; non-finite entries counted, never averaged."""
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


def group(items, keyfn):
    out = {}
    for it in items:
        out.setdefault(keyfn(it), []).append(it)
    return out


# ------------------------------------------------------------------ loading
def arm_label(algo, clip):
    c = 0.0 if clip is None else float(clip)
    base = "skrtrl" if str(algo).startswith("skrtrl") else str(algo)
    return "%s clip %g" % (base, c)


def read_run(path, tag, notes):
    name = os.path.basename(path)
    where = "%s/%s" % (tag, name)
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as e:                                          # noqa: BLE001
        notes.append(("unreadable", where, "%s: %s" % (type(e).__name__, e)))
        return None
    a = d.get("args")
    if not isinstance(a, dict) or "env_len" not in a:
        notes.append(("not_m5", where, "args carry no `env_len`; not a run_m5 file"))
        return None
    records = d.get("records") or []
    cs = d.get("cert_summary")
    r = {
        "file": name, "source": tag, "corridor": int(a["env_len"]),
        "algo": a.get("algo"), "seed": a.get("seed"), "steps": a.get("steps"),
        "clip": a.get("clip"), "shadow": a.get("shadow"),
        "arm": arm_label(a.get("algo"), a.get("clip")),
        "n_records": len(records), "records": records,
        "cert": cs if isinstance(cs, dict) else None,
        "wall_s": d.get("wall_s"), "peak_MB": d.get("peak_MB"),
    }
    if r["cert"] is None:
        notes.append(("no_certificate", where, "cert_summary absent or not an object"))

    # --- completeness: the last record must reach `steps`
    last_step = records[-1].get("step") if records else None
    r["last_step"] = last_step
    if not (isnum(last_step) and isnum(r["steps"]) and last_step >= r["steps"]):
        notes.append(("incomplete", where,
                      "last logged step %s < steps %s" % (last_step, r["steps"])))

    # --- task performance over the last 20% window
    for key in ("success", "ret", "ep_len"):
        m, n, lo = _window_mean(records, r["steps"], key)
        r["win_" + key] = m
        r["win_n"] = n
        r["win_lo"] = lo
        if m is None:
            notes.append(("window_empty", where,
                          "no finite %r inside the last-20%% window (step >= %g)"
                          % (key, lo)))

    # --- anomaly scan over the logged records
    an = {k: 0 for k in ("nan_loss", "nan_ret", "e_t_nonfinite", "e_t_points",
                         "bound_nonfinite", "bound_points", "rho_g_nonfinite",
                         "rho_g_points", "tight_nonfinite", "tight_points",
                         "no_signal_points", "signal_points")}
    for x in records:
        if "loss" in x and not finite(x.get("loss")):
            an["nan_loss"] += 1
        if "ret" in x and not finite(x.get("ret")):
            an["nan_ret"] += 1
        dn, gh = x.get("delta_norm"), x.get("ghat_norm")
        has_signal = finite(dn) and dn > 0 and finite(gh) and gh > 0
        if "ghat_norm" in x:
            an["signal_points" if has_signal else "no_signal_points"] += 1
        for key, pre in (("e_t", "e_t"), ("bound_norm", "bound"),
                         ("rho_g", "rho_g"), ("tight", "tight")):
            if key in x and x.get(key) is not None:
                an[pre + "_points"] += 1
                if not finite(x.get(key)):
                    an[pre + "_nonfinite"] += 1
    r["anom"] = an
    # finite-bound fraction over the SIGNAL-BEARING log points (proxy, see docstring)
    denom = sum(1 for x in records
                if finite(x.get("delta_norm")) and x["delta_norm"] > 0
                and finite(x.get("ghat_norm")) and x["ghat_norm"] > 0
                and x.get("bound_norm") is not None)
    num = sum(1 for x in records
              if finite(x.get("delta_norm")) and x["delta_norm"] > 0
              and finite(x.get("ghat_norm")) and x["ghat_norm"] > 0
              and finite(x.get("bound_norm")))
    r["frac_bound_finite_log"] = (num / denom) if denom else None
    r["n_bound_finite_log"] = num
    r["n_bound_log"] = denom
    return r


def scan(dirs, notes):
    runs = []
    for tag, path in dirs:
        p = rel_path(path)
        if not os.path.isdir(p):
            notes.append(("dir_missing", "%s (%s)" % (path, tag), "does not exist"))
            continue
        files = sorted(glob.glob(os.path.join(p, "*.json")))
        for f in files:
            if f.endswith(PARTIAL_SUFFIX):
                notes.append(("partial_snapshot", "%s/%s" % (tag, os.path.basename(f)),
                              "mid-run snapshot -- skipped, the finished run is used"))
                continue
            r = read_run(f, tag, notes)
            if r is not None:
                runs.append(r)
    return runs


# ------------------------------------------------------------------ cert accessors
def cf(r, key):
    """One per-run certificate fraction / count, or None when absent."""
    cs = r.get("cert")
    if not cs:
        return None
    return cs.get(key)


# ------------------------------------------------------------------ sections
def sec_coverage(runs, notes, dirs):
    L = ["## Data coverage", "",
         "Window rule: last %g%% of the run, source `%s`.  `*%s` files are mid-run "
         "snapshots and are skipped.  Nothing below is imputed -- an absent field is "
         "printed as `n/a`."
         % (100 * (1 - WINDOW_FRAC), _D4_SOURCE, PARTIAL_SUFFIX), ""]
    by_src = group(runs, lambda r: r["source"])
    rows = []
    for tag, path in dirs:
        rs = by_src.get(tag, [])
        rows.append([tag, path, "yes" if os.path.isdir(rel_path(path)) else "MISSING",
                     len(rs), sum(1 for r in rs if r["cert"]),
                     ", ".join(sorted({r["arm"] for r in rs})) or "-",
                     ", ".join(str(c) for c in sorted({r["corridor"] for r in rs}))
                     or "-"])
    L += md_table(["tag", "path", "exists", "runs", "with cert", "arm(s)",
                   "corridor(s)"], rows)
    rows = []
    for arm in ARM_ORDER:
        for corr in sorted({r["corridor"] for r in runs}):
            rs = [r for r in runs
                  if r["arm"] == arm and r["corridor"] == corr]
            if not rs:
                continue
            rows.append([arm, corr, len(rs),
                         ",".join(str(r["seed"]) for r in sorted(
                             rs, key=lambda x: (x["seed"] is None, x["seed"]))),
                         ",".join(str(r["steps"]) for r in rs[:1]),
                         "yes" if all(r["shadow"] for r in rs) else "MIXED",
                         "yes" if all(isnum(r["last_step"]) and isnum(r["steps"])
                                      and r["last_step"] >= r["steps"]
                                      for r in rs) else "NO"])
    L += ["### Cells (arm x corridor)", ""]
    L += md_table(["arm", "corridor", "n_seeds", "seeds", "steps", "exact shadow",
                   "complete"], rows)
    if notes:
        L += ["### Notes", ""]
        for kind, what, why in notes:
            L.append("- `%s` **%s**: %s" % (what, kind, why))
        L.append("")
    else:
        L += ["No coverage note: every file parsed, every run complete.", ""]
    return L


CERT_COLS = [("frac_rhobar_lt1", "rhobar<1", 4),
             ("frac_rel_lt1", "rho^g<1", 4),
             ("frac_rel_lt05", "rho^g<0.5", 4),
             ("frac_tight_le10", "T<=10", 4)]


def sec_per_run(runs):
    L = ["## Table RL-A -- every run (the seed is the unit)", "",
         "Per-run certificate fractions exactly as the runner counted them: "
         "`rhobar<1`, `rho^g<1`, `rho^g<0.5` over `n_steps` (signal-bearing steps), "
         "`T<=10` over `n_tight_checked`.  `bnd fin (log)` is the finite-`bound_norm` "
         "fraction over the signal-bearing LOG POINTS (run_m5 writes no step-level "
         "`n_bound_finite`).  `viol` / `inval` are the Theorem-1 and Lemma-1 counters, "
         "both of which must be 0.  `success` / `ret` are last-20%-window means.", ""]
    head = (["corridor", "arm", "seed", "steps", "n_steps", "n_nosig"]
            + [lbl for _, lbl, _ in CERT_COLS]
            + ["bnd fin (log)", "viol", "viol_fp", "inval", "success", "ret",
               "wall_s"])
    rows = []
    for r in sorted(runs, key=lambda x: (x["corridor"], ARM_ORDER.index(x["arm"])
                                        if x["arm"] in ARM_ORDER else 9,
                                        x["seed"])):
        rows.append([r["corridor"], r["arm"], r["seed"], r["steps"],
                     cf(r, "n_steps"), cf(r, "n_no_signal")]
                    + [fmt(cf(r, k), nd) for k, _, nd in CERT_COLS]
                    + ["%s (%d/%d)" % (fmt(r["frac_bound_finite_log"], 3),
                                       r["n_bound_finite_log"], r["n_bound_log"]),
                       cf(r, "n_bound_violation"), cf(r, "n_bound_violation_fp"),
                       cf(r, "n_cert_invalid"),
                       fmt(r["win_success"], 4), fmt(r["win_ret"], 3),
                       fmt(r["wall_s"], 0)])
    L += md_table(head, rows)
    return L


def sec_per_cell(runs):
    L = ["## Table RL-B -- corridor x arm, mean +- s.d. over the 3 seeds", ""]
    head = (["corridor", "arm", "n_seeds"] + [lbl for _, lbl, _ in CERT_COLS]
            + ["bnd fin (log)", "mean rho^g", "mean T", "viol", "inval",
               "success", "ret", "wall_s"])
    rows = []
    for corr in sorted({r["corridor"] for r in runs}):
        for arm in ARM_ORDER:
            rs = [r for r in runs if r["corridor"] == corr and r["arm"] == arm]
            if not rs:
                continue
            rows.append([corr, arm, len(rs)]
                        + [agg([cf(r, k) for r in rs], nd) for k, _, nd in CERT_COLS]
                        + [agg([r["frac_bound_finite_log"] for r in rs], 3),
                           agg([cf(r, "mean_rel_g") for r in rs], 2),
                           agg([cf(r, "mean_tightness") for r in rs], 2),
                           "%d" % sum(cf(r, "n_bound_violation") or 0 for r in rs),
                           "%d" % sum(cf(r, "n_cert_invalid") or 0 for r in rs),
                           agg([r["win_success"] for r in rs], 4),
                           agg([r["win_ret"] for r in rs], 3),
                           agg([r["wall_s"] for r in rs], 0)])
    L += md_table(head, rows)
    return L


def sec_family(runs):
    L = ["## Table RL-C -- the Table-7 rows (family average over the 9 runs)", "",
         "One row per arm, averaging the per-run fractions over all three corridors "
         "and three seeds (9 runs), which is the aggregation Table 7 of the "
         "manuscript uses for the other families.  The per-corridor split is Table "
         "RL-B; the corridor-10/20-only numbers are given as well, because the RL row "
         "in the current draft is those six runs.", ""]
    head = ["arm", "scope", "n_runs"] + [lbl for _, lbl, _ in CERT_COLS] + [
        "counted steps", "viol", "viol_fp", "inval"]
    rows = []
    scopes = [("corridors 10/20/40", (10, 20, 40)), ("corridors 10/20", (10, 20))]
    for arm in ARM_ORDER:
        for label, corrs in scopes:
            rs = [r for r in runs if r["arm"] == arm and r["corridor"] in corrs]
            if not rs:
                continue
            rows.append([arm, label, len(rs)]
                        + [agg([cf(r, k) for r in rs], nd) for k, _, nd in CERT_COLS]
                        + ["%d" % sum(cf(r, "n_steps") or 0 for r in rs),
                           "%d" % sum(cf(r, "n_bound_violation") or 0 for r in rs),
                           "%d" % sum(cf(r, "n_bound_violation_fp") or 0 for r in rs),
                           "%d" % sum(cf(r, "n_cert_invalid") or 0 for r in rs)])
    L += md_table(head, rows)
    return L


def sec_stages(runs):
    L = ["## Table RL-D -- early / mid / late thirds of the run", "",
         "`cert_summary.stages`, so a certificate that is informative only early is "
         "not hidden by the run average.", ""]
    head = ["corridor", "arm", "n_seeds", "rho^g<1 early", "rho^g<1 mid",
            "rho^g<1 late", "rhobar<1 early", "rhobar<1 mid", "rhobar<1 late"]
    rows = []
    for corr in sorted({r["corridor"] for r in runs}):
        for arm in ARM_ORDER:
            rs = [r for r in runs if r["corridor"] == corr and r["arm"] == arm
                  and isinstance(cf(r, "stages"), list) and len(cf(r, "stages")) == 3]
            if not rs:
                continue
            cells = []
            for key in ("frac_rel_lt1", "frac_rhobar_lt1"):
                for i in range(3):
                    cells.append(agg([cf(r, "stages")[i].get(key) for r in rs], 4))
            rows.append([corr, arm, len(rs)] + cells)
    L += md_table(head, rows)
    return L


def sec_clip_cost(runs):
    L = ["## Table RL-E -- what clip 0.5 costs the task (paired by corridor and seed)",
         "",
         "SK-RTRL r16 only.  `d` is clip 0.5 minus clip 0 on the last-20%-window mean; "
         "the percentage is `d / |clip-0 value|` (the T-maze return is negative, so a "
         "plain ratio of the two returns is not readable).  A negative `d` on either "
         "metric is a cost.", ""]
    head = ["corridor", "seed", "success clip0", "success clip0.5", "d success",
            "d success %", "ret clip0", "ret clip0.5", "d ret", "d ret %"]
    rows = []
    paired = {"success": [], "ret": [], "success_pct": [], "ret_pct": []}
    per_corr = {}
    for corr in sorted({r["corridor"] for r in runs}):
        for seed in sorted({r["seed"] for r in runs}):
            a = [r for r in runs if r["corridor"] == corr and r["seed"] == seed
                 and r["arm"] == "skrtrl clip 0"]
            b = [r for r in runs if r["corridor"] == corr and r["seed"] == seed
                 and r["arm"] == "skrtrl clip 0.5"]
            if not a or not b:
                continue
            a, b = a[0], b[0]
            cells = [corr, seed]
            for key in ("success", "ret"):
                x, y = a["win_" + key], b["win_" + key]
                d = (y - x) if (finite(x) and finite(y)) else None
                pct = (100.0 * d / abs(x)) if (d is not None and x) else None
                cells += [fmt(x, 4), fmt(y, 4), fmt(d, 4),
                          ("%s%%" % fmt(pct, 1)) if pct is not None else "n/a"]
                if d is not None:
                    paired[key].append(d)
                    per_corr.setdefault((corr, key), []).append(d)
                if pct is not None:
                    paired[key + "_pct"].append(pct)
                    per_corr.setdefault((corr, key + "_pct"), []).append(pct)
            rows.append([cells[0], cells[1]] + cells[2:6] + cells[6:10])
    L += md_table(head, rows)
    L += ["### Paired difference, aggregated", ""]
    rows = [["all corridors", len(paired["success"]),
             agg(paired["success"], 4), agg(paired["success_pct"], 1),
             agg(paired["ret"], 3), agg(paired["ret_pct"], 1)]]
    for corr in sorted({c for c, _ in per_corr}):
        rows.append(["corridor %d" % corr, len(per_corr.get((corr, "success"), [])),
                     agg(per_corr.get((corr, "success"), []), 4),
                     agg(per_corr.get((corr, "success_pct"), []), 1),
                     agg(per_corr.get((corr, "ret"), []), 3),
                     agg(per_corr.get((corr, "ret_pct"), []), 1)])
    L += md_table(["scope", "n pairs", "d success", "d success %", "d ret", "d ret %"],
                  rows)
    return L


def sec_snap1(runs):
    L = ["## Table RL-F -- SnAp-1 reference at the same corridor (both unclipped)", "",
         "SnAp-1 carries the same certificate machinery, so it is a control for the "
         "certificate as well as for the task.  Differences are SK-RTRL minus SnAp-1, "
         "unpaired across seeds (mean of one arm minus mean of the other).", ""]
    head = ["corridor", "metric", "skrtrl clip 0", "snap1 clip 0", "difference",
            "skrtrl clip 0.5"]
    rows = []
    for corr in sorted({r["corridor"] for r in runs}):
        pick = {arm: [r for r in runs if r["corridor"] == corr and r["arm"] == arm]
                for arm in ARM_ORDER}
        for key, nd in (("success", 4), ("ret", 3), ("wall_s", 0)):
            vals = {}
            for arm, rs in pick.items():
                vals[arm] = [r["wall_s"] if key == "wall_s" else r["win_" + key]
                             for r in rs]
            mk = lambda arm: agg(vals.get(arm, []), nd)             # noqa: E731
            fa = [v for v in vals.get("skrtrl clip 0", []) if finite(v)]
            fb = [v for v in vals.get("snap1 clip 0", []) if finite(v)]
            diff = (fmt(statistics.fmean(fa) - statistics.fmean(fb), nd)
                    if fa and fb else "n/a")
            rows.append([corr, key, mk("skrtrl clip 0"), mk("snap1 clip 0"), diff,
                         mk("skrtrl clip 0.5")])
    L += md_table(head, rows)
    return L


def sec_anomalies(runs):
    L = ["## Table RL-G -- anomaly scan over the logged records", "",
         "Counted at the log points (every `--log_every` steps), so these are not "
         "step-level counters.  `e_t !fin` / `rho^g !fin` / `T !fin` are log points at "
         "which the certificate quantity is not finite -- the fp32 accumulator having "
         "overflowed -- which is what makes a bound valid but vacuous.  `no signal` "
         "are log points with `||delta||=0` or `||ghat||=0`.", ""]
    head = ["corridor", "arm", "n_runs", "log pts", "nan loss", "nan ret", "no signal",
            "e_t !fin", "bound !fin", "rho^g !fin", "T !fin"]
    rows = []
    for corr in sorted({r["corridor"] for r in runs}):
        for arm in ARM_ORDER:
            rs = [r for r in runs if r["corridor"] == corr and r["arm"] == arm]
            if not rs:
                continue
            s = lambda k: sum(r["anom"][k] for r in rs)              # noqa: E731
            def frac(num, den):
                return "%d (%s)" % (s(num), fmt(s(num) / s(den), 3)) if s(den) else "0"
            rows.append([corr, arm, len(rs), sum(r["n_records"] for r in rs),
                         s("nan_loss"), s("nan_ret"),
                         frac("no_signal_points", "signal_points"),
                         frac("e_t_nonfinite", "e_t_points"),
                         frac("bound_nonfinite", "bound_points"),
                         frac("rho_g_nonfinite", "rho_g_points"),
                         frac("tight_nonfinite", "tight_points")])
    L += md_table(head, rows)
    tot_v = sum(cf(r, "n_bound_violation") or 0 for r in runs)
    tot_i = sum(cf(r, "n_cert_invalid") or 0 for r in runs)
    tot_f = sum(cf(r, "n_bound_violation_fp") or 0 for r in runs)
    tot_s = sum(cf(r, "n_steps") or 0 for r in runs)
    L += ["Over all %d runs and %d counted steps: `n_bound_violation` = %d, "
          "`n_bound_violation_fp` = %d, `n_cert_invalid` = %d."
          % (len(runs), tot_s, tot_v, tot_f, tot_i), ""]
    return L


def sec_bullets(runs):
    """Auto-generated numeric bullets -- no sentence states a number it did not read."""
    L = ["## Read-off", ""]

    def cell(arm, corr, key):
        rs = [r for r in runs if r["arm"] == arm and r["corridor"] == corr]
        return [cf(r, key) for r in rs]

    for arm in ARM_ORDER:
        corrs = sorted({r["corridor"] for r in runs
                        if r["arm"] == arm})
        if not corrs:
            continue
        parts = []
        for c in corrs:
            parts.append("corridor %d %s" % (c, agg(cell(arm, c, "frac_rel_lt1"), 4)))
        L.append("- **%s**, rho^g<1: %s." % (arm, "; ".join(parts)))
        parts = []
        for c in corrs:
            parts.append("corridor %d %s"
                         % (c, agg(cell(arm, c, "frac_rhobar_lt1"), 4)))
        L.append("- **%s**, rhobar<1: %s." % (arm, "; ".join(parts)))
    L.append("")
    return L


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rl", default="results/r2/rl")
    ap.add_argument("--rl-clip05", default="results/r2/rl_clip05")
    ap.add_argument("--out", default="results/r2/RL_SUMMARY.md")
    args = ap.parse_args()

    dirs = [("rl", args.rl), ("rl_clip05", args.rl_clip05)]
    notes = []
    runs = scan(dirs, notes)
    print("RL report: %d run(s) loaded, %d with a certificate summary"
          % (len(runs), sum(1 for r in runs if r["cert"])))
    if not runs:
        print("no run found -- nothing written")
        return 1

    L = ["# R2 D2 -- RL (T-maze) certificate and task summary", "",
         "Generated by `make_r2_rl_report.py` from %d run file(s) in %s.  "
         "Definitions, the counting rule and the seed-is-the-unit aggregation are in "
         "the module docstring and are identical to `make_r2_d2_report.py`."
         % (len(runs), " + ".join("`%s`" % p for _, p in dirs)), ""]
    L += sec_coverage(runs, notes, dirs)
    L += sec_family(runs)
    L += sec_per_cell(runs)
    L += sec_per_run(runs)
    L += sec_stages(runs)
    L += sec_clip_cost(runs)
    L += sec_snap1(runs)
    L += sec_anomalies(runs)
    L += sec_bullets(runs)

    out = rel_path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L).rstrip() + "\n")
    print("wrote %s" % out)
    print("%d coverage note(s)" % len(notes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
