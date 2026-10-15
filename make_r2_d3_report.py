"""Aggregate and plot the D3 adaptive-rank results.

This produces Table 8 and Fig 5 of the paper, plus the extra tables of Section S6.

Two sub-campaigns, run in this order (saturation diagnosis first, then the
one-at-a-time sweep):

  d3_diag  ceiling / saturation diagnosis.  rotation + anbn, the three kernel variants
           (c(r) rule vs fixed c=8 vs pre-projection off) crossed with the three cap
           cells (r_max=32 at n=64; r_max=64 at n=64 with --force_preproject; r_max=64
           at n=128), 3 seeds -- plus a same-trajectory fixed-rank sweep
           r in {4,8,16,24,32,48,64} that says which rank actually suffices.
  d3_oat   local sensitivity.  rotation/henon/sunspot/anbn/mackeyglass x {default + 9 single-knob
           variants: tau_high {0.01,0.04}, tau_low {0.0015,0.006}, M {2,5}, K {50,200},
           rank interval [2,64]} x 3 seeds.

Every configuration is identified from the run's own `args` block (never from the file
name): the file name only supplies task/method/seed/tag for bookkeeping, and the tag is
reported as a label hint.  A run whose args do not match any roster cell is still
tabulated, in an explicit "off-roster" section, so nothing is silently dropped.

Field tolerance (the D3 instrumentation of run_adaptive.py landed in stages)
---------------------------------------------------------------------------
Top-level `frac_at_cap`, `n_up`, `n_down`, `n_ctrl_checks`, `rank_changes` and the
`--c` / `--force_preproject` / `--preproject` args are read when present.  When they
are absent (any run produced before that instrumentation) the script falls back to the
D3 definitions evaluated on the logged `records` -- fraction of logged checkpoints at
the cap, and up/down moves counted from the logged rank sequence -- and marks every
such number with a trailing `*`.  Trigger attribution (tau_c vs tau_r) has no fallback:
without `rank_changes` it is reported as `n/a`, not guessed.  All substitutions are
counted and listed in the report's "Field tolerance" section.

Outputs
  <outmd>                             (default: <indir_diag>/../D3_SUMMARY.md)
    missing roster, Table A (ceiling diagnosis + fixed-rank sweep vs controller rank),
    Table B (OAT sensitivity, mean +- s.d. over seeds, with delta-vs-default columns and
    a global error-/rank-range header row), field-tolerance notes, numbers-only bullets.
  <figdir>/fig_r2_adaptive_diag.pdf   rotation rank trajectories, n=64 / n=128 panels,
                                      one curve per (kernel variant, cap), cap annotated
  <figdir>/fig_r2_adaptive_oat.pdf    one panel per OAT task: final error (left axis) and
                                      mean rank (right axis) per variant, default filled

A table or figure whose inputs are missing is skipped and listed. The style helpers
(Okabe-Ito palette, print-physical figure sizes, legend outside the axes, _save) come
from figstyle.py and are reused here rather than re-derived.

Usage
  python make_r2_d3_report.py                 # real data: results/r2/d3_diag + d3_oat
  python make_r2_d3_report.py --indir_diag <d> --indir_oat <d> --outmd <p> --figdir <d>
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                 # figstyle.py and fig_style_r1.py live next to this file

EPS = 1e-12


def _show(p):
    """A path for the report: relative to the working directory, forward slashes."""
    try:
        p = os.path.relpath(p)
    except ValueError:
        pass
    return str(p).replace("\\", "/")

# --------------------------------------------------------------------------------------
# protocol constants of the D3 campaign
# --------------------------------------------------------------------------------------
DIAG_TASKS = ["rotation", "anbn"]
OAT_TASKS = ["rotation", "henon", "sunspot", "anbn", "mackeyglass"]
SEEDS = [0, 1, 2]

# controller defaults = run_adaptive.py's argparse defaults; the OAT sweep is defined as
# one-at-a-time departures from exactly these values.
CTRL_DEFAULT = {"tau_high": 0.02, "tau_low": 0.003, "M": 3, "K": 100,
                "r_min": 4, "r_max": 32, "ctrl": "eta"}

# knob -> the non-default values the plan sweeps.  `interval` is the (r_min, r_max) pair,
# swept as a single knob because the plan changes both ends together ([4,32] -> [2,64]).
OAT_GRID = [
    ("tau_high", [0.01, 0.04]),
    ("tau_low", [0.0015, 0.006]),
    ("M", [2, 5]),
    ("K", [50, 200]),
    ("interval", [(2, 64)]),
]
# `ctrl` is not part of the planned 9-variant grid, but a run that changes only --ctrl is
# still a legitimate single-knob departure (the controller ablation), so it is accepted as an
# off-roster OAT variant instead of being rejected as unidentifiable.
OAT_KNOBS = ["tau_high", "tau_low", "M", "K", "interval", "ctrl"]

# kernel axis of the saturation diagnosis, keyed on what the args say about c / preproject
KERNEL_KEYS = ["cr", "c8", "ppoff"]
KERNEL_LABEL = {"cr": "c(r) rule", "c8": "fixed c=8", "ppoff": "pre-projection off"}
# figure labels are kept short so they fit the printed panel width.
KERNEL_LABEL_EN = {"cr": "c(r) rule", "c8": "fixed c=8", "ppoff": "pre-proj. off"}

# Panel titles for the OAT figure: each panel is ~0.5 in wide at the printed size, so a
# task name longer than ~8 characters at 7.5 pt would run into its neighbour.
OAT_TASK_LABEL = {"mackeyglass": "m-glass", "anbn": r"$a^nb^n$", "sunspot": "sunspot",
                  "rotation": "rotation", "henon": "hénon", "lorenz": "lorenz",
                  "laser": "laser", "copy": "copy", "adding": "adding"}
# cap axis: (n, r_max, force_preproject)
DIAG_CAPS = [(64, 32, False), (64, 64, True), (128, 64, False)]
CAP_LABEL = {(64, 32, False): "n=64, r_max=32",
             (64, 64, True): "n=64, r_max=64 (force_pp)",
             (128, 64, False): "n=128, r_max=64"}

FIXED_SWEEP_R = [4, 8, 16, 24, 32, 48, 64]
FIXED_SWEEP_N = 64          # the same-trajectory sweep is rostered at n=64

# metric direction: `metric` is accuracy for cross-entropy tasks and MSE otherwise
# (skrtrl/train.py loss_fn), so a single "error" column needs the conversion made explicit.
CE_TASKS = {"copy", "anbn"}

FNAME_RE = re.compile(r"^(?P<task>[a-z0-9]+)_"
                      r"(?P<method>adaptive(?:-[a-z_]+)?|fixed\d+)_"
                      r"s(?P<seed>\d+)(?:_(?P<tag>.+))?\.json$")

# every fallback actually used, name -> count of runs
TOLERANCE = defaultdict(int)
TOL_NOTE = {
    "frac_at_cap": "top-level frac_at_cap missing -> replaced by the fraction of checkpoints in records with rank>=r_max",
    "n_up_down": "top-level n_up/n_down missing -> replaced by the up/down move counts of the rank sequence in records",
    "rank_changes": "rank_changes missing -> trigger (tau_c/tau_r) reported as n/a, not guessed",
    "peak_MB": "peak_MB missing or null (CPU run)",
    "c_args": "args have no --c/--force_preproject/--preproject (older run_adaptive.py)"
              " -> treated as the defaults c=-1 / 0 / auto",
    "n_ctrl_checks": "top-level n_ctrl_checks missing -> estimated as steps//K",
    "avg_rank": "top-level avg_rank missing -> replaced by the mean avg_rank over records",
}


# --------------------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------------------
def mean_std(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return None, None, 0
    m = sum(v) / len(v)
    if len(v) == 1:
        return m, None, 1
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
    return m, math.sqrt(var), len(v)


def tail_mean(recs, field, frac=0.2):
    """Mean of `field` over the last `frac` of the logged checkpoints (the "tail")."""
    v = [r[field] for r in recs
         if r.get(field) is not None
         and not (isinstance(r[field], float) and math.isnan(r[field]))]
    if not v:
        return None
    k = max(1, int(round(len(v) * frac)))
    return sum(v[-k:]) / k


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    return f"{x:.{nd}f}"


def fmt_ms(m, s, nd=3, star=""):
    if m is None:
        return "n/a"
    if s is None:
        return f"{m:.{nd}f}{star}"
    return f"{m:.{nd}f}±{s:.{nd}f}{star}"


def fmt_pct(x, nd=1):
    return "n/a" if x is None else f"{100.0 * x:.{nd}f}%"


def to_error(task, metric):
    """One 'error' scale for both loss types: MSE as-is, 1-accuracy for the ce tasks."""
    if metric is None:
        return None
    return (1.0 - metric) if task in CE_TASKS else metric


# --------------------------------------------------------------------------------------
# loading + configuration identification (everything from `args`, nothing from the name)
# --------------------------------------------------------------------------------------
def meta_of(d, fallback_task=None, fallback_seed=None):
    a = d.get("args", {}) or {}
    have_c_args = any(k in a for k in ("c", "force_preproject", "preproject"))
    if not have_c_args:
        TOLERANCE["c_args"] += 1
    m = {
        "task": a.get("task", fallback_task),
        "seed": a.get("seed", fallback_seed),
        "n": a.get("n"),
        "steps": a.get("steps"),
        "fixed_r": a.get("fixed_r", -1),
        "r_min": a.get("r_min"),
        "r_max": a.get("r_max"),
        "K": a.get("K"),
        "M": a.get("M"),
        "tau_low": a.get("tau_low"),
        "tau_high": a.get("tau_high"),
        "ctrl": a.get("ctrl", "eta"),
        "c": a.get("c", -1),
        "force_preproject": int(a.get("force_preproject", 0) or 0),
        "preproject": a.get("preproject", "auto"),
        "c_effective": a.get("c_effective"),
        "preproject_effective": a.get("preproject_effective"),
        "tag": a.get("tag", "") or "",
        "have_c_args": have_c_args,
    }
    m["is_adaptive"] = (m["fixed_r"] is None) or (m["fixed_r"] < 0)
    m["forced_pp"] = bool(m["force_preproject"]) or (m["preproject"] == "on")
    return m


def kernel_key(m):
    """Kernel variant of the saturation diagnosis, derived from the args only."""
    if m["preproject"] == "off":
        return "ppoff"
    c = m["c"]
    if c is not None and c > 0:
        return f"c{int(c)}"
    return "cr"


def kernel_disp(key, m=None):
    if key in KERNEL_LABEL:
        lab = KERNEL_LABEL[key]
    elif re.fullmatch(r"c\d+", key or ""):
        lab = f"fixed c={key[1:]}"
    else:
        lab = str(key)
    if m is not None and m.get("c_effective") is not None:
        lab += f" [c_eff={int(m['c_effective'])}]"
    return lab


def cap_key(m):
    return (m["n"], m["r_max"], bool(m["forced_pp"]))


def cap_disp(key):
    n, rmax, fpp = key
    return CAP_LABEL.get(key, f"n={n}, r_max={rmax}" + (" (force_pp)" if fpp else ""))


def oat_variant(m):
    """(variant_key, display, n_knobs_changed).  'default' when nothing departs."""
    diffs = []
    for k in OAT_KNOBS:
        if k == "interval":
            cur = (m["r_min"], m["r_max"])
            ref = (CTRL_DEFAULT["r_min"], CTRL_DEFAULT["r_max"])
            if None not in cur and cur != ref:
                diffs.append(("interval", cur))
        else:
            cur, ref = m.get(k), CTRL_DEFAULT[k]
            if cur is None:
                continue
            if isinstance(ref, (int, float)) and not isinstance(ref, bool):
                same = abs(cur - ref) <= 1e-12 * max(1.0, abs(ref))
            else:
                same = (cur == ref)
            if not same:
                diffs.append((k, cur))
    if not diffs:
        return "default", "default", 0
    if len(diffs) == 1:
        k, v = diffs[0]
        if k == "interval":
            return f"interval[{v[0]},{v[1]}]", f"interval [{v[0]},{v[1]}]", 1
        vs = f"{v:g}" if isinstance(v, float) else str(v)
        return f"{k}={vs}", f"{k}={vs}", 1
    key = "+".join(f"{k}={v}" for k, v in diffs)
    return f"multi:{key}", f"several knobs changed ({key})", len(diffs)


def load_dir(indir):
    """-> (runs, bad).  runs = list of {path, meta, data}; bad = [(path, reason)]."""
    runs, bad = [], []
    if not indir or not os.path.isdir(indir):
        return runs, bad
    for p in sorted(glob.glob(os.path.join(indir, "*.json"))):
        base = os.path.basename(p)
        fm = FNAME_RE.match(base)
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception as e:                       # still being written, truncated, ...
            bad.append((p, f"read failed: {type(e).__name__}: {e}"))
            continue
        if not isinstance(d, dict) or "args" not in d:
            bad.append((p, "no args field, not run_adaptive.py output"))
            continue
        if not d.get("records"):
            bad.append((p, "no records (possibly still running)"))
            continue
        m = meta_of(d, fm["task"] if fm else None, int(fm["seed"]) if fm else None)
        m["file"] = base
        m["fname_tag"] = (fm["tag"] or "") if fm else ""
        m["fname_method"] = fm["method"] if fm else ""
        if m["task"] is None or m["seed"] is None:
            bad.append((p, "no task/seed in args and the file name cannot be parsed"))
            continue
        runs.append({"path": p, "meta": m, "data": d})
    return runs, bad


# --------------------------------------------------------------------------------------
# per-run metrics, with the documented fallbacks
# --------------------------------------------------------------------------------------
def run_metrics(run):
    d, m = run["data"], run["meta"]
    recs = d.get("records", [])
    out = {"flags": set()}

    out["metric_raw"] = tail_mean(recs, "metric")
    out["err"] = to_error(m["task"], out["metric_raw"])
    out["grad_cos"] = tail_mean(recs, "grad_cos")
    # certificate value e_t, same tail window as the error (present in every D3 record)
    out["e_t"] = tail_mean(recs, "e_t")

    if d.get("avg_rank") is not None:
        out["avg_rank"] = float(d["avg_rank"])
    else:
        out["avg_rank"] = tail_mean(recs, "avg_rank", frac=1.0)
        out["flags"].add("avg_rank")
        TOLERANCE["avg_rank"] += 1

    # --- frac_at_cap -------------------------------------------------------------------
    if d.get("frac_at_cap") is not None:
        out["frac_at_cap"] = float(d["frac_at_cap"])
    else:
        rmax = m["r_max"]
        ranks = [r.get("rank") for r in recs if r.get("rank") is not None]
        out["frac_at_cap"] = (sum(1 for r in ranks if r >= rmax) / len(ranks)
                              if (ranks and rmax) else None)
        out["flags"].add("frac_at_cap")
        TOLERANCE["frac_at_cap"] += 1

    # --- n_up / n_down ------------------------------------------------------------------
    if d.get("n_up") is not None or d.get("n_down") is not None:
        out["n_up"], out["n_down"] = d.get("n_up"), d.get("n_down")
    else:
        ranks = [r.get("rank") for r in recs if r.get("rank") is not None]
        if ranks:
            out["n_up"] = sum(1 for a, b in zip(ranks, ranks[1:]) if b > a)
            out["n_down"] = sum(1 for a, b in zip(ranks, ranks[1:]) if b < a)
        else:
            out["n_up"] = out["n_down"] = None
        out["flags"].add("n_up_down")
        TOLERANCE["n_up_down"] += 1

    # --- controller checks --------------------------------------------------------------
    if d.get("n_ctrl_checks") is not None:
        out["n_checks"] = d["n_ctrl_checks"]
    else:
        out["n_checks"] = (m["steps"] // m["K"]) if (m["steps"] and m["K"]) else None
        out["flags"].add("n_ctrl_checks")
        TOLERANCE["n_ctrl_checks"] += 1

    # --- trigger attribution: NO fallback ------------------------------------------------
    rc = d.get("rank_changes")
    if rc is None:
        out["trig"] = None
        out["flags"].add("rank_changes")
        TOLERANCE["rank_changes"] += 1
    else:
        dom = [c.get("dominant") for c in rc if c.get("dominant") in ("tau_c", "tau_r")]
        ups = [c for c in rc if c.get("direction") == "up"]
        dom_up = [c.get("dominant") for c in ups if c.get("dominant") in ("tau_c", "tau_r")]
        tc = [c.get("tau_c_norm") for c in rc if c.get("tau_c_norm") is not None]
        tr = [c.get("tau_r_norm") for c in rc if c.get("tau_r_norm") is not None]
        out["trig"] = {
            "n_changes": len(rc),
            "n_dom": len(dom),
            "frac_tau_c": (dom.count("tau_c") / len(dom)) if dom else None,
            "frac_tau_c_up": (dom_up.count("tau_c") / len(dom_up)) if dom_up else None,
            "tau_c_norm": mean_std(tc)[0],
            "tau_r_norm": mean_std(tr)[0],
            "frac_changes_at_cap": (sum(1 for c in rc if c.get("at_cap")) / len(rc)
                                    if rc else None),
        }

    out["peak_MB"] = d.get("peak_MB")
    if out["peak_MB"] is None:
        TOLERANCE["peak_MB"] += 1
    out["cert_viol"] = d.get("cert_violations")
    out["wall_s"] = d.get("wall_s")
    return out


def agg(runs_):
    """Aggregate the runs of one configuration over seeds."""
    ms = [run_metrics(r) for r in runs_]
    flags = set().union(*[m["flags"] for m in ms]) if ms else set()
    a = {"n_seeds": len(runs_),
         "seeds": sorted(r["meta"]["seed"] for r in runs_),
         "flags": flags,
         "tags": sorted({(r["meta"]["tag"] or r["meta"]["fname_tag"]) for r in runs_} - {""}),
         "metas": [r["meta"] for r in runs_]}
    for f in ("err", "grad_cos", "avg_rank", "frac_at_cap", "peak_MB", "metric_raw",
              "n_up", "n_down", "e_t"):
        a[f] = mean_std([m[f] for m in ms])
    a["cert_viol"] = sum((m["cert_viol"] or 0) for m in ms)
    trigs = [m["trig"] for m in ms if m["trig"]]
    if trigs:
        a["trig"] = {
            "n_changes": sum(t["n_changes"] for t in trigs),
            "frac_tau_c": mean_std([t["frac_tau_c"] for t in trigs])[0],
            "frac_tau_c_up": mean_std([t["frac_tau_c_up"] for t in trigs])[0],
            "tau_c_norm": mean_std([t["tau_c_norm"] for t in trigs])[0],
            "tau_r_norm": mean_std([t["tau_r_norm"] for t in trigs])[0],
            "n_runs": len(trigs),
        }
    else:
        a["trig"] = None
    return a


# --------------------------------------------------------------------------------------
# Table A: ceiling diagnosis + fixed-rank sweep
# --------------------------------------------------------------------------------------
def build_table_a(runs):
    adaptive, fixed = defaultdict(list), defaultdict(list)
    off = []
    for r in runs:
        m = r["meta"]
        if m["is_adaptive"]:
            key = (m["task"], cap_key(m), kernel_key(m))
            adaptive[key].append(r)
            if (key[1] not in DIAG_CAPS) or (key[2] not in KERNEL_KEYS) \
                    or (m["task"] not in DIAG_TASKS) or (m["seed"] not in SEEDS):
                off.append((r, "adaptive"))
        else:
            key = (m["task"], m["n"], int(m["fixed_r"]))
            fixed[key].append(r)
            if (m["task"] not in DIAG_TASKS) or (key[2] not in FIXED_SWEEP_R) \
                    or (m["n"] != FIXED_SWEEP_N) or (m["seed"] not in SEEDS):
                off.append((r, "fixed"))
    return {"adaptive": {k: agg(v) for k, v in adaptive.items()},
            "fixed": {k: agg(v) for k, v in fixed.items()},
            "offroster": off}


def min_rank_analysis(ta, target_cos, err_tol):
    """For each (task, n) present in the fixed-rank sweep: the smallest swept r meeting
    (a) grad_cos >= target_cos and (b) error <= (1+err_tol) x best error over the sweep,
    plus the controller's actual mean rank in every cap cell of the same task/width."""
    out = {}
    by_tn = defaultdict(dict)
    for (task, n, r), a in ta["fixed"].items():
        by_tn[(task, n)][r] = a
    for (task, n), rr in by_tn.items():
        rs = sorted(rr)
        errs = {r: rr[r]["err"][0] for r in rs if rr[r]["err"][0] is not None}
        coss = {r: rr[r]["grad_cos"][0] for r in rs if rr[r]["grad_cos"][0] is not None}
        best = min(errs.values()) if errs else None
        # reference for a *relative* error target: the best error attained in the sweep
        if best is None:
            thr = None
        elif best > 0:
            thr = best * (1.0 + err_tol)
        else:
            thr = best + err_tol          # non-positive best: fall back to additive slack
        r_cos = next((r for r in rs if coss.get(r) is not None and coss[r] >= target_cos),
                     None)
        r_err = (next((r for r in rs if errs.get(r) is not None and errs[r] <= thr), None)
                 if thr is not None else None)
        ctrl = {}
        for (t2, cap, ker), a in ta["adaptive"].items():
            if t2 == task and cap[0] == n:
                ctrl[(cap, ker)] = a
        out[(task, n)] = {"ranks": rs, "errs": errs, "coss": coss, "best_err": best,
                          "err_thr": thr, "r_cos": r_cos, "r_err": r_err,
                          "ctrl": ctrl, "aggs": rr}
    return out


def _diag_sort_key(k):
    task, cap, ker = k
    return (DIAG_TASKS.index(task) if task in DIAG_TASKS else 9, str(task),
            DIAG_CAPS.index(cap) if cap in DIAG_CAPS else 9, str(cap),
            KERNEL_KEYS.index(ker) if ker in KERNEL_KEYS else 9, str(ker))


def render_table_a(ta, mra, target_cos, err_tol):
    L = ["## Table A Ceiling diagnosis (D3 saturation diagnosis)", ""]
    if not ta["adaptive"] and not ta["fixed"]:
        L.append("_No d3_diag result files; this table is skipped._")
        return "\n".join(L)

    L.append("Tail = mean over the last 20% of logged checkpoints. Error: MSE for mse tasks, "
             "1−accuracy for ce tasks (copy/anbn). `*` = number derived by the records fallback "
             "(see \"Field tolerance\").")
    L.append("")
    L.append("### A-1 Adaptive controller (each cell mean±s.d. over seeds)")
    L.append("")
    L.append("| task | cap cell | kernel variant | seeds | frac_at_cap | n_up | n_down | mean rank | "
             "tail error | grad_cos | peak memory MB | τ_c trigger share | τ_c/τ_r normalised mean |")
    L.append("|" + "---|" * 13)
    for key in sorted(ta["adaptive"], key=_diag_sort_key):
        task, cap, ker = key
        a = ta["adaptive"][key]
        s_cap = "*" if "frac_at_cap" in a["flags"] else ""
        s_ud = "*" if "n_up_down" in a["flags"] else ""
        t = a["trig"]
        if not t or t["frac_tau_c"] is None:
            trig_c = "n/a"
        else:
            trig_c = fmt_pct(t["frac_tau_c"])
            if t["frac_tau_c_up"] is not None:
                trig_c += f" (rank up {fmt_pct(t['frac_tau_c_up'])})"
        taus = ("n/a" if not t or t["tau_c_norm"] is None else
                f"{fmt(t['tau_c_norm'], 4)} / {fmt(t['tau_r_norm'], 4)}")
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            task, cap_disp(cap), kernel_disp(ker, a["metas"][0]),
            ",".join(map(str, a["seeds"])),
            fmt_ms(*a["frac_at_cap"][:2], nd=3, star=s_cap),
            fmt_ms(*a["n_up"][:2], nd=1, star=s_ud),
            fmt_ms(*a["n_down"][:2], nd=1, star=s_ud),
            fmt_ms(*a["avg_rank"][:2], nd=2),
            fmt_ms(*a["err"][:2], nd=4),
            fmt_ms(*a["grad_cos"][:2], nd=4),
            fmt_ms(*a["peak_MB"][:2], nd=1),
            trig_c, taus))
    L.append("")

    L.append("### A-2 Same-trajectory fixed-rank sweep vs the controller's actual mean rank")
    L.append("")
    if not mra:
        L.append("_No fixed-r sweep results; skipped._")
        L.append("")
    for (task, n), d in sorted(mra.items()):
        L.append(f"**{task}, n={n}** — completed ranks {d['ranks']}; best tail error in the sweep "
                 f"{fmt(d['best_err'], 5)}; relative threshold (1+{err_tol:g})×best = "
                 f"{fmt(d['err_thr'], 5)}")
        L.append("")
        L.append("| fixed r | tail error | grad_cos | mean rank (check) | seeds |")
        L.append("|" + "---|" * 5)
        for r in d["ranks"]:
            a = d["aggs"][r]
            L.append("| {} | {} | {} | {} | {} |".format(
                r, fmt_ms(*a["err"][:2], nd=5), fmt_ms(*a["grad_cos"][:2], nd=4),
                fmt_ms(*a["avg_rank"][:2], nd=2), ",".join(map(str, a["seeds"]))))
        L.append("")
        L.append(f"- smallest r reaching grad_cos ≥ {target_cos:g}: **"
                 + (str(d["r_cos"]) if d["r_cos"] is not None
                    else "n/a (no completed rank reaches it)") + "**")
        L.append(f"- smallest r with tail error ≤ (1+{err_tol:g})×best: **"
                 + (str(d["r_err"]) if d["r_err"] is not None else "n/a") + "**")
        miss = [r for r in FIXED_SWEEP_R if r not in d["ranks"]]
        if miss:
            L.append(f"- Note: ranks {miss} are still missing, so the \"smallest r\" above holds only within "
                     f"the completed ranks {d['ranks']} and may change once they are filled in.")
        for (cap, ker), a in sorted(d["ctrl"].items(), key=lambda kv: str(kv[0])):
            s_cap = "*" if "frac_at_cap" in a["flags"] else ""
            L.append(f"- controller [{cap_disp(cap)} / {kernel_disp(ker)}] actual mean rank "
                     f"{fmt_ms(*a['avg_rank'][:2], nd=2)}, frac_at_cap "
                     f"{fmt_ms(*a['frac_at_cap'][:2], nd=3, star=s_cap)}")
        if not d["ctrl"]:
            L.append("- (no adaptive run for this task/width to compare with yet)")
        L.append("")

    if ta["offroster"]:
        L.append("### A-3 d3_diag runs outside the roster (listed, not dropped)")
        L.append("")
        L.append("These runs are still counted in the matching cells above (the seeds column shows "
                 "the seeds actually used). This list only marks where they depart from the roster:")
        L.append("")
        for r, kind in ta["offroster"]:
            m = r["meta"]
            why = []
            if m["task"] not in DIAG_TASKS:
                why.append(f"task {m['task']} not in the diagnosis roster")
            if m["seed"] not in SEEDS:
                why.append(f"seed {m['seed']} outside {SEEDS}")
            if kind == "adaptive":
                desc = f"adaptive, {cap_disp(cap_key(m))}, {kernel_disp(kernel_key(m), m)}"
                if cap_key(m) not in DIAG_CAPS:
                    why.append(f"cap cell {cap_key(m)} not in the roster")
                if kernel_key(m) not in KERNEL_KEYS:
                    why.append(f"kernel variant {kernel_key(m)} not in the roster")
            else:
                desc = f"fixed_r={m['fixed_r']}, n={m['n']}"
                if int(m["fixed_r"]) not in FIXED_SWEEP_R:
                    why.append(f"fixed rank {m['fixed_r']} not among the sweep points {FIXED_SWEEP_R}")
                if m["n"] != FIXED_SWEEP_N:
                    why.append(f"n={m['n']} is not the sweep width {FIXED_SWEEP_N}")
            tg = m["tag"] or m["fname_tag"]
            L.append(f"- `{m['file']}` — {m['task']}, s{m['seed']}, {desc}"
                     + (f", tag=`{tg}`" if tg else "")
                     + "; departs in: " + ("; ".join(why) if why else "unknown"))
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# Table B: OAT sensitivity
# --------------------------------------------------------------------------------------
def expected_oat_variants():
    out = ["default"]
    for k, vals in OAT_GRID:
        for v in vals:
            if k == "interval":
                out.append(f"interval[{v[0]},{v[1]}]")
            elif isinstance(v, float):
                out.append(f"{k}={v:g}")
            else:
                out.append(f"{k}={v}")
    return out


def build_table_b(runs):
    cells = defaultdict(list)
    multi = []
    # Every actual down move, with the control signal that produced it.
    # Read straight off `rank_changes`; runs without that field contribute nothing here
    # (the records fallback cannot recover c_t), and are counted separately.
    downs, no_rc = [], 0
    for r in runs:
        m = r["meta"]
        rc = r["data"].get("rank_changes")
        if rc is None:
            no_rc += 1
        else:
            for c in rc:
                if c.get("direction") == "down":
                    downs.append((m, c))
        if not m["is_adaptive"]:
            multi.append((r, f"fixed_r={m['fixed_r']} run found in the OAT directory, not counted in any OAT cell"))
            continue
        key, disp, nk = oat_variant(m)
        if nk > 1:
            multi.append((r, disp))
            continue
        cells[(m["task"], key)].append(r)
    return {"cells": {k: agg(v) for k, v in cells.items()}, "multi": multi,
            "downs": downs, "n_runs": len(runs), "no_rc": no_rc}


def _oat_task_order(cells):
    tasks = [t for t in OAT_TASKS if any(k[0] == t for k in cells)]
    return tasks + sorted({k[0] for k in cells} - set(OAT_TASKS))


def _oat_variant_order(cells, task):
    exp = expected_oat_variants()
    keys = [v for v in exp if (task, v) in cells]
    return keys + sorted({k[1] for k in cells if k[0] == task} - set(keys))


def render_table_b(tb):
    L = ["## Table B OAT one-at-a-time sensitivity (D3 local sensitivity)", ""]
    cells = tb["cells"]
    if not cells:
        L.append("_No d3_oat result files; this table is skipped._")
        return "\n".join(L)

    tasks = _oat_task_order(cells)
    exp = expected_oat_variants()
    L.append("⚑ = task or variant outside the D3 OAT roster (for example the controller ablation that "
             "changes only `--ctrl`). It is still listed as a legitimate single-knob variant, for reference.")
    L.append("")

    L.append("### B-0 Range across all variants (completed variants only)")
    L.append("")
    L.append("| task | completed variants | error min→max | error range | mean rank min→max | rank range |")
    L.append("|" + "---|" * 6)
    for t in tasks:
        es = {k[1]: cells[k]["err"][0] for k in cells
              if k[0] == t and cells[k]["err"][0] is not None}
        rk = {k[1]: cells[k]["avg_rank"][0] for k in cells
              if k[0] == t and cells[k]["avg_rank"][0] is not None}
        n_done = len([k for k in cells if k[0] == t])
        t = t if t in OAT_TASKS else f"{t} ⚑"
        e_rng = f"{fmt(min(es.values()), 5)} → {fmt(max(es.values()), 5)}" if es else "n/a"
        e_sp = fmt(max(es.values()) - min(es.values()), 5) if es else "n/a"
        r_rng = f"{fmt(min(rk.values()), 2)} → {fmt(max(rk.values()), 2)}" if rk else "n/a"
        r_sp = fmt(max(rk.values()) - min(rk.values()), 2) if rk else "n/a"
        L.append(f"| {t} | {n_done}/{len(exp)} | {e_rng} | {e_sp} | {r_rng} | {r_sp} |")
    L.append("")

    L.append("### B-1 Per task × variant (mean±s.d. over seeds)")
    L.append("")
    L.append("| task | variant | seeds | tail error | Δerror vs default | mean rank | Δrank vs default | "
             "frac_at_cap | grad_cos | n_up | n_down | certificate e_t | Δe_t | peak_MB | Δpeak |")
    L.append("|" + "---|" * 15)
    for t in tasks:
        base = cells.get((t, "default"))
        be = base["err"][0] if base else None
        br = base["avg_rank"][0] if base else None
        bc = base["e_t"][0] if base else None
        bp = base["peak_MB"][0] if base else None
        for v in _oat_variant_order(cells, t):
            a = cells[(t, v)]
            if be is None or a["err"][0] is None:
                de = "n/a"
            else:
                de = f"{a['err'][0] - be:+.5f}"
                if abs(be) > EPS:
                    de += f" ({100 * (a['err'][0] - be) / abs(be):+.1f}%)"
            if br is None or a["avg_rank"][0] is None:
                dr = "n/a"
            else:
                dr = f"{a['avg_rank'][0] - br:+.2f}"
                if abs(br) > EPS:
                    dr += f" ({100 * (a['avg_rank'][0] - br) / abs(br):+.1f}%)"
            def _rel(cur, bas):
                """Relative change vs the default cell, or n/a when either side is missing."""
                if bas is None or cur is None:
                    return "n/a"
                s = f"{cur - bas:+.4g}"
                if abs(bas) > EPS:
                    s += f" ({100 * (cur - bas) / abs(bas):+.1f}%)"
                return s
            dc = _rel(a["e_t"][0], bc)
            dp = _rel(a["peak_MB"][0], bp)
            star = "*" if "frac_at_cap" in a["flags"] else ""
            ud_star = "*" if "n_up_down" in a["flags"] else ""
            vlab = ("**default**" if v == "default"
                    else (v if v in exp else f"{v} ⚑"))
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                t if t in OAT_TASKS else f"{t} ⚑", vlab, ",".join(map(str, a["seeds"])),
                fmt_ms(*a["err"][:2], nd=5), de,
                fmt_ms(*a["avg_rank"][:2], nd=2), dr,
                fmt_ms(*a["frac_at_cap"][:2], nd=3, star=star),
                fmt_ms(*a["grad_cos"][:2], nd=4),
                fmt_ms(*a["n_up"][:2], nd=1, star=ud_star),
                fmt_ms(*a["n_down"][:2], nd=1, star=ud_star),
                fmt_ms(*a["e_t"][:2], nd=4), dc,
                fmt_ms(*a["peak_MB"][:2], nd=1), dp))
    L.append("")

    # --- B-1b: did shrinking ever fire? -------------------------------------------------
    L.append("### B-1b Does rank reduction ever fire?")
    L.append("")
    downs, nr = tb["downs"], tb["n_runs"]
    if tb["no_rc"]:
        L.append(f"Note: {tb['no_rc']} runs have no `rank_changes` field, so their rank reductions cannot be "
                 "counted and they are excluded (`c_t` has no fallback).")
        L.append("")
    if not downs:
        L.append(f"**None of the {nr} OAT runs reduces its rank (n_down = 0 for all).** "
                 "Within this sweep the controller is a monotone ratchet.")
    else:
        nrun = len({(m["file"]) for m, _ in downs})
        L.append(f"**Rank reduction does fire: of the {nr} OAT runs, {nrun} reduce their rank, "
                 f"{len(downs)} times in total.** Each one is listed, with `c_t` the control signal at that moment:")
        L.append("")
        L.append("| run file | task | τ_low | M | K | step | r change | c_t | τ_c | τ_r |")
        L.append("|" + "---|" * 10)
        for m, c in sorted(downs, key=lambda x: (str(x[0]["task"]), x[0]["file"], x[1]["step"])):
            def g(k):
                v = c.get(k)
                return "n/a" if v is None else f"{v:.4g}"
            L.append(f"| `{m['file']}` | {m['task']} | {m['tau_low']:g} | {m['M']} | "
                     f"{m['K']} | {c['step']} | {c['r_before']}→{c['r_after']} | "
                     f"{g('c_t')} | {g('tau_c_norm')} | {g('tau_r_norm')} |")
    L.append("")
    L.append("Scale of the control signal (recorded only at rank changes; see the table above and the "
             "trigger columns of Table A): whether a rank reduction can happen depends on how the "
             "default τ_low = 0.003 compares with the observed `c_t`.")
    L.append("")

    if tb["multi"]:
        L.append("### B-2 Multi-knob or unclassifiable runs (listed, left out of the OAT aggregation)")
        L.append("")
        for r, why in tb["multi"]:
            L.append(f"- `{r['meta']['file']}` — {why}")
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# roster / missing lists
# --------------------------------------------------------------------------------------
def diag_roster():
    exp = [("adaptive", t, cap, ker, s)
           for t in DIAG_TASKS for cap in DIAG_CAPS for ker in KERNEL_KEYS for s in SEEDS]
    exp += [("fixed", t, FIXED_SWEEP_N, r, s)
            for t in DIAG_TASKS for r in FIXED_SWEEP_R for s in SEEDS]
    return exp


def oat_roster():
    return [(t, v, s) for t in OAT_TASKS for v in expected_oat_variants() for s in SEEDS]


def missing_diag(runs):
    have = set()
    for r in runs:
        m = r["meta"]
        if m["is_adaptive"]:
            have.add(("adaptive", m["task"], cap_key(m), kernel_key(m), m["seed"]))
        else:
            have.add(("fixed", m["task"], m["n"], int(m["fixed_r"]), m["seed"]))
    return [e for e in diag_roster() if e not in have]


def missing_oat(runs):
    have = set()
    for r in runs:
        m = r["meta"]
        if not m["is_adaptive"]:
            continue
        key, _, nk = oat_variant(m)
        if nk <= 1:
            have.add((m["task"], key, m["seed"]))
    return [e for e in oat_roster() if e not in have]


def render_missing(md, mo, n_diag, n_oat):
    L = ["## Missing files", ""]
    L.append(f"d3_diag: {n_diag} usable files read; roster {len(diag_roster())} entries, "
             f"{len(md)} missing. d3_oat: {n_oat} usable files read; roster "
             f"{len(oat_roster())} entries, {len(mo)} missing.")
    L.append("")
    L.append("Note: the diagnosis roster is the full cross of kernel variant × cap cell, but the "
             "campaign only commits to some of these combinations (n=64 with fixed c=8 + forced pre-projection, "
             "n=128 with natural c=16), so some missing entries may be runs that were never planned.")
    L.append("")
    if md:
        L.append("**d3_diag missing**")
        L.append("")
        byc = defaultdict(list)
        for e in md:
            byc[(e[0], e[1], e[2])].append((e[3], e[4]))
        for k in sorted(byc, key=lambda x: (x[0], str(x[1]), str(x[2]))):
            if k[0] == "adaptive":
                items = ", ".join(f"{KERNEL_LABEL.get(a, a)}/s{b}"
                                  for a, b in sorted(byc[k], key=str))
                L.append(f"- adaptive · {k[1]} · {cap_disp(k[2])}: {items}")
            else:
                items = ", ".join(f"r{a}/s{b}" for a, b in sorted(byc[k]))
                L.append(f"- fixed · {k[1]} · n={k[2]}: {items}")
        L.append("")
    else:
        L.append("d3_diag roster complete.")
        L.append("")
    if mo:
        L.append("**d3_oat missing**")
        L.append("")
        byt = defaultdict(lambda: defaultdict(list))
        for t, v, s in mo:
            byt[t][v].append(s)
        for t in sorted(byt):
            items = "; ".join(f"{v}(s{','.join(map(str, sorted(ss)))})"
                              for v, ss in sorted(byt[t].items()))
            L.append(f"- **{t}**: {items}")
        L.append("")
    else:
        L.append("d3_oat roster complete.")
        L.append("")
    return "\n".join(L)


def render_tolerance(bad_diag, bad_oat):
    L = ["## Field tolerance", ""]
    if not TOLERANCE:
        L.append("All runs carry the fields added for D3 (frac_at_cap / n_up / n_down / n_ctrl_checks / "
                 "rank_changes / --c / --force_preproject / --preproject); no fallback was used.")
    else:
        L.append("The fields below are missing in some runs and fall back to the D3 definitions. "
                 "Numbers obtained by a fallback carry a `*` in the tables:")
        L.append("")
        for k, n in sorted(TOLERANCE.items(), key=lambda kv: -kv[1]):
            L.append(f"- `{k}` — {n} runs — {TOL_NOTE.get(k, '')}")
    L.append("")
    L.append("How the fallbacks are computed:")
    L.append("")
    L.append("- `frac_at_cap`: the top-level field is counted per step. The fallback can only use the "
             "`rank` checkpoints in `records`, one every `log_every` steps. Its resolution is much lower "
             "and it can over- or underestimate brief contact with the cap, so treat it as an order of magnitude only.")
    L.append("- `n_up`/`n_down`: the controller can change the rank every `K` steps, but the log records it only "
             "every `log_every` steps, so the fallback count is a lower bound (an up and a down move within one "
             "logging interval cancel).")
    L.append("- The trigger (τ_c dominant / τ_r dominant) and the normalised τ_c, τ_r exist only in "
             "`rank_changes`. There is no substitute, so when they are missing they are reported as `n/a` and not inferred.")
    L.append("- `peak_MB` is written as null by run_adaptive.py under `--device cpu`; this is expected.")
    L.append("- The kernel variant is decided from the `--preproject` / `--c` arguments first. If "
             "`c_effective` / `preproject_effective` (written back by run_adaptive.py) are present, they are "
             "added as a note on the value the kernel actually used.")
    if bad_diag or bad_oat:
        L.append("")
        L.append("**Files that cannot be parsed or have no records** (possibly still being written):")
        L.append("")
        for p, why in list(bad_diag) + list(bad_oat):
            L.append(f"- `{os.path.basename(p)}`: {why}")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# numbers-only bullets
# --------------------------------------------------------------------------------------
def render_summary(ta, mra, tb, target_cos, err_tol):
    B = []
    for key in sorted(ta["adaptive"], key=_diag_sort_key):
        task, cap, ker = key
        a = ta["adaptive"][key]
        if a["frac_at_cap"][0] is None and a["avg_rank"][0] is None:
            continue
        star = "*" if "frac_at_cap" in a["flags"] else ""
        star_ud = "*" if "n_up_down" in a["flags"] else ""
        B.append(f"{task} / {cap_disp(cap)} / {kernel_disp(ker)}: frac_at_cap "
                 f"{fmt_pct(a['frac_at_cap'][0])}{star}, mean rank "
                 f"{fmt(a['avg_rank'][0], 2)} (cap {cap[1]}), tail error "
                 f"{fmt(a['err'][0], 5)}, n_up/n_down {fmt(a['n_up'][0], 1)}/"
                 f"{fmt(a['n_down'][0], 1)}{star_ud}, peak memory "
                 f"{fmt(a['peak_MB'][0], 1)} MB")
        t = a["trig"]
        if t and t["frac_tau_c"] is not None:
            B.append(f"{task} / {cap_disp(cap)} / {kernel_disp(ker)}: "
                     f"{t['n_changes']} rank changes in total, τ_c dominant in {fmt_pct(t['frac_tau_c'])}"
                     f" (among rank increases {fmt_pct(t['frac_tau_c_up'])}), normalised τ_c mean "
                     f"{fmt(t['tau_c_norm'], 4)}, τ_r mean {fmt(t['tau_r_norm'], 4)}")
    for (task, n), d in sorted(mra.items()):
        ctrl = [a["avg_rank"][0] for a in d["ctrl"].values() if a["avg_rank"][0] is not None]
        s = (f"{task}, n={n}: in the fixed-rank sweep the smallest r reaching grad_cos ≥ {target_cos:g} = "
             f"{d['r_cos'] if d['r_cos'] is not None else 'n/a'}; smallest r with tail error ≤ "
             f"(1+{err_tol:g})×best ({fmt(d['best_err'], 5)}) = "
             f"{d['r_err'] if d['r_err'] is not None else 'n/a'}; completed ranks {d['ranks']}")
        if ctrl:
            s += f"; controller mean rank for the same task and width {fmt(min(ctrl), 2)}–{fmt(max(ctrl), 2)}"
        B.append(s)
    cells = tb["cells"]
    for t in _oat_task_order(cells):
        es = {k[1]: cells[k]["err"][0] for k in cells
              if k[0] == t and cells[k]["err"][0] is not None}
        rk = {k[1]: cells[k]["avg_rank"][0] for k in cells
              if k[0] == t and cells[k]["avg_rank"][0] is not None}
        if es:
            lo, hi = min(es, key=es.get), max(es, key=es.get)
            if abs(es[hi] - es[lo]) < 5e-6:
                B.append(f"OAT {t}: the tail errors of all {len(es)} variants agree to 5 decimals"
                         f" ({fmt(es[lo], 5)}), range {fmt(es[hi] - es[lo], 5)}")
            else:
                B.append(f"OAT {t}: tail error of the {len(es)} variants goes from {fmt(es[lo], 5)} ({lo}) to "
                         f"{fmt(es[hi], 5)} ({hi}), range {fmt(es[hi] - es[lo], 5)}")
        if rk:
            lo, hi = min(rk, key=rk.get), max(rk, key=rk.get)
            if abs(rk[hi] - rk[lo]) < 5e-3:
                B.append(f"OAT {t}: the mean ranks of all {len(rk)} variants are equal ({fmt(rk[lo], 2)}), "
                         f"range {fmt(rk[hi] - rk[lo], 2)}")
            else:
                B.append(f"OAT {t}: mean rank goes from {fmt(rk[lo], 2)} ({lo}) to {fmt(rk[hi], 2)}"
                         f" ({hi}), range {fmt(rk[hi] - rk[lo], 2)}")
    if not B:
        B = ["No completed result files yet, so no numeric summary can be generated."]
    return "\n".join(f"- {b}" for b in B)


# --------------------------------------------------------------------------------------
# figures (reuses figstyle.py: Okabe-Ito palette, print-physical
# sizes, legend outside the axes, _save -> both .pdf and .png)
# --------------------------------------------------------------------------------------
# _kernel_style (colour+marker from the kernel) and _cap_ls (dash from r_max) used to
# provide the series style here.  Together they broke the two-attribute rule: two series with the
# same kernel shared both colour and marker and differed only in the dash pattern, so they
# were indistinguishable in grey-scale print.  fig_style_r1.distinct_table() now issues one
# flat table over the series that actually appear and asserts the >=2-attribute guarantee.


def _fig_var_label(v):
    """Compact ASCII tick label for an OAT variant (the full key stays in the table)."""
    return (v.replace("tau_high=", "th=").replace("tau_low=", "tl=")
             .replace("interval", "").replace("multi:", "multi "))


def rank_traj(runs_):
    """Mean +- s.d. over seeds of the logged running-mean rank, on the common step grid."""
    series = []
    for r in runs_:
        pts = {}
        for x in r["data"].get("records", []):
            v = x.get("avg_rank", x.get("rank"))
            if v is not None and x.get("step") is not None:
                pts[x["step"]] = v
        if pts:
            series.append(pts)
    if not series:
        return [], [], []
    common = set(series[0])
    for s in series[1:]:
        common &= set(s)
    steps = sorted(common) if common else sorted(series[0])
    xs, mu, sd = [], [], []
    for s in steps:
        m, d, _ = mean_std([ser[s] for ser in series if s in ser])
        if m is not None:
            xs.append(s); mu.append(m); sd.append(d or 0.0)
    return xs, mu, sd


def fig_adaptive_diag(ra, plt, np, diag_runs, task="rotation"):
    name = "fig_r2_adaptive_diag"
    groups = defaultdict(list)
    for r in diag_runs:
        m = r["meta"]
        if m["is_adaptive"] and m["task"] == task:
            groups[(m["n"], m["r_max"], bool(m["forced_pp"]), kernel_key(m))].append(r)
    if not groups:
        return name, f"no adaptive d3_diag runs for {task}"
    widths = sorted({k[0] for k in groups if k[0] is not None})
    panels = [w for w in (64, 128) if w in widths] or widths
    # Drawn at the width the paper prints it at (0.49\linewidth of a figure* = 3.37 in),
    # not at 0.96 x \textwidth: the old size was scaled down by 0.53 on the page, which
    # printed the 7.5 pt labels at 4 pt.  Figure 5 is the reference for this rule.
    fs.register(ra, name, height=2.10)
    fig, axs = plt.subplots(1, len(panels), figsize=ra.figsize(name), squeeze=False)
    axes = list(axs[0])
    # The old table took both colour and marker from the kernel and only the dash
    # pattern from r_max, so "fixed c=8, r_max=32" and "fixed c=8, r_max=64, force_pp"
    # shared two of the three visual attributes.  One flat table over the series that
    # actually appear, asserted pairwise-distinct in >= 2 attributes, removes that.
    series_keys = sorted({(k[1], bool(k[2]), k[3]) for k in groups}, key=str)
    SER = fs.distinct_table(series_keys, oi=ra.OI)
    for ax, w in zip(axes, panels):
        caps_here = sorted({k[1] for k in groups if k[0] == w and k[1] is not None})
        drew = False
        for si, key in enumerate(sorted([k for k in groups if k[0] == w], key=str)):
            _, rmax, fpp, ker = key
            xs, mu, sd = rank_traj(groups[key])
            if not xs:
                continue
            c, _ser_ls, mk = SER[(rmax, bool(fpp), ker)]
            # the cap is always named: the key is merged across panels, so a label that
            # omitted it would collide with the same kernel variant at another cap
            lbl = KERNEL_LABEL_EN.get(ker, ker) + f", $r_{{\\max}}$={rmax}"
            if fpp:
                lbl += ", force_pp"
            # stagger the marker phase per series so curves that coincide stay visible
            me = max(1, len(xs) // 8)
            ax.plot(xs, mu, color=c, linestyle=_ser_ls, marker=mk,
                    markevery=(si * max(1, me // 6) % me, me), label=lbl)
            if any(s > 0 for s in sd):
                ax.fill_between(xs, [a - b for a, b in zip(mu, sd)],
                                [a + b for a, b in zip(mu, sd)],
                                color=c, alpha=0.16, linewidth=0)
            drew = True
        for rmax in caps_here:
            ax.axhline(rmax, color=ra.OI["black"], linestyle=(0, (1, 3)), linewidth=0.7)
            ax.annotate(f"$r_{{\\max}}$={rmax}", xy=(0.985, rmax),
                        xycoords=("axes fraction", "data"), ha="right", va="bottom",
                        fontsize=6.0, color=ra.OI["black"])
        if drew and caps_here:
            # headroom above the highest cap (set after axhline, which re-autoscales) so the
            # r_max annotation never collides with the top frame
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, max(hi, max(caps_here) * 1.5))
        ax.set_title(f"{task}, n={w}")
        ax.set_xlabel("step")
        ax.set_ylabel("mean rank $r_t$")
        if drew:
            ax.set_yscale("log", base=2)
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
            # 1.55 in panels cannot carry five 5-digit step labels at 6.8 pt
            ax.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(4))
    fig.tight_layout()
    h, l = ra._merged_handles(axes)
    if h:
        fs.legend_below_fit(fig, axes, h, l, ncol=min(2, len(l)), name=name,
                            handlelength=1.9, columnspacing=0.7, handletextpad=0.35)
    ra._save(fig, name)
    return name, None


def fig_adaptive_oat(ra, plt, np, tb):
    name = "fig_r2_adaptive_oat"
    cells = tb["cells"]
    tasks = _oat_task_order(cells)
    if not tasks:
        return name, "no d3_oat results"
    # This figure is printed at 0.49\linewidth of a figure* = 3.37 in (Figure-5 rule:
    # authored at print size).  In the old orientation the OAT variants were x ticks, and
    # 12 rotated labels need >= 1.2 in of axis per panel -- five of those plus two y-axis
    # gutters each is ~10 in, which is why the figure was drawn at 6.5 in and then scaled
    # down by 0.52, printing its 5.4 pt tick labels at 2.8 pt (unreadable in print).
    # The panels are therefore transposed: the variants run down a single shared
    # y axis, labelled once and horizontally, and each task keeps its own panel with its
    # own error scale (bottom axis) and rank scale (top axis).  Same tasks, same
    # variants, same two quantities, same per-task scaling -- only the axes swap roles.
    all_keys = _oat_variant_order(cells, tasks[0])
    for t in tasks[1:]:                        # union, in first-task order then extras
        for v in _oat_variant_order(cells, t):
            if v not in all_keys:
                all_keys.append(v)
    row_h = max(0.115 * len(all_keys) + 0.95, 2.05)
    ra.PRINT_SIZE[name] = fs.print_size(name, height=row_h)
    fig, axs = plt.subplots(1, len(tasks), figsize=ra.figsize(name), squeeze=False,
                            sharey=True)
    axes = list(axs[0])
    c_err, c_rk = ra.OI["blue"], ra.OI["verm"]
    DY = 0.15          # vertical offset: error below the tick, mean rank above it
    ypos = {v: i for i, v in enumerate(all_keys)}
    twins = []
    for ci, (ax, t) in enumerate(zip(axes, tasks)):
        keys = _oat_variant_order(cells, t)
        axr = ax.twiny()
        twins.append(axr)
        for v in keys:
            a = cells[(t, v)]
            i = ypos[v]
            filled = (v == "default")
            # The default row carries two visual channels, not one.  Fill alone
            # (filled = default, open = variant) is a single attribute, and the
            # rule behind fig_style_r1.distinct_table asks for two; size is the second,
            # and it is Figure 5's own vocabulary rather than a new invention --
            # FIG5["lines"]["point_markersize"] is 7.0 for "a single highlighted point"
            # against a 3.6 default, which is exactly what the default row is here.
            ms = 6.2 if filled else 3.6
            if a["err"][0] is not None:
                ax.errorbar([a["err"][0]], [i - DY], xerr=[a["err"][1] or 0.0], color=c_err,
                            marker="o", markerfacecolor=(c_err if filled else "none"),
                            markersize=ms, linestyle="none", capsize=1.8, elinewidth=0.7,
                            markeredgewidth=0.9)
            if a["avg_rank"][0] is not None:
                axr.errorbar([a["avg_rank"][0]], [i + DY], xerr=[a["avg_rank"][1] or 0.0],
                             color=c_rk, marker="s",
                             markerfacecolor=(c_rk if filled else "none"),
                             markersize=ms, linestyle="none", capsize=1.8, elinewidth=0.7,
                             markeredgewidth=0.9)
        ax.set_yticks(list(range(len(all_keys))))
        ax.set_yticklabels([_fig_var_label(v) for v in all_keys], fontsize=6.0)
        ax.set_ylim(-0.8, len(all_keys) - 0.2)
        # No set_title here: matplotlib lifts a title clear of the top axis
        # *per panel*, and the rotated rank tick labels have different heights, so
        # the five titles came out at five different heights.  They are placed by
        # hand below, on one line, after the layout is final.
        # A 0.5 in wide panel holds two numeric ticks at most, and only if they are
        # rotated so that neighbouring labels cannot touch.
        for a_, col, lab in ((ax, c_err, "final error"), (axr, c_rk, "mean rank")):
            a_.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(2))
            a_.tick_params(axis="x", colors=col, labelrotation=90, labelsize=6.0)
            if ci == 0:
                a_.set_xlabel(lab, color=col)
        axr.set_ylim(ax.get_ylim())
        axr.grid(False)
        ax.margins(x=0.22)
        axr.margins(x=0.22)
    # The key states both channels the default row differs by (fill and size), so the
    # reader is not left to infer the second one from the panels.
    handles = [
        plt.Line2D([], [], color=c_err, marker="o", linestyle="none", markersize=6.2,
                   markerfacecolor=c_err, label="final error, default (filled, large)"),
        plt.Line2D([], [], color=c_err, marker="o", linestyle="none", markersize=3.6,
                   markerfacecolor="none", label="final error, variant (open, small)"),
        plt.Line2D([], [], color=c_rk, marker="s", linestyle="none", markersize=6.2,
                   markerfacecolor=c_rk, label="mean rank, default (filled, large)"),
        plt.Line2D([], [], color=c_rk, marker="s", linestyle="none", markersize=3.6,
                   markerfacecolor="none", label="mean rank, variant (open, small)"),
    ]
    fig.tight_layout()
    # one title row, all five at the same height, above everything the panels drew
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    # NB: the mean-rank axis is a separate Axes (twiny), and its rotated tick labels sit
    # above the panel -- so it has to be measured too, or the titles land on top of them.
    _top = max(a.get_tightbbox(_r).transformed(fig.transFigure.inverted()).y1
               for a in list(axes) + twins)
    for ax, t in zip(axes, tasks):
        p = ax.get_position()
        fig.text(0.5 * (p.x0 + p.x1), _top + 0.012, OAT_TASK_LABEL.get(t, t),
                 ha="center", va="bottom", fontsize=7.5)
    fs.legend_below_fit(fig, axes, handles, [h.get_label() for h in handles], ncol=2,
                        name=name, pad=0.05, handlelength=1.4, columnspacing=0.7,
                        handletextpad=0.35)
    ra._save(fig, name)
    return name, None


def make_figures(diag_runs, tb, figdir, task_diag="rotation"):
    names = ("fig_r2_adaptive_diag", "fig_r2_adaptive_oat")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import figstyle as ra
        import fig_style_r1 as fs
        globals()["fs"] = fs
        fs.bind(ra)
        fs.apply_rc(plt)
    except Exception as e:
        return [], [(n, f"plotting dependencies unavailable: {type(e).__name__}: {e}") for n in names]
    ra.OUT = figdir
    ra.GRAY = ""
    os.makedirs(figdir, exist_ok=True)
    wrote, skipped = [], []
    jobs = [(fig_adaptive_diag, lambda: fig_adaptive_diag(ra, plt, np, diag_runs,
                                                          task=task_diag)),
            (fig_adaptive_oat, lambda: fig_adaptive_oat(ra, plt, np, tb))]
    for fn, call in jobs:
        try:
            nm, why = call()
        except Exception as e:
            skipped.append((fn.__name__, f"exception: {type(e).__name__}: {e}"))
            continue
        if why:
            skipped.append((nm, why))
        else:
            wrote.append(nm)
    return wrote, skipped


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Aggregate and plot the D3 adaptive-rank results (saturation diagnosis + OAT sensitivity)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir_diag", default=os.path.join("results", "r2", "d3_diag"))
    ap.add_argument("--indir_oat", default=os.path.join("results", "r2", "d3_oat"))
    ap.add_argument("--outmd", default=None, help="default: <parent of indir_diag>/D3_SUMMARY.md")
    ap.add_argument("--figdir", default=None,
                    help="default: <code>/results/figures")
    ap.add_argument("--target_cos", type=float, default=0.95,
                    help="grad_cos target of the fixed-rank sweep (default 0.95)")
    ap.add_argument("--err_tol", type=float, default=0.05,
                    help="relative error target of the fixed-rank sweep: error <= (1+tol) x best in the sweep (default 0.05)")
    ap.add_argument("--fig_task", default="rotation", help="task for the rank-trajectory figure (default rotation)")
    ap.add_argument("--no_figs", action="store_true")
    ap.add_argument("--skip_oat", action="store_true",
                    help="diagnosis part only: do not read --indir_oat, mark Table B / the OAT figure as pending"
                         " (for use while the 90 OAT runs are incomplete, so partial data is not taken as a result)")
    args = ap.parse_args()

    indir_diag, indir_oat = args.indir_diag, args.indir_oat
    outmd = args.outmd or os.path.join(
        os.path.dirname(os.path.normpath(os.path.abspath(indir_diag))),
        "D3_SUMMARY.md")
    figdir = args.figdir or os.path.join("results", "figures")

    diag_runs, bad_diag = load_dir(indir_diag)
    oat_runs, bad_oat = ([], []) if args.skip_oat else load_dir(indir_oat)

    ta = build_table_a(diag_runs)
    mra = min_rank_analysis(ta, args.target_cos, args.err_tol)
    tb = build_table_b(oat_runs)
    md = missing_diag(diag_runs)
    mo = missing_oat(oat_runs)

    wrote, skipped = (([], []) if args.no_figs
                      else make_figures(diag_runs, tb, figdir, task_diag=args.fig_task))

    L = ["# D3 adaptive rank: saturation diagnosis + sensitivity — aggregated report", ""]
    L += [f"Diagnosis data directory: `{_show(indir_diag)}` ({len(diag_runs)} usable files)", "",
          (f"OAT data directory: `{_show(indir_oat)}` — **pending (--skip_oat: not read in this run, no OAT output)**"
           if args.skip_oat else
           f"OAT data directory: `{_show(indir_oat)}` ({len(oat_runs)} usable files)"), "",
          f"Criteria: grad_cos target {args.target_cos:g}; relative error target ≤ "
          f"(1+{args.err_tol:g})×best in the sweep. Tail = mean over the last 20% of logged checkpoints.", ""]
    L.append(render_missing(md, mo, len(diag_runs), len(oat_runs)))
    L.append(render_table_a(ta, mra, args.target_cos, args.err_tol))
    L.append("")
    if args.skip_oat:
        L += ["## Table B OAT one-at-a-time sensitivity (D3 local sensitivity)", "",
              "**Pending — this run used `--skip_oat` and aggregates only the diagnosis part (d3_diag).**",
              "", "The OAT sensitivity runs (`" + indir_oat + "`, roster " + str(len(oat_roster())) +
              " entries) are still queued. This report contains no OAT numbers. "
              "Rerun this script without `--skip_oat` once they are complete.", ""]
    else:
        L.append(render_table_b(tb))
    L.append("")
    L += ["## Key numbers (auto-generated, numbers only)", "",
          render_summary(ta, mra, tb, args.target_cos, args.err_tol), ""]
    L.append(render_tolerance(bad_diag, bad_oat))
    L += ["## Figures", ""]
    for nm in wrote:
        fp = os.path.join(figdir, nm + ".pdf")
        try:
            fp = os.path.relpath(fp)
        except ValueError:
            fp = os.path.abspath(fp)
        L.append(f"- `{_show(fp)}` — written")
    for nm, why in skipped:
        L.append(f"- `{nm}` — skipped: {why}")
    if args.no_figs:
        L.append("- (--no_figs: no figures drawn in this run)")
    L.append("")

    os.makedirs(os.path.dirname(os.path.abspath(outmd)) or ".", exist_ok=True)
    with open(outmd, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")

    print("wrote", outmd)
    print(f"diag: {len(diag_runs)} runs read, {len(md)}/{len(diag_roster())} roster missing, "
          f"{len(bad_diag)} unreadable")
    print(f"oat : {len(oat_runs)} runs read, {len(mo)}/{len(oat_roster())} roster missing, "
          f"{len(bad_oat)} unreadable")
    if TOLERANCE:
        print("field fallbacks used:", dict(TOLERANCE))
    print("figures:", wrote, "skipped:", [n for n, _ in skipped])


if __name__ == "__main__":
    main()
