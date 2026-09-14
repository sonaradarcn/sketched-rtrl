"""E4b (D4b on the reviewer's own numbering) common-trajectory report.

Reads the 27 run jsons written by `run_e4b_common.py` (9 tasks x seeds {0,1,2}, n=64,
20000 steps, exact RTRL trains a single trajectory per (task, seed) and every other
estimator is attached as a passive tracker that never updates a parameter -- see that
script's docstring and `R2_EXPERIMENT_PLAN.md` D4's "common-trajectory tracking
comparison" addition) and compares them against `results/r2/D4_TABLES.md` (each
estimator trains, and is measured on, its OWN trajectory; 10 seeds).

Purpose: D4 confounds "how accurate is estimator X" with "where did estimator X's own
training take the network" (SnAp-1's own trajectory drifts differently from SK-RTRL's).
E4b removes the confound: every estimator here sees the identical (A_t, imm_t, delta_t)
and the identical lane resets on the SAME exact-RTRL trajectory, so any grad_cos
difference from D4's own-trajectory number is attributable to the trajectory-choice
confound, not to the estimator itself.

Per-run value: `tail_summary[tracker]["grad_cos" | "rel_err"]` from the run json --
already the mean over the trailing `tail_frac=0.2` log points (steps >= 0.8*args.steps
with log_every=250 giving 80 log points total, so this is the same "last 20% of
training" window D4_TABLES.md's `window20` aggregation uses; the two are directly
comparable without re-deriving anything from `records`).
Across seeds (3, e4b) / (10, D4): mean +/- POPULATION std (ddof=0) -- D4_TABLES.md's own
convention, kept here for apples-to-apples deltas.

Win/loss (SK-RTRL rank vs baseline, per task): paired by seed on the SAME run json (both
trackers see the identical trajectory in the identical seed's file, so the pairing is
exact, not an assumption) -- diff_s = cos(skrtrl) - cos(baseline) for s in {0,1,2}; a
paired bootstrap (10000 resamples with replacement over the 3 seeds) 95% CI on mean(diff)
decides win (CI lower bound > 0), loss (CI upper bound < 0), or mixed (CI straddles 0).
Three seeds make this a coarse instrument (a unanimous 3/3 sign is the practical floor for
a bootstrap CI to exclude zero); mixed is reported honestly, not resolved by eyeballing.

Spearman: per task, rank the 7 common trackers by grad_cos under e4b vs under D4 and
report rho (n=7); also one pooled rho over all 63 (task, tracker) points.

Output: results/r2/E4B_SUMMARY.md.  Reads only; touches no experiment code, no GPU, no
paper/response-letter files.

Usage
  python make_r2_e4b_report.py
  python make_r2_e4b_report.py --e4b_dir results/r2/e4b --d4_tables results/r2/D4_TABLES.md \
      --out results/r2/E4B_SUMMARY.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

TASKS = ["copy", "adding", "rotation", "anbn",            # diagnostic
         "henon", "mackeyglass", "lorenz",                # chaotic
         "sunspot", "laser"]                               # real
DIAG_TASKS = {"copy", "adding", "rotation", "anbn"}
CHAOS_TASKS = {"henon", "mackeyglass", "lorenz"}
REAL_TASKS = {"sunspot", "laser"}
SEEDS = [0, 1, 2]

# e4b tracker key -> (D4 estimator label as printed in D4_TABLES.md, short display name)
TRACKERS = [
    ("skrtrl-r32", "SK-RTRL r=32", "r32"),
    ("skrtrl-r16", "SK-RTRL r=16", "r16"),
    ("skrtrl-r4",  "SK-RTRL r=4",  "r4"),
    ("snap1",      "SnAp-1",       "snap1"),
    ("kfrtrl",     "KF-RTRL",      "kfrtrl"),
    ("rflo",       "RFLO",         "rflo"),
    ("uoro",       "UORO",         "uoro"),
]
RANKS = ["skrtrl-r4", "skrtrl-r16", "skrtrl-r32"]
BASELINES = ["snap1", "kfrtrl", "rflo", "uoro"]
KEY2LABEL = {k: lbl for k, lbl, _ in TRACKERS}
KEY2SHORT = {k: s for k, _, s in TRACKERS}

BOOT_N = 10000
BOOT_SEED = 0


# --------------------------------------------------------------------------------------
# e4b loading
# --------------------------------------------------------------------------------------
def load_e4b(e4b_dir):
    """-> per_run[(task, tracker, seed)] = {"grad_cos":.., "rel_err":..}; issues[list]"""
    per_run = {}
    issues = []
    found_tasks = set()
    for task in TASKS:
        for seed in SEEDS:
            path = os.path.join(e4b_dir, f"{task}_n64_s{seed}.json")
            if not os.path.exists(path):
                issues.append(f"MISSING file: {os.path.basename(path)}")
                continue
            try:
                d = json.load(open(path))
            except Exception as e:
                issues.append(f"UNREADABLE {os.path.basename(path)}: {e}")
                continue
            found_tasks.add(task)
            args = d.get("args", {})
            if args.get("steps") != 20000:
                issues.append(f"{os.path.basename(path)}: steps={args.get('steps')} (expected 20000)")
            ts = d.get("tail_summary", {})
            trackers_meta = set(d.get("meta", {}).get("trackers", []))
            for key, _, _ in TRACKERS:
                if key not in trackers_meta:
                    issues.append(f"{os.path.basename(path)}: tracker {key} not run")
                    continue
                v = ts.get(key)
                if not v:
                    issues.append(f"{os.path.basename(path)}: tracker {key} has no tail_summary")
                    continue
                gc, re_ = v.get("grad_cos"), v.get("rel_err")
                for name, val in (("grad_cos", gc), ("rel_err", re_)):
                    if val is None or not (isinstance(val, (int, float)) and math.isfinite(val)):
                        issues.append(f"{os.path.basename(path)}: {key} {name}={val}")
                per_run[(task, key, seed)] = {"grad_cos": gc, "rel_err": re_}
    missing_tasks = [t for t in TASKS if t not in found_tasks]
    if missing_tasks:
        issues.append(f"MISSING TASKS entirely: {missing_tasks}")
    return per_run, issues


def agg_seeds(per_run, task, key, field):
    vals = [per_run[(task, key, s)][field] for s in SEEDS
            if (task, key, s) in per_run and per_run[(task, key, s)][field] is not None]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / max(1, len(vals)))  # population, ddof=0
    return {"mean": m, "sd": sd, "n": len(vals)}


# --------------------------------------------------------------------------------------
# D4_TABLES.md parsing (grad_cos sections only)
# --------------------------------------------------------------------------------------
def _parse_table_block(text, start):
    """Given text and an index right after a '### ...' heading line, parse the
    contiguous markdown table that follows -> (header_cells, {row_label: [cells...]})."""
    lines = text[start:].splitlines()
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s:
            if rows:
                break
            continue
        if not s.startswith("|"):
            if rows:
                break
            continue
        rows.append(s)
    if not rows:
        return [], {}
    def cells(row):
        c = [x.strip() for x in row.strip("|").split("|")]
        return c
    header = cells(rows[0])[1:]
    out = {}
    for r in rows[2:]:  # skip header + separator
        c = cells(r)
        if not c:
            continue
        label = c[0].strip().strip("*").strip()
        out[label] = c[1:]
    return header, out


def _extract_meanstd(cell):
    if cell is None:
        return None
    s = cell.strip().strip("*").strip()
    if s in ("--", "-", ""):
        return None
    m = re.search(r"([\d.]+)\s*\+/-\s*([\d.]+)\s*\(n=(\d+)\)", s)
    if not m:
        return None
    return {"mean": float(m.group(1)), "sd": float(m.group(2)), "n": int(m.group(3))}


def load_d4_cosines(d4_tables_path):
    """-> d4[(task, estimator_label)] = {"mean":, "sd":, "n":} using D4_TABLES.md's own
    'mean +/- sd (n=N)' cells, from the three grad_cos sections (Table 3/6/7)."""
    text = open(d4_tables_path, encoding="utf-8").read()
    out = {}
    parse_errors = []

    def grab(heading, tasks):
        idx = text.find(heading)
        if idx < 0:
            parse_errors.append(f"heading not found: {heading!r}")
            return
        idx2 = idx + len(heading)
        header, rows = _parse_table_block(text, idx2)
        # header columns should be exactly `tasks` in some order
        col_of = {}
        for i, h in enumerate(header):
            for t in tasks:
                if t.lower() in h.lower():
                    col_of[t] = i
        missing = [t for t in tasks if t not in col_of]
        if missing:
            parse_errors.append(f"{heading!r}: columns not found for {missing} (header={header})")
        for label, cells in rows.items():
            for t in tasks:
                if t not in col_of or col_of[t] >= len(cells):
                    continue
                v = _extract_meanstd(cells[col_of[t]])
                if v is not None:
                    out[(t, label)] = v

    grab("### grad_cos, last-window mean per run, mean +/- std over seeds",
         ["copy", "adding", "rotation", "anbn"])
    grab("### grad_cos per system (the LaTeX table pools these into one column)",
         ["henon", "mackeyglass", "lorenz"])
    grab("### grad_cos per series", ["sunspot", "laser"])
    return out, parse_errors


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------
def paired_bootstrap_ci(diffs, n_boot=BOOT_N, seed=BOOT_SEED, alpha=0.05):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diffs.mean()), float(lo), float(hi)


def win_loss(per_run):
    """-> rows: list of dicts, one per (rank, baseline, task); plus a totals table."""
    rows = []
    for rank in RANKS:
        for base in BASELINES:
            for task in TASKS:
                diffs = []
                for s in SEEDS:
                    a = per_run.get((task, rank, s), {}).get("grad_cos")
                    b = per_run.get((task, base, s), {}).get("grad_cos")
                    if a is None or b is None:
                        continue
                    diffs.append(a - b)
                if len(diffs) < 3:
                    rows.append({"rank": rank, "base": base, "task": task, "n": len(diffs),
                                 "verdict": "n/a"})
                    continue
                mean_d, lo, hi = paired_bootstrap_ci(diffs)
                if lo > 0:
                    verdict = "win"
                elif hi < 0:
                    verdict = "loss"
                else:
                    verdict = "mixed"
                rows.append({"rank": rank, "base": base, "task": task, "n": len(diffs),
                             "mean_diff": mean_d, "ci_lo": lo, "ci_hi": hi,
                             "verdict": verdict})
    totals = defaultdict(lambda: {"win": 0, "loss": 0, "mixed": 0, "na": 0})
    for r in rows:
        k = (r["rank"], r["base"])
        v = r["verdict"]
        if v == "n/a":
            totals[k]["na"] += 1
        else:
            totals[k][v] += 1
    return rows, totals


def spearman_per_task(per_run, d4):
    out = {}
    all_e4b, all_d4 = [], []
    for task in TASKS:
        xs, ys = [], []
        for key, label, _ in TRACKERS:
            a = agg_seeds(per_run, task, key, "grad_cos")
            b = d4.get((task, label))
            if a is None or b is None:
                continue
            xs.append(a["mean"]); ys.append(b["mean"])
        if len(xs) >= 3:
            rho, p = stats.spearmanr(xs, ys)
            out[task] = {"rho": rho, "p": p, "n": len(xs)}
            all_e4b += xs; all_d4 += ys
        else:
            out[task] = None
    pooled = None
    if len(all_e4b) >= 3:
        rho, p = stats.spearmanr(all_e4b, all_d4)
        pooled = {"rho": rho, "p": p, "n": len(all_e4b)}
    return out, pooled


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def build_report(per_run, e4b_issues, d4, d4_parse_errors, out_path, e4b_dir, d4_path):
    lines = []
    A = lines.append
    A("# E4b (common-trajectory passive tracking) summary")
    A("")
    A(f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by `code/make_r2_e4b_report.py`.")
    A("")
    A("Protocol (see `run_e4b_common.py` docstring and `R2_EXPERIMENT_PLAN.md` D4's "
      "\"common-trajectory tracking comparison\" addition): one trajectory per (task, "
      "seed) is generated by **exact RTRL training**; SK-RTRL r in {4,16,32}, SnAp-1, "
      "UORO, KF-RTRL, RFLO are attached to it as **passive trackers** that receive the "
      "identical (A_t, imm_t, delta_t) and the identical lane resets and never update a "
      "parameter. This isolates estimator accuracy from the confound in the D4 own-"
      "trajectory tables, where each estimator trains -- and is graded on -- its own "
      "network. TBPTT is not applicable (no per-step ghat_t; windowed autograd gradient "
      "only).")
    A("")
    A(f"- e4b source: `{os.path.relpath(e4b_dir, HERE)}` -- {len(TASKS)} tasks x seeds "
      f"{SEEDS} x n=64, 20000 steps, per-run value = `tail_summary` (mean over the "
      "trailing 20% of logged steps, i.e. step >= 0.8*20000 -- same window convention "
      "as D4's `window20`).")
    A(f"- D4 (own-trajectory) reference: `{os.path.relpath(d4_path, HERE)}` -- 10 seeds "
      "(0-9), same window convention, values copied from its published grad_cos tables, "
      "not re-derived.")
    A("- across-seed aggregation: mean +/- population std (ddof=0), matching D4_TABLES.md.")
    A("")

    # ---- anomalies ----
    A("## Anomalies")
    A("")
    if not e4b_issues and not d4_parse_errors:
        A(f"None. All {len(TASKS) * len(SEEDS)} expected e4b run files "
          f"({len(TASKS)} tasks x {len(SEEDS)} seeds) are present, all 7 trackers ran in "
          "every file, and `grad_cos`/`rel_err` are finite in every trailing-window "
          "summary used below.")
    else:
        for iss in e4b_issues:
            A(f"- e4b: {iss}")
        for iss in d4_parse_errors:
            A(f"- D4_TABLES.md parse: {iss}")
    A("")
    A("Aside (not an anomaly in grad_cos/rel_err, not used below): on some (task, seed) "
      "the SK-RTRL/SnAp-1 certificate `e_t` overflows float32 well before the tail window "
      "(e.g. henon s0 reaches `e_t ~ 5.6e35`), consistent with the certificate-saturation "
      "behaviour already documented for D2/D4 without spectral clipping; it affects only "
      "`rho_g`/`tight`, which this report does not use.")
    A("")

    # ---- main table ----
    A("## Task x estimator: grad_cos and rel_err on the common trajectory (e4b, 3 seeds) "
      "vs. own trajectory (D4, 10 seeds)")
    A("")
    A("delta = e4b mean - D4 mean (positive: the estimator looks BETTER when scored on "
      "the exact-RTRL trajectory than on its own trained trajectory).")
    A("")
    for group_name, group_tasks in (("Diagnostic", ["copy", "adding", "rotation", "anbn"]),
                                    ("Chaotic", ["henon", "mackeyglass", "lorenz"]),
                                    ("Real time series", ["sunspot", "laser"])):
        A(f"### {group_name}")
        A("")
        A("| Task | Estimator | e4b cos (n=3) | D4 cos (n=10) | delta | e4b rel_err (n=3) |")
        A("|---|---|---|---|---|---|")
        for task in group_tasks:
            for key, label, _ in TRACKERS:
                e = agg_seeds(per_run, task, key, "grad_cos")
                r = agg_seeds(per_run, task, key, "rel_err")
                dref = d4.get((task, label))
                e_s = f"{fmt(e['mean'])} +/- {fmt(e['sd'])}" if e else "n/a"
                d_s = f"{fmt(dref['mean'])} +/- {fmt(dref['sd'])}" if dref else "n/a"
                delta_s = fmt(e["mean"] - dref["mean"]) if (e and dref) else "n/a"
                r_s = f"{fmt(r['mean'])} +/- {fmt(r['sd'])}" if r else "n/a"
                A(f"| {task} | {label} | {e_s} | {d_s} | {delta_s} | {r_s} |")
        A("")

    # ---- spearman ----
    rank_by_task, pooled = spearman_per_task(per_run, d4)
    A("## Ranking stability (Spearman rho, e4b cosine order vs. D4 cosine order, n=7 "
      "estimators per task)")
    A("")
    A("| Task | rho | p | n |")
    A("|---|---|---|---|")
    for task in TASKS:
        v = rank_by_task.get(task)
        if v is None:
            A(f"| {task} | n/a | n/a | n/a |")
        else:
            A(f"| {task} | {fmt(v['rho'])} | {fmt(v['p'], 4)} | {v['n']} |")
    if pooled:
        A(f"| **pooled (all tasks x estimators)** | {fmt(pooled['rho'])} | "
          f"{fmt(pooled['p'], 4)} | {pooled['n']} |")
    A("")

    # ---- win/loss ----
    rows, totals = win_loss(per_run)
    A("## SK-RTRL rank vs. baseline win/loss on the common trajectory (paired by seed, "
      "n=3 seeds, 95% paired-bootstrap CI on mean(cos_skrtrl - cos_baseline))")
    A("")
    A("win = CI lower bound > 0; loss = CI upper bound < 0; mixed = CI straddles 0 "
      "(with 3 seeds this typically means the 3 signs are not unanimous).")
    A("")
    A("### Totals (out of 9 tasks each)")
    A("")
    A("| Rank | Baseline | Win | Mixed | Loss | n/a |")
    A("|---|---|---|---|---|---|")
    for rank in RANKS:
        for base in BASELINES:
            t = totals[(rank, base)]
            A(f"| {KEY2LABEL[rank]} | {KEY2LABEL[base]} | {t['win']} | {t['mixed']} | "
              f"{t['loss']} | {t['na']} |")
    A("")
    A("### Per task detail")
    A("")
    A("| Rank | Baseline | Task | mean(diff) | 95% CI | verdict |")
    A("|---|---|---|---|---|---|")
    for r in rows:
        if r["verdict"] == "n/a":
            A(f"| {KEY2LABEL[r['rank']]} | {KEY2LABEL[r['base']]} | {r['task']} | n/a | "
              f"n/a | n/a (n={r['n']}) |")
        else:
            A(f"| {KEY2LABEL[r['rank']]} | {KEY2LABEL[r['base']]} | {r['task']} | "
              f"{fmt(r['mean_diff'])} | [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] | "
              f"{r['verdict']} |")
    A("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return rank_by_task, pooled, totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4b_dir", default=os.path.join(HERE, "results", "r2", "e4b"))
    ap.add_argument("--d4_tables", default=os.path.join(HERE, "results", "r2", "D4_TABLES.md"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "r2", "E4B_SUMMARY.md"))
    args = ap.parse_args()

    per_run, issues = load_e4b(args.e4b_dir)
    d4, d4_errs = load_d4_cosines(args.d4_tables)
    rank_by_task, pooled, totals = build_report(per_run, issues, d4, d4_errs, args.out,
                                                args.e4b_dir, args.d4_tables)

    print(f"wrote {args.out}")
    print(f"  e4b issues: {len(issues)}, D4_TABLES.md parse errors: {len(d4_errs)}")
    if pooled:
        print(f"  pooled Spearman rho (e4b vs D4 cosine ranking) = {pooled['rho']:.3f} "
              f"(p={pooled['p']:.4f}, n={pooled['n']})")
    for rank in RANKS:
        for base in BASELINES:
            t = totals[(rank, base)]
            print(f"  {rank:<11} vs {base:<7} win/mixed/loss/na = "
                  f"{t['win']}/{t['mixed']}/{t['loss']}/{t['na']}")


if __name__ == "__main__":
    main()
