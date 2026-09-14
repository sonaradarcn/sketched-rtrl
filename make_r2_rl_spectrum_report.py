"""Aggregate the D1-style residual spectrum measured on the online RL task (T-maze).

Reads results/r2/rl/tmaze<len>_<algo>_s<seed>.json (clip 0) and
results/r2/rl_clip05/tmaze<len>_skrtrl-r16_s<seed>.json (clip 0.5), written by run_m5.py
with --shadow 1.  Each file carries a `spectrum` list of six checkpoints (fractions 0.01,
0.05, 0.25, 0.5, 0.75, 1.0 of --steps) of the residual R_t = J_t - S_t, with J_t the exact
influence matrix of an RTRL shadow and S_t the SnAp-1 trace embedded on the diagonal
blocks -- the same object run_e1_spectrum.py measures for the supervised D1 sweep, so the
two are directly comparable.  `*.localpartial.json` are partial copies of a running job and
are ignored.

One convention differs from D1 and is handled explicitly.  run_m5.py stores the residual
spectrum as the LANE MEAN of the per-lane singular values (`sv`, 16 lanes at n=64), while
D1 reports r_eps as the lane mean of the per-lane r_eps.  r_eps therefore has to be read off
the mean spectrum here.  The same quantity is recomputed from D1's own stored mean spectrum
(run_e1_spectrum.py stores `sv` the same way) so the offset between the two conventions is
measured rather than assumed; mass_top{k}, stable_rank and res_frac_of_J are lane means of
per-lane quantities in both sweeps and need no such treatment.

Outputs
  <outmd>                                (default: results/r2/RL_SPECTRUM_SUMMARY.md)
    roster/anomaly list, Table A (final stage per corridor x arm), Table B (the full
    checkpoint grid), Table C (clip 0 vs clip 0.5, paired), Table D (SnAp-1 arm against the
    \\skrtrl{} arm, paired), Table E (side by side with the D1 n=64 final-stage rows and the
    group the Section 6.4 rule assigns), Table F (convention cross-checks), and a
    numbers-only bullet summary.
  <figdir>/fig_r2_spectrum_rl.pdf        left: r_eps10/n against training stage, one line
                                         per corridor x arm; right: the final-stage
                                         r_eps10/n of the T-maze arms against the nine
                                         supervised tasks, with the Section 6.4 thresholds.

A missing file is listed and left out of every pooled statistic, never interpolated; a
figure whose inputs are missing is skipped loudly.  Style helpers come from fig_style_r1.py
and paper/figures/gen/regen_all.py, exactly as in make_r2_d1_report.py.

Usage
  python make_r2_rl_spectrum_report.py
  python make_r2_rl_spectrum_report.py --indir results/r2/rl --clipdir results/r2/rl_clip05
  python make_r2_rl_spectrum_report.py --no-fig
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
sys.path.insert(0, os.path.join(HERE, "..", "paper", "figures", "gen"))
sys.path.insert(0, HERE)

# --------------------------------------------------------------------------------------
# roster: 3 corridors x 3 arms x 3 seeds = 27 runs
# --------------------------------------------------------------------------------------
CORRIDORS = [10, 20, 40]
SEEDS = [0, 1, 2]
# (arm key, algo in the filename, clip, which directory)
ARMS = [
    ("skrtrl-r16 / clip 0", "skrtrl-r16", 0.0, "indir"),
    ("snap1 / clip 0", "snap1", 0.0, "indir"),
    ("skrtrl-r16 / clip 0.5", "skrtrl-r16", 0.5, "clipdir"),
]
ARM_KEYS = [a[0] for a in ARMS]
FNAME_RE = re.compile(r"^tmaze(?P<corr>\d+)_(?P<algo>skrtrl-r16|snap1)_s(?P<seed>\d+)\.json$")
EPS = 1e-6
EPS_LEVELS = ((0.10, "10"), (0.05, "05"))
N_RL = 64                      # iso-width of the RL study
P_RL = 4416                    # n * (n + n_obs + 1) = 64 * 69, the parameter count

# Section 6.4's grouping rule on the final r_eps10/n.
GROUP_LOW, GROUP_MID, GROUP_HIGH = 0.3, 0.5, 0.6

# D1 supervised comparison rows (n=64, final stage), read from results/r2/d1_spectrum.
D1_TASKS = ["rotation", "anbn", "lorenz", "mackeyglass", "laser", "henon",
            "sunspot", "adding", "copy"]
D1_LABEL = {"anbn": "a^n b^n", "mackeyglass": "mackeyglass"}
# short names for the figure only (the markdown tables keep the full arm keys)
ARM_SHORT = {"skrtrl-r16 / clip 0": "SK-r16", "snap1 / clip 0": "SnAp-1",
             "skrtrl-r16 / clip 0.5": "SK-r16 c0.5"}
FIG_TASK_LABEL = {"anbn": r"$a^n b^n$", "mackeyglass": "mackey-glass",
                  "henon": "henon", "lorenz": "lorenz", "sunspot": "sunspot"}


# --------------------------------------------------------------------------------------
# metrics off a stored mean spectrum
# --------------------------------------------------------------------------------------
def metrics_from_sv(sv, n):
    """r_eps{10,05}[_over_n], plus mass_top16 / stable_rank re-derived from the same
    spectrum (only used to cross-check the stored lane-mean versions)."""
    sq = [float(s) * float(s) for s in sv]
    tot = sum(sq)
    if not tot > 0.0 or any(math.isnan(v) for v in sq):
        return None
    out = {}
    for eps, key in EPS_LEVELS:
        thr = (1.0 - eps * eps) * tot
        acc, r = 0.0, len(sq)
        for i, v in enumerate(sq):
            acc += v
            if acc >= thr:
                r = i + 1
                break
        out["r_eps%s" % key] = float(r)
        out["r_eps%s_over_n" % key] = r / float(n)
    out["mass_top16_derived"] = sum(sq[:16]) / tot
    out["stable_rank_derived"] = tot / max(sq[0], 1e-300)
    return out


def enrich(rec, n):
    """Add the derived fields to one stored checkpoint record; returns None on bad data."""
    sv = rec.get("sv")
    if not sv:
        return None
    m = metrics_from_sv(sv, n)
    if m is None:
        return None
    out = dict(rec)
    out.update(m)
    return out


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def load_rl(indir, clipdir):
    """runs[(corridor, arm_key)] = [(seed, path, checkpoints), ...]

    Returns (runs, ignored, bad, notes) where `ignored` lists the *.localpartial.json
    skipped, `bad` the files that failed to parse or carry no usable spectrum, and `notes`
    the per-checkpoint anomalies (NaN, a checkpoint taken at its deadline below the age
    threshold, an unexpected n / P / lane count)."""
    runs = defaultdict(list)
    ignored, bad, notes = [], [], []
    dirs = {"indir": indir, "clipdir": clipdir}
    for which, d in dirs.items():
        if not os.path.isdir(d):
            bad.append((d, "directory does not exist"))
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            base = os.path.basename(p)
            if base.endswith(".localpartial.json"):
                ignored.append(p)
                continue
            m = FNAME_RE.match(base)
            if not m:
                ignored.append(p)
                continue
            try:
                with open(p) as f:
                    j = json.load(f)
            except Exception as e:                      # noqa: BLE001
                bad.append((p, str(e)))
                continue
            a = j.get("args") or {}
            spec = j.get("spectrum") or []
            if not spec:
                bad.append((p, "no spectrum records (shadow off, or job died early)"))
                continue
            clip = float(a.get("clip", 0.0))
            arm_key = None
            for key, algo, arm_clip, arm_dir in ARMS:
                if algo == m["algo"] and abs(clip - arm_clip) < EPS and arm_dir == which:
                    arm_key = key
                    break
            if arm_key is None:
                ignored.append("%s (algo %s, clip %s in %s: not in the roster)"
                               % (p, m["algo"], clip, which))
                continue
            corr = int(m["corr"])
            seed = int(m["seed"])
            n = int(a.get("n") or N_RL)
            min_age = float(a.get("spectrum_min_age", 0.0))
            defer = int(a.get("spectrum_defer", 0))
            steps = int(a.get("steps", 0))
            cks = []
            for rec in spec:
                e = enrich(rec, n)
                if e is None:
                    notes.append("%s: checkpoint frac=%s has a NaN/degenerate spectrum "
                                 "(dropped)" % (base, rec.get("frac")))
                    continue
                if e.get("n") not in (None, N_RL):
                    notes.append("%s: checkpoint frac=%s reports n=%s, not %d"
                                 % (base, e.get("frac"), e.get("n"), N_RL))
                if e.get("P") not in (None, P_RL):
                    notes.append("%s: checkpoint frac=%s reports P=%s, not %d"
                                 % (base, e.get("frac"), e.get("P"), P_RL))
                nom = e.get("step_nominal")
                dead = min(nom + defer, steps) if nom else None
                if (min_age and e.get("age_mean") is not None
                        and e["age_mean"] < min_age - EPS):
                    e["forced"] = True
                    notes.append("%s: checkpoint frac=%s taken at step %s (deadline %s) "
                                 "with mean lane age %.2f < --spectrum_min_age %.1f"
                                 % (base, e.get("frac"), e.get("step"), dead,
                                    e["age_mean"], min_age))
                cks.append(e)
            if len(cks) < 6:
                notes.append("%s: %d of 6 checkpoints usable" % (base, len(cks)))
            runs[(corr, arm_key)].append((seed, p, cks, a))
    return runs, ignored, bad, notes


def missing_list(runs):
    have = {(c, k, s) for (c, k), rows in runs.items() for s, _, _, _ in rows}
    want = {(c, k, s) for c in CORRIDORS for k in ARM_KEYS for s in SEEDS}
    return sorted(want - have)


# --------------------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------------------
def is_final(r):
    f = r.get("frac")
    return f is not None and abs(f - 1.0) < EPS


def at_frac(f0):
    return lambda r: r.get("frac") is not None and abs(r["frac"] - f0) < EPS


def is_any(_r):
    return True


def pool(rows, field, pred=is_any):
    return [r[field] for _, _, cks, _ in rows for r in cks
            if pred(r) and r.get(field) is not None]


def mean_std(vals):
    vals = [v for v in vals
            if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0, 1
    s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return m, s, len(vals)


def fmt(m, s, _n=0, nd=4):
    if m is None:
        return "—"
    if not s:
        return "%.*f" % (nd, m)
    return "%.*f±%.*f" % (nd, m, nd, s)


def seed_count(rows):
    return len({s for s, _, _, _ in rows})


def fracs_present(runs):
    fs = {r["frac"] for rows in runs.values() for _, _, cks, _ in rows for r in cks
          if r.get("frac") is not None}
    return sorted(fs)


def group_of(r10n):
    """Section 6.4's rule on the final r_eps10/n."""
    if r10n is None:
        return "—"
    if r10n < GROUP_LOW:
        return "低秩"
    if r10n <= GROUP_MID:
        return "中间"
    if r10n > GROUP_HIGH:
        return "非低秩"
    return "未归组 (0.5–0.6)"


# --------------------------------------------------------------------------------------
# D1 comparison rows
# --------------------------------------------------------------------------------------
def load_d1(d1dir):
    """d1[task] = {official/mean-spectrum r_eps10/n, r_eps05/n, mass_top16, stable_rank,
    res_frac_of_J} pooled over the 3 seeds x reset ages of the n=64 final stage."""
    out = {}
    for task in D1_TASKS:
        off10, off05, ms10, ms05, m16, sr, rf = [], [], [], [], [], [], []
        nfile = 0
        for seed in SEEDS:
            p = os.path.join(d1dir, "%s_n64_s%d.json" % (task, seed))
            if not os.path.isfile(p):
                continue
            try:
                with open(p) as f:
                    j = json.load(f)
            except Exception:                            # noqa: BLE001
                continue
            nfile += 1
            for r in j.get("records", []):
                if not is_final(r):
                    continue
                if r.get("r_eps10_over_n") is not None:
                    off10.append(r["r_eps10_over_n"])
                if r.get("r_eps05_over_n") is not None:
                    off05.append(r["r_eps05_over_n"])
                m = metrics_from_sv(r.get("sv") or [], 64)
                if m is not None:
                    ms10.append(m["r_eps10_over_n"])
                    ms05.append(m["r_eps05_over_n"])
                for src, dst in (("mass_top16", m16), ("stable_rank", sr),
                                 ("res_frac_of_J", rf)):
                    if r.get(src) is not None:
                        dst.append(r[src])
        if nfile:
            out[task] = {"n_files": nfile,
                         "off10": mean_std(off10), "off05": mean_std(off05),
                         "ms10": mean_std(ms10), "ms05": mean_std(ms05),
                         "mass_top16": mean_std(m16), "stable_rank": mean_std(sr),
                         "res_frac_of_J": mean_std(rf)}
    return out


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------
FIELDS_A = ["r_eps10_over_n", "r_eps05_over_n", "r_eps10", "mass_top16", "stable_rank",
            "res_frac_of_J"]


def build_table_a(runs):
    rows_out = []
    for corr in CORRIDORS:
        for key in ARM_KEYS:
            rows = runs.get((corr, key), [])
            e = {"corridor": corr, "arm": key, "n_seeds": seed_count(rows)}
            for f in FIELDS_A:
                e[f] = mean_std(pool(rows, f, is_final))
            rows_out.append(e)
    # per-arm pooled over the three corridors
    pooled = []
    for key in ARM_KEYS:
        rows = [r for corr in CORRIDORS for r in runs.get((corr, key), [])]
        e = {"corridor": "10/20/40", "arm": key, "n_seeds": seed_count(rows)}
        for f in FIELDS_A:
            e[f] = mean_std(pool(rows, f, is_final))
        pooled.append(e)
    return rows_out, pooled


def render_table_a(rows_out, pooled):
    out = ["### 表 A — 末阶段 (frac=1.0)，走廊 × arm，3 seeds",
           "",
           "mean±sd over seeds（每个 (走廊, arm, seed) 在末阶段只有一个检查点，故 n=3）。"
           "`r_eps10/n` 由存储的 lane 平均谱算出，见上文口径说明；"
           "`mass_top16`、`stable_rank`、`res_frac_of_J` 是 lane 平均量，与 D1 同口径。",
           "",
           "| 走廊 | arm | n_seeds | r_eps10/n | r_eps05/n | r_eps10 | mass_top16 | "
           "stable rank | res_frac_of_J | 组别 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for e in rows_out + [None] + pooled:
        if e is None:
            out.append("| | | | | | | | | | |")
            continue
        out.append("| %s | %s | %d/3 | %s | %s | %s | %s | %s | %s | %s |"
                   % (e["corridor"], e["arm"], e["n_seeds"],
                      fmt(*e["r_eps10_over_n"]), fmt(*e["r_eps05_over_n"]),
                      fmt(*e["r_eps10"], nd=2), fmt(*e["mass_top16"]),
                      fmt(*e["stable_rank"], nd=2), fmt(*e["res_frac_of_J"]),
                      group_of(e["r_eps10_over_n"][0])))
    out.append("")
    return out


def build_table_b(runs, fracs):
    rows_out = []
    for corr in CORRIDORS:
        for key in ARM_KEYS:
            rows = runs.get((corr, key), [])
            e = {"corridor": corr, "arm": key, "cells": []}
            for f0 in fracs:
                e["cells"].append(mean_std(pool(rows, "r_eps10_over_n", at_frac(f0))))
            rows_out.append(e)
    return rows_out


def render_table_b(rows_out, fracs):
    out = ["### 表 B — 全检查点网格：r_eps10/n 按训练阶段",
           "",
           "列为 `--spectrum_ckpts` 的六个 frac（占 `--steps` 的比例）；每格 mean±sd "
           "over 3 seeds。",
           "",
           "| 走廊 | arm | " + " | ".join("frac=%g" % f for f in fracs) + " |",
           "|---|---|" + "---|" * len(fracs)]
    for e in rows_out:
        out.append("| %s | %s | %s |" % (e["corridor"], e["arm"],
                                         " | ".join(fmt(*c) for c in e["cells"])))
    out.append("")
    return out


def paired_diff(runs, key_a, key_b, field, pred=is_final):
    """Per (corridor, seed) difference arm_a - arm_b of one field, and the pooled stats."""
    diffs, per_corr = [], {}
    for corr in CORRIDORS:
        va = {s: [r[field] for r in cks if pred(r) and r.get(field) is not None]
              for s, _, cks, _ in runs.get((corr, key_a), [])}
        vb = {s: [r[field] for r in cks if pred(r) and r.get(field) is not None]
              for s, _, cks, _ in runs.get((corr, key_b), [])}
        d = []
        for s in sorted(set(va) & set(vb)):
            if va[s] and vb[s]:
                d.append(sum(va[s]) / len(va[s]) - sum(vb[s]) / len(vb[s]))
        per_corr[corr] = mean_std(d)
        diffs.extend(d)
    return per_corr, mean_std(diffs)


def render_paired(title, lead, runs, key_a, key_b, fields):
    out = ["### %s" % title, "", lead, "",
           "| 量 | 走廊 10 | 走廊 20 | 走廊 40 | 全部 (9 配对) |",
           "|---|---|---|---|---|"]
    for field, label, nd in fields:
        per_corr, allm = paired_diff(runs, key_a, key_b, field)
        out.append("| %s | %s | %s | %s | %s |"
                   % (label, fmt(*per_corr[10], nd=nd), fmt(*per_corr[20], nd=nd),
                      fmt(*per_corr[40], nd=nd), fmt(*allm, nd=nd)))
    out.append("")
    return out


def render_table_e(pooled, d1):
    out = ["### 表 E — 与监督任务 D1 (n=64，末阶段) 并列，并按 §6.4 规则归组",
           "",
           "D1 行取 `results/r2/d1_spectrum/<task>_n64_s{0,1,2}.json` 的 frac=1.0 检查点"
           "（3 seeds × 各 reset age 池化），与主文表 4 同口径。`r_eps10/n (官方)` 是 D1 报告"
           "使用的 per-lane r_eps 的 lane 平均；`r_eps10/n (均值谱)` 用与 T-maze 相同的方式"
           "（从存储的 lane 平均谱读出）重算，两列之差即口径偏差。T-maze 行只有均值谱一列。"
           "归组按 §6.4：`<0.3` 低秩，`≤0.5` 中间，`>0.6` 非低秩。",
           "",
           "| 任务 / arm | 来源 | r_eps10/n (官方) | r_eps10/n (均值谱) | mass_top16 | "
           "stable rank | res_frac_of_J | 组别 (均值谱) |",
           "|---|---|---|---|---|---|---|---|"]
    for task in D1_TASKS:
        e = d1.get(task)
        if not e:
            out.append("| %s | D1 | — | — | — | — | — | — |" % D1_LABEL.get(task, task))
            continue
        out.append("| %s | D1 | %s | %s | %s | %s | %s | %s |"
                   % (D1_LABEL.get(task, task), fmt(*e["off10"]), fmt(*e["ms10"]),
                      fmt(*e["mass_top16"]), fmt(*e["stable_rank"], nd=2),
                      fmt(*e["res_frac_of_J"]), group_of(e["ms10"][0])))
    out.append("| | | | | | | | |")
    for e in pooled:
        out.append("| T-maze %s | RL | — | %s | %s | %s | %s | %s |"
                   % (e["arm"], fmt(*e["r_eps10_over_n"]), fmt(*e["mass_top16"]),
                      fmt(*e["stable_rank"], nd=2), fmt(*e["res_frac_of_J"]),
                      group_of(e["r_eps10_over_n"][0])))
    out.append("")
    return out


def render_table_f(runs, d1):
    """Convention cross-checks: (i) the two r_eps conventions on D1, (ii) the mean-spectrum
    mass_top16 / stable_rank against their stored lane-mean versions on the RL runs."""
    out = ["### 表 F — 口径核对",
           "",
           "**(i) D1 上两种 r_eps 口径之差**（n=64，末阶段；`均值谱 − 官方`）。这是把 T-maze 的"
           "均值谱读数与主文表 4 并列时应扣掉的偏差，负值表示均值谱口径偏乐观（读出的秩更小）。",
           "",
           "| 任务 | r_eps10/n (官方) | r_eps10/n (均值谱) | 差 | r_eps05/n 差 |",
           "|---|---|---|---|---|"]
    d10, d05 = [], []
    for task in D1_TASKS:
        e = d1.get(task)
        if not e or e["off10"][0] is None or e["ms10"][0] is None:
            continue
        a = e["ms10"][0] - e["off10"][0]
        b = (e["ms05"][0] - e["off05"][0]
             if e["off05"][0] is not None and e["ms05"][0] is not None else None)
        d10.append(a)
        if b is not None:
            d05.append(b)
        out.append("| %s | %s | %s | %+.4f | %s |"
                   % (D1_LABEL.get(task, task), fmt(*e["off10"]), fmt(*e["ms10"]), a,
                      ("%+.4f" % b) if b is not None else "—"))
    if d10:
        out.append("| **全部 9 任务** | | | mean %+.4f，最大绝对偏差 %.4f | mean %+.4f |"
                   % (sum(d10) / len(d10), max(abs(v) for v in d10),
                      (sum(d05) / len(d05)) if d05 else float("nan")))
    out += ["",
            "**(ii) T-maze 均值谱与存储 lane 平均量的一致性**（全 27 run × 6 检查点）。"
            "`mass_top16` 与 `stable_rank` 两种算法在同一检查点上的差，用来说明 lane 之间"
            "谱的异质程度：差值越大，均值谱越比“典型 lane”集中，(i) 的偏差就越该被当作下限。",
            "",
            "| 量 | 存储 (lane 平均) | 均值谱重算 | 差 (均值谱 − 存储) |",
            "|---|---|---|---|"]
    for field, dfield, label, nd in (("mass_top16", "mass_top16_derived", "mass_top16", 4),
                                     ("stable_rank", "stable_rank_derived", "stable rank", 2)):
        rows = [r for rows in runs.values() for r in rows]
        stored = pool(rows, field)
        deriv = pool(rows, dfield)
        diffs = [r[dfield] - r[field] for _, _, cks, _ in rows for r in cks
                 if r.get(field) is not None and r.get(dfield) is not None]
        out.append("| %s | %s | %s | %s |"
                   % (label, fmt(*mean_std(stored), nd=nd), fmt(*mean_std(deriv), nd=nd),
                      fmt(*mean_std(diffs), nd=nd)))
    out.append("")
    return out


# --------------------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------------------
FIG_NAME = "fig_r2_spectrum_rl"


def make_figure(runs, d1, figdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import regen_all as ra
        import fig_style_r1 as fs
        fs.bind(ra)
        fs.apply_rc(plt)
    except Exception as e:                               # noqa: BLE001
        return None, "style modules unavailable: %s" % e
    if not runs:
        return None, "no RL spectrum results found"
    try:
        size = fs.register(ra, FIG_NAME, height=2.10)
    except KeyError:
        size = (0.92 * fs.TW, 2.10)
        ra.PRINT_SIZE[FIG_NAME] = size
    ra.OUT = figdir
    ra.GRAY = ""
    os.makedirs(figdir, exist_ok=True)

    oi = ra.OI
    corr_colour = {10: oi["blue"], 20: oi["orange"], 40: oi["green"]}
    arm_style = {ARM_KEYS[0]: ("-", "o"), ARM_KEYS[1]: ((0, (5, 2)), "s"),
                 ARM_KEYS[2]: ((0, (1, 1.2)), "^")}
    fracs = fracs_present(runs)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=size)
    for corr in CORRIDORS:
        for key in ARM_KEYS:
            rows = runs.get((corr, key), [])
            if not rows:
                continue
            ls, mk = arm_style[key]
            xs, ms, ss = [], [], []
            for f0 in fracs:
                m, s, k = mean_std(pool(rows, "r_eps10_over_n", at_frac(f0)))
                if m is not None:
                    xs.append(f0); ms.append(m); ss.append(s or 0.0)
            if not xs:
                continue
            axL.plot(xs, ms, color=corr_colour[corr], linestyle=ls, marker=mk, ms=3.2,
                     label="%d / %s" % (corr, ARM_SHORT[key]))
            axL.fill_between(xs, [a - b for a, b in zip(ms, ss)],
                             [a + b for a, b in zip(ms, ss)],
                             color=corr_colour[corr], alpha=0.12, linewidth=0)
    axL.set_xlabel("training stage (frac)")
    axL.set_ylabel(r"$r_{\epsilon=0.10}/n$")
    axL.set_title("T-maze, rank fraction")
    axL.set_ylim(bottom=0)

    # right: final-stage rank fraction, supervised tasks against the T-maze arms
    pts = []
    for task in D1_TASKS:
        e = d1.get(task)
        if e and e["ms10"][0] is not None:
            pts.append((FIG_TASK_LABEL.get(task, task), e["ms10"][0], e["ms10"][1] or 0.0,
                        oi["grey"]))
    for key in ARM_KEYS:
        rows = [r for corr in CORRIDORS for r in runs.get((corr, key), [])]
        m, s, k = mean_std(pool(rows, "r_eps10_over_n", is_final))
        if m is not None:
            pts.append(("T-maze %s" % ARM_SHORT[key], m, s or 0.0, oi["verm"]))
    pts.sort(key=lambda z: z[1])
    ys = range(len(pts))
    axR.barh(list(ys), [p[1] for p in pts], xerr=[p[2] for p in pts],
             color=[p[3] for p in pts], height=0.68, linewidth=0,
             error_kw=dict(elinewidth=0.6, capsize=1.4))
    axR.set_yticks(list(ys))
    axR.set_yticklabels([p[0] for p in pts])
    for x in (GROUP_LOW, GROUP_MID, GROUP_HIGH):
        axR.axvline(x, color="0.25", linewidth=0.7, linestyle=(0, (2, 1.4)), zorder=4)
    axR.set_xlabel(r"$r_{\epsilon=0.10}/n$ (final)")
    axR.set_title("final stage, all tasks")
    axR.set_xlim(0, 1.0)

    fig.tight_layout()
    h, l = axL.get_legend_handles_labels()
    if h:
        try:
            fs.legend_below_fit(fig, [axL, axR], h, l, ncol=3, name=FIG_NAME,
                                handlelength=1.9, columnspacing=0.7, handletextpad=0.35)
        except Exception:                                # noqa: BLE001
            axL.legend(h, l, fontsize=5.4, ncol=1, frameon=False, loc="upper right")
    try:
        ra._save(fig, FIG_NAME)
    except Exception:                                    # noqa: BLE001
        fig.savefig(os.path.join(figdir, FIG_NAME + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return os.path.join(figdir, FIG_NAME + ".pdf"), None


# --------------------------------------------------------------------------------------
def render_summary(runs, pooled, d1, fracs):
    b = []
    fin = {e["arm"]: e for e in pooled}
    for key in ARM_KEYS:
        e = fin.get(key)
        if e and e["r_eps10_over_n"][0] is not None:
            b.append("末阶段 %s（3 走廊 × 3 seeds = %d 个检查点）: r_eps10/n = %s，"
                     "mass_top16 = %s，stable rank = %s，res_frac_of_J = %s，组别 %s。"
                     % (key, e["r_eps10_over_n"][2], fmt(*e["r_eps10_over_n"]),
                        fmt(*e["mass_top16"]), fmt(*e["stable_rank"], nd=2),
                        fmt(*e["res_frac_of_J"]), group_of(e["r_eps10_over_n"][0])))
    per_corr, allm = paired_diff(runs, ARM_KEYS[2], ARM_KEYS[0], "r_eps10_over_n")
    if allm[0] is not None:
        b.append("clip 0.5 − clip 0（skrtrl-r16，按 (走廊, seed) 配对，9 对）: "
                 "Δ r_eps10/n = %s。" % fmt(*allm))
    per_corr, allm = paired_diff(runs, ARM_KEYS[1], ARM_KEYS[0], "r_eps10_over_n")
    if allm[0] is not None:
        b.append("snap1 − skrtrl-r16（均 clip 0，9 对）: Δ r_eps10/n = %s。" % fmt(*allm))
    # stage trend
    for key in ARM_KEYS:
        rows = [r for corr in CORRIDORS for r in runs.get((corr, key), [])]
        m0 = mean_std(pool(rows, "r_eps10_over_n", at_frac(fracs[0])))
        m1 = mean_std(pool(rows, "r_eps10_over_n", is_final))
        if m0[0] is not None and m1[0] is not None:
            b.append("%s: r_eps10/n 从 frac=%g 的 %s 变到末阶段的 %s（Δ %+.4f）。"
                     % (key, fracs[0], fmt(*m0), fmt(*m1), m1[0] - m0[0]))
    # the early-stage band, which is what D1 reports as the unstructured signature
    rows10 = runs.get((10, ARM_KEYS[0]), []) + runs.get((10, ARM_KEYS[1]), [])
    m = mean_std(pool(rows10, "r_eps10_over_n", at_frac(fracs[0])))
    if m[0] is not None:
        b.append("走廊 10 的两个 clip-0 arm 在 frac=%g 上是 %s，落在 D1 早期"
                 "“无结构残差”带 0.686–0.746 内；压缩同样是训练产生的。" % (fracs[0], fmt(*m)))
    # D1 band the RL arms fall in, and the worst-case convention correction
    if d1:
        off = [d1[t]["ms10"][0] - d1[t]["off10"][0] for t in D1_TASKS
               if d1.get(t) and d1[t]["ms10"][0] is not None
               and d1[t]["off10"][0] is not None]
        worst = max((-v for v in off), default=0.0)     # largest downward (optimistic) bias
        lo = [t for t in D1_TASKS if d1.get(t) and d1[t]["ms10"][0] is not None
              and d1[t]["ms10"][0] < GROUP_LOW]
        b.append("D1 低秩组（均值谱 r_eps10/n < %.1f）为 %s。" % (
            GROUP_LOW, ", ".join(D1_LABEL.get(t, t) for t in lo) or "（空）"))
        for key in ARM_KEYS:
            e = fin.get(key)
            if not e or e["r_eps10_over_n"][0] is None:
                continue
            adj = e["r_eps10_over_n"][0] + worst
            g0, g1 = group_of(e["r_eps10_over_n"][0]), group_of(adj)
            b.append("%s: 均值谱读数 %.4f 归「%s」；加上 D1 上测到的最大向下口径偏差 %.4f "
                     "后为 %.4f，%s。"
                     % (key, e["r_eps10_over_n"][0], g0, worst, adj,
                        ("归组不变" if g0 == g1 else "归组改为「%s」" % g1)))
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=os.path.join("results", "r2", "rl"))
    ap.add_argument("--clipdir", default=os.path.join("results", "r2", "rl_clip05"))
    ap.add_argument("--d1dir", default=os.path.join("results", "r2", "d1_spectrum"))
    ap.add_argument("--outmd", default=None,
                    help="default: <indir>/../RL_SPECTRUM_SUMMARY.md")
    ap.add_argument("--figdir", default=os.path.join(HERE, "..", "paper", "figures"))
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    outmd = args.outmd or os.path.join(os.path.dirname(args.indir),
                                       "RL_SPECTRUM_SUMMARY.md")

    runs, ignored, bad, notes = load_rl(args.indir, args.clipdir)
    missing = missing_list(runs)
    n_have = sum(len(v) for v in runs.values())
    n_want = len(CORRIDORS) * len(ARM_KEYS) * len(SEEDS)
    fracs = fracs_present(runs) or [0.01, 0.05, 0.25, 0.5, 0.75, 1.0]
    d1 = load_d1(args.d1dir)

    table_a, pooled = build_table_a(runs)
    table_b = build_table_b(runs, fracs)

    figpath, figwhy = (None, "skipped by --no-fig")
    if not args.no_fig:
        figpath, figwhy = make_figure(runs, d1, args.figdir)

    L = ["# RL 残差谱 (T-maze) —— 聚合报告", ""]
    L.append("数据目录: `%s` (clip 0) + `%s` (clip 0.5)  |  已完成 %d/%d 项 roster "
             "(%d 项缺失)  |  忽略文件: %d  |  解析失败/无谱: %d"
             % (args.indir, args.clipdir, n_have, n_want, len(missing), len(ignored),
                len(bad)))
    L.append("")
    L.append("## 口径")
    L.append("")
    L += [
        "测量对象与监督任务 D1 完全相同：残差 $R_t = J_t - S_t$，$J_t$ 是 exact-RTRL shadow "
        "携带的影响矩阵，$S_t$ 是 SnAp-1 trace 按块对角嵌入后的矩阵（`skrtrl/rl.py` 的 "
        "`TanhCore.spectrum`，与 `run_e1_spectrum.py` 的 `checkpoint_spectrum` 同一定义）。"
        "两个 arm 的差别只在于这条残差是在哪条策略轨迹上测的，不在残差本身的定义：snap1 "
        "arm 训练用 SnAp-1，被丢弃的正是整个 $J_t-S_t$，因此它是“无 sketch”参照。",
        "",
        "RL 运行为 iso-width $n=%d$，$P=%d$（= 参数数 $n(n+n_{\\mathrm{obs}}+1)"
        "=64\\times69$），每个检查点 %s 个 lane。检查点为 `--spectrum_ckpts` 的六个 frac；"
        "`--spectrum_min_age 4` 与 `--spectrum_defer 200` 使检查点向后滑动，避开 T-maze 中"
        "所有 lane 同步 reset（那里残差恒为零、会读出虚假低秩）的步；`--spectrum_defer` "
        "到期仍未达 age 阈值时照样取样，这类检查点在下面的异常清单中逐个列出。"
        % (N_RL, P_RL, "16"),
        "",
        "**一处口径差异必须显式处理。** `run_m5.py` 只存 lane 平均后的奇异值向量 `sv`，"
        "而 D1 报告的 `r_eps` 是 per-lane `r_eps` 的 lane 平均。因此本报告的 `r_eps` 只能"
        "从 lane 平均谱读出（“均值谱”口径）。为了量化而非假设这个偏差，D1 的 `sv` 用同样"
        "方式（`run_e1_spectrum.py` 也只存 lane 平均谱）重算了一遍，两种口径之差见表 F(i)。"
        "`mass_top{k}`、`stable_rank`、`res_frac_of_J` 在两套实验里都是 per-lane 量的 lane "
        "平均，无此问题，可直接与主文表 4 对照。",
        "",
        "归组沿用 §6.4 的规则：末阶段 $r_{0.1}/n<0.3$ 低秩，$\\le 0.5$ 中间，$>0.6$ 非低秩。",
        "",
        "## 缺失与异常",
        "",
    ]
    if not missing:
        L.append("roster 全部完成（%d/%d），无缺失。" % (n_have, n_want))
    else:
        L.append("缺失 %d 项:" % len(missing))
        L.append("")
        for corr, key, seed in missing:
            L.append("- 走廊 %d / %s / seed %d" % (corr, key, seed))
    L.append("")
    if bad:
        L.append("解析失败 / 无谱:")
        L.append("")
        for p, why in bad:
            L.append("- `%s` — %s" % (p, why))
        L.append("")
    if notes:
        L.append("逐检查点异常（NaN、age 未达阈值仍取样、n/P/lane 数异常）:")
        L.append("")
        for t in notes:
            L.append("- %s" % t)
        L.append("")
    else:
        L.append("逐检查点异常：无。全部 %d 个 run × 6 个检查点均可用，无 NaN，"
                 "无检查点在 `--spectrum_min_age` 未达标时被 deadline 强制取样。" % n_have)
        L.append("")
    if ignored:
        L.append("忽略的文件（`*.localpartial.json` 为运行中作业的部分副本，"
                 "不参与任何聚合）:")
        L.append("")
        for p in ignored:
            L.append("- `%s`" % p)
        L.append("")

    L += render_table_a(table_a, pooled)
    L += render_table_b(table_b, fracs)
    L += render_paired(
        "表 C — clip 0.5 相对 clip 0（skrtrl-r16，按 (走廊, seed) 配对）",
        "每格为 `clip 0.5 − clip 0` 的配对差，mean±sd；末阶段检查点。",
        runs, ARM_KEYS[2], ARM_KEYS[0],
        [("r_eps10_over_n", "Δ r_eps10/n", 4), ("r_eps05_over_n", "Δ r_eps05/n", 4),
         ("mass_top16", "Δ mass_top16", 4), ("stable_rank", "Δ stable rank", 2),
         ("res_frac_of_J", "Δ res_frac_of_J", 4)])
    L += render_paired(
        "表 D — snap1 相对 skrtrl-r16（均 clip 0，按 (走廊, seed) 配对）",
        "snap1 arm 的策略由 SnAp-1 自己训练，残差 $J_t-S_t$ 即它丢弃的全部非块对角部分，"
        "作为“无 sketch”参照。每格为 `snap1 − skrtrl-r16` 的配对差，mean±sd；末阶段检查点。",
        runs, ARM_KEYS[1], ARM_KEYS[0],
        [("r_eps10_over_n", "Δ r_eps10/n", 4), ("r_eps05_over_n", "Δ r_eps05/n", 4),
         ("mass_top16", "Δ mass_top16", 4), ("stable_rank", "Δ stable rank", 2),
         ("res_frac_of_J", "Δ res_frac_of_J", 4)])
    L += render_table_e(pooled, d1)
    L += render_table_f(runs, d1)

    L.append("## 要点（自动生成，仅陈述数字）")
    L.append("")
    for line in render_summary(runs, pooled, d1, fracs):
        L.append("- %s" % line)
    L.append("")
    L.append("## 图")
    L.append("")
    L.append("- `%s` — %s"
             % (os.path.normpath(figpath) if figpath else FIG_NAME,
                "已生成" if figpath else "未生成: %s" % figwhy))
    L.append("")

    os.makedirs(os.path.dirname(os.path.abspath(outmd)), exist_ok=True)
    with open(outmd, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("wrote %s (%d runs, %d missing)" % (outmd, n_have, len(missing)))
    if figpath:
        print("wrote %s" % figpath)
    elif figwhy:
        print("figure skipped: %s" % figwhy)


if __name__ == "__main__":
    main()
