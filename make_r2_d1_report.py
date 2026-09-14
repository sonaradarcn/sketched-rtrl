"""Aggregate + plot D1 residual-spectrum results (R2, Reviewer #3 point 1).

Reads results/r2/d1_spectrum/<task>_n<n>_s<seed>[_tag].json, written by
run_e1_spectrum.py (R_t = J_t - S_t, plus the R1-comparable bd_ = J_t - blockdiag(J_t)
quantities). The real job runs on GPU1 and lands files gradually; this script is written
to be robust to that -- any missing file is simply absent from the pooled statistics and
listed at the top of the markdown report, never fabricated or interpolated.

Outputs
  <outmd>                              (default: <indir>/D1_SUMMARY.md)
    missing-file roster, Table A (n=64, all 9 tasks), Table B (width sweep, 4 diagnostic
    tasks x n in {64,128,256,512}), Table C (main J-S_t vs bd J-blockdiag(J) on r_eps10),
    and an auto-generated, numbers-only bullet summary.
  <figdir>/fig_r2_spectrum_stage.pdf   n=64, r_eps10/n and mass_top16 vs training stage
  <figdir>/fig_r2_spectrum_width.pdf   diagnostic tasks, r_eps10 (abs) and r_eps10/n vs n,
                                       pre-training vs converged
  <figdir>/fig_r2_spectrum_age.pdf     rotation, final stage, r_eps10/n and res_frac_of_J
                                       vs age-within-reset-interval

A figure whose inputs are entirely missing is skipped loudly (never fabricated), matching
the convention in paper/figures/gen/regen_all.py -- whose style helpers (Okabe-Ito palette,
per-task colour/dash/marker, legend-outside-axes, print-size figures) this script imports
and reuses rather than re-deriving.

Usage
  python make_r2_d1_report.py                  # real data: results/r2/d1_spectrum
  python make_r2_d1_report.py --smoke           # dev: results/r2/d1_smoke (same layout)
  python make_r2_d1_report.py --indir <dir> --outmd <path> --figdir <dir>
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
sys.path.insert(0, HERE)                 # fig_style_r1.py lives next to this file

# --------------------------------------------------------------------------------------
# roster (mirrors make_r2_d1_jobs.py: n=64 all tasks + width sweep on the 4 diagnostics)
# --------------------------------------------------------------------------------------
ALL_TASKS = ["adding", "copy", "anbn", "rotation",          # diagnostic
             "henon", "mackeyglass", "lorenz",              # chaotic
             "sunspot", "laser"]                            # real
DIAGNOSTIC = ["adding", "copy", "anbn", "rotation"]
CHAOTIC = ["henon", "mackeyglass", "lorenz"]
REAL = ["sunspot", "laser"]
FAMILY = {**{t: "diagnostic" for t in DIAGNOSTIC},
          **{t: "chaotic" for t in CHAOTIC},
          **{t: "real" for t in REAL}}
SEEDS = [0, 1, 2]
WIDTHS = [64, 128, 256, 512]
EXTRA_WIDTHS = [128, 256, 512]        # width sweep beyond the base n=64 grid

FNAME_RE = re.compile(r"^(?P<task>[a-z0-9]+)_n(?P<n>\d+)_s(?P<seed>\d+)"
                       r"(?:_(?P<tag>[A-Za-z0-9]+))?\.json$")
EPS = 1e-6


def expected_roster():
    exp = [(t, 64, s) for t in ALL_TASKS for s in SEEDS]
    exp += [(t, n, s) for n in EXTRA_WIDTHS for t in DIAGNOSTIC for s in SEEDS]
    return exp


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def load_indir(indir):
    """runs[(task, n)] = [(seed, path, json_dict), ...] for untagged files.
    tagged = [(path, task, n, seed, tag, json_dict), ...] (e.g. *_r1cmp.json) -- not
    aggregated into any table; the parsed dict is carried along only so the separate
    tagged-file accounting further down can report their numbers without re-reading disk.
    bad = [(path, error_str), ...] for files that failed to parse."""
    runs = defaultdict(list)
    tagged, bad = [], []
    for p in sorted(glob.glob(os.path.join(indir, "*.json"))):
        m = FNAME_RE.match(os.path.basename(p))
        if not m:
            continue
        task, n, seed, tag = m["task"], int(m["n"]), int(m["seed"]), m["tag"]
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            bad.append((p, str(e)))
            continue
        if not d.get("records"):
            bad.append((p, "no records (job likely still running / crashed early)"))
            continue
        if tag:
            tagged.append((p, task, n, seed, tag, d))
            continue
        runs[(task, n)].append((seed, p, d))
    return runs, tagged, bad


def missing_list(runs):
    have = set()
    for (task, n), rows in runs.items():
        for seed, p, d in rows:
            have.add((task, n, seed))
    return sorted(set(expected_roster()) - have)


# --------------------------------------------------------------------------------------
# record predicates / pooling
# --------------------------------------------------------------------------------------
def _frac(r):
    return r.get("frac")


def is_final(r):
    f = _frac(r)
    return f is not None and abs(f - 1.0) < EPS


def is_early(r):
    f = _frac(r)
    return f is not None and f <= 0.05 + EPS


def is_mid(r):
    f = _frac(r)
    return f is not None and 0.25 - EPS <= f <= 0.5 + EPS


def is_late(r):
    f = _frac(r)
    return f is not None and f >= 0.75 - EPS


def is_any(r):
    return True


def pred_at(f0):
    return lambda r: _frac(r) is not None and abs(_frac(r) - f0) < EPS


def fracs_present(rows):
    return sorted({_frac(r) for _, _, d in rows for r in d.get("records", [])
                   if _frac(r) is not None})


def stage_bounds(rows):
    fs = fracs_present(rows)
    return (fs[0], fs[-1]) if fs else (None, None)


def pool_field(rows, field, pred=is_any):
    return [r[field] for _, _, d in rows for r in d.get("records", [])
            if pred(r) and r.get(field) is not None]


def pool_pair_diff(rows, f1, f2, pred=is_any):
    return [r[f1] - r[f2] for _, _, d in rows for r in d.get("records", [])
            if pred(r) and r.get(f1) is not None and r.get(f2) is not None]


def mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
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
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f}±{s:.{nd}f}"


def seed_count(rows):
    return len({seed for seed, _, _ in rows})


# --------------------------------------------------------------------------------------
# Table A: n=64, all 9 tasks
# --------------------------------------------------------------------------------------
def build_table_a(runs):
    fields = ["mass_top16", "stable_rank", "r_eps10_over_n", "r_eps05_over_n", "res_frac_of_J"]
    rows_out = []
    for task in ALL_TASKS:
        rows = runs.get((task, 64), [])
        nseed = seed_count(rows)
        entry = {"task": task, "family": FAMILY[task], "n_seeds": nseed}
        for f in fields:
            entry[f"final_{f}"] = mean_std(pool_field(rows, f, is_final))
            entry[f"all_{f}"] = mean_std(pool_field(rows, f, is_any))
        entry["early_r10n"] = mean_std(pool_field(rows, "r_eps10_over_n", is_early))
        entry["mid_r10n"] = mean_std(pool_field(rows, "r_eps10_over_n", is_mid))
        entry["late_r10n"] = mean_std(pool_field(rows, "r_eps10_over_n", is_late))
        rows_out.append(entry)
    return rows_out


def render_table_a(rows_out):
    lines = [
        "### 表 A — n=64，9 任务，末阶段 (frac=1.0) 与全阶段聚合",
        "",
        "mean±std over seeds x ages (末阶段列) 或 over seeds x ages x fracs (全阶段列)。"
        " `n_seeds` = 该任务已完成的 seed 数 (/3)。",
        "",
        "| task | family | n_seeds | mass_top16 (末/全) | stable_rank (末/全) | "
        "r_eps10/n (末/全) | r_eps05/n (末/全) | res_frac_of_J (末/全) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in rows_out:
        if e["n_seeds"] == 0:
            lines.append(f"| {e['task']} | {e['family']} | 0/3 | — | — | — | — | — |")
            continue

        def cell(f):
            return f"{fmt(*e[f'final_{f}'])} / {fmt(*e[f'all_{f}'])}"

        lines.append(f"| {e['task']} | {e['family']} | {e['n_seeds']}/3 | "
                     f"{cell('mass_top16')} | {cell('stable_rank')} | "
                     f"{cell('r_eps10_over_n')} | {cell('r_eps05_over_n')} | "
                     f"{cell('res_frac_of_J')} |")
    lines += [
        "",
        "早 (frac<=0.05) / 中 (0.25-0.5) / 晚 (>=0.75) 三档 r_eps10/n:",
        "",
        "| task | 早 | 中 | 晚 |",
        "|---|---|---|---|",
    ]
    for e in rows_out:
        if e["n_seeds"] == 0:
            lines.append(f"| {e['task']} | — | — | — |")
            continue
        lines.append(f"| {e['task']} | {fmt(*e['early_r10n'])} | {fmt(*e['mid_r10n'])} | "
                     f"{fmt(*e['late_r10n'])} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Table B: width sweep, 4 diagnostic tasks x n in {64,128,256,512}
# --------------------------------------------------------------------------------------
def build_table_b(runs):
    rows_out = []
    for task in DIAGNOSTIC:
        for n in WIDTHS:
            rows = runs.get((task, n), [])
            nseed = seed_count(rows)
            entry = {"task": task, "n": n, "n_seeds": nseed}
            if nseed:
                for f in ["mass_top16", "mass_topn4", "stable_rank", "r_eps10",
                         "r_eps10_over_n", "r_eps05_over_n"]:
                    entry[f"final_{f}"] = mean_std(pool_field(rows, f, is_final))
                lo, hi = stage_bounds(rows)
                entry["stage_lo"], entry["stage_hi"] = lo, hi
                if lo is not None:
                    entry["pre_r10n"] = mean_std(pool_field(rows, "r_eps10_over_n", pred_at(lo)))
                if hi is not None:
                    entry["post_r10n"] = mean_std(pool_field(rows, "r_eps10_over_n", pred_at(hi)))
            rows_out.append(entry)
    return rows_out


def render_table_b(rows_out):
    lines = [
        "### 表 B — 宽度扫描，诊断 4 任务 x n∈{64,128,256,512}，末阶段 (frac=1.0)",
        "",
        "| task | n | n_seeds | mass_top16 | mass_topn4 | stable_rank | r_eps10 | "
        "r_eps10/n | r_eps05/n |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in rows_out:
        if e["n_seeds"] == 0:
            lines.append(f"| {e['task']} | {e['n']} | 0/3 | — | — | — | — | — | — |")
            continue
        lines.append(f"| {e['task']} | {e['n']} | {e['n_seeds']}/3 | "
                     f"{fmt(*e['final_mass_top16'])} | {fmt(*e['final_mass_topn4'])} | "
                     f"{fmt(*e['final_stable_rank'], nd=2)} | "
                     f"{fmt(*e['final_r_eps10'], nd=2)} | "
                     f"{fmt(*e['final_r_eps10_over_n'])} | {fmt(*e['final_r_eps05_over_n'])} |")
    lines += [
        "",
        "训练前 (第一个已跑到的阶段, frac=stage_lo) vs 收敛端 (frac=1.0) 的 r_eps10/n"
        " —— 直接读出所需秩是否随宽度线性增长、训练是否压缩它:",
        "",
        "| task | n | frac(前) | r_eps10/n (训练前) | r_eps10/n (收敛端) | Δ (收敛-训练前) |",
        "|---|---|---|---|---|---|",
    ]
    for e in rows_out:
        if e["n_seeds"] == 0 or "pre_r10n" not in e:
            lines.append(f"| {e['task']} | {e['n']} | — | — | — | — |")
            continue
        pre_m = e["pre_r10n"][0]
        post_m = e.get("post_r10n", (None, None, 0))[0]
        delta = (post_m - pre_m) if (pre_m is not None and post_m is not None) else None
        lines.append(f"| {e['task']} | {e['n']} | {e['stage_lo']:.2f} | "
                     f"{fmt(*e['pre_r10n'])} | {fmt(*e.get('post_r10n', (None, None, 0)))} | "
                     f"{'—' if delta is None else f'{delta:+.4f}'} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Table B' : separate accounting for the tagged files (never enters Table A/B/C)
# --------------------------------------------------------------------------------------
# Tagged files are deliberately excluded from the pooled tables above, and that stays
# true: the Table A/B/C convention is "the three roster seeds, one file per seed", so if
# whatever reruns happen to be sitting on disk were silently folded in, a cell's n_seeds
# -- and therefore its mean -- would change from one regeneration of this report to the
# next without anything in the paper changing. Keeping the roster fixed is what makes the
# tables quotable.
#
# The cost of that convention is that a rerun campaign shows up in the report only as a
# list of filenames, which is exactly wrong when the reruns are the evidence. The
# adding/n=512 campaign is that case: the extra seeds were launched *because* the 3-seed
# roster cell happens to contain a run that destabilises late in training, and the
# manuscript wants to quote what the extra seeds say. So the tagged files get their own
# accounting here -- per-file numbers, a mean over the tagged set, and, since the claim
# the paper makes is about all the seeds together, a mean over the union of untagged +
# tagged split by whether the run trained to completion. Every number in this section is
# additional reading; none of it feeds any Table A/B/C cell.

STAB_RATIO = 1.5     # final/best training metric above which a run is called destabilised


def run_stability(d):
    """Classify one run as trained-to-completion or destabilised, from its own loss curve.

    `curve` is the training metric logged every --log_every steps, so a run can be judged
    without re-deciding what "diverged" means per task: a run that trained to completion
    ends at (or within logging noise of) its best value, whereas a run that blew up ends
    far above it, and final/best is therefore a single scale-free test. On the adding
    n=512 campaign the two families separate very widely -- every run that trained through
    ends within 1.5% of its best, the two that blew up end 2.3x and 2.5x above it -- so
    the threshold is not doing any delicate work there.

    It is deliberately NOT offered as a general health check on every run in the roster,
    and this is why the verdict is only ever printed inside a single (task, width) cell
    rather than as a column of Table A/B. On tasks whose training metric plateaus at a
    very small value (lorenz, mackeyglass, anbn) the last few logged points wander by tens
    of percent of a tiny number, so a purely relative test calls them destabilised when
    nothing has gone wrong; conversely the spectral signature of the adding blow-up
    (stable_rank collapsing towards 1, mass_top16 -> 1) is the *normal* converged state for
    rotation and anbn, so it cannot be used as a cross-task test either. Read the verdict
    as "which runs in this cell are comparable with each other", not as an absolute one.

    Also reported, because "destabilises in the last quarter of training" is a claim about
    *when*: `best_step` is the last step that was still the best seen (where the run turned
    round) and `cross_step` the first later step whose metric exceeds STAB_RATIO x best.
    Both are given as a fraction of the run's own step budget so runs of different length
    stay comparable. A non-finite final metric counts as destabilised outright.
    """
    curve = [c for c in (d.get("curve") or [])
             if c.get("metric") is not None and c.get("step") is not None]
    steps = (d.get("args") or {}).get("steps") or (curve[-1]["step"] if curve else None)
    out = {"steps": steps, "best": None, "best_step": None, "best_frac": None,
           "final": None, "ratio": None, "cross_step": None, "cross_frac": None,
           "unstable": None}
    if not curve:
        return out
    finite = [c for c in curve if math.isfinite(c["metric"])]
    out["final"] = curve[-1]["metric"]
    if not finite:
        out["unstable"] = True
        return out
    best = min(finite, key=lambda c: c["metric"])
    out["best"], out["best_step"] = best["metric"], best["step"]
    if steps:
        out["best_frac"] = best["step"] / steps
    if not math.isfinite(out["final"]):
        out["ratio"], out["unstable"] = float("inf"), True
    elif best["metric"] > 0:
        out["ratio"] = out["final"] / best["metric"]
        out["unstable"] = out["ratio"] > STAB_RATIO
    if out["unstable"]:
        thr = best["metric"] * STAB_RATIO
        later = [c for c in curve if c["step"] > best["step"]
                 and (not math.isfinite(c["metric"]) or c["metric"] > thr)]
        if later:
            out["cross_step"] = later[0]["step"]
            if steps:
                out["cross_frac"] = later[0]["step"] / steps
    return out


TAGGED_FIELDS = ["r_eps10_over_n", "r_eps05_over_n", "r_eps10", "stable_rank", "mass_top16"]


def build_tagged(runs, tagged):
    """rows_out[(task, n)] = {"files": [...], "groups": [(label, note, rows), ...]}.

    `files` is one entry per file in the cell -- the tagged ones plus, marked `roster`, the
    untagged ones that Table B pools -- each with its final-stage numbers and stability
    verdict. The roster files are listed even though they are not tagged because the split
    below classifies them too: if one of them is the run that destabilised, a reader has to
    be able to see which one and where it turned, and that is precisely the case the extra
    seeds were run to settle. Listing them here changes no Table B number; Table B is still
    built from `runs` alone by build_table_b.

    `groups` are the pooled readings we want side by side: the roster (untagged) set that
    Table B actually reports, the tagged set on its own, the union of the two, and the
    union split into the runs that trained to completion and the ones that did not. The
    union is pooled per *file*, not per seed, and a seed that appears in more than one file
    (an original plus a longer rerun, say) is flagged in the note rather than silently
    de-duplicated, because which of the two should represent that seed is a judgement for
    the text, not for this script."""
    by_cell = defaultdict(list)
    for p, task, n, seed, tag, d in tagged:
        by_cell[(task, n)].append((p, seed, tag, d))
    rows_out = {}
    for (task, n), items in sorted(by_cell.items()):
        items.sort(key=lambda x: (x[1], x[2]))
        untagged_rows = list(runs.get((task, n), []))
        listing = [(p, seed, "roster", d) for seed, p, d in sorted(untagged_rows)] + items
        files = []
        for p, seed, tag, d in listing:
            rows1 = [(seed, p, d)]
            ent = {"path": p, "seed": seed, "tag": tag, "stab": run_stability(d)}
            for f in TAGGED_FIELDS:
                ent[f] = mean_std(pool_field(rows1, f, is_final))
            files.append(ent)

        tagged_rows = [(seed, p, d) for p, seed, tag, d in items]
        stable_rows = [r for r in untagged_rows + tagged_rows if not run_stability(r[2])["unstable"]]
        unstable_rows = [r for r in untagged_rows + tagged_rows if run_stability(r[2])["unstable"]]

        def grp(label, rows, note=""):
            ent = {"label": label, "note": note, "n_files": len(rows),
                   "n_seeds": seed_count(rows)}
            for f in TAGGED_FIELDS:
                ent[f] = mean_std(pool_field(rows, f, is_final))
            return ent

        def seed_note(rows):
            """Spell out which files a split contains. The stable/unstable split is the
            one the manuscript quotes a seed count from, and a seed can contribute two
            files, so "7 files" alone is ambiguous about how many seeds that is and which
            budget each ran under -- name them and the arithmetic is checkable."""
            return "含 " + ", ".join(sorted(os.path.basename(p).replace(".json", "")
                                            for _, p, _ in rows))

        seeds_untagged = {s for s, _, _ in untagged_rows}
        seeds_tagged = {s for s, _, _ in tagged_rows}
        overlap = sorted(seeds_untagged & seeds_tagged)
        note_union = ("" if not overlap else
                      "seed " + ",".join(str(s) for s in overlap) + " 同时出现在未 tag 与 tagged 文件中，两份都计入")
        groups = [
            grp("roster（未 tag，= 表 B 该行）", untagged_rows),
            grp("tagged 集合", tagged_rows),
            grp("并集（全部文件）", untagged_rows + tagged_rows, note_union),
            grp("并集中跑完 (final/best <= %.2f)" % STAB_RATIO, stable_rows,
                seed_note(stable_rows)),
            grp("并集中失稳 (final/best > %.2f)" % STAB_RATIO, unstable_rows,
                seed_note(unstable_rows)),
        ]
        rows_out[(task, n)] = {"files": files, "groups": groups}
    return rows_out


def _step_frac(step, frac):
    """A step, annotated with where it falls in the run's own budget (blank if unknown)."""
    if step is None:
        return "—"
    return str(step) if frac is None else f"{step} ({frac:.2f})"


def render_tagged(rows_out):
    lines = [
        "### 表 B' — tagged 文件的单独核算（**不进入表 A/B/C 的任何均值**）",
        "",
        "表 A/B/C 只读 roster 约定的未 tag 文件（每任务每宽度 3 个 seed，每 seed 一个文件）；"
        "本节把带后缀的文件单独列出，供正文引用额外 seed 时使用。稳定性判据取自各自的训练"
        f"曲线：final/best > {STAB_RATIO:.2f} 记为失稳，`turn` 为曲线最后一次取得最优的 step"
        "（开始转折处），`cross` 为此后首次超过阈值的 step，括号内为占该 run 自身步数预算的比例。"
        "该判据只用于在**同一 (task, n) cell 内**区分哪些 run 彼此可比；它不是跨任务的通用"
        "健康检查（训练指标收敛到极小值的任务，末段相对抖动本来就可能超过阈值），"
        "因此不作为表 A/B 的列出现。",
        "",
    ]
    if not rows_out:
        lines.append("本次没有任何 tagged 文件，无需单独核算。")
        return "\n".join(lines)
    for (task, n), cell in rows_out.items():
        lines += [
            f"**{task}, n={n}** — 逐文件（末阶段 frac=1.0，池化该文件的 5 个 age"
            " checkpoint）；tag 栏为 `roster` 的行即表 B 所池化的未 tag 文件，列在此处只为"
            "让下面的稳定性划分可核对，其数值与表 B 完全一致:",
            "",
            "| file | seed | tag | steps | r_eps10/n | r_eps05/n | r_eps10 | stable_rank | "
            "mass_top16 | best | final | final/best | turn | cross | 判定 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for e in cell["files"]:
            st = e["stab"]
            steps = "—" if st["steps"] is None else str(st["steps"])
            best = "—" if st["best"] is None else f"{st['best']:.4f}"
            final = "—" if st["final"] is None else f"{st['final']:.4f}"
            ratio = "—" if st["ratio"] is None else f"{st['ratio']:.3f}"
            turn = _step_frac(st["best_step"], st["best_frac"])
            cross = _step_frac(st["cross_step"], st["cross_frac"])
            verdict = "失稳" if st["unstable"] else "跑完"
            lines.append(
                f"| `{os.path.basename(e['path'])}` | {e['seed']} | {e['tag']} | {steps} | "
                f"{fmt(*e['r_eps10_over_n'])} | {fmt(*e['r_eps05_over_n'])} | "
                f"{fmt(*e['r_eps10'], nd=2)} | {fmt(*e['stable_rank'], nd=2)} | "
                f"{fmt(*e['mass_top16'])} | {best} | {final} | {ratio} | "
                f"{turn} | {cross} | {verdict} |")
        lines += [
            "",
            f"**{task}, n={n}** — 各口径池化对照（末阶段 frac=1.0）:",
            "",
            "| 口径 | n_files | n_seeds | r_eps10/n | r_eps05/n | r_eps10 | stable_rank | "
            "mass_top16 | 备注 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for g in cell["groups"]:
            if not g["n_files"]:
                lines.append(f"| {g['label']} | 0 | 0 | — | — | — | — | — | {g['note']} |")
                continue
            lines.append(
                f"| {g['label']} | {g['n_files']} | {g['n_seeds']} | "
                f"{fmt(*g['r_eps10_over_n'])} | {fmt(*g['r_eps05_over_n'])} | "
                f"{fmt(*g['r_eps10'], nd=2)} | {fmt(*g['stable_rank'], nd=2)} | "
                f"{fmt(*g['mass_top16'])} | {g['note']} |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Table C: main (J - S_t) vs bd (J - blockdiag J) on r_eps10, n=64
# --------------------------------------------------------------------------------------
def build_table_c(runs):
    rows_out = []
    families = {"diagnostic": DIAGNOSTIC, "chaotic": CHAOTIC, "real": REAL, "all": ALL_TASKS}
    for fam, tasks in families.items():
        rows = []
        for t in tasks:
            rows += runs.get((t, 64), [])
        main_vals = pool_field(rows, "r_eps10", is_any)
        bd_vals = pool_field(rows, "bd_r_eps10", is_any)
        diffs = pool_pair_diff(rows, "r_eps10", "bd_r_eps10", is_any)
        m_main, _, n_main = mean_std(main_vals)
        m_bd, _, n_bd = mean_std(bd_vals)
        m_diff, s_diff, n_diff = mean_std(diffs)
        pct = (100.0 * m_diff / m_bd) if (m_diff is not None and m_bd not in (None, 0)) else None
        rows_out.append({"family": fam, "n_records": n_diff, "main": (m_main, None),
                         "bd": (m_bd, None), "diff": (m_diff, s_diff), "pct": pct})
    return rows_out


def render_table_c(rows_out):
    lines = [
        "### 表 C — 主口径 (J−S_t) vs bd 口径 (J−blockdiag(J)) 在 r_eps10 上的差异，n=64，全阶段全 age 池化",
        "",
        "diff = r_eps10 − bd_r_eps10，逐 checkpoint 配对后再取 mean±std；pct = diff 均值"
        " 相对 bd 均值的百分比。",
        "",
        "| 任务族 | n_records | r_eps10 (main) | r_eps10 (bd) | diff (main−bd) | diff / bd |",
        "|---|---|---|---|---|---|",
    ]
    for e in rows_out:
        if not e["n_records"]:
            lines.append(f"| {e['family']} | 0 | — | — | — | — |")
            continue
        pct = "—" if e["pct"] is None else f"{e['pct']:+.1f}%"
        lines.append(f"| {e['family']} | {e['n_records']} | {fmt(*e['main'], nd=2)} | "
                     f"{fmt(*e['bd'], nd=2)} | {fmt(*e['diff'], nd=2)} | {pct} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# auto-generated, numbers-only bullet summary
# --------------------------------------------------------------------------------------
def render_summary(table_a, table_b, table_c):
    bullets = []
    have_a = [e for e in table_a if e["n_seeds"] > 0]
    if have_a:
        m16 = [(e["task"], e["final_mass_top16"][0]) for e in have_a
               if e["final_mass_top16"][0] is not None]
        if m16:
            hi = max(m16, key=lambda x: x[1])
            lo = min(m16, key=lambda x: x[1])
            bullets.append(f"n=64 末阶段 mass_top16：最高 {hi[0]} ({hi[1]:.4f})，"
                           f"最低 {lo[0]} ({lo[1]:.4f})，{len(m16)}/9 任务已有数据。")
        r10 = [(e["task"], e["late_r10n"][0]) for e in have_a if e["late_r10n"][0] is not None]
        r10e = [(e["task"], e["early_r10n"][0]) for e in have_a if e["early_r10n"][0] is not None]
        if r10 and r10e:
            r10d = {t: v for t, v in r10}
            r10ed = {t: v for t, v in r10e}
            deltas = [(t, r10d[t] - r10ed[t]) for t in r10d if t in r10ed]
            if deltas:
                mean_delta = sum(v for _, v in deltas) / len(deltas)
                bullets.append(f"晚 (>=0.75) 相对早 (<=0.05) 阶段 r_eps10/n 的平均变化: "
                               f"{mean_delta:+.4f}（{len(deltas)} 个任务，逐任务见表 A）。")

    have_b = [e for e in table_b if e["n_seeds"] > 0 and "final_r_eps10_over_n" in e]
    if have_b:
        by_task = defaultdict(list)
        for e in have_b:
            by_task[e["task"]].append((e["n"], e["final_r_eps10_over_n"][0]))
        for t, pts in by_task.items():
            pts = sorted(pts)
            if len(pts) >= 2:
                (n0, v0), (n1, v1) = pts[0], pts[-1]
                bullets.append(f"{t}: r_eps10/n 在 n={n0}→{n1} 间从 {v0:.4f} 变为 {v1:.4f} "
                               f"({len(pts)}/{len(WIDTHS)} 宽度已有数据)。")

    have_c = [e for e in table_c if e["n_records"] and e["family"] != "all"]
    for e in have_c:
        bullets.append(f"{e['family']} 任务族 (n=64, 全阶段池化, {e['n_records']} 条记录): "
                       f"r_eps10 主口径均值 {e['main'][0]:.2f}，bd 口径均值 {e['bd'][0]:.2f}，"
                       f"差值 {e['diff'][0]:+.2f}"
                       + ("" if e["pct"] is None else f" ({e['pct']:+.1f}%)") + "。")
    all_c = [e for e in table_c if e["family"] == "all" and e["n_records"]]
    if all_c:
        e = all_c[0]
        bullets.append(f"全部任务 (n=64, {e['n_records']} 条记录): r_eps10 主口径均值 "
                       f"{e['main'][0]:.2f} vs bd 口径均值 {e['bd'][0]:.2f}，差值 "
                       f"{e['diff'][0]:+.2f}" + ("" if e["pct"] is None else
                                                  f" ({e['pct']:+.1f}%)") + "。")
    if not bullets:
        bullets = ["尚无任何已完成的结果文件，无法生成数值要点。"]
    return "\n".join(f"- {b}" for b in bullets)


# --------------------------------------------------------------------------------------
# figures (reuses paper/figures/gen/regen_all.py style: Okabe-Ito palette, per-task
# colour/dash/marker, print-physical-size figures, legend-outside-axes)
# --------------------------------------------------------------------------------------
EXTRA_TASK_STYLE = {}     # filled in _init_styles() once `ra` is imported
EXTRA_TASK_LABEL = {"henon": "henon", "mackeyglass": "mackey-glass", "lorenz": "lorenz",
                    "sunspot": "sunspot", "laser": "laser"}


def _init_styles(ra):
    """henon/mackeyglass/lorenz/sunspot/laser aren't in regen_all.TASK_STYLE (only the 4
    diagnostic tasks + rotrecall24 are); give them the remaining Okabe-Ito colours so all
    9 D1 tasks stay visually distinct (colour AND marker AND, mostly, linestyle differ)."""
    global EXTRA_TASK_STYLE
    EXTRA_TASK_STYLE = {
        "henon":       (ra.OI["sky"],    (0, (4, 2, 1, 2)),        "d"),
        "mackeyglass": (ra.OI["yellow"], (0, (1, 1)),              "x"),
        "lorenz":      (ra.OI["teal"],   (0, (3, 1, 1, 1)),        "*"),
        "sunspot":     (ra.OI["purple"], (0, (5, 1, 1, 1, 1, 1)),  "P"),
        "laser":       (ra.OI["grey"],   (0, (2, 2)),              "p"),
    }


def _d1_task_kw(ra, task):
    if task in ra.TASK_STYLE:
        return ra.task_kw(task)
    c, ls, mk = EXTRA_TASK_STYLE[task]
    return dict(color=c, linestyle=ls, marker=mk, label=EXTRA_TASK_LABEL.get(task, task))


def stage_series(rows, field, fracs):
    xs, means, stds = [], [], []
    for f in fracs:
        seed_means = []
        for seed, p, d in rows:
            vals = [r[field] for r in d.get("records", [])
                    if r.get(field) is not None and abs((r.get("frac") or -9) - f) < EPS]
            if vals:
                seed_means.append(sum(vals) / len(vals))
        if seed_means:
            m, s, _ = mean_std(seed_means)
            xs.append(f); means.append(m); stds.append(s or 0.0)
    return xs, means, stds


def age_series(rows, field, pred=is_final):
    ages = sorted({r.get("target_age") for _, _, d in rows for r in d.get("records", [])
                   if pred(r) and r.get("target_age") is not None})
    xs, means, stds = [], [], []
    for a in ages:
        seed_means = []
        for seed, p, d in rows:
            vals = [r[field] for r in d.get("records", [])
                    if pred(r) and r.get("target_age") == a and r.get(field) is not None]
            if vals:
                seed_means.append(sum(vals) / len(vals))
        if seed_means:
            m, s, _ = mean_std(seed_means)
            xs.append(a); means.append(m); stds.append(s or 0.0)
    return xs, means, stds


def fig_spectrum_stage(ra, plt, runs, figdir):
    name = "fig_r2_spectrum_stage"
    tasks_with_data = [t for t in ALL_TASKS if runs.get((t, 64))]
    if not tasks_with_data:
        return name, "no n=64 D1 results found"
    fs.register(ra, name, height=2.10)      # printed width from the .tex (Figure-5 rule)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=ra.figsize(name))
    for t in tasks_with_data:
        rows = runs[(t, 64)]
        fracs = fracs_present(rows)
        kw = _d1_task_kw(ra, t)
        xs, m, s = stage_series(rows, "r_eps10_over_n", fracs)
        if xs:
            axL.plot(xs, m, ms=3.6, **kw)
            axL.fill_between(xs, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)],
                             color=kw["color"], alpha=0.15, linewidth=0)
        kw2 = dict(kw); kw2.pop("label")
        xs2, m2, s2 = stage_series(rows, "mass_top16", fracs)
        if xs2:
            axR.plot(xs2, m2, ms=3.6, **kw2, label=kw["label"])
            axR.fill_between(xs2, [a - b for a, b in zip(m2, s2)],
                             [a + b for a, b in zip(m2, s2)],
                             color=kw["color"], alpha=0.15, linewidth=0)
    # Titles are kept to the width a 1.55 in panel can hold at 7.5 pt, exactly as in
    # Figure 5 (whose left panel carries no title at all): the long form lived in the
    # title only because the figure used to be drawn 1.85x too wide, and the caption
    # already says which panel is which.
    axL.set_xlabel("training stage (frac)")
    axL.set_ylabel(r"$r_{\epsilon=0.10}/n$")
    axL.set_title("rank fraction")
    axL.set_ylim(bottom=0)
    axR.set_xlabel("training stage (frac)")
    axR.set_ylabel("mass$_{top16}$")
    axR.set_title("top-16 mass")
    axR.set_ylim(0, 1.02)
    fig.tight_layout()
    h, l = ra._merged_handles([axL, axR])
    if h:
        fs.legend_below_fit(fig, [axL, axR], h, l, ncol=min(4, len(l)), name=name,
                            handlelength=1.9, columnspacing=0.7, handletextpad=0.35)
    ra._save(fig, name)
    return name, None


def fig_spectrum_width(ra, plt, runs, figdir):
    name = "fig_r2_spectrum_width"
    have = [t for t in DIAGNOSTIC if any(runs.get((t, n)) for n in WIDTHS)]
    if not have:
        return name, "no diagnostic-task D1 results at any width"
    fs.register(ra, name, height=2.10)      # printed width from the .tex (Figure-5 rule)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=ra.figsize(name))
    for t in have:
        c, ls, mk = ra.TASK_STYLE.get(t, (ra.OI["grey"], "-", "o"))
        pre_n, pre_abs, pre_abs_s, pre_rn, pre_rn_s = [], [], [], [], []
        post_n, post_abs, post_abs_s, post_rn, post_rn_s = [], [], [], [], []
        for n in WIDTHS:
            rows = runs.get((t, n))
            if not rows:
                continue
            lo, hi = stage_bounds(rows)
            if lo is None:
                continue
            m0, s0, _ = mean_std(pool_field(rows, "r_eps10", pred_at(lo)))
            m0n, s0n, _ = mean_std(pool_field(rows, "r_eps10_over_n", pred_at(lo)))
            if m0 is not None:
                pre_n.append(n); pre_abs.append(m0); pre_abs_s.append(s0 or 0.0)
                pre_rn.append(m0n); pre_rn_s.append(s0n or 0.0)
            m1, s1, _ = mean_std(pool_field(rows, "r_eps10", is_final))
            m1n, s1n, _ = mean_std(pool_field(rows, "r_eps10_over_n", is_final))
            if m1 is not None:
                post_n.append(n); post_abs.append(m1); post_abs_s.append(s1 or 0.0)
                post_rn.append(m1n); post_rn_s.append(s1n or 0.0)
        label = ra.TASK_LABEL.get(t, t)
        if pre_n:
            axL.errorbar(pre_n, pre_abs, yerr=pre_abs_s, color=c, linestyle="--", marker=mk,
                         markerfacecolor="none", ms=4.0, capsize=2, lw=1.0,
                         label=f"{label} (pre-training)")
            axR.errorbar(pre_n, pre_rn, yerr=pre_rn_s, color=c, linestyle="--", marker=mk,
                         markerfacecolor="none", ms=4.0, capsize=2, lw=1.0)
        if post_n:
            axL.errorbar(post_n, post_abs, yerr=post_abs_s, color=c, linestyle="-", marker=mk,
                         ms=4.0, capsize=2, lw=1.3, label=f"{label} (converged)")
            axR.errorbar(post_n, post_rn, yerr=post_rn_s, color=c, linestyle="-", marker=mk,
                         ms=4.0, capsize=2, lw=1.3)
    axL.set_xlabel("width $n$")
    axL.set_ylabel(r"$r_{\epsilon=0.10}$")
    axL.set_title("absolute rank")
    axL.set_xscale("log", base=2)
    axL.set_xticks(WIDTHS); axL.set_xticklabels([str(w) for w in WIDTHS])
    axR.set_xlabel("width $n$")
    axR.set_ylabel(r"$r_{\epsilon=0.10}/n$")
    axR.set_title("rank fraction")
    axR.set_xscale("log", base=2)
    axR.set_xticks(WIDTHS); axR.set_xticklabels([str(w) for w in WIDTHS])
    axR.set_ylim(bottom=0)
    fig.tight_layout()
    h, l = ra._merged_handles([axL, axR])
    if h:
        fs.legend_below_fit(fig, [axL, axR], h, l, ncol=min(2, len(l)), name=name,
                            handlelength=1.9, columnspacing=0.7, handletextpad=0.35)
    ra._save(fig, name)
    return name, None


def fig_spectrum_age(ra, plt, runs, figdir):
    name = "fig_r2_spectrum_age"
    rows = runs.get(("rotation", 64), [])
    if not rows:
        return name, "no rotation_n64_s*.json in D1 results"
    fs.register(ra, name, height=2.05)      # printed width from the .tex (Figure-5 rule)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=ra.figsize(name))
    c, ls, mk = ra.TASK_STYLE.get("rotation", (ra.OI["blue"], "-", "o"))
    xs, m, s = age_series(rows, "r_eps10_over_n")
    if xs:
        axL.plot(xs, m, color=c, linestyle=ls, marker=mk, ms=4.0, lw=1.3)
        axL.fill_between(xs, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)],
                         color=c, alpha=0.18, linewidth=0)
    xs2, m2, s2 = age_series(rows, "res_frac_of_J")
    if xs2:
        axR.plot(xs2, m2, color=c, linestyle=ls, marker=mk, ms=4.0, lw=1.3)
        axR.fill_between(xs2, [a - b for a, b in zip(m2, s2)], [a + b for a, b in zip(m2, s2)],
                         color=c, alpha=0.18, linewidth=0)
    axL.set_xlabel("age in reset interval")
    axL.set_ylabel(r"$r_{\epsilon=0.10}/n$")
    axL.set_title("rank fraction")
    axL.set_ylim(bottom=0)
    axR.set_xlabel("age in reset interval")
    axR.set_ylabel(r"$\|R\|_F/\|J\|_F$")
    axR.set_title("residual mass")
    axR.set_ylim(bottom=0)
    fig.tight_layout()
    ra._save(fig, name)
    return name, None


def make_figures(runs, figdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import regen_all as ra
        import fig_style_r1 as fs
        globals()["fs"] = fs
        fs.bind(ra)
        fs.apply_rc(plt)
    except Exception as e:
        return [], [(f, str(e)) for f in
                    ("fig_r2_spectrum_stage", "fig_r2_spectrum_width", "fig_r2_spectrum_age")]
    _init_styles(ra)
    ra.OUT = figdir
    ra.GRAY = ""
    os.makedirs(figdir, exist_ok=True)
    wrote, skipped = [], []
    for fn in (fig_spectrum_stage, fig_spectrum_width, fig_spectrum_age):
        try:
            name, why = fn(ra, plt, runs, figdir)
        except Exception as e:
            skipped.append((fn.__name__, f"exception: {e}"))
            continue
        if why:
            skipped.append((name, why))
        else:
            wrote.append(name)
    return wrote, skipped


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=os.path.join("results", "r2", "d1_spectrum"))
    ap.add_argument("--smoke", action="store_true",
                    help="shortcut for --indir results/r2/d1_smoke (dev/testing)")
    ap.add_argument("--outmd", default=None, help="default: <indir>/D1_SUMMARY.md")
    ap.add_argument("--figdir", default=os.path.join(HERE, "..", "paper", "figures"))
    args = ap.parse_args()

    indir = os.path.join("results", "r2", "d1_smoke") if args.smoke else args.indir
    outmd = args.outmd or os.path.join(indir, "D1_SUMMARY.md")

    runs, tagged, bad = load_indir(indir)
    missing = missing_list(runs)
    n_have = sum(len(v) for v in runs.values())
    n_expected = len(expected_roster())

    table_a = build_table_a(runs)
    table_b = build_table_b(runs)
    table_b_tagged = build_tagged(runs, tagged)
    table_c = build_table_c(runs)

    wrote, skipped_figs = make_figures(runs, args.figdir)

    lines = ["# D1 残差谱实验 (R2, Reviewer #3 point 1) —— 聚合报告", ""]
    if os.path.basename(os.path.normpath(indir)) == "d1_smoke":
        lines.append("**注：本报告基于 smoke 输出生成，仅用于验证聚合/作图流程，"
                     "不代表正式结果。**")
        lines.append("")
    lines.append(f"数据目录: `{indir}`  |  已完成 {n_have}/{n_expected} 项 roster "
                 f"({len(missing)} 项缺失)  |  tagged 文件（未参与聚合）: {len(tagged)}  |  "
                 f"解析失败/无记录: {len(bad)}")
    lines.append("")

    lines.append("## 缺失清单")
    lines.append("")
    if not missing:
        lines.append("roster 全部完成，无缺失。")
    else:
        by_task = defaultdict(list)
        for t, n, s in missing:
            by_task[t].append((n, s))
        for t in ALL_TASKS:
            if t not in by_task:
                continue
            items = ", ".join(f"n{n}_s{s}" for n, s in sorted(by_task[t]))
            lines.append(f"- **{t}**: {items}")
    if tagged:
        lines.append("")
        lines.append("Tagged 文件（文件名含后缀，如 `_r1cmp`，跳过聚合，仅作对照参考；"
                     "逐文件数值见下方表 B'）:")
        for p, t, n, s, tag, _d in tagged:
            lines.append(f"- `{os.path.basename(p)}`")
    if bad:
        lines.append("")
        lines.append("解析失败 / 无 records 的文件（可能仍在写入中）:")
        for p, err in bad:
            lines.append(f"- `{os.path.basename(p)}`: {err}")
    lines.append("")

    lines.append(render_table_a(table_a))
    lines.append("")
    lines.append(render_table_b(table_b))
    lines.append("")
    if tagged:
        # Only emitted when tagged files exist, so a clean roster-only run produces the
        # exact same report it always did.
        lines.append(render_tagged(table_b_tagged))
        lines.append("")
    lines.append(render_table_c(table_c))
    lines.append("")

    lines.append("## 要点（自动生成，仅陈述数字）")
    lines.append("")
    lines.append(render_summary(table_a, table_b, table_c))
    lines.append("")

    lines.append("## 图")
    lines.append("")
    for name in wrote:
        fp = os.path.join(args.figdir, name + ".pdf")
        try:
            fp = os.path.relpath(fp)
        except ValueError:
            fp = os.path.abspath(fp)
        lines.append(f"- `{fp}` — 已生成")
    for name, why in skipped_figs:
        lines.append(f"- `{name}` — 跳过：{why}")
    lines.append("")

    os.makedirs(os.path.dirname(outmd) or ".", exist_ok=True)
    with open(outmd, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    print(f"wrote {outmd}")
    print(f"roster: {n_have}/{n_expected} done, {len(missing)} missing, "
         f"{len(tagged)} tagged skipped, {len(bad)} unreadable")
    print(f"figures: wrote {wrote}, skipped {[n for n, _ in skipped_figs]}")


if __name__ == "__main__":
    main()
