#!/usr/bin/env python
"""Build reproduction_archive_r2.zip for the SK-RTRL Neurocomputing R2 revision.

Read-only with respect to the source tree: the only thing this script writes
is the output zip (default: ../paper/revise_r2/reproduction_archive_r2.zip).
The manifest (ARCHIVE_MANIFEST.md) is generated in memory and written
straight into the zip's top level -- no loose copy is left in the repo.

What goes in (see paper/revise_r2/PACKAGING_CHECKLIST.md section 2.2(e) and
section 5 item 4 for the underlying "formal vs. smoke/probe/dev" call):

  1. results/r2/<FORMAL_DIRS>/*.json  -- every json in the 13 directories that
     correspond to a named, report-backed R2 experiment (d1_spectrum,
     d2_noreset, d2_clip, d3_diag, d3_oat, d4_tune, d4_stage2, d4_eval,
     d4_holdout, d5_gated, rl, rl_clip05, e4b). Excludes the smoke/probe/dev/
     timing/svd_equiv directories sitting alongside them (a5_*, a8_*, a9_*,
     a11_*, a13_*, d1_probe, d1_smoke, d1_timing, figs_dev, smoke*,
     svd_equiv, timing).
     rl_clip05 was reclassified from "excluded variant" to formal on
     2026-09-15: RL_SUMMARY.md and RL_SPECTRUM_SUMMARY.md are both generated
     from results/r2/rl PLUS results/r2/rl_clip05, and the clip-0.5 numbers
     they carry are quoted in the manuscript (secs/D_rlcase.tex, the
     "$0.945\\pm0.038$ of steps at clip $0.5$" sentence) and in the
     supplement (secs/S7_diagnostics.tex, secs/S8_scaling_rl.tex). Shipping
     those summaries without the clip-0.5 runs left half the underlying data
     for reported numbers out of the archive.
  2. results/r2/*.md  -- every top-level summary report (D2_SUMMARY*.md,
     D3_SUMMARY*.md, D4_TABLES.md, D4_STATS.md, D5_SUMMARY.md,
     HOLDOUT_SUMMARY.md, D4_SELECT_stage1(.prev/.prev2)/stage2.md,
     svd_driver_equivalence.md). Files *inside* a formal dir (e.g.
     d1_spectrum/D1_SUMMARY.md) are NOT duplicated here; the per-directory
     json set above already carries that directory's own data.
  3. jobs/*.txt -- the formal job-launch scripts, i.e. the ones that produced
     data under one of FORMAL_DIRS or fed svd_driver_equivalence.md.
     Excluded:
       - scheduling-shard splits of a formal job list across workers/GPUs/
         rounds: filename matches _tail, _mid, _gpu<N>, or .round<N>
         (e.g. d4_eval_gpu0.txt, d4_eval_gpu1.txt, d4_stage2_tail.txt,
         rl_tail.txt, d4_boundary.round1.txt). The complete, non-sharded job
         file is archived instead.
       - orphans with no corresponding archived directory/summary: smoke.txt,
         timing128.txt, timing_final.txt (results/r2/timing and .../smoke are
         not part of this archive's scope).

Usage:
    python make_reproduction_archive.py [--out PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent  # .../RTRL/code
RESULTS_R2 = REPO_ROOT / "results" / "r2"
JOBS_DIR = REPO_ROOT / "jobs"
DEFAULT_OUT = REPO_ROOT.parent / "paper" / "revise_r2" / "reproduction_archive_r2.zip"

FORMAL_DIRS = [
    "d1_spectrum", "d2_noreset", "d2_clip", "d3_diag", "d3_oat",
    "d4_tune", "d4_stage2", "d4_eval", "d4_holdout", "d5_gated",
    "rl", "rl_clip05", "e4b",
]

# Table/figure mapping for the manifest narrative only -- sourced 2026-09-14
# from results/r2/*.md report headers and paper/secs/*.tex \label/\Cref/\input
# sites (grep, read-only). Not authoritative on the paper; cross-check
# secs/6_experiments.tex, secs/C_protocol.tex, secs/Z_suppstats.tex and the
# tab_r2_*.tex files directly before citing this in the submission.
DIR_PAPER_MAP = {
    "d1_spectrum": (
        "D1 residual-spectrum experiment (Reviewer #3 pt.1). "
        "Table tab:r2spectrum (secs/6_experiments.tex); "
        "Fig. fig_r2_spectrum_stage, fig_r2_spectrum_width."
    ),
    "d2_clip": (
        "D2 certificate informativeness (R3-2), clip-level sweep. "
        "Table tab:r2cert, tab:r2clipcost, tab:r2certtask; "
        "Fig. fig_r2_cert_frac_vs_clip, fig_r2_cert_cdf."
    ),
    "d2_noreset": (
        "D2 certificate informativeness (R3-2), no-reset variant "
        "(D2_SUMMARY_noreset.md). Table tab:tightness (secs/tab_r2_tightness.tex)."
    ),
    "d3_diag": (
        "D3 adaptive-rank ceiling diagnosis (Reviewer #3 pt.3). "
        "Table tab:adaptive (secs/tab_r2_adaptive.tex); "
        "Fig. fig_r2_adaptive_diag, fig_adaptive_trajectory."
    ),
    "d3_oat": (
        "D3 one-at-a-time sensitivity sweep (Reviewer #3 pt.3). "
        "Table tab:cor3 (secs/6_experiments.tex); "
        "Fig. fig_r2_adaptive_oat, fig_cert_c2sweep."
    ),
    "d4_tune": (
        "D4 stage-1 learning-rate tuning grid. Selects D4_SELECT_stage1*.json; "
        "not itself a table/figure source -- feeds the D4 unified tables via "
        "the LR selection."
    ),
    "d4_stage2": (
        "D4 stage-2 tuning refinement (subset of tuned pairs). "
        "Selects D4_SELECT_stage2.json, used by D4_TABLES.md/D4_STATS.md."
    ),
    "d4_eval": (
        "D4 unified-protocol main evaluation (10 seeds, tuned LR). "
        "Tables 3/6/7 replacements: tab:r2fidelity, tab:r2timeseries, "
        "tab:r2realts, tab:r2lr, tab:r2stats/r2stats2/r2stats3 "
        "(secs/tab_r2_*.tex, secs/S1_suppstats.tex); "
        "Fig. fig_fidelity_bars, fig_horizon_nmse, fig_fidelity_vs_error, "
        "fig_scaling, fig_memory_time_pareto."
    ),
    "d4_holdout": (
        "A11 temporal hold-out on sunspot/laser. Table tab:r2holdout "
        "(secs/tab_r2_holdout.tex, \\input at secs/6_experiments.tex:764; "
        "\\Cref also in secs/C_protocol.tex, secs/8_conclusion.tex)."
    ),
    "d5_gated": (
        "D5 gated cells (GRU/LSTM) applicability. Table tab:gated "
        "(secs/6_experiments.tex), tab:app-gated-sizes (secs/E_gated_appendix.tex)."
    ),
    "rl": (
        "RL case study (T-maze), unclipped arm. Table tab:rl, Fig. "
        "fig_rl_curves (secs/D_rlcase.tex, \\input at "
        "secs/6_experiments.tex:654); residual spectrum in "
        "secs/S7_diagnostics.tex (Fig. fig_r2_spectrum_rl), memory--time and "
        "full case study in secs/S8_scaling_rl.tex. Backs RL_SUMMARY.md and "
        "RL_SPECTRUM_SUMMARY.md together with rl_clip05."
    ),
    "rl_clip05": (
        "RL case study (T-maze), clip-0.5 operating point -- the arm that "
        "makes the certificate informative. Quoted in secs/D_rlcase.tex "
        "(the $0.945\\pm0.038$-of-steps sentence), secs/S7_diagnostics.tex "
        "(clip-$0.5$ row of the RL spectrum table) and "
        "secs/S8_scaling_rl.tex. Aggregated jointly with rl into "
        "RL_SUMMARY.md and RL_SPECTRUM_SUMMARY.md."
    ),
    "e4b": (
        "D4b common-trajectory estimator comparison (passive trackers sharing "
        "one exact-RTRL trajectory), reported in E4B_SUMMARY.md. Cited from "
        "secs/6_experiments.tex:69 (the common-trajectory control is "
        "consistent with the tuned-run ranking), with the protocol in "
        "secs/C_protocol.tex:83 and secs/S2_environment.tex:37 and the "
        "seed count in secs/S9_discussion.tex:299."
    ),
}

JOBS_SHARD_RE = re.compile(r"(_tail|_mid|_gpu\d*|\.round\d+)")
JOBS_ORPHAN = {"smoke", "timing128", "timing_final"}


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def iter_formal_json(results_dir: Path):
    """Yield (dirname, dir_path, sorted json file list) for each formal dir."""
    for name in FORMAL_DIRS:
        d = results_dir / name
        if not d.is_dir():
            print(f"WARNING: formal dir missing, skipped: {d}", file=sys.stderr)
            yield name, d, []
            continue
        files = sorted(d.glob("*.json"))
        yield name, d, files


def iter_top_md(results_dir: Path):
    return sorted(results_dir.glob("*.md"))


def iter_formal_jobs(jobs_dir: Path):
    out = []
    if not jobs_dir.is_dir():
        print(f"WARNING: jobs dir missing: {jobs_dir}", file=sys.stderr)
        return out
    for f in sorted(jobs_dir.glob("*.txt")):
        if f.stem in JOBS_ORPHAN:
            continue
        if JOBS_SHARD_RE.search(f.name):
            continue
        out.append(f)
    return out


def build_manifest(dir_entries, top_md, job_files, out_path: Path) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# Reproduction Archive Manifest (R2)")
    lines.append("")
    lines.append(f"Generated {now} by `code/make_reproduction_archive.py`.")
    lines.append(
        "Scope and inclusion/exclusion rules are documented in "
        "`paper/revise_r2/PACKAGING_CHECKLIST.md` (sections 2.2(e) and 5.4) "
        "and in this script's module docstring."
    )
    lines.append("")
    lines.append(
        "**Status note:** at generation time, the `rl` and `e4b` experiments "
        "had not finished running (jobs still in flight on the GPU workers). "
        "The counts below reflect the partial state at generation time; this "
        "archive must be regenerated once those two runs complete, before "
        "final submission."
    )
    lines.append("")
    lines.append("## 1. Formal experiment directories (`results/r2/<dir>/*.json`)")
    lines.append("")
    lines.append("| directory | json files | size | paper table(s) / figure(s) |")
    lines.append("|---|---:|---:|---|")
    total_json = 0
    total_json_size = 0
    for name, d, files in dir_entries:
        size = sum(f.stat().st_size for f in files)
        total_json += len(files)
        total_json_size += size
        mapping = DIR_PAPER_MAP.get(name, "(no mapping recorded)")
        status = "" if d.is_dir() else " **(directory missing)**"
        lines.append(f"| `{name}`{status} | {len(files)} | {human_size(size)} | {mapping} |")
    lines.append(f"| **total** | **{total_json}** | **{human_size(total_json_size)}** | |")
    lines.append("")
    lines.append("## 2. Top-level summary reports (`results/r2/*.md`)")
    lines.append("")
    lines.append("| file | size |")
    lines.append("|---|---:|")
    total_md_size = 0
    for f in top_md:
        size = f.stat().st_size
        total_md_size += size
        lines.append(f"| `{f.name}` | {human_size(size)} |")
    lines.append(f"| **total ({len(top_md)} files)** | **{human_size(total_md_size)}** |")
    lines.append("")
    lines.append(
        "`svd_driver_equivalence.md` (A1 -- SVD-driver equivalence regression, "
        "`skrtrl/algos.py::_robust_svd`, backing the QR-based `gesvd` driver "
        "sentence in `secs/4_method.tex` and `secs/S4_method_details.tex`) is "
        "included in this set. `RL_SUMMARY.md` and `RL_SPECTRUM_SUMMARY.md` "
        "aggregate `rl` + `rl_clip05`; `E4B_SUMMARY.md` reports the `e4b` "
        "common-trajectory control."
    )
    lines.append("")
    lines.append("## 3. Job launch scripts (`jobs/*.txt`, formal subset)")
    lines.append("")
    lines.append(
        "Excludes scheduling-shard splits (`_tail`, `_mid`, `_gpu<N>`, "
        "`.round<N>` -- the complete job list is archived instead of the "
        "per-worker/per-round fragment) and orphans with no archived "
        "directory/summary in this package (`smoke.txt`, `timing128.txt`, "
        "`timing_final.txt`)."
    )
    lines.append("")
    lines.append("| file | size |")
    lines.append("|---|---:|")
    total_jobs_size = 0
    for f in job_files:
        size = f.stat().st_size
        total_jobs_size += size
        lines.append(f"| `{f.name}` | {human_size(size)} |")
    lines.append(f"| **total ({len(job_files)} files)** | **{human_size(total_jobs_size)}** |")
    lines.append("")
    excluded_shards = sorted(
        f.name for f in JOBS_DIR.glob("*.txt")
        if JOBS_SHARD_RE.search(f.name) and f.stem not in JOBS_ORPHAN
    ) if JOBS_DIR.is_dir() else []
    excluded_orphans = sorted(
        f.name for f in JOBS_DIR.glob("*.txt") if f.stem in JOBS_ORPHAN
    ) if JOBS_DIR.is_dir() else []
    if excluded_shards:
        lines.append("Excluded as scheduling shards: " + ", ".join(f"`{n}`" for n in excluded_shards))
        lines.append("")
    if excluded_orphans:
        lines.append("Excluded as orphans (no matching archived dir/summary): " + ", ".join(f"`{n}`" for n in excluded_orphans))
        lines.append("")
    lines.append("## 4. Excluded directories under `results/r2/`")
    lines.append("")
    lines.append(
        "Smoke/probe/dev/timing/svd_equiv directories are intentionally left "
        "out of this archive: `a5_cpu`, `a5_tbptt`, `a8_smoke`, `a9_smoke`, "
        "`a11_smoke`, `a13_probe`, `d1_probe`, `d1_smoke`, `d1_timing`, "
        "`figs_dev`, `smoke`, `smoke_queue`, `svd_equiv`, `timing`. "
        "`rl_clip05` is NOT excluded: it is the clip-0.5 arm of the RL case "
        "study and is archived as a formal directory (see section 1)."
    )
    lines.append("")
    lines.append(f"Archive built to: `{out_path}`")
    lines.append("")
    return "\n".join(lines)


def build(out_path: Path, dry_run: bool = False) -> None:
    dir_entries = list(iter_formal_json(RESULTS_R2))
    top_md = iter_top_md(RESULTS_R2)
    job_files = iter_formal_jobs(JOBS_DIR)

    manifest = build_manifest(dir_entries, top_md, job_files, out_path)

    total_entries = sum(len(files) for _, _, files in dir_entries) + len(top_md) + len(job_files) + 1
    print(f"Planned entries: {total_entries} "
          f"({sum(len(f) for _, _, f in dir_entries)} json, {len(top_md)} md, "
          f"{len(job_files)} jobs, 1 manifest)")

    if dry_run:
        print(manifest)
        print("[dry-run] not writing zip")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, d, files in dir_entries:
            for f in files:
                arcname = f"results/r2/{name}/{f.name}"
                zf.write(f, arcname.replace("\\", "/"))
        for f in top_md:
            arcname = f"results/r2/{f.name}"
            zf.write(f, arcname.replace("\\", "/"))
        for f in job_files:
            arcname = f"jobs/{f.name}"
            zf.write(f, arcname.replace("\\", "/"))
        zf.writestr("ARCHIVE_MANIFEST.md", manifest)

    zip_size = out_path.stat().st_size
    print(f"Wrote {out_path} ({human_size(zip_size)}, {zip_size} bytes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output zip path")
    ap.add_argument("--dry-run", action="store_true", help="print manifest and counts, do not write the zip")
    args = ap.parse_args()
    build(args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
