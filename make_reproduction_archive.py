#!/usr/bin/env python
"""Pack the second-campaign result records into one zip, for readers who want them without git.

Contents: the run JSON of the 13 directories under results/r2/, the committed reports and selection
records next to them, the table fragments in results/tables/, the launch manifests in jobs/
(top level, shards/ and dev/ left out), REPRODUCIBILITY.md and a generated ARCHIVE_MANIFEST.md.
Local leftovers that git ignores (partial run copies, superseded report versions) are left out.

Usage:  python make_reproduction_archive.py [--out results/reproduction_archive.zip] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_R2 = REPO_ROOT / "results" / "r2"
TABLES = REPO_ROOT / "results" / "tables"
JOBS_DIR = REPO_ROOT / "jobs"
DEFAULT_OUT = REPO_ROOT / "results" / "reproduction_archive.zip"

# directory -> (released run count, what it backs in the final manuscript)
FORMAL_DIRS = {
    "d1_spectrum": (70, "Table 4, Fig 4, Fig S7, Section S7"),
    "d2_clip": (135, "Tables 5-7, Table S5, Fig S3"),
    "d2_noreset": (15, "Tables 5-6, Fig S3, §6.5, Section S6"),
    "d3_diag": (72, "Table 8, Fig 5, Section S6"),
    "d3_oat": (150, "Fig 5, Section S6"),
    "d4_tune": (824, "learning-rate selection, stage 1 (Table 13)"),
    "d4_stage2": (324, "learning-rate selection, stage 2 (Table 13)"),
    "d4_eval": (810, "Fig 3, Tables 5-7, 13, S1-S3, S6-S8, Figs S3, S6 right"),
    "d4_holdout": (50, "Table 9"),
    "d5_gated": (168, "Table 11"),
    "e4b": (27, "common-trajectory control, §6.2 and Section S7"),
    "rl": (18, "Table 5 (RL rows), Table S9, Fig S7"),
    "rl_clip05": (9, "Table 5 (RL rows), Table S9, Fig S7"),
}
SKIP_SUFFIXES = (".localpartial.json",)
SKIP_NAMES = {"D2_SUMMARY_noreset.md", "D3_SUMMARY.3task.md", "D3_SUMMARY.diag_only.md",
              "D3_SUMMARY_dev.md"}


def keep(p: Path) -> bool:
    return not (p.name.endswith(SKIP_SUFFIXES) or ".prev" in p.name or p.name in SKIP_NAMES)


def collect():
    runs, problems = {}, []
    for name, (n, _) in FORMAL_DIRS.items():
        files = sorted(f for f in (RESULTS_R2 / name).glob("*.json") if keep(f))
        if len(files) != n:
            problems.append("%s: %d run files, expected %d" % (name, len(files), n))
        runs[name] = files
    reports = sorted(f for f in RESULTS_R2.glob("*.md") if keep(f))
    reports += sorted(f for f in RESULTS_R2.glob("D4_*.json") if keep(f))
    reports.append(RESULTS_R2 / "d1_spectrum" / "D1_SUMMARY.md")
    tables = sorted(TABLES.glob("*.tex"))
    jobs = sorted(JOBS_DIR.glob("*.txt"))
    return runs, reports, tables, jobs, problems


def manifest(runs, reports, tables, jobs):
    L = ["# Reproduction archive", "",
         "Built %s by `make_reproduction_archive.py`. Table and figure numbers are those of the "
         "final manuscript; REPRODUCIBILITY.md maps each one to its tree, manifest and generator."
         % datetime.datetime.now().strftime("%Y-%m-%d"), "",
         "| directory | runs | used for |", "|---|---:|---|"]
    for name, files in runs.items():
        L.append("| `results/r2/%s` | %d | %s |" % (name, len(files), FORMAL_DIRS[name][1]))
    L.append("| total (%d directories) | %d | |" % (len(runs), sum(len(f) for f in runs.values())))
    L += ["", "Reports and selection records: " + ", ".join("`%s`" % p.name for p in reports), "",
          "Table fragments: " + ", ".join("`%s`" % p.name for p in tables), "",
          "Launch manifests: " + ", ".join("`%s`" % p.name for p in jobs), ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true", help="print the manifest, write nothing")
    args = ap.parse_args()
    runs, reports, tables, jobs, problems = collect()
    if problems:
        sys.exit("incomplete release:\n  " + "\n  ".join(problems))
    text = manifest(runs, reports, tables, jobs)
    if args.dry_run:
        print(text)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for files in runs.values():
            for f in files:
                zf.write(f, f.relative_to(REPO_ROOT).as_posix())
        for f in reports + tables + jobs + [REPO_ROOT / "REPRODUCIBILITY.md"]:
            zf.write(f, f.relative_to(REPO_ROOT).as_posix())
        zf.writestr("ARCHIVE_MANIFEST.md", text)
    print("wrote %s (%.1f MB)" % (args.out, args.out.stat().st_size / 2 ** 20))


if __name__ == "__main__":
    main()
