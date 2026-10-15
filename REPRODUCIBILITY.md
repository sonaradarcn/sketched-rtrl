# Reproducing the paper

This guide follows the final manuscript: main paper Tables 1-14 and Figures 1-6, supplementary
material Tables S1-S10 and Figures S1-S9. Section numbers (§) refer to the main paper, S-numbers to
the supplement.

Every number in the paper was produced with the implementation at commit `109e698`. This release
(tag `v1.0-neurocomputing`) adds the result records, the launch manifests, this guide and the
release check. It leaves the implementation unchanged: `git diff 109e698 -- skrtrl/ run_*.py`
shows comments, docstrings, help text and error-message text only.

All commands run from the repository root. The manuscript cites paths as `code/...`; here the
repository is the root, so `code/results/r2/` is `results/r2/`.

Commands come from three sources, labelled throughout this guide:

- **Manifest.** The second-campaign blocks (the 13 trees of section 5, 2,672 runs) have per-run
  launch manifests in `jobs/`: one command per line, each annotated with its output file.
  `check_release.py` compares every flag of every line with the `args` stored in that file
  (section 2).
- **Legacy.** The first-campaign blocks the paper still reports (Table 10, Fig 6, Figs S2, S4, S5,
  S6 left, S8, Table S10 and Fig S9, and the runs cited in §4.2, §6.3, §6.6, §7.4 and S9.3) have
  their commands in `docs/legacy_round1.md`, with their known run conditions. These are the loops
  we ran; no per-run manifest was kept.
- **Reconstructed.** The argument lists of the §4.3 kernel benchmark were not saved. Its commands
  in `docs/legacy_round1.md` were rebuilt from the fields recorded in the result files and are
  marked as reconstructed. (The second round of boundary extensions, section 4, step 3, is a
  manifest rebuilt the same way, and is checked flag by flag like the others.)

## 1. Environments

The runs come from two campaigns and one re-measurement.

| Batch | Machine | Software | Blocks |
|---|---|---|---|
| First campaign | Pool of NVIDIA TITAN Xp (12 GB) x3, T4, RTX 3060, plus two RTX 3080 (20 GB) | Python 3.10 (patch not recorded), PyTorch 2.3.0 + CUDA 12.1, NumPy 1.26.4, Matplotlib 3.9.0; cuDNN not recorded | `docs/legacy_round1.md` except its last section: Fig 6, Figs S2, S4, S5, S6 (left), S8, S9, Table S10, and the runs cited in §4.2, §6.3, §6.6, §7.4 and S9.3 |
| Re-measurement | Windows 10 workstation, one RTX 3080 (20 GB), driver 560.94 | Python 3.10.20, PyTorch 2.6.0+cu124, CUDA 12.4, cuDNN 9.1.0, NumPy 1.26.4 | Table 10 and the exact-path check (`results/revise_timing/`), the kernel benchmark of §4.3 (`results/revise_kernel/`), the re-run in `results/revise_repro/` |
| Second campaign, workstation | Same Windows 10 workstation, two RTX 3080 (20 GB) as GPU0 and GPU1, driver 560.94 | Python 3.10.20, PyTorch 2.6.0+cu124, CUDA 12.4, cuDNN 9.1.0, NumPy 1.26.4 | see section 5, column "ran on" |
| Second campaign, Linux | Ubuntu 24.04, one RTX 3080 (10 GB), driver 595.84 | Python 3.12.3, PyTorch 2.6.0+cu124, CUDA 12.4, cuDNN 9.1.0, NumPy 1.26.4 | see section 5 |
| Second campaign, rented | AutoDL node, two RTX 3080 (10 GB) | PyTorch 2.5.1+cu124; Python, NumPy and driver not recorded | five runs of `results/r2/rl_clip05` (corridor 20 seeds 1-2, corridor 40 seeds 0-2) |

Sources: the first-campaign versions are the pinned environment of the first submission; the
workstation and Linux versions were read from the interpreters the job queue used
(`D:\Anaconda\envs\multilingual_lora` and `env/` on the Linux machine, both installed before the
runs); the rented node's PyTorch version is in the launch log. Which machine ran which run was read
from the per-job queue logs, which are not part of the release. Every second-campaign run peaks
below 4.3 GB of GPU memory.

Three things can be reproduced, and they need different things.

1. **Reports, tables and figures from the released JSON.** `requirements.txt` (the second-campaign
   versions, with SciPy 1.15.3 and Matplotlib 3.10.9) is enough, on a CPU. With it the generators
   rewrite the committed reports and table fragments identically, apart from timestamp lines, and
   `check_release.py` passes. This is what the release guarantees.
2. **Re-running training.** This needs a CUDA GPU, the same code and a stated environment:
   section 1 above, per batch. Install the CUDA build of PyTorch with
   `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`. GPU training is
   not bit-reproducible: the runners do not switch on PyTorch's deterministic algorithms, the
   order of floating-point reductions on the GPU can change from run to run, even on the same
   card and library, and last-bit differences grow over 20,000 steps of a recurrent network.
3. **Agreement of a re-run with the released record.** Expect agreement within the seed-to-seed
   spread of the block, not digit for digit. `docs/acceptance_v1.0.md` shows one such re-run and
   the size of the difference.

## 2. Check the release

```bash
python check_release.py                                  # about two minutes, no GPU
python -m pytest tests -q --import-mode=importlib       # 16 tests, about one minute on CPU
```

`check_release.py` covers part of the paper, not every table and figure. This section lists what it
compares, what it only draws, and what it does not cover. Each output line starts with `ok`, `SKIP`
or `FAIL`, and the last line counts them (`check_release: N ok, M skipped, K failed`). The check
exits non-zero on any `FAIL`; a `SKIP` is a step that was not run and is reported as such.
`--layer 1|m|2` runs one layer.

**Inputs.** The 24 input trees must be present with their released run counts (`TREES` in the
script), together with the two selection records and the four pilot-spectrum files.

**Layer 1: regenerate and compare.** 15 generator commands run on the released trees, writing into
`results/_check/`. Their output is compared line by line with the committed copies, ignoring
timestamp lines and tex comment lines:

| Compared with the committed copy | Float |
|---|---|
| `results/r2/D4_SELECT_stage1.{md,json}`, `D4_SELECT_stage2.{md,json}` | Table 13 (selection) |
| `results/r2/D4_TABLES.md`, `D4_STATS.md`, `D4_STATS.json` | Tables 13, S1-S3, S6-S8 |
| `results/tables/tab_r2_{lr,stats,fidelity,timeseries,realts}.tex` | Tables 13, S1-S3, S6, S7, S8 |
| `results/tables/tab_r2_{adaptive,tightness}.tex` | Tables 8, S5 |
| `results/tables/tab_r2_clipcost.tex` | Table 7 |
| `results/r2/d1_spectrum/D1_SUMMARY.md` | Table 4 |
| `results/r2/D2_SUMMARY.md`, `RL_SUMMARY.md` | Tables 5, 6; §6.5 washout-free numbers |
| `results/r2/D3_SUMMARY.md` | §6.6, S6 |
| `results/r2/HOLDOUT_SUMMARY.md` | Table 9 |
| `results/r2/D5_SUMMARY.md` | Table 11 |
| `results/r2/RL_SPECTRUM_SUMMARY.md` | Table S9 |
| `results/r2/E4B_SUMMARY.md` | §6.2 common-trajectory control |
| `results/REPORT_m5iso.md` (`make_m5_report.py`) | Table S10 |

That is 16 reports and 8 table fragments. A generator that exits with an error or prints a `SKIP`
line for a figure fails the check.

**Figures: drawn, not compared.** The 17 figure files of Figs 3, 4, 5, 6 and S2-S9
(`fig_fidelity_bars`, `fig_rinterp_rotation`, `fig_r2_spectrum_stage`, `fig_r2_spectrum_width`,
`fig_r2_adaptive_diag`, `fig_r2_adaptive_oat`, `fig_scaling`, `fig_cert_c2sweep`,
`fig_r2_cert_frac_vs_clip`, `fig_r2_cert_cdf`, `fig_adaptive_trajectory`, `fig_pilot_residual`,
`fig_horizon_nmse`, `fig_fidelity_vs_error`, `fig_r2_spectrum_rl`, `fig_memory_time_pareto`,
`fig_rl_curves`) must be written. The check does not compare curves, panels or pixels with the
PDFs in the manuscript. What ties a figure to its data is that it is drawn from the same trees
whose reports and table cells are compared.

**Against the manuscript, when `../paper/secs` exists.** `compare_with_paper()` runs only when the
manuscript sources sit next to the repository, as in our working copy. It compares the bodies of
the 7 fragments the manuscript inputs (Tables 8, 13, S1-S3, S5-S8), the rows of Table 7 in
`secs/6_experiments.tex`, and the 18 success-rate cells of Table S10 in `secs/S8_scaling_rl.tex`
(recomputed from `results/m5iso`). In a clone without the manuscript it prints

```
  SKIP ../paper/secs not found next to the repository: table fragments, Table 7 rows and Table S10 NOT compared with the manuscript
```

and counts one skipped check. It reads the LaTeX sources, not the submitted PDFs.

**Manifest layer.** For every command line of every manifest in `jobs/` (not `jobs/shards/` or
`jobs/dev/`), it loads the run file named by the line's `# out=` annotation and compares every flag
on the line (`--task`, `--algo`, `--seed`, `--lr`, `--steps`, `--n`, `--batch`, `--clip`, `--cell`,
`--svd_driver`, `--tbptt_window`, `--env_len`, `--outdir`, `--tag` and the rest) with the `args`
stored in the file. It also checks that no file is named by two lines, that every run file of the
13 second-campaign trees is named by a line, that `d4_eval.txt` covers exactly 81
(task, estimator) pairs x seeds 0-9, that seeds 100-101 occur only in `d4_tune` and `d4_stage2`
and those trees hold no other seed, and that `make_r2_d4_select.py emit` re-creates
`jobs/d4_stage2.txt` and `jobs/d4_eval.txt` line for line from the committed selection records.
On this release: 18 manifests, 2,672 lines, 2,672 records, no mismatch.

**Layer 2: printed numbers.** It recomputes from the run JSON the numbers listed in
`paper_numbers.json` and compares them with the values stored there, within the tolerance stored
there: the 108 of 108 fidelity wins, the win-tie-loss records 24-18-3 / 26-16-3 / 30-12-3, the
certificate counters, the 93 Lemma-1 failures, the cells of Tables 5, 6, 9 and 11, the adaptive
mean ranks of Table 8, the 13.5x memory ratio, the §6.3 random-projection cosines and the numbers
§6.5, §7, §8 and S6 quote for the washout-free runs. `paper_numbers.json` is a copy of the values
the manuscript prints, transcribed by hand; each entry stores its source line. It is not extracted
from the PDF, so a change to a printed number has to be mirrored in it, or layer 2 compares the
records with the old value.

**Not covered by the script, and how each was checked by hand.**

| Float | Why not automatic | Hand check (2026-10-15) |
|---|---|---|
| Table 4 residual spectrum, Table S9 T-maze spectrum | typeset by hand from `D1_SUMMARY.md` and `RL_SPECTRUM_SUMMARY.md`, which layer 1 regenerates | all 88 cells compared with the reports: 82 equal at the printed precision; 6 sit on a rounding boundary of the reports' four-decimal display and agree within one unit of the last printed digit |
| Table 10 factored vs textbook exact | values read from `results/revise_timing/*.json`, no generator | time, s.d. and memory of the 9 rows compared with the JSON rows (three cells rounded at the last digit, section 10); throughput recomputed as 1000 / time; FLOPs are analytic, factored = 4 x textbook in every row |
| Table 12 protocol | a protocol summary, not a computed table | rows compared with the manifests and records (section 5); the manifest layer checks every launched value, but does not read Table 12 |
| Tables 1, 2, 3, S4 | analytic or notation, no data | none needed |
| Table 14 gated-cell constants | analytic | state size, per-unit and total parameter counts checked against `skrtrl/cells_gated.py` at n = 64, m = 1 |
| Figs 1, 2, S1 | hand-drawn diagrams, no data | none needed |

Checked on a fresh clone: `docs/acceptance_v1.0.md` records a clone of this release, the
installation, the tests, `check_release.py`, the figures, a GPU re-run of the worked example of
section 5 compared with its released record, and a dry run of `jobs/d4_eval.txt`, with commands,
return codes and times.

## 3. Tables and figures

"Manifest" is the launch file in `jobs/`; "legacy" means the command is in
`docs/legacy_round1.md`. Generators write to `results/figures/` and `results/tables/` unless told
otherwise.

### Main paper

| Float | Result tree | Manifest | Generator (output) |
|---|---|---|---|
| Table 1 unified comparison | none | none | analytic |
| Table 2 notation | none | none | none |
| Table 3 per-step cost | none | none | analytic |
| Fig 1 overview, Fig 2 decomposition | none | none | hand-drawn |
| Fig 3 fidelity bars, rank interpolation | `results/r2/d4_eval` | `d4_eval.txt` | `make_paper_figures.py` (`fig_fidelity_bars`, `fig_rinterp_rotation`) |
| Table 4 residual spectrum | `results/r2/d1_spectrum` | `d1_spectrum.txt` | `make_r2_d1_report.py` (`d1_spectrum/D1_SUMMARY.md`, tables A-C) |
| Fig 4 residual spectrum | `results/r2/d1_spectrum` | `d1_spectrum.txt` | `make_r2_d1_report.py` (`fig_r2_spectrum_stage`, `fig_r2_spectrum_width`) |
| Table 5 certificate vs clip | `d4_eval` (r=16), `d2_noreset`, `d2_clip`, `rl`, `rl_clip05` | `d4_eval.txt`, `d2_noreset.txt`, `d2_clip.txt`, `rl.txt` | `make_r2_d2_report.py` (`D2_SUMMARY.md` table B), `make_r2_rl_report.py` (`RL_SUMMARY.md` table RL-C) |
| Table 6 certificate per task | `d4_eval` (r=16), `d2_noreset`, `d2_clip` | as Table 5 | `make_r2_d2_report.py` (`D2_SUMMARY.md` table A) |
| §6.5, §7, §8, S6 washout-free runs (text) | `results/r2/d2_noreset`, `d4_eval` (r=16) | `d2_noreset.txt`, `d4_eval.txt` | `make_r2_d2_report.py` (`D2_SUMMARY.md` tables D and G) |
| Table 7 clip cost | `d4_eval` (r=16, seeds 0-4), `d2_clip` | `d4_eval.txt`, `d2_clip.txt` | `make_r2_clipcost_table.py` (`tab_r2_clipcost.tex`) |
| Table 8 adaptive rank | `results/r2/d3_diag` | `d3_diag.txt` | `make_r2_w1b_tables.py` (`tab_r2_adaptive.tex`) |
| Fig 5 adaptive diagnosis, sensitivity | `d3_diag`, `d3_oat` | `d3_diag.txt`, `d3_oat.txt`, `d3_oat_extra.txt` | `make_r2_d3_report.py` (`fig_r2_adaptive_diag`, `fig_r2_adaptive_oat`, `D3_SUMMARY.md`) |
| Table 9 temporal hold-out | `results/r2/d4_holdout` | `d4_holdout.txt` | `make_r2_holdout_report.py` (`HOLDOUT_SUMMARY.md`) |
| Table 10 factored vs textbook exact | `results/revise_timing` | legacy | values read from the two JSON files |
| Fig 6 scaling | `results/m3` (rotation), `scale`, `scale256`, `membench` | legacy | `make_paper_figures.py` (`fig_scaling`) |
| Table 11 gated cells | `results/r2/d5_gated` (+ `d4_eval` for the tanh reference) | `d5_gated.txt` | `make_r2_d5_report.py` (`D5_SUMMARY.md`) |
| Table 12 protocol | all manifests | all | run counts, section 5 |
| Table 13 selected learning rates | `d4_tune`, `d4_stage2` via `D4_SELECT_stage2.json` | section 4 | `make_r2_d4_tables.py` (`tab_r2_lr.tex`) |
| Table 14 gated-cell constants | none | none | analytic |

### Supplement

| Float | Result tree | Manifest | Generator (output) |
|---|---|---|---|
| Tables S1-S3 paired statistics | `results/r2/d4_eval` | `d4_eval.txt` | `make_r2_d4_tables.py` (`tab_r2_stats.tex`, `D4_STATS.md`, `D4_STATS.json`) |
| Table S4 notation | none | none | none |
| Table S5 certificate tightness | `results/r2/d2_clip` | `d2_clip.txt` | `make_r2_w1b_tables.py` (`tab_r2_tightness.tex`) |
| Table S6 diagnostic fidelity | `results/r2/d4_eval` | `d4_eval.txt` | `make_r2_d4_tables.py` (`tab_r2_fidelity.tex`) |
| Table S7 chaotic series | `results/r2/d4_eval` | `d4_eval.txt` | `make_r2_d4_tables.py` (`tab_r2_timeseries.tex`) |
| Table S8 real series | `results/r2/d4_eval` | `d4_eval.txt` | `make_r2_d4_tables.py` (`tab_r2_realts.tex`) |
| Table S9 T-maze residual spectrum | `rl`, `rl_clip05` | `rl.txt` | `make_r2_rl_spectrum_report.py` (`RL_SPECTRUM_SUMMARY.md`) |
| Table S10 T-maze success | `results/m5iso` | legacy | `make_m5_report.py results/m5iso` |
| Fig S1 pipeline | none | none | hand-drawn |
| Fig S2 older tightness sweep | `results/c2sweep` | legacy | `make_paper_figures.py` (`fig_cert_c2sweep`) |
| Fig S3 certificate vs clip | `d4_eval`, `d2_clip`, `d2_noreset` | `d4_eval.txt`, `d2_clip.txt`, `d2_noreset.txt` | `make_r2_d2_report.py` (`fig_r2_cert_frac_vs_clip`, `fig_r2_cert_cdf`) |
| Fig S4 adaptive trajectory | `results/round1/traj` | legacy | `make_paper_figures.py` (`fig_adaptive_trajectory`) |
| Fig S5 residual pilot | `results/m1_spectrum_*.json` | legacy | `make_paper_figures.py` (`fig_pilot_residual`) |
| Fig S6 left horizons | `results/round1/horizon`, `results/ts` (h=1) | legacy | `make_paper_figures.py` (`fig_horizon_nmse`) |
| Fig S6 right cosine vs error | `results/r2/d4_eval` | `d4_eval.txt` | `make_paper_figures.py` (`fig_fidelity_vs_error`) |
| Fig S7 T-maze spectrum | `rl`, `rl_clip05`, `d1_spectrum` | `rl.txt` | `make_r2_rl_spectrum_report.py` (`fig_r2_spectrum_rl`) |
| Fig S8 memory-time trade-off | `results/membench` | legacy | `make_paper_figures.py` (`fig_memory_time_pareto`) |
| Fig S9 T-maze curves | `results/m5iso` | legacy | `make_paper_figures.py` (`fig_rl_curves`) |

Regenerate everything without a GPU:

```bash
python make_r2_d4_tables.py --outdir results/r2       # Tables 13, S1-S3, S6-S8
python make_r2_w1b_tables.py                           # Tables 8, S5
python make_r2_clipcost_table.py                       # Table 7
python make_r2_d1_report.py                            # Table 4, Fig 4
python make_r2_d2_report.py                            # Tables 5-6, Fig S3, §6.5 no-reset numbers
python make_r2_d3_report.py                            # Fig 5
python make_r2_holdout_report.py                       # Table 9
python make_r2_d5_report.py                            # Table 11
python make_r2_rl_report.py                            # RL rows of Table 5
python make_r2_rl_spectrum_report.py                   # Table S9, Fig S7
python make_r2_e4b_report.py                           # common-trajectory control, §6.2
python make_paper_figures.py                           # Figs 3, 6, S2, S4, S5, S6, S8, S9
python make_m5_report.py results/m5iso                 # Table S10
```

The reports overwrite the committed copies in `results/r2/`; `git diff` should then show timestamp
lines only. Tables 4, 5, 6, 9, 10, 11 and S9 are typeset by hand from these reports.
Six figures drawn from first-campaign trees (Figs 6, S2, S4, S6 left, S8, S9) come out a few points
wider or taller than the PDFs in the manuscript, which were drawn before a later layout change;
the plotted data are the same.

## 4. Learning-rate tuning (Tables 13, S1-S3, S6-S8)

Every (task, estimator) pair is tuned on seeds 100 and 101 and evaluated on seeds 0-9. The chain:

1. Stage-1 grid. `jobs/d4_tune.txt`: 9 tasks x 9 estimators x lr {1e-4, 3e-4, 1e-3, 3e-3} x
   seeds {100, 101}, 6,000 steps; TBPTT also over windows {25, 50}. 720 runs, written by
   `make_r2_jobs.py --stage d4_tune`.
2. Stage-1 selection. `python make_r2_d4_select.py select --stage 1` scores the mean task metric
   over the last 20% of each run, averaged over the two seeds (never the gradient cosine), and
   writes `results/r2/D4_SELECT_stage1.{md,json}`.
3. Boundary extensions. A winner at a grid edge gets one more rung on the 1-3 ladder, at most
   twice. `jobs/d4_boundary.round1.txt` (39 configs, 78 runs) and `jobs/d4_boundary.round2.txt`
   (13 configs, 26 runs) were emitted by `make_r2_d4_select.py emit --what boundary`; the third
   emission, `jobs/d4_boundary.txt`, has no jobs. The round-2 file was rebuilt from the arguments
   recorded in its 26 run files, because the launch copy was overwritten by the third emission.
   All boundary runs write into `results/r2/d4_tune`, which therefore holds 824 runs
   (412 configs x 2 seeds), and step 2 is repeated after each extension.
4. Stage 2. `jobs/d4_stage2.txt`: the best two stage-1 configs of every pair, 20,000 steps, seeds
   {100, 101}; 324 runs in `results/r2/d4_stage2`.
5. Stage-2 selection. `python make_r2_d4_select.py select --stage 2` writes
   `results/r2/D4_SELECT_stage2.{md,json}`, the winner of each of the 81 pairs.
6. Evaluation. `make_r2_d4_select.py emit --what eval` writes `jobs/d4_eval.txt`: the selected rate
   (and TBPTT window) of each pair x seeds 0-9 x 20,000 steps, shadow on; 810 runs in
   `results/r2/d4_eval`. The tuning seeds never appear in it.
7. Tables. `python make_r2_d4_tables.py --outdir results/r2` reads `d4_eval` and
   `D4_SELECT_stage2.json` and writes Table 13 (`tab_r2_lr.tex`), Tables S6-S8 and Tables S1-S3
   (`tab_r2_stats.tex`, `D4_STATS.md`, `D4_STATS.json`).

The temporal hold-out (`jobs/d4_holdout.txt`) and the clip sweep (`jobs/d2_clip.txt`) read their
rates from the same `D4_SELECT_stage2.json`.

## 5. Second-campaign blocks (rows of Table 12)

Every block in this section has a per-run launch manifest in `jobs/`.

All blocks use a tanh cell (except the gated block), n=64, batch 8, Adam, `--svd_driver auto` and
no spectral clip unless the row says otherwise. "Ran on": W0/W1 = workstation GPU0/GPU1, L = Linux
machine, A = rented node; counts are runs.

| Block (Table 12 row) | Manifest | Runs | Steps | Seeds | lr | Other settings | Ran on |
|---|---|---|---|---|---|---|---|
| Estimator comparison | `d4_eval.txt` | 810 | 20,000 | 0-9 | tuned | 9 tasks x 9 estimators, shadow on | W0 324, L 486 |
| LR tuning, stage 1 | `d4_tune.txt`, `d4_boundary.round1.txt`, `d4_boundary.round2.txt` | 824 | 6,000 | 100-101 | grid | section 4 | W0 295, L 529 |
| LR tuning, stage 2 | `d4_stage2.txt` | 324 | 20,000 | 100-101 | top two | section 4 | W0 92, L 232 |
| Temporal hold-out | `d4_holdout.txt` | 50 | 20,000 | 0-4 | tuned | Sunspot, laser; exact, SK-RTRL r=16/32, SnAp-1, TBPTT; `--holdout_frac 0.2` | L 50 |
| Residual spectrum | `d1_spectrum.txt` (+ `d1_adding512_extra.txt`, `d1_adding512_s0long.txt`) | 63 (+7) | 20,000 (two at 40,000) | 0-2 (+3-7) | 1e-3 | exact-RTRL training, passive SnAp-1; n=64 on 9 tasks and n=128/256/512 on 4; batch 4 at n=512 | W1 63, L 7 |
| Certificate clip sweep | `d2_clip.txt` | 135 | 20,000 | 0-4 | tuned r=16 rate | SK-RTRL r=16, clip {0.35, 0.7, 0.9}, 9 tasks | L 135 |
| Certificate, no resets | `d2_noreset.txt` | 15 | 20,000 | 0-2 | 1e-3 | SK-RTRL r=16, `--washout 0`, 5 series | W1 15 |
| Adaptive rank, ceiling | `d3_diag.txt` | 72 | 12,000 | 0-2 | 1e-3 | clip 0.5; rotation, a^nb^n; 5 controller variants (30) + fixed r in {4,8,16,24,32,48,64} (42); variant C at n=128 | W1 72 |
| Controller sensitivity | `d3_oat.txt`, `d3_oat_extra.txt` | 90 + 60 | 12,000 | 0-2 | 1e-3 | clip 0.5; 10 one-at-a-time settings x 5 tasks | W0 60, W1 90 |
| Gated cells | `d5_gated.txt` | 168 | 20,000 | 0-2 | 1e-3 | `--cell gru` and `--cell lstm`; rotation, a^nb^n, Henon, Sunspot; 7 estimators | GRU W1 84, LSTM L 84 |
| Common-trajectory control | `e4b_common.txt` | 27 | 20,000 | 0-2 | 1e-3 | exact-RTRL trainer, 7 passive trackers, 9 tasks | L 27 |
| Online RL, certificate | `rl.txt` | 27 | 60k / 100k / 60k | 0-2 | 3e-4 | T-maze 10/20/40, batch 16, shadow on; SK-RTRL r=16 at clip 0 and 0.5, SnAp-1 at clip 0 | W0/W1 22, A 5 |

Two rates differ from the tuned rate of Table 13: the no-reset block ran at 1e-3 on all five series
(the tuned r=16 rate is 3e-4 on Lorenz and Mackey-Glass and 1e-4 on Sunspot), and the
common-trajectory trainer ran at 1e-3 on all nine tasks. The paper states both.

Run one manifest line, for example the first r=16 evaluation run:

```bash
python run_m3.py --task adding --algo skrtrl-r16 --seed 0 --lr 0.003 --steps 20000 --n 64 \
  --batch 8 --log_every 250 --shadow 1 --svd_driver auto --outdir results/rerun/d4_eval --tag lr3e-03
```

Run a whole manifest into a fresh directory (the runners skip a run whose output file exists, so
pointing them at the released tree does nothing):

```bash
grep -v '^#' jobs/d4_eval.txt | sed -e 's/  # out=.*//' -e 's/^python //' \
  -e 's#results/r2/#results/rerun/#' | while read -r cmd; do python $cmd; done
```

Change `d4_eval` for any other manifest. Lines in `rl.txt` start with `python`, which the `sed`
removes. `run_r2_queue.sh` and `run_r2_queue.ps1` split a manifest across GPUs
(`run_r2_queue.sh WORKER NWORKERS DEV JOBS LOGDIR [TIMEOUT_S] [PY]`); they skip a job whose
`# out=` file exists. The per-worker splits we used are in `jobs/shards/`; `jobs/dev/` holds smoke
tests, timing runs and the SVD-driver check; their outputs stay out of the repository.

On the real series, `--causal 1` selects the one-off window normalisation of S2.7 (the flag name is
older than that description).

## 6. First-campaign blocks

These blocks have no per-run manifest. `docs/legacy_round1.md` lists the commands, with their
known run conditions, for the floats still drawn from first-campaign trees (Table 10, Table S10,
Figs 6, S2, S4, S5, S6 left, S8, S9) and for the runs cited in the text (§4.2 width sweep, §4.3
kernel benchmark, §6.3 random-projection control, §6.6 control-signal ablation, §7.4 and S9.3
rank x clip and normalisation ablations). Its index table marks each block as an original run, a
re-measurement or a reconstruction; only the §4.3 kernel benchmark is reconstructed. These runs
used `--svd_driver gesvd`, the default of `run_m3.py`, `run_m5.py` and `run_adaptive.py`.

One first-campaign run is missing: `results/scale` has no rotation, SK-RTRL r=4, seed 0 run at
n=128, so the r=4 point of Fig 6 (left) at n=128 rests on seed 1 alone. No other legacy tree used
by a float lacks a run of its loop.

## 7. The certificate counts

Three statements, kept apart in the paper and here.

- Theorem 1 bounds the gradient bias in exact arithmetic, under its assumptions.
- In fp32, the gradient inequality ||g_t - ghat_t|| <= ||delta_t|| e_t was checked on every counted
  step of the 267 certified runs (the 240 supervised runs of `d4_eval` r=16, `d2_clip` and
  `d2_noreset`, and the 27 runs of `rl`/`rl_clip05`): 0 failures over 5,589,060 counted steps. On
  2,575,736 of the 3,614,460 counted supervised steps the fp32 bound was finite; on the others it
  had overflowed and holds trivially. The RL runs do not record finiteness.
- The intermediate Lemma-1 inequality ||E_t||_F <= e_t (1 + 1e-4) fails on 93 of the 1,980 counted
  steps of one unclipped run, copy r=16 seed 2 (`d4_eval/copy_skrtrl-r16_s2_*.json`).

The counters are `cert_summary.n_steps`, `n_bound_finite`, `n_bound_violation` (Theorem 1) and
`n_cert_invalid` (Lemma 1) in each run file; `make_r2_d2_report.py` tabulates them (table E) and
`check_release.py` recounts them. The "0 violations" line printed by `tests/test_numerics.py` is a
different check: the Lemma-1 inequality on a small float64 problem.

## 8. Gated cells and the GRU/LSTM baselines

Two scripts run GRU and LSTM cells, for different purposes.

- `run_m3.py --cell gru|lstm` runs SK-RTRL and the other online estimators on a gated cell
  (`skrtrl/cells_gated.py`). This is the block of Table 11 (`jobs/d5_gated.txt`). KF-RTRL and TBPTT
  are refused for gated cells.
- `run_arch.py` trains GRU, LSTM and ESN forecasters with truncated BPTT or a ridge readout. These
  are first-submission baselines in `results/round1/arch`; the final paper does not use them.

## 9. Result files

Each run file holds `args` (the full argument vector), `records` (the logged steps), `wall_s` and,
where a GPU was used, `peak_MB`. Second-campaign files add `cert_summary`, `svd_stats` (how many
SVD calls each driver served) and block-specific fields
(`checkpoints`, `spectrum`, `metric_holdout`, `avg_rank`); no released run that records
`svd_stats` needed a fallback driver. `results/scale_large.json`,
`results/revise_timing/*.json` and `results/revise_kernel/*.json` are lists of measurement rows.

The per-run value in every table is the mean over the records in the last 20% of the run; seeds
are the unit of aggregation. Tables 5, 6 and 8 report the sample s.d. over runs; Tables 9, 11
and S6-S8 the population s.d.

Directories under `results/` that are not in the tables above (`round1/real`, `round1/arch`, `c2`,
`round1/STATS_*`, `TIMESERIES_TABLES.md`, `m0_profile.json`, `cor3_timing.json`) are kept from the
first submission and are not used by the final paper.

## 10. Notes

- `--svd_driver auto` (every second-campaign run) calls the library's default SVD driver first and
  falls back to gesvd, a jittered gesvd and a CPU SVD only if that call fails or returns non-finite
  factors; `gesvd` starts at gesvd. `run_svd_driver_equiv.py` and `results/r2/svd_driver_equivalence.md`
  compare the two.
- Fig S2 plots `results/c2sweep` (12 runs, 1,192 logged points). `results/c2` is not plotted.
- The §6.3 random-projection runs are seeds 0-2 (`results/m3/*skrtrl-rp*`).
- Table 10 rounds three cells at the last digit (n=192 factored 133.45 ms printed 133.5; s.d.
  0.0075 and 0.0135 printed 0.008 and 0.014).
- Re-running a training block gives the same numbers only up to floating-point nondeterminism,
  on the same card and library as on another; the release check compares the released records,
  not new runs. Section 1 states what to expect from a re-run, and `docs/acceptance_v1.0.md`
  shows one.
