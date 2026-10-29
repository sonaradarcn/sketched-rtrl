# Sketched RTRL (SK-RTRL)

Code, result records and launch manifests for the paper *Certified Low-Rank Real-Time Recurrent
Learning for Dense Recurrent Neural Networks*.

SK-RTRL keeps the RTRL influence matrix as the exact SnAp-1 block-diagonal trace `S` plus a
deterministic rank-r sketch `L Rᵀ` of the off-diagonal residual. One rank knob interpolates between
SnAp-1 (r = 0) and exact RTRL (r = n). The discarded mass of each truncation feeds a running scalar
`e_t` that bounds the influence error and, through Theorem 1, the gradient bias.

Every number in the paper was produced with the implementation at commit `109e698`. This release
(tag `v1.0-neurocomputing`) adds the result records, the launch manifests, the reproduction guide
and a release check; `git diff 109e698 -- skrtrl/ run_*.py` shows comments, docstrings, help text
and error-message text only. The materials cited by the paper are those at tag
`v1.0-neurocomputing` (commit `583dc9f`); the second acceptance section of `docs/acceptance_v1.0.md` and the helper scripts under `tools/`
were added on `main` afterwards, without changing code, data or the release check.

Table and figure numbers below are those of the final manuscript (main paper Tables 1-14 and
Figs 1-6, supplement Tables S1-S10 and Figs S1-S9). [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
maps each of them to its result tree, launch manifest and generator, and gives the protocol of
every block.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or later. Two uses of the release need different things.

- Regenerating the reports, table fragments and figures from the released JSON, and running
  `check_release.py`, needs only `requirements.txt` and a CPU. With those versions the reports
  and fragments come out identical to the committed copies, apart from timestamp lines.
- Re-running training needs a CUDA GPU, the same code and a stated environment. The paper's runs
  used PyTorch 2.3.0 + CUDA 12.1 (first campaign) and PyTorch 2.6.0 + CUDA 12.4 (second campaign;
  five runs under 2.5.1); REPRODUCIBILITY.md, section 1, lists the versions per batch. GPU training
  is not bit-reproducible, even on the same card and library, so a re-run agrees with the
  released record only up to floating-point nondeterminism, and the agreement to expect is within
  the seed-to-seed spread of the block, not digit for digit.

## Checking the release

```bash
python -m pytest tests -q --import-mode=importlib    # 16 tests, about a minute on CPU
python check_release.py                              # about two minutes on CPU, no GPU
```

The tests cover exact RTRL against autograd BPTT, SK-RTRL at r = n against exact RTRL, the
Lemma-1 inequality on a small float64 problem, the amortised kernel against the released one, the
GRU/LSTM Jacobians and the SVD fallback chain.

`check_release.py` covers part of the paper, not every table and figure. Exactly:

- Layer 1 re-runs 15 generator commands on the released trees and compares 16 reports and 8 table
  fragments line by line with the committed copies (Tables 7, 8, 13, S1-S3, S5-S8 as fragments;
  the reports behind Tables 4, 5, 6, 9, 11, S9 and S10). It checks that the 17 figure files of
  Figs 3-6 and S2-S9 are drawn, but not what they show.
- When the manuscript sources sit next to the repository (`../paper/secs`), layer 1 also compares
  the fragment bodies, the rows of Table 7 and the 18 cells of Table S10 with the manuscript.
  Without them it prints a `SKIP` line and does not compare.
- The manifest layer checks every launch line in `jobs/*.txt` against the record it names (below).
- Layer 2 recomputes the numbers listed in `paper_numbers.json` from the run JSON: the cells of
  Tables 5, 6, 9 and 11, the mean ranks of Table 8, and headline counts quoted in the text.
  `paper_numbers.json` is a copy of the printed values, transcribed by hand, with a tolerance per
  entry; it is not read from the PDF, so a change to a printed number has to be made there too.
- Not checked by the script: Tables 4 and S9 (typeset by hand from the reports), Table 10 (read
  from `results/revise_timing/*.json`), Table 12 (the protocol table), the analytic and notation
  Tables 1-3, 14 and S4, and the hand-drawn Figs 1, 2 and S1. REPRODUCIBILITY.md, section 2, says
  how each was checked by hand.

The last line of the output counts the checks that passed, were skipped and failed; the check
fails on a missing input, a skipped figure or a mismatch.

Checked on a fresh clone: [`docs/acceptance_v1.0.md`](docs/acceptance_v1.0.md) records the
installation, the tests, the release check, the figures, a GPU re-run of the worked example below
and a dry run of its manifest, with commands, return codes and times, once at the parent commit of
the tag and once on a clone checked out at the tag. The re-run at the tag is compared with its released record
by `python tools/compare_rerun.py` and the manifest by `python tools/dry_run_manifest.py
jobs/d4_eval.txt`, both run from the repository root. The tag does not contain `tools/`; on a
clone checked out at the tag, copy the scripts in from `main` first and remove them afterwards:

```
mkdir -p tools
git show 8a549fe:tools/compare_rerun.py > tools/compare_rerun.py
git show 8a549fe:tools/dry_run_manifest.py > tools/dry_run_manifest.py
python tools/compare_rerun.py   # after the GPU re-run of the worked example below
python tools/dry_run_manifest.py jobs/d4_eval.txt
rm -r tools
git status --short       # prints nothing
```

## Layout

```
skrtrl/                  the library: cells, gated cells, estimators, tasks, RL environment
skrtrl/algos.py          exact RTRL, SK-RTRL (r = 0 is SnAp-1), the SVD fallback chain
skrtrl/algos_amortised.py   amortised-rotation kernel of §4.3 (same results, no timing in the paper uses it)
skrtrl/data/             Sunspot and Santa Fe laser series (see THIRD_PARTY_DATA.md)
run_m3.py                supervised runs: diagnostic tasks, chaotic and real series, gated cells
run_adaptive.py          certificate-guided adaptive rank
run_e1_spectrum.py       residual spectrum R_t = J_t - S_t under exact-RTRL training
run_e4b_common.py        common-trajectory control (passive trackers on one exact-RTRL run)
run_m5.py                online T-maze actor-critic
run_cor3_timing_revise.py, run_width_sweep.py, run_amortised_kernel_bench.py   measurement blocks
jobs/                    launch manifests, one command per line (shards/ and dev/ hold worker splits and smoke tests)
make_r2_*.py             reports and table fragments for the second-campaign blocks
make_paper_figures.py    Figs 3, 6, S2, S4, S5, S6, S8, S9
check_release.py, paper_numbers.json   the release check
results/                 one JSON per run, plus the reports
docs/legacy_round1.md    commands for the first-campaign runs the paper still uses
```

## Where each table and figure comes from

| Paper item | Generator |
|---|---|
| Fig 3 | `make_paper_figures.py` |
| Table 4, Fig 4 | `make_r2_d1_report.py` |
| Tables 5, 6, Fig S3 | `make_r2_d2_report.py` (RL rows of Table 5: `make_r2_rl_report.py`) |
| Table 7 | `make_r2_clipcost_table.py` |
| Table 8, Table S5 | `make_r2_w1b_tables.py` |
| Fig 5 | `make_r2_d3_report.py` |
| Table 9 | `make_r2_holdout_report.py` |
| Table 10 | `results/revise_timing/*.json` (docs/legacy_round1.md) |
| Fig 6, Figs S2, S4, S5, S6, S8, S9 | `make_paper_figures.py` |
| Table 11 | `make_r2_d5_report.py` |
| Table 13, Tables S1-S3, S6-S8 | `make_r2_d4_tables.py` |
| Table S9, Fig S7 | `make_r2_rl_spectrum_report.py` |
| Table S10 | `make_m5_report.py results/m5iso` |
| Tables 1-3, 12, 14, S4; Figs 1, 2, S1 | analytic, protocol or hand-drawn; no data |

None of these needs a GPU. Figures go to `results/figures/`, tex fragments to `results/tables/`.

## Results shipped

Second campaign, `results/r2/` (13 directories, 2,672 runs):

| Directory | Runs | Used for |
|---|---|---|
| `d4_tune`, `d4_stage2` | 824, 324 | learning-rate selection (Table 13) |
| `d4_eval` | 810 | Fig 3, Tables 5-7, 13, S1-S3, S6-S8, Figs S3, S6 right |
| `d4_holdout` | 50 | Table 9 |
| `d1_spectrum` | 70 | Table 4, Fig 4, Fig S7, Section S7 |
| `d2_clip`, `d2_noreset` | 135, 15 | Tables 5-7, S5, Fig S3 |
| `d3_diag`, `d3_oat` | 72, 150 | Table 8, Fig 5, Section S6 |
| `d5_gated` | 168 | Table 11 |
| `e4b` | 27 | common-trajectory control, §6.2 |
| `rl`, `rl_clip05` | 18, 9 | Table 5, Table S9, Fig S7 |

The reports built from them (`results/r2/*.md`, `D4_STATS.json`, the two `D4_SELECT_stage*`
selection records) and the table fragments (`results/tables/`) are committed.

First campaign: `m3` (202 runs; Fig 6, §6.3), `scale` and `scale256` (23, 24; Fig 6),
`membench` (16; Fig 6, Fig S8), `c2sweep` (12; Fig S2), `round1/traj` (1; Fig S4),
`m1_spectrum_*.json` (4; Fig S5), `round1/horizon` and `ts` (225, 240; Fig S6 left), `m5iso`
(54; Table S10, Fig S9), `round1/adapt_ablate` (18; §6.6), `round1/ablate_rank`,
`round1/ablate_norm` and `revise_repro` (§7.4, S9.3), `revise_timing` (Table 10),
`revise_kernel` (§4.3), `scale_large.json` (§4.2). `round1/real`, `round1/arch`, `c2` and the
`STATS_*` files are kept from the first submission and are not used by the final paper.

## Re-running an experiment

Commands come from three sources, and the guide labels each.

- Second campaign (13 result trees, 2,672 runs): per-run launch manifests in `jobs/`, one command
  per line, each naming its output file. `check_release.py` checks every flag of every line
  against the `args` stored in that file.
- First campaign, for the blocks the paper still reports (Table 10, Fig 6, Figs S2, S4, S5,
  S6 left, S8, Table S10 and Fig S9, and runs cited in the text): the commands are in
  `docs/legacy_round1.md`, with their known run conditions. These are the loops we ran; no
  per-run manifest was kept.
- The §4.3 kernel benchmark: its argument lists were not saved. The commands in
  `docs/legacy_round1.md` were reconstructed from the fields recorded in the result files and are
  marked as reconstructed.

Each line of a manifest in `jobs/` is one run. For example, the first SK-RTRL r=16 run of the
estimator comparison:

```bash
python run_m3.py --task adding --algo skrtrl-r16 --seed 0 --lr 0.003 --steps 20000 --n 64 \
  --batch 8 --log_every 250 --shadow 1 --svd_driver auto --outdir results/rerun/d4_eval --tag lr3e-03
```

For another run, take its line from the manifest and change `results/r2/` to a new directory; the
runners skip a run whose output file already exists. REPRODUCIBILITY.md gives the loop that runs a
whole manifest, the learning-rate tuning chain (stage-1 grid, boundary extensions, stage 2,
evaluation on seeds 0-9 with tuning seeds 100-101 kept apart), and the steps, seeds, clip, cell,
width and machine of every block.

## The certificate in the code

- Theorem 1 holds in exact arithmetic, under its assumptions.
- In fp32 the gradient inequality was checked on every counted step of the 267 certified runs:
  0 failures over 5,589,060 steps. On 2,575,736 of the 3,614,460 counted supervised steps the fp32
  bound was finite; on the rest it had overflowed and holds trivially.
- The intermediate Lemma-1 inequality fails on 93 of the 1,980 counted steps of one unclipped run
  (copy, r = 16, seed 2).

The counters are in each run's `cert_summary`; `check_release.py` recounts them. The
"0 violations" printed by `tests/test_numerics.py` is the Lemma-1 check on a small float64 problem,
not these counts.

## GRU and LSTM

`run_m3.py --cell gru|lstm` runs SK-RTRL and the other online estimators on a gated cell; this is
the structural check of Table 11 (`jobs/d5_gated.txt`). `run_arch.py` is a different thing: GRU,
LSTM and ESN forecasters trained with truncated BPTT or a ridge readout, first-submission baselines
that the final paper does not use.

## Data

`skrtrl/data/sunspot.txt` is the SILSO monthly mean total sunspot number (`SN_m_tot_V2.0`,
1749-01 to 2026-05) and `skrtrl/data/laser.txt` is data set A of the Santa Fe competition. Both are
third-party data; see [`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md) for sources, checksums and terms.
Every other task is generated from the run seed.

## License

MIT for the code and the result records (see [`LICENSE`](LICENSE)). The two data files are not
covered by it.

## Citation

```bibtex
@article{SK-RTRL,
title = {Certified low-rank real-time recurrent learning for dense recurrent neural networks},
journal = {Neurocomputing},
pages = {135543},
year = {2026},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2026.135543},
url = {https://www.sciencedirect.com/science/article/pii/S0925231226029413},
author = {Junfei Yi and Yuxiang Wang},
keywords = {Real-time recurrent learning, Online learning, Recurrent neural networks, Low-rank approximation, Gradient estimation, Chaotic time series},
abstract = {Learning from an unbounded stream means updating weights at each observation, never revisiting the past. Real-Time Recurrent Learning (RTRL) is exact but costs quartic time and cubic memory in a dense network’s width; cheap successors add uncontrolled variance, discard cross-unit credit without an error bound, or abandon dense recurrence. We introduce Sketched RTRL (SK-RTRL), keeping the influence matrix as an exact block-diagonal part plus a deterministic low-rank sketch of the discarded residual, one rank knob spanning that part alone to exact RTRL. The same shrinkage yields, at no extra asymptotic cost, a running scalar certificate bounding the gradient bias at every step, to our knowledge the first such bound for an RTRL approximation. Truncation records its discarded mass, so the rank resizes online without breaking the bound. The sketch holds a consistent gradient-cosine advantage over four approximate estimators on all nine tasks and three ranks tested, 108 of 108 paired comparisons, and cuts online training memory 13.5 times at width 512. Task-level benefit varies: higher fidelity need not lower prediction error, no consistent advantage over tuned truncated BPTT, and rotation memory is an outright loss. The certificate holds in exact arithmetic and on every counted fp32 step, but is informative only at the clipped operating points measured, essentially vacuous without clipping (exactly zero on the chaotic and real-world tasks, 0.092 on copy) and on both gated cells. The low-rank premise holds for some task groups, not others; training-level evidence is on tanh units, the gated results structural.}
}
```
