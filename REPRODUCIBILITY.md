# Reproducibility — SK-RTRL (Neurocomputing submission)

Most experiments use synthetic / standard chaotic-system data generated deterministically from
the run seed; the two *real* benchmarks (SILSO monthly mean total sunspot number and the Santa Fe
laser series) are redistributed as text files under `skrtrl/data/` and are never fabricated
— the loader raises if a file is missing. A single 12 GB GPU is sufficient (the largest single
run, exact RTRL at n=512, peaks at 8.3 GB of allocated tensors).

> **Path note.** In the manuscript and in the authors' working tree this repository sits in a
> `code/` subdirectory next to `paper/`. Here it *is* the repository root, so every `code/…` path
> quoted in the paper maps to the same path without the `code/` prefix: `code/skrtrl/data/` →
> `skrtrl/data/`, `code/results/round1/` → `results/round1/`, and so on. All commands below are
> run from the repository root.

## Environments

Two environments are involved; they are reported separately rather than merged.

| | Python | PyTorch | Hardware | Used for |
|---|---|---|---|---|
| **Paper environment** (results reported in the manuscript) | 3.10 | 2.3.0 + CUDA 12.1 | 5 remote GPUs (TITAN Xp ×3, T4, RTX 3060) for the round-1 campaign (E1/E2/E6/E7); 2 local RTX 3080 for E3/E5/F3 and the clean `results/membench` sweep | all tables and figures |
| **Revision re-measurement environment** (2026-08) | 3.10.20 | 2.6.0 + CUDA 12.4 | 2 × RTX 3080 (20 GB) | independent re-run of the E6 rank/clip ablation and the Fig. 9 memory benchmark, archived under `results/revise_repro/`; the Table 8 timing re-measurement (`results/revise_timing/`) and the amortised-kernel benchmark (`results/revise_kernel/`) | revision-phase measurements |

`requirements.txt` pins the paper environment. Its Python minor version is recorded only
indirectly — the byte-compiled modules left by the local RTX 3080 runs are all
`*.cpython-310.pyc` — so it is stated as 3.10 rather than a full patch version, and the remote
pool's patch version was not captured.

The revision re-measurement deliberately used a newer PyTorch to check that the reported numbers
are not an artefact of one toolchain. Outcome: the numerics tests reproduce to the same digits
(exact RTRL ≡ BPTT at 4.409e-16, SK-RTRL(r=n) ≡ exact at 1.585e-15, 0 certificate violations),
and every peak-memory figure is bit-identical, since `torch.cuda.max_memory_allocated()` counts
allocator-tracked tensor bytes and is therefore independent of the GPU model. Wall-clock times
differ, as expected. See `results/revise_repro/COMPARISON.md` for the point-by-point table.
NumPy 1.26.4 and SciPy 1.15.3 in the revision environment; Matplotlib is needed only by the
`make_*.py` figure scripts, not by any run script.

## Layout
```
skrtrl/               library: cells, algos (SKRTRL/ExactRTRL/baselines), tasks, train, envs, rl
skrtrl/algos_amortised.py  amortised-rotation kernel of §4.3 (deferred rotations, O(n^2 r) per step)
skrtrl/data/          REAL benchmark series: sunspot.txt (SILSO monthly *mean total*, SN_m_tot_V2.0,
                        3329 values 1749-01..2026-05), laser.txt (Santa Fe set A, 1000 points)
run_m3.py             fidelity + time-series online runs (gradient cosine vs exact shadow, NMSE; --horizon/--causal)
run_arch.py           forecasting-architecture baselines (GRU/LSTM via TBPTT, ESN ridge readout)
run_adaptive.py       certificate-guided adaptive-rank controller + fixed-r baselines (--ctrl eta|e_t|oracle)
run_m5.py             online RL (T-maze actor-critic)
run_m0_profile.py     memory/time profile across n
run_cor3_timing.py    factored-exact (r=n) vs textbook-exact timing (original round-1 measurement)
run_cor3_timing_revise.py    hardened re-measurement of the same (CUDA events, 20 warmup, 5 repeats)
run_amortised_kernel_bench.py  naive vs amortised kernel, svd and randproj pre-projection modes
make_figures.py       regenerates the core paper figures from results/*/*.json
make_round1_figures.py  horizon / fidelity-vs-error / adaptive-trajectory / memory-time-pareto / scaling
make_stats.py         paired bootstrap CI + Wilcoxon signed-rank + Holm + Cohen d_z for the tables
make_kernel_table.py  renders results/revise_kernel/*.json into SUMMARY.md
make_revise_repro_report.py  builds results/revise_repro/COMPARISON.{md,json}
tests/                numerics unit tests (exact RTRL == BPTT; SK-RTRL(r=n) == exact; certificate validity)
tests/test_amortised_kernel.py  A1–A10 equivalence of the amortised kernel vs the kernel of record
results/              one JSON per run (args + per-step records + wall_s + peak_MB)
```

## Released result trees

The `results/` tree ships with the code so that every reported number can be re-derived without
re-running anything. What is in it:

| Tree | Files | Backs |
|---|---|---|
| `results/round1/` | 517 JSON + 2 stats MD | the round-1 campaign: `horizon/`, `real/`, `ablate_rank/`, `ablate_norm/`, `adapt_ablate/`, `arch/`, `traj/`, plus `STATS_gradcos.*` / `STATS_nmse.*` |
| `results/membench/` | 16 JSON | the clean shadow-OFF memory/time sweep behind Fig. 9 and the scaling/Pareto figures |
| `results/revise_timing/` | 2 JSON | re-measurement of Table 8 (Cor. 3 factored-exact vs textbook exact); the `rel_err_grad` field quoted in §6.6 lives here |
| `results/revise_kernel/` | 3 JSON + SUMMARY.md + equivalence_test.txt | naive vs amortised-rotation kernel benchmark of §4.3 |
| `results/revise_repro/` | 22 JSON + 5 logs + COMPARISON.{md,json} | the 2026-08 independent re-run of E6 (rank×clip) and the Fig. 9 memory benchmark on a newer toolchain, with the point-by-point comparison against the manuscript |

`results/round1/` + `results/membench/` = **533 raw JSON records**, the figure quoted in the
response letter. Every other run tree (scratch sweeps, local smoke tests, figure output) is
git-ignored and regenerated by the commands below.

Each run JSON has the same four top-level keys: `args` (the full argument namespace of the run),
`records` (the per-step log), `wall_s`, and `peak_MB` (`torch.cuda.max_memory_allocated()`).

## Remote orchestration scripts

The round-1 campaign was dispatched to five rented GPUs by a set of `launch_*.sh` / `run_*.sh` /
`run_*_local.ps1` wrappers. Those wrappers are **not published**: they embed rented-instance
hostnames, ports and working directories, i.e. environment credentials with no scientific
content. Each one is a thin loop around the `python run_*.py …` commands listed below, which are
the complete and sufficient description of what was run.

## Numerical correctness (run first)
```bash
python -m tests.test_numerics           # exact RTRL ≡ BPTT (4e-16); SK-RTRL(r=n) ≡ exact (1.6e-15); certificate 0 violations
python -m tests.test_m5_traces          # diagonal-cell (LRU/RTU) eligibility traces vs autograd
python -m tests.test_amortised_kernel   # amortised kernel ≡ kernel of record (A1–A10)
```

## Core experiment commands (one block each)
```bash
# Gradient fidelity (diagnostic tasks, 3 seeds, exact shadow)
for t in copy adding rotation anbn; do for a in exact skrtrl-r4 skrtrl-r16 skrtrl-r64 snap1 uoro kfrtrl rflo tbptt; do for s in 0 1 2; do
  python run_m3.py --task $t --algo $a --seed $s --steps 20000 --n 64 --outdir results/m3; done; done; done

# Online chaotic time-series (Henon/Mackey-Glass/Lorenz, 10 seeds; tab:timeseries)
for t in henon mackeyglass lorenz; do for a in exact skrtrl-r4 skrtrl-r16 snap1 uoro kfrtrl rflo; do for s in 0 1 2 3 4 5 6 7 8 9; do
  python run_m3.py --task $t --algo $a --seed $s --steps 15000 --n 64 --shadow 1 --outdir results/ts; done; done; done
# (skrtrl-r32 and tbptt at 5 seeds: same loop with `for s in 0 1 2 3 4` and those algos)

# Real benchmarks: Sunspot + Santa Fe laser (10 seeds, causal normalization; tab:realts)
# requires skrtrl/data/{sunspot.txt, laser.txt}; the task raises if absent (no fabrication)
for t in sunspot laser; do for a in exact skrtrl-r4 skrtrl-r16 snap1 uoro rflo; do for s in 0 1 2 3 4 5 6 7 8 9; do
  python run_m3.py --task $t --algo $a --seed $s --steps 15000 --n 64 --shadow 1 --causal 1 --outdir results/round1/real --tag real; done; done; done

# Multi-step horizons h in {5,10,25} (h=1 = the runs above; fig:horizon)
for t in henon mackeyglass lorenz; do for h in 5 10 25; do for a in exact snap1 skrtrl-r4 skrtrl-r16 rflo; do for s in 0 1 2 3 4; do
  python run_m3.py --task $t --algo $a --seed $s --horizon $h --steps 15000 --n 64 --outdir results/round1/horizon --tag h$h; done; done; done; done

# Forecasting-architecture baselines (GRU/LSTM via TBPTT, ESN ridge; context only)
for t in henon mackeyglass lorenz sunspot laser; do for arch in gru lstm esn; do for s in 0 1 2 3 4; do
  python run_arch.py --task $t --arch $arch --seed $s --steps 15000 --n 64 --outdir results/round1/arch --tag arch; done; done; done

# Adaptive-controller ablation: eta (ours) vs e_t (naive) vs oracle (hindsight)
for t in rotation anbn; do for c in eta e_t oracle; do for s in 0 1 2; do
  python run_adaptive.py --task $t --ctrl $c --seed $s --steps 20000 --n 64 --shadow 1 --outdir results/round1/adapt_ablate --tag e5; done; done; done

# Certificate validity/tightness sweep (spectral clip)
for clip in 0.2 0.35 0.5 0.7 0.9; do for t in adding anbn rotrecall24; do
  python run_m3.py --task $t --algo skrtrl-r16 --seed 0 --steps 25000 --clip $clip --outdir results/c2sweep --tag clip$clip; done; done

# Certificate-guided adaptive rank (vs fixed r), 5 seeds
for t in mackeyglass lorenz copy; do for s in 0 1 2 3 4; do
  python run_adaptive.py --task $t --seed $s --steps 15000 --n 64 --r_min 4 --r_max 32 --outdir results/adaptive
  for fr in 4 16 32; do python run_adaptive.py --task $t --seed $s --steps 15000 --fixed_r $fr --outdir results/adaptive; done; done; done

# Memory/time scaling and Corollary-3 timing
python run_m0_profile.py                       # peak memory + ms/step across n in {64..512}
python run_cor3_timing.py                      # factored-exact vs textbook-exact across n (round-1)
python run_cor3_timing_revise.py               # hardened re-measurement -> results/revise_timing/

# Clean (shadow-OFF) memory/time benchmark for fig:scaling and fig:pareto (n in {128,256,384,512})
for n in 128 256 384 512; do for a in exact snap1 skrtrl-r4 skrtrl-r16; do
  python run_m3.py --task anbn --algo $a --seed 0 --n $n --steps 1500 --shadow 0 --outdir results/membench --tag n$n; done; done

# Amortised-rotation kernel benchmark (Sec. 4.3), both pre-projection modes
python run_amortised_kernel_bench.py --mode svd      --out results/revise_kernel/kernel_bench_svd.json
python run_amortised_kernel_bench.py --mode randproj --out results/revise_kernel/kernel_bench_randproj.json

# Online RL case study (iso-width n=64, 3 seeds)
for s in 0 1 2; do for len in 10 20 40; do for a in rtu lru snap1 skrtrl-r16 exact tbptt; do
  python run_m5.py --env_len $len --algo $a --seed $s --n 64 --outdir results/m5iso; done; done; done
```

## Figures and tables
```bash
python make_figures.py results                 # fig_fidelity_bars, fig_rinterp_*, fig_cert_*, fig_curve_*
python make_round1_figures.py                  # fig_horizon_nmse, fig_fidelity_vs_error, fig_adaptive_trajectory,
                                               #   fig_memory_time_pareto, fig_scaling (clean, from results/membench)
python make_m3_report.py results/ts            # time-series NMSE table (mean±std)
python make_m5_report.py results/m5iso         # RL success-rate table
python make_kernel_table.py                    # results/revise_kernel/SUMMARY.md (Sec. 4.3 kernel table)
python make_revise_repro_report.py             # results/revise_repro/COMPARISON.{md,json}
# Paired statistics for tab:timeseries / tab:realts (bootstrap CI + Wilcoxon + Holm + Cohen d_z)
python make_stats.py --dirs results/round1/real results/ts --metric grad_cos --higher_better 1 --out results/round1/STATS_gradcos
python make_stats.py --dirs results/round1/real results/ts --metric metric   --higher_better 0 --out results/round1/STATS_nmse
```

## Notes
- `run_m3.py`/`run_m5.py`/`run_adaptive.py` skip a run if its output JSON already exists (safe restart / multi-worker split by seed).
  Since the released `results/` tree is already populated, delete (or redirect with `--outdir`) the target
  directory before re-running, or every run will be skipped.
- Reported `metric` for time-series tasks is the running normalized MSE (series scaled to unit variance ⇒ MSE ≈ NMSE).
- The exact shadow (for gradient cosine) is enabled for `n ≤ 256`; beyond that only memory/NMSE are recorded.
