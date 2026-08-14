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

`requirements.txt` pins the paper environment, with one stated exception: SciPy was not recorded
at run time, so it carries a tested floor (`scipy>=1.11`) rather than a false pin. Note also that
`pip install -r requirements.txt` alone installs `torch==2.3.0` from PyPI, not the CUDA 12.1
build; take `pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121` first if
you want that build. Its Python minor version is recorded only
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
skrtrl/algos_amortised.py  amortised-rotation kernel of §4.3 (deferred rotations). O(n^2 r) is the
                      amortised rotation term under c=Theta(r) with r=O(sqrt n) (or c=O(1)); the
                      shipped kernel caps the deferred width at w <= 5r, so it runs at K=O(1) and
                      the released configuration is still O(n^3+n^2 r^2) per step (Sec. 4.2)
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
make_paper_figures.py   the single script behind the paper's data figures: one colour-blind-safe
                      style, print-size axes, a hard check that any two methods differ in >=2
                      visual attributes, and a loud skip when an input tree is absent
make_figures.py       older/plainer figure entry point, kept for reference
make_round1_figures.py  horizon / fidelity-vs-error / adaptive-trajectory / memory-time-pareto / scaling
make_stats.py         paired bootstrap CI + Wilcoxon signed-rank + Holm + Cohen d_z for the tables
make_kernel_table.py  renders results/revise_kernel/*.json into SUMMARY.md
make_revise_repro_report.py  builds results/revise_repro/COMPARISON.{md,json}
tests/                numerics unit tests (exact RTRL == BPTT; SK-RTRL(r=n) == exact; certificate validity)
tests/test_amortised_kernel.py  A1–A10 equivalence of the amortised kernel vs the kernel of record
results/              one JSON per run (args + per-step records + wall_s + peak_MB)
```

## Released result trees

The `results/` tree ships with the code. It covers Tables 3, 4, 6, 7, 8, 9, the amortised-kernel
benchmark and all ten data figures, so those numbers can be recomputed without re-running
anything. It does **not** cover everything the paper reports; the exceptions are listed below the
table. What is in it:

| Tree | Files | Backs |
|---|---|---|
| `results/round1/` | 515 run JSON + 2 stats MD + 2 stats JSON | the round-1 campaign: `horizon/`, `real/`, `ablate_rank/`, `ablate_norm/`, `adapt_ablate/`, `arch/`, `traj/`, plus `STATS_gradcos.*` / `STATS_nmse.*` |
| `results/ts/` | 240 JSON | the chaotic time-series campaign (Table 6, Figs. 7 and 8) and, with `round1/real`, the paired statistics |
| `results/m3/` | 202 JSON | diagnostic-task gradient fidelity: Table 3 and Fig. 5 |
| `results/membench/` | 16 JSON | the clean shadow-OFF memory/time sweep behind Fig. 9 and the scaling/Pareto figures |
| `results/m5iso/` | 54 JSON | the online T-maze study: Table 9 and Fig. 11 |
| `results/c2sweep/` | 12 JSON | the certificate validity/tightness sweep: Table 4 and Fig. 6 |
| `results/scale/`, `results/scale256/` | 23 + 24 JSON | the n=128 / n=256 fidelity points of Fig. 9 (left) |
| `results/m1_spectrum_*.json` | 4 JSON | the residual-spectrum pilot of Fig. 2 |
| `results/scale_large.json` | 1 JSON | the extended width sweep n ∈ {384, 512, 768, 1024} quoted in §4.2 |
| `results/m0_profile.json`, `results/cor3_timing.json` | 2 JSON | round-1 memory/time profile; original Cor.-3 timing |
| `results/revise_timing/` | 2 JSON | re-measurement of Table 8 (Cor. 3 factored-exact vs textbook exact); the `rel_err_grad` field quoted in §6.6 lives here |
| `results/revise_kernel/` | 3 JSON + SUMMARY.md + equivalence_test.txt | naive vs amortised-rotation kernel benchmark of §4.3 |
| `results/revise_repro/` | 22 JSON + 5 logs + COMPARISON.{md,json} | the 2026-08 independent re-run of E6 (rank×clip) and the Fig. 9 memory benchmark on a newer toolchain, with the point-by-point comparison against the manuscript |

`results/round1/` + `results/membench/` = **533 JSON files** — 531 run records plus the two
aggregate `STATS_*.json` — the figure quoted in the response letter. Every other run tree
(scratch sweeps, local smoke tests, figure output) is git-ignored.

**What is not released, and why.**

- *Table 5 (adaptive vs fixed rank).* Those runs are not in this tree. The command that
  reproduces the block is listed below; the neighbouring controller ablation it is discussed with
  (η_t vs e_t vs oracle) *is* released, under `results/round1/adapt_ablate/` and
  `results/round1/traj/`.
- *A clip-0.9 stability sweep* at 30k steps. No table or figure in the paper reports it.
- *Longer-horizon diagnostic reruns* (60k / 200k steps, and a `rotrecall` variant). These are
  separate experiments from Table 3's 20k-step block and are not reported in the paper; leaving
  them out keeps `results/m3/` in one-to-one correspondence with Table 3.
- *The remote orchestration wrappers*, for the reason given in the next section.

**JSON shape.** Most run JSONs have four top-level keys: `args` (the full argument namespace of
the run), `records` (the per-step log), `wall_s`, and `peak_MB`
(`torch.cuda.max_memory_allocated()`). Three groups differ:

- the `run_arch.py` baselines in `results/round1/arch/` (75 files) carry `final_nmse` instead of
  `peak_MB`, plus `esn_res` for the 25 ESN runs;
- the `run_adaptive.py` runs in `results/round1/adapt_ablate/` and `results/round1/traj/` (19
  files) add `avg_rank`, `cert_violations` and `cert_checks_with_shadow`;
- runs logged without a CUDA device (the `tbptt` runs in `results/ts/`, the T-maze runs) have no
  `peak_MB`.

`results/scale_large.json`, `results/m0_profile.json` and `results/cor3_timing.json` are flat
lists of measurement rows rather than run records, and `STATS_*.json`, `COMPARISON.json`,
`cor3_timing_revise*.json` and `kernel_bench_*.json` are report/benchmark files with their own
schemas. Every released script reads the shape it needs and ignores the rest.

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
# Gradient fidelity (diagnostic tasks, 5 seeds, exact shadow; tab:fidelity, fig:fidelity)
for t in copy adding rotation anbn; do for a in exact skrtrl-r4 skrtrl-r16 skrtrl-r64 snap1 uoro kfrtrl rflo tbptt; do for s in 0 1 2 3 4; do
  python run_m3.py --task $t --algo $a --seed $s --steps 20000 --n 64 --outdir results/m3; done; done; done
# Four released runs (copy/adding x kfrtrl/uoro, seed 0) come from an earlier pass logged at
# --steps 8000. Table 3 averages the tail of the logged cosine, which has plateaued long before
# then; `python make_fidelity_bars.py` recomputes all 24 cells from results/m3 and fails if any
# of them disagrees with the published value, so the effect is checkable rather than asserted.

# Online chaotic time-series (Henon/Mackey-Glass/Lorenz, 10 seeds; tab:timeseries)
for t in henon mackeyglass lorenz; do for a in exact skrtrl-r4 skrtrl-r16 snap1 uoro kfrtrl rflo; do for s in 0 1 2 3 4 5 6 7 8 9; do
  python run_m3.py --task $t --algo $a --seed $s --steps 15000 --n 64 --shadow 1 --outdir results/ts; done; done; done
# (skrtrl-r32 and tbptt at 5 seeds: same loop with `for s in 0 1 2 3 4` and those algos)

# Real benchmarks: Sunspot + Santa Fe laser (10 seeds, causal normalization; tab:realts)
# requires skrtrl/data/{sunspot.txt, laser.txt}; the task raises if absent (no fabrication)
for t in sunspot laser; do for a in exact skrtrl-r4 skrtrl-r16 snap1 uoro rflo; do for s in 0 1 2 3 4 5 6 7 8 9; do
  python run_m3.py --task $t --algo $a --seed $s --steps 15000 --n 64 --shadow 1 --causal 1 --outdir results/round1/real --tag real; done; done; done
# KF-RTRL sits at 5 seeds in tab:realts, and at 5 seeds in the released tree:
for t in sunspot laser; do for s in 0 1 2 3 4; do
  python run_m3.py --task $t --algo kfrtrl --seed $s --steps 15000 --n 64 --shadow 1 --causal 1 --outdir results/round1/real --tag real; done; done

# Multi-step horizons h in {5,10,25} (h=1 = the runs above; fig:horizon)
for t in henon mackeyglass lorenz; do for h in 5 10 25; do for a in exact snap1 skrtrl-r4 skrtrl-r16 rflo; do for s in 0 1 2 3 4; do
  python run_m3.py --task $t --algo $a --seed $s --horizon $h --steps 15000 --n 64 --outdir results/round1/horizon --tag h$h; done; done; done; done

# Forecasting-architecture baselines (GRU/LSTM via TBPTT, ESN ridge; context only)
for t in henon mackeyglass lorenz sunspot laser; do for arch in gru lstm esn; do for s in 0 1 2 3 4; do
  python run_arch.py --task $t --arch $arch --seed $s --steps 15000 --n 64 --outdir results/round1/arch --tag arch; done; done; done

# Adaptive-controller ablation: eta (ours) vs e_t (naive) vs oracle (hindsight)
for t in rotation anbn; do for c in eta e_t oracle; do for s in 0 1 2; do
  python run_adaptive.py --task $t --ctrl $c --seed $s --steps 20000 --n 64 --shadow 1 --outdir results/round1/adapt_ablate --tag e5; done; done; done

# Certificate validity/tightness sweep (spectral clip; tab:tightness, fig:cert). Twelve runs,
# one seed each; the five (rho_bar, tightness) points of tab:tightness are read across them.
for clip in 0.2 0.35 0.5 0.7; do for t in adding anbn rotrecall24; do
  python run_m3.py --task $t --algo skrtrl-r16 --seed 0 --steps 25000 --clip $clip --outdir results/c2sweep --tag clip$clip; done; done

# Certificate-guided adaptive rank vs fixed r (tab:adaptive). NOT in the released tree.
for t in rotation anbn; do for s in 0 1 2 3 4; do
  python run_adaptive.py --task $t --seed $s --n 64 --clip 0.5 --shadow 1 --r_min 4 --r_max 32 --outdir results/adaptive
  for fr in 4 16 32; do python run_adaptive.py --task $t --seed $s --n 64 --clip 0.5 --shadow 1 --fixed_r $fr --outdir results/adaptive; done; done; done

# Memory/time scaling and Corollary-3 timing
python run_m0_profile.py                       # peak memory + ms/step across n in {64..512}
python run_cor3_timing.py                      # factored-exact vs textbook-exact across n (round-1)
# Hardened re-measurement behind tab:cor3 -- two invocations, small then large widths:
python run_cor3_timing_revise.py
python run_cor3_timing_revise.py --ns 256,320,384,512 --steps 20 --warmup 10 --repeats 3   --out results/revise_timing/cor3_timing_revise_large.json

# Clean (shadow-OFF) memory/time benchmark for fig:scaling and fig:pareto (n in {128,256,384,512})
for n in 128 256 384 512; do for a in exact snap1 skrtrl-r4 skrtrl-r16; do
  python run_m3.py --task anbn --algo $a --seed 0 --n $n --steps 1500 --shadow 0 --outdir results/membench --tag n$n; done; done

# Amortised-rotation kernel benchmark (Sec. 4.3), both pre-projection modes
python run_amortised_kernel_bench.py --mode svd      --out results/revise_kernel/kernel_bench_svd.json
python run_amortised_kernel_bench.py --mode randproj --out results/revise_kernel/kernel_bench_randproj.json

# Online RL case study (iso-width n=64, 3 seeds; tab:rl, fig:rlcurves). The step budget grows
# with the corridor and must be passed: run_m5.py's default is 300k, which is not the protocol.
for s in 0 1 2; do for a in rtu lru snap1 skrtrl-r16 exact tbptt; do
  python run_m5.py --env_len 10 --steps 60000  --algo $a --seed $s --n 64 --outdir results/m5iso
  python run_m5.py --env_len 20 --steps 100000 --algo $a --seed $s --n 64 --outdir results/m5iso
  python run_m5.py --env_len 40 --steps 150000 --algo $a --seed $s --n 64 --outdir results/m5iso; done; done
```

## Figures and tables
```bash
python make_paper_figures.py                   # all ten data figures -> results/figures/
python make_fidelity_bars.py                   # fig_fidelity_bars alone, with the tab:fidelity check
python make_m3_report.py results/ts            # time-series NMSE table (mean±std)
python make_m5_report.py results/m5iso         # RL success-rate table (tab:rl)
python c2sweep_points.py results/c2sweep       # certificate tightness points (tab:tightness)
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
