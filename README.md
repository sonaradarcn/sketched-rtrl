# Sketched RTRL (SK-RTRL)

**Certified low-rank Real-Time Recurrent Learning for dense recurrent neural networks.**

SK-RTRL maintains the RTRL influence matrix as an exact SnAp-1 block-diagonal part `S` plus a
deterministic two-sided low-rank sketch `L Rᵀ` of the off-diagonal residual. A single rank knob `r`
interpolates between SnAp-1 (`r = 0`) and exact RTRL (`r = n`). The same shrinkage that bounds the
sketch error yields, at no extra asymptotic cost, a running scalar **certificate**
`eₜ = ρ̄ₜ·eₜ₋₁ + ηₜ` that upper-bounds the gradient bias at every step — turning an approximate online
gradient into one whose error is known as it is computed.

This repository contains the reference implementation, the scripts that produce the paper's
figures and tables, **and the raw result records behind most of them** (`results/`, see below).
The released trees cover Tables 3, 4, 6, 7, 8 and 9, the amortised-kernel benchmark and all
eight numbered data figures (2, 5, 6, 7, 8, 9, 10, 11 — ten figure files, since two are
multi-panel), so those numbers can be recomputed without re-running anything. Figures 1, 3 and 4
are hand-drawn schematics and contain no measured data. Two run
trees are *not* released — the Table 5 adaptive-vs-fixed-rank runs and a clip-0.9 stability
sweep; the table below says exactly what is and is not there. The figure scripts skip a figure
whose input tree is absent and say so, rather than drawing a partial panel silently.

> **Paper:** *Certified Low-Rank Real-Time Recurrent Learning for Dense Recurrent Neural Networks*
> (under review). The citation will be finalized on publication.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full protocol: both environments, the
released result trees, and the exact command for every experiment in the paper.

## Requirements

- Python ≥ 3.10 (the paper runs used 3.10; see `REPRODUCIBILITY.md`, "Environments")
- PyTorch 2.3 (CUDA 12.1). A single 12 GB GPU covers every training and scaling run; the one
  exception is the isolated per-step benchmark of Table 8, whose factored-exact path allocates
  14.7 GB at n=512 and was measured on a 20 GB card. The unit tests and small runs work on CPU
  with `--device cpu`
- NumPy, SciPy, Matplotlib

```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1 build
pip install -r requirements.txt
```

Installing `requirements.txt` on its own gives you `torch==2.3.0` from PyPI, which is not the
CUDA 12.1 build the paper ran on; take the first line above if you want that build.

Versions used for the paper: `torch==2.3.0` (CUDA 12.1), `numpy==1.26.4`, `matplotlib==3.9.0`,
and SciPy for the paired Wilcoxon signed-rank test. SciPy is the one dependency that is *not*
pinned to the original version: it was not recorded at run time, so `requirements.txt` carries a
tested floor (`scipy>=1.11`) instead of a false pin. All data is synthetic or standard public
series generated deterministically from a seed.

## Repository layout

```
skrtrl/             library: cells, algorithms (SK-RTRL / exact RTRL / baselines),
                    tasks, training loop, RL envs, diagonal-exact cells
skrtrl/algos.py     the kernel of record (eager rank-r rotation, O(n²r²) per step)
skrtrl/algos_amortised.py   amortised-rotation kernel of §4.3 (deferred rotations). Its rotation
                    term is O(n²r) amortised only under the append budget c=Θ(r) with r=O(√n)
                    (or a constant budget c=O(1)); the shipped kernel caps the deferred width at
                    w ≤ 5r, so it runs at K=O(1) and the released configuration is still
                    O(n³+n²r²) per step, as §4.2 and Table 2 state. It is an equivalent kernel,
                    not a lower-order one — see results/revise_kernel/SUMMARY.md
skrtrl/data/        real benchmark series: sunspot.txt (SILSO monthly), laser.txt (Santa Fe set A)
run_m3.py           gradient fidelity + online time-series (cosine vs exact shadow, NMSE; --horizon/--causal)
run_arch.py         forecasting-architecture baselines (GRU/LSTM via TBPTT, ESN ridge readout)
run_adaptive.py     certificate-guided adaptive-rank controller + fixed-r baselines (--ctrl eta|e_t|oracle)
run_m5.py           online RL (T-maze actor-critic)
run_m0_profile.py   memory/time profile across n
run_cor3_timing.py  factored-exact (r=n) vs textbook-exact timing (original round-1 measurement)
run_cor3_timing_revise.py     hardened re-measurement behind Table 8 (CUDA events, 20 warmup, 5 repeats)
run_amortised_kernel_bench.py naive vs amortised kernel (svd / randproj pre-projection modes)
run_width_sweep.py  extended width sweep of §4.2 -> results/scale_large.json
make_timeseries_tables.py    rebuilds Tables 6 and 7 from the raw runs and checks every cell
                    against the published value
make_paper_figures.py    the single script behind the paper's data figures: one
                    colour-blind-safe style, print-size axes, a hard check that any two
                    methods differ in >=2 visual attributes, and a loud skip when an input
                    tree is missing
make_figures.py     older/plainer figure entry point, kept for reference
make_round1_figures.py   horizon / fidelity-vs-error / adaptive-trajectory / memory-time / scaling
make_stats.py       paired bootstrap CI + Wilcoxon signed-rank + Holm + Cohen's d_z for the tables
make_kernel_table.py          renders results/revise_kernel/*.json into SUMMARY.md
make_revise_repro_report.py   builds results/revise_repro/COMPARISON.{md,json}
tests/              numerics unit tests (exact RTRL ≡ BPTT; SK-RTRL(r=n) ≡ exact; certificate validity)
tests/test_amortised_kernel.py  A1–A10 equivalence of the amortised kernel vs the kernel of record
results/            the released raw result records — one JSON per run (see below)
```

## Released results

Most run JSONs have four top-level keys: `args` (the run's full argument namespace), `records`
(the per-step log), `wall_s`, and `peak_MB` (`torch.cuda.max_memory_allocated()`). Three groups
differ, and the plotting/report scripts handle all of them:

- the `run_arch.py` baselines in `results/round1/arch/` carry `final_nmse` instead of `peak_MB`
  (and `esn_res` as well, for the ESN runs), because they are CPU/ridge runs with no CUDA peak;
- the `run_adaptive.py` runs in `results/round1/adapt_ablate/` and `results/round1/traj/` add
  `avg_rank`, `cert_violations` and `cert_checks_with_shadow`;
- runs logged without a CUDA device or from before `peak_MB` was recorded have only `args`,
  `records` and `wall_s`: the `tbptt` runs in `results/ts/`, all of `results/m5iso/`,
  `results/c2sweep/`, `results/c2/`, `results/m1_spectrum_*.json` and 134 of the 202 files in
  `results/m3/`.

`results/scale_large.json`, `results/m0_profile.json` and `results/cor3_timing.json` are flat
lists of measurement rows rather than run records, and `STATS_*.json`, `COMPARISON.json`,
`cor3_timing_revise*.json` and `kernel_bench_*.json` are report/benchmark files with their own
schemas. Every released script reads the shape it needs.

| Tree | Contents | Backs |
|---|---|---|
| `results/round1/` | 515 run JSON + `STATS_gradcos.{md,json}` / `STATS_nmse.{md,json}` | the round-1 campaign: `horizon/`, `real/`, `ablate_rank/`, `ablate_norm/`, `adapt_ablate/`, `arch/`, `traj/` |
| `results/ts/` | 240 JSON | the chaotic time-series campaign (Table 6) and, with `round1/real`, the paired statistics |
| `results/m3/` | 202 JSON | the diagnostic-task gradient fidelity of Table 3 and Fig. 5 |
| `results/membench/` | 16 JSON | the clean shadow-OFF memory/time sweep behind the scaling and Pareto figures |
| `results/m5iso/` | 54 JSON | the online T-maze study of Table 9 and Fig. 11 (3 seeds x 3 corridors x 6 methods) |
| `results/c2sweep/`, `results/c2/` | 12 + 18 JSON | the certificate sweep. Together they are the 3340 logged points over which §6.5, §7.4 and §8 report zero violations (1192 + 2148); `c2sweep` alone gives Table 4 and Fig. 6 |
| `results/scale/`, `results/scale256/` | 23 + 24 JSON | the n=128 / n=256 fidelity points of the scaling figure |
| `results/m1_spectrum_*.json` | 4 JSON | the residual-spectrum pilot of Fig. 2 |
| `results/scale_large.json` | 1 JSON | the extended width sweep (n = 384..1024) quoted in §4.2 |
| `results/m0_profile.json`, `results/cor3_timing.json` | 2 JSON | the round-1 memory/time profile and the original Cor.-3 timing |
| `results/revise_timing/` | 2 JSON | the Table 8 re-measurement (Cor. 3 factored-exact vs textbook exact), incl. the `rel_err_grad` field quoted in §6.6 |
| `results/revise_kernel/` | 3 JSON + `SUMMARY.md` + `equivalence_test.txt` | the §4.3 naive-vs-amortised kernel benchmark |
| `results/revise_repro/` | 22 JSON + 5 logs + `COMPARISON.{md,json}` | the 2026-08 independent re-run of the E6 rank×clip ablation and the memory benchmark on a newer toolchain |

`results/round1/` + `results/membench/` = **533 JSON files** (531 run records plus the two
aggregate `STATS_*.json`), the figure quoted in the response letter.

**Not released.** Two run trees are held back: the Table 5 adaptive-vs-fixed-rank runs, and a set
of longer-horizon diagnostic reruns (60k/200k steps, and a `rotrecall` variant) that are separate
experiments from Table 3's block and that the paper does not report. Everything else the paper reports is
backed by a tree above, and `python make_paper_figures.py` rebuilds all ten data figures from
this tree with nothing missing. Any run tree you create yourself (scratch sweeps, figure output)
is git-ignored.

> The round-1 campaign was dispatched to rented GPUs by a set of `launch_*.sh` / `run_*.sh` /
> `run_*_local.ps1` wrappers. Those are **not published**: they embed rented-instance hostnames,
> ports and working directories — environment credentials with no scientific content. Each is a
> thin loop around the `python run_*.py …` commands listed below, which fully describe what was run.

## Quick start — numerical correctness (run first)

```bash
python -m tests.test_numerics         # exact RTRL ≡ BPTT (4e-16); SK-RTRL(r=n) ≡ exact RTRL (1.6e-15); certificate: 0 violations
python -m tests.test_m5_traces        # diagonal-cell (LRU/RTU) eligibility traces vs autograd
python -m tests.test_amortised_kernel # amortised-rotation kernel ≡ kernel of record (A1–A10)
```

A first real run (gradient fidelity on the copy task). Write it to a fresh directory: the runners
skip a run whose output JSON already exists, and `results/m3/` ships populated, so pointing this at
`results/m3` would print `exists, skip` and do nothing.

```bash
python run_m3.py --task copy --algo skrtrl-r16 --seed 0 --steps 2000 --n 64 --outdir results/smoke
# add --device cpu if you have no GPU
```

## Reproducing the paper

Each block writes one JSON per run into `results/…`. A rerun skips a run whose output already
exists, so the loops are restart-safe and can be split across workers by seed. **Because the
released `results/` tree is already populated, delete the target directory (or redirect it with
`--outdir`) before re-running, or every run will be skipped as already done.**

```bash
# Gradient fidelity (diagnostic tasks, 5 seeds, exact shadow) -> Table 3, Fig. 5
for t in copy adding rotation anbn; do for a in exact skrtrl-r4 skrtrl-r16 skrtrl-r64 snap1 uoro kfrtrl rflo tbptt; do for s in 0 1 2 3 4; do
  python run_m3.py --task $t --algo $a --seed $s --steps 20000 --n 64 --outdir results/m3; done; done; done
# Six of the 202 released runs (copy/adding x kfrtrl/uoro/tbptt, seed 0) were logged in an
# earlier pass at --steps 8000; four of them enter Table 3, the two tbptt ones do not. They are
# kept because Table 3 averages the tail of the logged cosine, which has plateaued well before
# then: `python make_fidelity_bars.py` recomputes all 24 cells from results/m3 and fails if any
# of them disagrees with the published value.

# Online chaotic time-series (Henon / Mackey-Glass / Lorenz, 10 seeds)
for t in henon mackeyglass lorenz; do for a in exact skrtrl-r4 skrtrl-r16 snap1 uoro kfrtrl rflo; do for s in 0 1 2 3 4 5 6 7 8 9; do
  python run_m3.py --task $t --algo $a --seed $s --steps 15000 --n 64 --shadow 1 --outdir results/ts; done; done; done
# (skrtrl-r32 and tbptt at 5 seeds: same loop with `for s in 0 1 2 3 4` and those algos)

# Real benchmarks: Sunspot + Santa Fe laser (10 seeds, causal normalization) -> Table 7
for t in sunspot laser; do for a in exact skrtrl-r4 skrtrl-r16 snap1 uoro rflo; do for s in 0 1 2 3 4 5 6 7 8 9; do
  python run_m3.py --task $t --algo $a --seed $s --steps 15000 --n 64 --shadow 1 --causal 1 --outdir results/round1/real --tag real; done; done; done
# KF-RTRL is in Table 7 at 5 seeds (it is the one method not run to 10 there):
for t in sunspot laser; do for s in 0 1 2 3 4; do
  python run_m3.py --task $t --algo kfrtrl --seed $s --steps 15000 --n 64 --shadow 1 --causal 1 --outdir results/round1/real --tag real; done; done

# Multi-step horizons h in {5,10,25}
for t in henon mackeyglass lorenz; do for h in 5 10 25; do for a in exact snap1 skrtrl-r4 skrtrl-r16 rflo; do for s in 0 1 2 3 4; do
  python run_m3.py --task $t --algo $a --seed $s --horizon $h --steps 15000 --n 64 --outdir results/round1/horizon --tag h$h; done; done; done; done

# Forecasting-architecture baselines (GRU/LSTM via TBPTT, ESN ridge readout; context only)
for t in henon mackeyglass lorenz sunspot laser; do for arch in gru lstm esn; do for s in 0 1 2 3 4; do
  python run_arch.py --task $t --arch $arch --seed $s --steps 15000 --n 64 --outdir results/round1/arch --tag arch; done; done; done

# Adaptive-controller ablation: eta (ours) vs e_t (naive) vs oracle (hindsight)
for t in rotation anbn; do for c in eta e_t oracle; do for s in 0 1 2; do
  python run_adaptive.py --task $t --ctrl $c --seed $s --steps 20000 --n 64 --shadow 1 --outdir results/round1/adapt_ablate --tag e5; done; done; done

# Certificate sweep -> Table 4, Fig. 6, and the 3340-point validity count of §6.5/§7.4/§8
for clip in 0.2 0.35 0.5 0.7; do for t in adding anbn rotrecall24; do
  python run_m3.py --task $t --algo skrtrl-r16 --seed 0 --steps 25000 --clip $clip --outdir results/c2sweep --tag clip$clip; done; done
for t in adding anbn rotrecall; do for a in skrtrl-r4 skrtrl-r16 snap1; do for s in 0 1; do
  python run_m3.py --task $t --algo $a --seed $s --steps 30000 --clip 0.9 --outdir results/c2; done; done; done
python c2sweep_points.py results/c2sweep     # prints rho_bar / rho_hat / tightness per run

# Certificate-guided adaptive rank vs fixed r (Table 5). These runs are NOT in the released
# tree; the command is listed so the block can be reproduced from scratch.
for t in rotation anbn; do for s in 0 1 2 3 4; do
  python run_adaptive.py --task $t --seed $s --n 64 --clip 0.5 --shadow 1 --r_min 4 --r_max 32 --outdir results/adaptive
  for fr in 4 16 32; do python run_adaptive.py --task $t --seed $s --n 64 --clip 0.5 --shadow 1 --fixed_r $fr --outdir results/adaptive; done; done; done

# Residual-spectrum pilot (Fig. 2)
for t in copy adding rotation anbn; do python run_m1_spectrum.py --task $t; done

# Rank-matched random-projection control (§6.3): the skrtrl-rp* runs in results/m3
for t in adding rotation; do for a in skrtrl-rp4 skrtrl-rp16 skrtrl-rp64; do for s in 0 1 2; do
  python run_m3.py --task $t --algo $a --seed $s --steps 20000 --n 64 --outdir results/m3; done; done; done

# Adaptive-rank trajectory for Fig. 7 (the single logged trajectory, not the ablation above)
python run_adaptive.py --task rotation --ctrl eta --seed 0 --steps 20000 --n 64 --shadow 1   --outdir results/round1/traj --tag traj

# Lorenz rank x spectral-clip ablation (§7.4) -> results/round1/ablate_rank (30 runs)
for r in 2 4 8 16 32; do for c in 0 0.9; do for s in 0 1 2; do
  python run_m3.py --task lorenz --algo skrtrl-r$r --seed $s --steps 15000 --n 64 --clip $c --outdir results/round1/ablate_rank --tag clip$c; done; done; done

# Normalization / washout ablation (§7.4) -> results/round1/ablate_norm (36 runs)
for t in mackeyglass sunspot laser; do for causal in 0 1; do for w in 0 200; do for s in 0 1 2; do
  python run_m3.py --task $t --algo skrtrl-r16 --seed $s --steps 15000 --n 64 --causal $causal --washout $w --outdir results/round1/ablate_norm --tag c${causal}w${w}; done; done; done; done

# n=128 / n=256 fidelity points of the scaling figure
for n in 128 256; do for t in rotation anbn; do for a in exact snap1 skrtrl-r4 skrtrl-r16 kfrtrl uoro; do
  python run_m3.py --task $t --algo $a --seed 0 --n $n --steps $([ $n = 128 ] && echo 8000 || echo 6000) --outdir results/scale$([ $n = 128 ] && echo "" || echo 256); done; done; done

# Extended width sweep of §4.2 -> results/scale_large.json
python run_width_sweep.py

# Memory/time scaling and Corollary-3 timing
python run_m0_profile.py                       # peak memory + ms/step across n in {64..512}
python run_cor3_timing.py                      # factored-exact vs textbook-exact across n (round-1)
# Hardened re-measurement behind Table 8: two invocations, small and large widths.
python run_cor3_timing_revise.py
python run_cor3_timing_revise.py --ns 256,320,384,512 --steps 20 --warmup 10 --repeats 3   --out results/revise_timing/cor3_timing_revise_large.json

# Clean (shadow-OFF) memory/time benchmark (n in {128,256,384,512})
for n in 128 256 384 512; do for a in exact snap1 skrtrl-r4 skrtrl-r16; do
  python run_m3.py --task anbn --algo $a --seed 0 --n $n --steps 1500 --shadow 0 --outdir results/membench --tag n$n; done; done

# Amortised-rotation kernel benchmark (§4.3), both pre-projection modes
python run_amortised_kernel_bench.py --mode svd      --out results/revise_kernel/kernel_bench_svd.json
python run_amortised_kernel_bench.py --mode randproj --out results/revise_kernel/kernel_bench_randproj.json

# Online RL case study (iso-width n=64, 3 seeds) -> Table 9, Fig. 11.
# The step budget grows with the corridor; run_m5.py's default (300k) is not the protocol.
for s in 0 1 2; do for a in rtu lru snap1 skrtrl-r16 exact tbptt; do
  python run_m5.py --env_len 10 --steps 60000  --algo $a --seed $s --n 64 --outdir results/m5iso
  python run_m5.py --env_len 20 --steps 100000 --algo $a --seed $s --n 64 --outdir results/m5iso
  python run_m5.py --env_len 40 --steps 150000 --algo $a --seed $s --n 64 --outdir results/m5iso; done; done
```

## Figures and statistics

All of these run on the released tree as cloned; none of them needs a GPU.

```bash
python make_paper_figures.py                   # all ten data figure files -> results/figures/
python make_fidelity_bars.py                   # Fig. 5 (left) alone, with the Table 3 cross-check
python make_timeseries_tables.py               # Tables 6 and 7, each cell checked against the paper
python make_m5_report.py results/m5iso         # RL success-rate table (Table 9)
python make_m3_report.py results/m3            # diagnostic-run summary (20%-tail convention;
                                               #   a browsing aid, NOT the statistic in the tables)
python c2sweep_points.py results/c2sweep       # certificate tightness points (Table 4)
python make_kernel_table.py                    # results/revise_kernel/SUMMARY.md (§4.3 kernel table)
python make_revise_repro_report.py             # results/revise_repro/COMPARISON.{md,json}
# Paired statistics for the time-series tables (bootstrap CI + Wilcoxon + Holm + Cohen's d_z)
python make_stats.py --dirs results/round1/real results/ts --metric grad_cos --higher_better 1 --out results/round1/STATS_gradcos
python make_stats.py --dirs results/round1/real results/ts --metric metric   --higher_better 0 --out results/round1/STATS_nmse
```

`make_timeseries_tables.py`, `make_fidelity_bars.py` and `make_m5_report.py` use the convention
the paper's tables use: per run, the mean of the last five logged records; across seeds, mean ±
population standard deviation. `make_m3_report.py` and `make_ts_report.py` are older browsing
aids on a 20%-tail mean and a sample standard deviation, so their numbers differ slightly from
the tables by construction — do not read them as table reproductions.

`make_kernel_table.py`, `make_revise_repro_report.py` and the two `make_stats.py` calls rewrite
files that are committed here, so `git status` staying clean after running them is itself a check
that the committed reports match the committed raw records. `make_figures.py` and
`make_round1_figures.py` are earlier, plainer entry points kept for reference;
`make_paper_figures.py` is the one that produced the figures in the manuscript.

## Data sources

- `skrtrl/data/sunspot.txt` — monthly **mean total** sunspot number, [SILSO](https://www.sidc.be/SILSO/),
  Royal Observatory of Belgium. Product `SN_m_tot_V2.0` (series column of
  <https://www.sidc.be/SILSO/INFO/snmtotcsv.php>), one raw monthly mean per line, 3329 values
  covering 1749-01 through 2026-05. It is **not** the 13-month smoothed product `SN_ms_tot_V2.0`:
  the first twelve values here are 96.7/104.3/116.7/92.8/141.7/139.2/158.0/110.5/126.5/125.8/
  264.3/142.0, which is SN_m_tot for 1749-01..1749-12, whereas SN_ms_tot is undefined (`-1.0`)
  for the first six months and reads 135.9 at 1749-07. The mean absolute month-to-month change
  is 19.2 (max 156.5), far larger than any 13-month moving average could produce.
- `skrtrl/data/laser.txt` — far-infrared laser intensity, Santa Fe time-series competition (data set A).

## Notes

- `run_m3.py` / `run_m5.py` / `run_adaptive.py` skip a run whose output JSON already exists
  (safe restart / multi-worker split by seed).
- The reported `metric` for time-series tasks is the running normalized MSE (each series is scaled to
  unit variance, so MSE ≈ NMSE).
- The exact shadow (used for the gradient cosine) is enabled for `n ≤ 256`; beyond that only memory
  and NMSE are recorded.

## License

Code and result records: MIT (see [`LICENSE`](LICENSE)). The two series under `skrtrl/data/` are
third-party data and are **not** covered by it. The SILSO sunspot series is redistributed under
SILSO's stated attribution terms; for the Santa Fe laser series we have not located an explicit
redistribution licence, and say so rather than implying one. See
[`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md) for each one's product, source, retrieval date,
byte count, SHA-256 and terms.

## Citation

```bibtex
@article{skrtrl,
  title  = {Certified Low-Rank Real-Time Recurrent Learning for Dense Recurrent Neural Networks},
  author = {Junfei Yi and Yuxiang Wang},
  note   = {Under review},
  year   = {2026}
}
```
