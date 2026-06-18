# Sketched RTRL (SK-RTRL)

**Certified low-rank Real-Time Recurrent Learning for dense recurrent neural networks.**

SK-RTRL maintains the RTRL influence matrix as an exact SnAp-1 block-diagonal part `S` plus a
deterministic two-sided low-rank sketch `L Rᵀ` of the off-diagonal residual. A single rank knob `r`
interpolates between SnAp-1 (`r = 0`) and exact RTRL (`r = n`). The same shrinkage that bounds the
sketch error yields, at no extra asymptotic cost, a running scalar **certificate**
`eₜ = ρ̄ₜ·eₜ₋₁ + ηₜ` that upper-bounds the gradient bias at every step — turning an approximate online
gradient into one whose error is known as it is computed.

This repository contains the reference implementation and the scripts that reproduce every figure
and table in the paper.

> **Paper:** *Certified Low-Rank Real-Time Recurrent Learning for Dense Recurrent Neural Networks*
> (under review). The citation will be finalized on publication.

## Requirements

- Python 3.12
- PyTorch 2.3 (CUDA 12.1) — a single 12 GB GPU is sufficient; the unit tests and small runs also
  work on CPU
- NumPy, SciPy, Matplotlib

```bash
pip install -r requirements.txt
```

Exact versions used for the paper: `torch==2.3.0` (CUDA 12.1), `numpy==1.26.4`, `matplotlib==3.9`,
and SciPy (paired Wilcoxon signed-rank). All data is synthetic or standard public series generated
deterministically from a seed.

## Repository layout

```
skrtrl/             library: cells, algorithms (SK-RTRL / exact RTRL / baselines),
                    tasks, training loop, RL envs, diagonal-exact cells
skrtrl/data/        real benchmark series: sunspot.txt (SILSO monthly), laser.txt (Santa Fe set A)
run_m3.py           gradient fidelity + online time-series (cosine vs exact shadow, NMSE; --horizon/--causal)
run_arch.py         forecasting-architecture baselines (GRU/LSTM via TBPTT, ESN ridge readout)
run_adaptive.py     certificate-guided adaptive-rank controller + fixed-r baselines (--ctrl eta|e_t|oracle)
run_m5.py           online RL (T-maze actor-critic)
run_m0_profile.py   memory/time profile across n
run_cor3_timing.py  factored-exact (r=n) vs textbook-exact timing
make_figures.py     regenerate the core paper figures from results/*/*.json
make_round1_figures.py   horizon / fidelity-vs-error / adaptive-trajectory / memory-time / scaling
make_stats.py       paired bootstrap CI + Wilcoxon signed-rank + Holm + Cohen's d_z for the tables
tests/              numerics unit tests (exact RTRL ≡ BPTT; SK-RTRL(r=n) ≡ exact; certificate validity)
results/            one JSON per run (git-ignored; created when you run experiments)
```

## Quick start — numerical correctness (run first)

```bash
python -m tests.test_numerics   # exact RTRL ≡ BPTT (4e-16); SK-RTRL(r=n) ≡ exact RTRL (1.6e-15); certificate: 0 violations
python -m tests.test_m5_traces  # diagonal-cell (LRU/RTU) eligibility traces vs autograd
```

A first real run (gradient fidelity on the copy task):

```bash
python run_m3.py --task copy --algo skrtrl-r16 --seed 0 --steps 20000 --n 64 --outdir results/m3
```

## Reproducing the paper

Each block writes one JSON per run into `results/…`. A rerun skips a run whose output already
exists, so the loops are restart-safe and can be split across workers by seed.

```bash
# Gradient fidelity (diagnostic tasks, 3 seeds, exact shadow)
for t in copy adding rotation anbn; do for a in exact skrtrl-r4 skrtrl-r16 skrtrl-r64 snap1 uoro kfrtrl rflo tbptt; do for s in 0 1 2; do
  python run_m3.py --task $t --algo $a --seed $s --steps 20000 --n 64 --outdir results/m3; done; done; done

# Online chaotic time-series (Henon / Mackey-Glass / Lorenz, 10 seeds)
for t in henon mackeyglass lorenz; do for a in exact skrtrl-r4 skrtrl-r16 snap1 uoro kfrtrl rflo; do for s in 0 1 2 3 4 5 6 7 8 9; do
  python run_m3.py --task $t --algo $a --seed $s --steps 15000 --n 64 --shadow 1 --outdir results/ts; done; done; done
# (skrtrl-r32 and tbptt at 5 seeds: same loop with `for s in 0 1 2 3 4` and those algos)

# Real benchmarks: Sunspot + Santa Fe laser (10 seeds, causal normalization)
for t in sunspot laser; do for a in exact skrtrl-r4 skrtrl-r16 snap1 uoro rflo; do for s in 0 1 2 3 4 5 6 7 8 9; do
  python run_m3.py --task $t --algo $a --seed $s --steps 15000 --n 64 --shadow 1 --causal 1 --outdir results/round1/real --tag real; done; done; done

# Multi-step horizons h in {5,10,25}
for t in henon mackeyglass lorenz; do for h in 5 10 25; do for a in exact snap1 skrtrl-r4 skrtrl-r16 rflo; do for s in 0 1 2 3 4; do
  python run_m3.py --task $t --algo $a --seed $s --horizon $h --steps 15000 --n 64 --outdir results/round1/horizon --tag h$h; done; done; done; done

# Forecasting-architecture baselines (GRU/LSTM via TBPTT, ESN ridge readout; context only)
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
python run_cor3_timing.py                      # factored-exact vs textbook-exact across n

# Clean (shadow-OFF) memory/time benchmark (n in {128,256,384,512})
for n in 128 256 384 512; do for a in exact snap1 skrtrl-r4 skrtrl-r16; do
  python run_m3.py --task anbn --algo $a --seed 0 --n $n --steps 1500 --shadow 0 --outdir results/membench --tag n$n; done; done

# Online RL case study (iso-width n=64, 3 seeds)
for s in 0 1 2; do for len in 10 20 40; do for a in rtu lru snap1 skrtrl-r16 exact tbptt; do
  python run_m5.py --env_len $len --algo $a --seed $s --n 64 --outdir results/m5iso; done; done; done
```

## Figures and statistics

```bash
python make_figures.py results                 # fidelity bars, rank-interpolation, certificate, curves
python make_round1_figures.py                  # horizon NMSE, fidelity-vs-error, adaptive trajectory,
                                               #   memory-time pareto, scaling
python make_m3_report.py results/ts            # time-series NMSE table (mean ± std)
python make_m5_report.py results/m5iso         # RL success-rate table
# Paired statistics for the time-series tables (bootstrap CI + Wilcoxon + Holm + Cohen's d_z)
python make_stats.py --dirs results/round1/real results/ts --metric grad_cos --higher_better 1 --out results/round1/STATS_gradcos
python make_stats.py --dirs results/round1/real results/ts --metric metric   --higher_better 0 --out results/round1/STATS_nmse
```

## Data sources

- `skrtrl/data/sunspot.txt` — monthly mean total sunspot number, [SILSO](https://www.sidc.be/SILSO/),
  Royal Observatory of Belgium.
- `skrtrl/data/laser.txt` — far-infrared laser intensity, Santa Fe time-series competition (data set A).

## Notes

- `run_m3.py` / `run_m5.py` / `run_adaptive.py` skip a run whose output JSON already exists
  (safe restart / multi-worker split by seed).
- The reported `metric` for time-series tasks is the running normalized MSE (each series is scaled to
  unit variance, so MSE ≈ NMSE).
- The exact shadow (used for the gradient cosine) is enabled for `n ≤ 256`; beyond that only memory
  and NMSE are recorded.

## Citation

```bibtex
@article{skrtrl,
  title  = {Certified Low-Rank Real-Time Recurrent Learning for Dense Recurrent Neural Networks},
  author = {Junfei Yi and Yuxiang Wang},
  note   = {Under review},
  year   = {2026}
}
```
