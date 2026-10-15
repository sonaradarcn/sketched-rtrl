# First-campaign commands still used by the paper

These runs were made for the first submission and still back the floats and text passages listed
below. They ran under PyTorch 2.3.0 + CUDA 12.1 on the GPU pool of REPRODUCIBILITY.md, section 1,
except the three re-measurements at the end, which ran under PyTorch 2.6.0 + CUDA 12.4. Every
command uses the defaults of its runner unless a flag says otherwise: n=64, batch 8, Adam at
lr 1e-3, `--log_every 250`, exact shadow on, no clip, `--svd_driver gesvd`.

The runners skip a run whose output file exists. To re-run a block, change `--outdir` (or
`--out`) to a new directory.

No per-run manifest was kept for these blocks, unlike the second-campaign blocks in `jobs/`. The
commands below are the loops we ran, with the run conditions we know: the environment above, the
runner defaults, and the flags each command shows. Each run file also stores its full argument
vector in `args`.

## Index

"Original run": the loop below is the one that produced the tree. "Re-measured": a later run
under PyTorch 2.6.0, with the command as run. "Reconstructed": the argument lists were not saved,
and the command was rebuilt from the fields recorded in the result files.

| Block | Used by | Tree (run files) | Category | Runs missing from the loop |
|---|---|---|---|---|
| Fidelity against width, n=64 | Fig 6 left | `results/m3` rotation (30) | original run | none |
| Scaling, n=128 and 256 | Fig 6 left | `results/scale` (23), `results/scale256` (24) | original run | rotation, SK-RTRL r=4, seed 0, n=128; the r=4 point of Fig 6 at n=128 is seed 1 alone |
| Memory and time | Fig 6 right, Fig S8 | `results/membench` (16) | original run | none |
| Older tightness sweep | Fig S2 | `results/c2sweep` (12) | original run | none |
| Adaptive-rank trajectory | Fig S4 | `results/round1/traj` (1) | original run | none |
| Residual pilot | Fig S5 | `results/m1_spectrum_*.json` (4) | original run | none |
| Multi-step horizons | Fig S6 left | `results/round1/horizon` (225), `results/ts` (240) | original run | none |
| Online T-maze | Table S10, Fig S9 | `results/m5iso` (54) | original run | none |
| Random-projection control | §6.3 text | `results/m3` (18 rp runs, r=16 seeds 0-2) | original run | none |
| Control-signal ablation | §6.6 text | `results/round1/adapt_ablate` (18) | original run | none |
| Rank x clip, normalisation and washout | §7.4, S9.3 text | `results/round1/ablate_rank` (30), `results/round1/ablate_norm` (36) | original run | none |
| Per-step timing and exact-path agreement | Table 10, exact-path check | `results/revise_timing` (2 files) | re-measured | none |
| Width sweep | §4.2 text | `results/scale_large.json` | re-measured | none |
| Kernel benchmark | §4.3 text | `results/revise_kernel` (3 files) | reconstructed (re-measured; commands rebuilt) | none |
| Re-run of rank x clip and memory benchmark | §7.4, S9.3 text | `results/revise_repro/e6_clip` (6), `results/revise_repro/membench` (16) | re-measured | none |

"None" means every run of the loop below is in the tree. The one missing scaling run is absent
from the final figures; nothing else is excluded from a float.

## Fig 6 (scaling) and Fig S8 (memory-time trade-off)

Fidelity against width, n=64 (rotation, seeds 0-4, 20,000 steps):

```bash
for a in exact snap1 skrtrl-r4 skrtrl-r16 kfrtrl uoro; do for s in 0 1 2 3 4; do
  python run_m3.py --task rotation --algo $a --seed $s --steps 20000 --outdir results/m3; done; done
```

n=128 and n=256 (rotation and a^nb^n, seeds 0-1):

```bash
for t in rotation anbn; do for a in exact snap1 skrtrl-r4 skrtrl-r16 kfrtrl uoro; do for s in 0 1; do
  python run_m3.py --task $t --algo $a --seed $s --n 128 --steps 8000 --log_every 200 --outdir results/scale
  python run_m3.py --task $t --algo $a --seed $s --n 256 --steps 6000 --log_every 200 --outdir results/scale256
done; done; done
```

`results/scale` has 23 runs: rotation, SK-RTRL r=4, seed 0 is missing. The n=256 runs were written
to `results/scale` and moved to `results/scale256` afterwards, so their `args` still name `results/scale`.

Memory and time, shadow off (Fig 6 right, Fig S8):

```bash
for n in 128 256 384 512; do for a in exact snap1 skrtrl-r4 skrtrl-r16; do
  python run_m3.py --task anbn --algo $a --seed 0 --n $n --steps 1500 --shadow 0 --outdir results/membench --tag n$n
done; done
```

Draw both figures with `python make_paper_figures.py`.

## Fig S2 (older certificate-tightness sweep)

```bash
for clip in 0.2 0.35 0.5 0.7; do for t in adding anbn rotrecall24; do
  python run_m3.py --task $t --algo skrtrl-r16 --seed 0 --steps 25000 --clip $clip --outdir results/c2sweep --tag clip$clip
done; done
```

12 runs, 1,192 logged points. `python c2sweep_points.py results/c2sweep` prints the points.

## Fig S4 (adaptive-rank trajectory)

```bash
python run_adaptive.py --task rotation --seed 0 --steps 20000 --outdir results/round1/traj --tag traj
```

`run_adaptive.py` defaults to clip 0.5, rank range [4, 32] and the eta controller.

## Fig S5 (residual pilot)

```bash
for t in copy adding rotation anbn; do python run_m1_spectrum.py --task $t; done
```

6,000 steps, seed 0; writes `results/m1_spectrum_<task>.json`.

## Fig S6, left panel (multi-step horizons)

Horizons 5, 10 and 25, shadow off:

```bash
for t in henon mackeyglass lorenz; do for h in 5 10 25; do
  for a in exact snap1 skrtrl-r4 skrtrl-r16 rflo; do for s in 0 1 2 3 4; do
    python run_m3.py --task $t --algo $a --seed $s --horizon $h --steps 15000 --shadow 0 --outdir results/round1/horizon --tag h$h
  done; done; done; done
```

The h=1 points come from `results/ts`. Seeds 0-4 were logged every 500 steps, seeds 5-9 every 250:

```bash
for t in henon mackeyglass lorenz; do for a in exact skrtrl-r4 skrtrl-r16 snap1 uoro kfrtrl rflo; do
  for s in 0 1 2 3 4; do python run_m3.py --task $t --algo $a --seed $s --steps 15000 --log_every 500 --outdir results/ts; done
  for s in 5 6 7 8 9; do python run_m3.py --task $t --algo $a --seed $s --steps 15000 --outdir results/ts; done
done; done
```

`results/ts` also holds SK-RTRL r=32 and TBPTT at seeds 0-4 (same command, `--log_every 500`),
which the figure does not plot.

## Table S10 and Fig S9 (online T-maze)

```bash
for s in 0 1 2; do for a in rtu lru snap1 skrtrl-r16 exact tbptt; do
  python run_m5.py --env_len 10 --steps 60000  --algo $a --seed $s --n 64 --outdir results/m5iso
  python run_m5.py --env_len 20 --steps 100000 --algo $a --seed $s --n 64 --outdir results/m5iso
  python run_m5.py --env_len 40 --steps 150000 --algo $a --seed $s --n 64 --outdir results/m5iso
done; done
python make_m5_report.py results/m5iso
```

Pass `--steps`; the default of `run_m5.py` is 300,000. `make_m5_report.py` takes the mean of the
last 20% of each run and the sample s.d. over seeds.

## Text: §6.3 random-projection control

```bash
for t in adding rotation; do for a in skrtrl-rp4 skrtrl-rp16 skrtrl-rp64; do for s in 0 1 2; do
  python run_m3.py --task $t --algo $a --seed $s --steps 20000 --outdir results/m3; done; done; done
```

The comparison uses SK-RTRL r=16 from the same tree at seeds 0-2 (the rotation runs above and
`--task adding` with the same command). `check_release.py` recomputes the four cosines.

## Text: §6.6 control-signal ablation

```bash
for t in rotation anbn; do for c in eta e_t oracle; do for s in 0 1 2; do
  python run_adaptive.py --task $t --ctrl $c --seed $s --steps 20000 --outdir results/round1/adapt_ablate --tag e5
done; done; done
```

## Text: §7.4 and S9.3 (rank x clip on Lorenz, normalisation and washout)

```bash
for r in 2 4 8 16 32; do for c in 0 0.9; do for s in 0 1 2; do
  python run_m3.py --task lorenz --algo skrtrl-r$r --seed $s --steps 15000 --shadow 0 --clip $c --outdir results/round1/ablate_rank --tag clip$c
done; done; done
for t in sunspot laser mackeyglass; do for c in 0 1; do for w in 0 200; do for s in 0 1 2; do
  python run_m3.py --task $t --algo skrtrl-r16 --seed $s --steps 15000 --shadow 0 --causal $c --washout $w --outdir results/round1/ablate_norm --tag c${c}w$w
done; done; done; done
```

## Re-measurements under PyTorch 2.6.0

Table 10 and the exact-path agreement, two invocations:

```bash
python run_cor3_timing_revise.py
python run_cor3_timing_revise.py --ns 256,320,384,512 --steps 20 --warmup 10 --repeats 3 --out results/revise_timing/cor3_timing_revise_large.json
```

§4.2 width sweep: `python run_width_sweep.py` (writes `results/scale_large.json`).

§4.3 kernel benchmark (**reconstructed**). The argument lists were not saved; the settings below
are read from the `env` and row fields of each file:

```bash
python run_amortised_kernel_bench.py --modes svd --rs 8,16,32 --steps 6 --warmup 3 --repeats 3 --sweep_K "" --out results/revise_kernel/kernel_bench_svd.json
python run_amortised_kernel_bench.py --modes randproj --rs 8,16,32,64 --sweep_K 512:32 --rscan 256:8,16,32,64,128,192,256 --out results/revise_kernel/kernel_bench_randproj.json
python make_kernel_table.py
```

`results/revise_kernel/kernel_rscan_n256.json` repeats the r-scan alone (20 timed steps, 10
warmup, 3 repeats).

Re-run of the rank x clip ablation and the memory benchmark, compared point by point with the
first-campaign values:

```bash
for c in 0 0.9; do for s in 0 1 2; do
  python run_m3.py --task lorenz --algo skrtrl-r8 --seed $s --steps 15000 --shadow 0 --clip $c --outdir results/revise_repro/e6_clip --tag clip$c
done; done
# memory benchmark: the membench loop above with --outdir results/revise_repro/membench
python make_revise_repro_report.py
```
