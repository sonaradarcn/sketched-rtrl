# D3 adaptive rank: saturation diagnosis + sensitivity — aggregated report

Diagnosis data directory: `results/r2/d3_diag` (72 usable files)

OAT data directory: `results/r2/d3_oat` (150 usable files)

Criteria: grad_cos target 0.95; relative error target ≤ (1+0.05)×best in the sweep. Tail = mean over the last 20% of logged checkpoints.

## Missing files

d3_diag: 72 usable files read; roster 96 entries, 24 missing. d3_oat: 150 usable files read; roster 150 entries, 0 missing.

Note: the diagnosis roster is the full cross of kernel variant × cap cell, but the campaign only commits to some of these combinations (n=64 with fixed c=8 + forced pre-projection, n=128 with natural c=16), so some missing entries may be runs that were never planned.

**d3_diag missing**

- adaptive · anbn · n=128, r_max=64: fixed c=8/s0, fixed c=8/s1, fixed c=8/s2, pre-projection off/s0, pre-projection off/s1, pre-projection off/s2
- adaptive · anbn · n=64, r_max=64 (force_pp): c(r) rule/s0, c(r) rule/s1, c(r) rule/s2, pre-projection off/s0, pre-projection off/s1, pre-projection off/s2
- adaptive · rotation · n=128, r_max=64: fixed c=8/s0, fixed c=8/s1, fixed c=8/s2, pre-projection off/s0, pre-projection off/s1, pre-projection off/s2
- adaptive · rotation · n=64, r_max=64 (force_pp): c(r) rule/s0, c(r) rule/s1, c(r) rule/s2, pre-projection off/s0, pre-projection off/s1, pre-projection off/s2

d3_oat roster complete.

## Table A Ceiling diagnosis (D3 saturation diagnosis)

Tail = mean over the last 20% of logged checkpoints. Error: MSE for mse tasks, 1−accuracy for ce tasks (copy/anbn). `*` = number derived by the records fallback (see "Field tolerance").

### A-1 Adaptive controller (each cell mean±s.d. over seeds)

| task | cap cell | kernel variant | seeds | frac_at_cap | n_up | n_down | mean rank | tail error | grad_cos | peak memory MB | τ_c trigger share | τ_c/τ_r normalised mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rotation | n=64, r_max=32 | c(r) rule [c_eff=8] | 0,1,2 | 0.975±0.000 | 3.0±0.0 | 0.0±0.0 | 31.43±0.00 | 0.1212±0.0005 | 0.9998±0.0000 | 58.8±0.0 | 100.0% (rank up 100.0%) | 0.1029 / 0.0243 |
| rotation | n=64, r_max=32 | fixed c=8 [c_eff=8] | 0,1,2 | 0.975±0.000 | 3.0±0.0 | 0.0±0.0 | 31.43±0.00 | 0.1212±0.0005 | 0.9998±0.0000 | 58.8±0.0 | 100.0% (rank up 100.0%) | 0.1029 / 0.0243 |
| rotation | n=64, r_max=32 | pre-projection off [c_eff=64] | 0,1,2 | 0.975±0.000 | 3.0±0.0 | 0.0±0.0 | 31.43±0.00 | 0.1212±0.0005 | 1.0000±0.0000 | 87.6±0.0 | 0.0% (rank up 0.0%) | 0.0000 / 0.1043 |
| rotation | n=64, r_max=64 (force_pp) | fixed c=8 [c_eff=8] | 0,1,2 | 0.967±0.000 | 4.0±0.0 | 0.0±0.0 | 62.36±0.00 | 0.1212±0.0005 | 0.9998±0.0000 | 69.8±0.0 | 100.0% (rank up 100.0%) | 0.0885 / 0.0182 |
| rotation | n=128, r_max=64 | c(r) rule [c_eff=16] | 0,1,2 | 0.000±0.000 | 2.0±0.0 | 0.0±0.0 | 15.81±0.04 | 0.1222±0.0006 | 1.0000±0.0000 | 307.7±0.0 | 66.7% (rank up 66.7%) | 0.0634 / 0.0429 |
| anbn | n=64, r_max=32 | c(r) rule [c_eff=8] | 0,1,2 | 0.000±0.000 | 0.3±0.6 | 0.0±0.0 | 5.19±2.06 | 0.1100±0.0024 | 1.0000±0.0000 | 51.7±0.8 | 0.0% (rank up 0.0%) | 0.0086 / 0.0122 |
| anbn | n=64, r_max=32 | fixed c=8 [c_eff=8] | 0,1,2 | 0.000±0.000 | 0.3±0.6 | 0.0±0.0 | 5.19±2.06 | 0.1100±0.0024 | 1.0000±0.0000 | 51.7±0.8 | 0.0% (rank up 0.0%) | 0.0086 / 0.0122 |
| anbn | n=64, r_max=32 | pre-projection off [c_eff=64] | 0,1,2 | 0.000±0.000 | 0.0±0.0 | 0.0±0.0 | 4.00±0.00 | 0.1100±0.0024 | 1.0000±0.0000 | 70.3±0.0 | n/a | n/a |
| anbn | n=64, r_max=64 (force_pp) | fixed c=8 [c_eff=8] | 0,1,2 | 0.000±0.000 | 0.3±0.6 | 0.0±0.0 | 5.19±2.06 | 0.1100±0.0024 | 1.0000±0.0000 | 51.5±0.4 | 0.0% (rank up 0.0%) | 0.0086 / 0.0122 |
| anbn | n=128, r_max=64 | c(r) rule [c_eff=16] | 0,1,2 | 0.000±0.000 | 0.3±0.6 | 0.0±0.0 | 4.31±0.54 | 0.1100±0.0024 | 0.9999±0.0001 | 285.2±1.8 | 0.0% (rank up 0.0%) | 0.0064 / 0.0152 |

### A-2 Same-trajectory fixed-rank sweep vs the controller's actual mean rank

**anbn, n=64** — completed ranks [4, 8, 16, 24, 32, 48, 64]; best tail error in the sweep 0.10996; relative threshold (1+0.05)×best = 0.11546

| fixed r | tail error | grad_cos | mean rank (check) | seeds |
|---|---|---|---|---|
| 4 | 0.10996±0.00235 | 1.0000±0.0000 | 4.00±0.00 | 0,1,2 |
| 8 | 0.10996±0.00235 | 1.0000±0.0000 | 8.00±0.00 | 0,1,2 |
| 16 | 0.10996±0.00235 | 1.0000±0.0000 | 16.00±0.00 | 0,1,2 |
| 24 | 0.10996±0.00235 | 1.0000±0.0000 | 24.00±0.00 | 0,1,2 |
| 32 | 0.10996±0.00235 | 1.0000±0.0000 | 32.00±0.00 | 0,1,2 |
| 48 | 0.10996±0.00235 | 1.0000±0.0000 | 48.00±0.00 | 0,1,2 |
| 64 | 0.10996±0.00235 | 1.0000±0.0000 | 64.00±0.00 | 0,1,2 |

- smallest r reaching grad_cos ≥ 0.95: **4**
- smallest r with tail error ≤ (1+0.05)×best: **4**
- controller [n=64, r_max=32 / fixed c=8] actual mean rank 5.19±2.06, frac_at_cap 0.000±0.000
- controller [n=64, r_max=32 / c(r) rule] actual mean rank 5.19±2.06, frac_at_cap 0.000±0.000
- controller [n=64, r_max=32 / pre-projection off] actual mean rank 4.00±0.00, frac_at_cap 0.000±0.000
- controller [n=64, r_max=64 (force_pp) / fixed c=8] actual mean rank 5.19±2.06, frac_at_cap 0.000±0.000

**rotation, n=64** — completed ranks [4, 8, 16, 24, 32, 48, 64]; best tail error in the sweep 0.12119; relative threshold (1+0.05)×best = 0.12724

| fixed r | tail error | grad_cos | mean rank (check) | seeds |
|---|---|---|---|---|
| 4 | 0.12163±0.00030 | 0.9954±0.0009 | 4.00±0.00 | 0,1,2 |
| 8 | 0.12119±0.00054 | 0.9998±0.0000 | 8.00±0.00 | 0,1,2 |
| 16 | 0.12119±0.00053 | 0.9998±0.0000 | 16.00±0.00 | 0,1,2 |
| 24 | 0.12119±0.00053 | 0.9998±0.0000 | 24.00±0.00 | 0,1,2 |
| 32 | 0.12119±0.00053 | 0.9998±0.0000 | 32.00±0.00 | 0,1,2 |
| 48 | 0.12119±0.00053 | 0.9998±0.0000 | 48.00±0.00 | 0,1,2 |
| 64 | 0.12119±0.00053 | 0.9998±0.0000 | 64.00±0.00 | 0,1,2 |

- smallest r reaching grad_cos ≥ 0.95: **4**
- smallest r with tail error ≤ (1+0.05)×best: **4**
- controller [n=64, r_max=32 / fixed c=8] actual mean rank 31.43±0.00, frac_at_cap 0.975±0.000
- controller [n=64, r_max=32 / c(r) rule] actual mean rank 31.43±0.00, frac_at_cap 0.975±0.000
- controller [n=64, r_max=32 / pre-projection off] actual mean rank 31.43±0.00, frac_at_cap 0.975±0.000
- controller [n=64, r_max=64 (force_pp) / fixed c=8] actual mean rank 62.36±0.00, frac_at_cap 0.967±0.000


## Table B OAT one-at-a-time sensitivity (D3 local sensitivity)

⚑ = task or variant outside the D3 OAT roster (for example the controller ablation that changes only `--ctrl`). It is still listed as a legitimate single-knob variant, for reference.

### B-0 Range across all variants (completed variants only)

| task | completed variants | error min→max | error range | mean rank min→max | rank range |
|---|---|---|---|---|---|
| rotation | 10/10 | 0.12119 → 0.12120 | 0.00001 | 26.78 → 58.47 | 31.69 |
| henon | 10/10 | 0.00477 → 0.00486 | 0.00009 | 4.00 → 28.80 | 24.80 |
| sunspot | 10/10 | 0.14293 → 0.14293 | 0.00000 | 2.00 → 4.00 | 2.00 |
| anbn | 10/10 | 0.10996 → 0.10996 | 0.00000 | 4.00 → 16.39 | 12.39 |
| mackeyglass | 10/10 | 0.00706 → 0.00708 | 0.00002 | 2.25 → 7.71 | 5.46 |

### B-1 Per task × variant (mean±s.d. over seeds)

| task | variant | seeds | tail error | Δerror vs default | mean rank | Δrank vs default | frac_at_cap | grad_cos | n_up | n_down | certificate e_t | Δe_t | peak_MB | Δpeak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rotation | **default** | 0,1,2 | 0.12119±0.00053 | +0.00000 (+0.0%) | 31.43±0.00 | +0.00 (+0.0%) | 0.975±0.000 | 0.9998±0.0000 | 3.0±0.0 | 0.0±0.0 | 4.0479±3.0256 | +0 (+0.0%) | 58.8±0.0 | +0 (+0.0%) |
| rotation | tau_high=0.01 | 0,1,2 | 0.12119±0.00053 | +0.00000 (+0.0%) | 31.43±0.00 | +0.00 (+0.0%) | 0.975±0.000 | 0.9998±0.0000 | 3.0±0.0 | 0.0±0.0 | 4.0479±3.0256 | +0 (+0.0%) | 58.8±0.0 | +0 (+0.0%) |
| rotation | tau_high=0.04 | 0,1,2 | 0.12119±0.00053 | +0.00000 (+0.0%) | 31.43±0.00 | +0.00 (+0.0%) | 0.975±0.000 | 0.9998±0.0000 | 3.0±0.0 | 0.0±0.0 | 4.0479±3.0256 | +0 (+0.0%) | 58.8±0.0 | +0 (+0.0%) |
| rotation | tau_low=0.0015 | 0,1,2 | 0.12119±0.00053 | +0.00000 (+0.0%) | 31.43±0.00 | +0.00 (+0.0%) | 0.975±0.000 | 0.9998±0.0000 | 3.0±0.0 | 0.0±0.0 | 4.0479±3.0256 | +0 (+0.0%) | 58.8±0.0 | +0 (+0.0%) |
| rotation | tau_low=0.006 | 0,1,2 | 0.12119±0.00053 | +0.00000 (+0.0%) | 31.43±0.00 | +0.00 (+0.0%) | 0.975±0.000 | 0.9998±0.0000 | 3.0±0.0 | 0.0±0.0 | 4.0479±3.0256 | +0 (+0.0%) | 58.8±0.0 | +0 (+0.0%) |
| rotation | M=2 | 0,1,2 | 0.12119±0.00053 | +0.00000 (+0.0%) | 31.43±0.00 | +0.00 (+0.0%) | 0.975±0.000 | 0.9998±0.0000 | 3.0±0.0 | 0.0±0.0 | 4.0479±3.0256 | +0 (+0.0%) | 58.8±0.0 | +0 (+0.0%) |
| rotation | M=5 | 0,1,2 | 0.12119±0.00053 | +0.00000 (+0.0%) | 31.43±0.00 | +0.00 (+0.0%) | 0.975±0.000 | 0.9998±0.0000 | 3.0±0.0 | 0.0±0.0 | 4.0479±3.0256 | +0 (+0.0%) | 58.8±0.0 | +0 (+0.0%) |
| rotation | K=50 | 0,1,2 | 0.12119±0.00053 | -0.00000 (-0.0%) | 31.71±0.00 | +0.28 (+0.9%) | 0.987±0.000 | 0.9998±0.0000 | 3.0±0.0 | 0.0±0.0 | 3.9971±3.0372 | -0.05082 (-1.3%) | 58.8±0.0 | +0 (+0.0%) |
| rotation | K=200 | 0,1,2 | 0.12120±0.00053 | +0.00000 (+0.0%) | 26.78±7.08 | -4.66 (-14.8%) | 0.694±0.443 | 0.9998±0.0000 | 3.0±0.0 | 0.0±0.0 | 3.9398±2.8781 | -0.1081 (-2.7%) | 58.8±0.0 | +0 (+0.0%) |
| rotation | interval[2,64] | 0,1,2 | 0.12120±0.00053 | +0.00001 (+0.0%) | 58.47±5.85 | +27.04 (+86.0%) | 0.853±0.183 | 0.9998±0.0000 | 5.0±0.0 | 0.0±0.0 | 3.3707±2.7959 | -0.6773 (-16.7%) | 69.8±0.0 | +11.01 (+18.7%) |
| henon | **default** | 0,1,2 | 0.00486±0.00074 | +0.00000 (+0.0%) | 7.57±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 1.0±0.0 | 0.0±0.0 | 0.1503±0.0026 | +0 (+0.0%) | 52.3±0.0 | +0 (+0.0%) |
| henon | tau_high=0.01 | 0,1,2 | 0.00479±0.00074 | -0.00007 (-1.4%) | 28.80±0.06 | +21.23 (+280.6%) | 0.858±0.000 | 1.0000±0.0000 | 3.3±0.6 | 0.3±0.6 | 0.1463±0.0017 | -0.004026 (-2.7%) | 55.1±0.0 | +2.75 (+5.3%) |
| henon | tau_high=0.04 | 0,1,2 | 0.00477±0.00069 | -0.00009 (-2.0%) | 4.00±0.00 | -3.57 (-47.1%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.2990±0.0121 | +0.1486 (+98.9%) | 50.9±0.0 | -1.445 (-2.8%) |
| henon | tau_low=0.0015 | 0,1,2 | 0.00486±0.00074 | +0.00000 (+0.0%) | 7.57±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 1.0±0.0 | 0.0±0.0 | 0.1503±0.0026 | +0 (+0.0%) | 52.3±0.0 | +0 (+0.0%) |
| henon | tau_low=0.006 | 0,1,2 | 0.00477±0.00069 | -0.00009 (-1.8%) | 4.50±0.00 | -3.07 (-40.5%) | 0.000±0.000 | 1.0000±0.0000 | 1.0±0.0 | 1.0±0.0 | 0.2997±0.0124 | +0.1493 (+99.3%) | 52.3±0.0 | +0 (+0.0%) |
| henon | M=2 | 0,1,2 | 0.00486±0.00074 | +0.00000 (+0.0%) | 7.57±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 1.0±0.0 | 0.0±0.0 | 0.1503±0.0026 | +0 (+0.0%) | 52.3±0.0 | +0 (+0.0%) |
| henon | M=5 | 0,1,2 | 0.00486±0.00074 | +0.00000 (+0.0%) | 7.57±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 1.0±0.0 | 0.0±0.0 | 0.1503±0.0026 | +0 (+0.0%) | 52.3±0.0 | +0 (+0.0%) |
| henon | K=50 | 0,1,2 | 0.00479±0.00076 | -0.00007 (-1.4%) | 7.84±0.24 | +0.28 (+3.7%) | 0.000±0.000 | 1.0000±0.0000 | 1.0±0.0 | 0.0±0.0 | 0.1506±0.0024 | +0.0002805 (+0.2%) | 52.3±0.0 | +0 (+0.0%) |
| henon | K=200 | 0,1,2 | 0.00477±0.00069 | -0.00009 (-2.0%) | 4.00±0.00 | -3.57 (-47.1%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.2990±0.0121 | +0.1486 (+98.9%) | 50.9±0.0 | -1.445 (-2.8%) |
| henon | interval[2,64] | 0,1,2 | 0.00477±0.00065 | -0.00009 (-1.9%) | 7.22±0.00 | -0.35 (-4.6%) | 0.000±0.000 | 1.0000±0.0000 | 2.0±0.0 | 0.0±0.0 | 0.1505±0.0055 | +0.0001359 (+0.1%) | 51.9±0.0 | -0.3906 (-0.7%) |
| sunspot | **default** | 0,1,2 | 0.14293±0.00143 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.1659±0.0023 | +0 (+0.0%) | 50.3±0.0 | +0 (+0.0%) |
| sunspot | tau_high=0.01 | 0,1,2 | 0.14293±0.00143 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.1659±0.0023 | +0 (+0.0%) | 50.3±0.0 | +0 (+0.0%) |
| sunspot | tau_high=0.04 | 0,1,2 | 0.14293±0.00143 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.1659±0.0023 | +0 (+0.0%) | 50.3±0.0 | +0 (+0.0%) |
| sunspot | tau_low=0.0015 | 0,1,2 | 0.14293±0.00143 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.1659±0.0023 | +0 (+0.0%) | 50.3±0.0 | +0 (+0.0%) |
| sunspot | tau_low=0.006 | 0,1,2 | 0.14293±0.00143 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.1659±0.0023 | +0 (+0.0%) | 50.3±0.0 | +0 (+0.0%) |
| sunspot | M=2 | 0,1,2 | 0.14293±0.00143 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.1659±0.0023 | +0 (+0.0%) | 50.3±0.0 | +0 (+0.0%) |
| sunspot | M=5 | 0,1,2 | 0.14293±0.00143 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.1659±0.0023 | +0 (+0.0%) | 50.3±0.0 | +0 (+0.0%) |
| sunspot | K=50 | 0,1,2 | 0.14293±0.00143 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.1659±0.0023 | +0 (+0.0%) | 50.3±0.0 | +0 (+0.0%) |
| sunspot | K=200 | 0,1,2 | 0.14293±0.00143 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.1659±0.0023 | +0 (+0.0%) | 50.3±0.0 | +0 (+0.0%) |
| sunspot | interval[2,64] | 0,1,2 | 0.14293±0.00141 | -0.00000 (-0.0%) | 2.00±0.00 | -2.00 (-50.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.1942±0.0017 | +0.0283 (+17.1%) | 50.1±0.0 | -0.2617 (-0.5%) |
| anbn | **default** | 0,1,2 | 0.10996±0.00235 | +0.00000 (+0.0%) | 5.19±2.06 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.3±0.6 | 0.0±0.0 | 0.4208±0.1477 | +0 (+0.0%) | 51.7±0.8 | +0 (+0.0%) |
| anbn | tau_high=0.01 | 0,1,2 | 0.10996±0.00235 | +0.00000 (+0.0%) | 16.39±9.77 | +11.20 (+215.8%) | 0.261±0.452 | 1.0000±0.0000 | 2.0±1.0 | 0.0±0.0 | 0.2536±0.0062 | -0.1672 (-39.7%) | 53.9±1.3 | +2.151 (+4.2%) |
| anbn | tau_high=0.04 | 0,1,2 | 0.10996±0.00235 | +0.00000 (+0.0%) | 4.00±0.00 | -1.19 (-22.9%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.4892±0.0314 | +0.06839 (+16.3%) | 51.3±0.0 | -0.4714 (-0.9%) |
| anbn | tau_low=0.0015 | 0,1,2 | 0.10996±0.00235 | +0.00000 (+0.0%) | 5.19±2.06 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.3±0.6 | 0.0±0.0 | 0.4208±0.1477 | +0 (+0.0%) | 51.7±0.8 | +0 (+0.0%) |
| anbn | tau_low=0.006 | 0,1,2 | 0.10996±0.00235 | +0.00000 (+0.0%) | 5.16±2.00 | -0.03 (-0.6%) | 0.000±0.000 | 1.0000±0.0000 | 0.3±0.6 | 0.3±0.6 | 0.4258±0.1390 | +0.005033 (+1.2%) | 51.7±0.8 | +0 (+0.0%) |
| anbn | M=2 | 0,1,2 | 0.10996±0.00235 | +0.00000 (+0.0%) | 5.19±2.06 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.3±0.6 | 0.0±0.0 | 0.4208±0.1477 | +0 (+0.0%) | 51.7±0.8 | +0 (+0.0%) |
| anbn | M=5 | 0,1,2 | 0.10996±0.00235 | +0.00000 (+0.0%) | 5.19±2.06 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.3±0.6 | 0.0±0.0 | 0.4208±0.1477 | +0 (+0.0%) | 51.7±0.8 | +0 (+0.0%) |
| anbn | K=50 | 0,1,2 | 0.10996±0.00235 | +0.00000 (+0.0%) | 6.40±2.11 | +1.21 (+23.3%) | 0.000±0.000 | 1.0000±0.0000 | 0.7±0.6 | 0.0±0.0 | 0.3440±0.1514 | -0.07674 (-18.2%) | 52.2±0.8 | +0.4714 (+0.9%) |
| anbn | K=200 | 0,1,2 | 0.10996±0.00235 | +0.00000 (+0.0%) | 5.18±2.04 | -0.01 (-0.2%) | 0.000±0.000 | 1.0000±0.0000 | 0.3±0.6 | 0.0±0.0 | 0.4208±0.1477 | -4.036e-06 (-0.0%) | 51.7±0.8 | +0 (+0.0%) |
| anbn | interval[2,64] | 0,1,2 | 0.10996±0.00235 | +0.00000 (+0.0%) | 6.05±1.99 | +0.86 (+16.6%) | 0.000±0.000 | 1.0000±0.0000 | 1.7±0.6 | 0.0±0.0 | 0.3537±0.1529 | -0.06706 (-15.9%) | 51.8±0.4 | +0.01302 (+0.0%) |
| mackeyglass | **default** | 0,1,2 | 0.00706±0.00104 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.2322±0.0129 | +0 (+0.0%) | 50.9±0.0 | +0 (+0.0%) |
| mackeyglass | tau_high=0.01 | 0,1,2 | 0.00707±0.00101 | +0.00001 (+0.2%) | 7.71±0.13 | +3.71 (+92.8%) | 0.000±0.000 | 1.0000±0.0000 | 1.3±0.6 | 0.3±0.6 | 0.1275±0.0063 | -0.1047 (-45.1%) | 52.3±0.0 | +1.444 (+2.8%) |
| mackeyglass | tau_high=0.04 | 0,1,2 | 0.00706±0.00104 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.2322±0.0129 | +0 (+0.0%) | 50.9±0.0 | +0 (+0.0%) |
| mackeyglass | tau_low=0.0015 | 0,1,2 | 0.00706±0.00104 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.2322±0.0129 | +0 (+0.0%) | 50.9±0.0 | +0 (+0.0%) |
| mackeyglass | tau_low=0.006 | 0,1,2 | 0.00706±0.00104 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.2322±0.0129 | +0 (+0.0%) | 50.9±0.0 | +0 (+0.0%) |
| mackeyglass | M=2 | 0,1,2 | 0.00706±0.00104 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.2322±0.0129 | +0 (+0.0%) | 50.9±0.0 | +0 (+0.0%) |
| mackeyglass | M=5 | 0,1,2 | 0.00706±0.00104 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.2322±0.0129 | +0 (+0.0%) | 50.9±0.0 | +0 (+0.0%) |
| mackeyglass | K=50 | 0,1,2 | 0.00708±0.00103 | +0.00002 (+0.3%) | 5.41±2.23 | +1.41 (+35.1%) | 0.000±0.000 | 1.0000±0.0000 | 1.0±0.0 | 0.7±0.6 | 0.1950±0.0532 | -0.03719 (-16.0%) | 52.3±0.0 | +1.445 (+2.8%) |
| mackeyglass | K=200 | 0,1,2 | 0.00706±0.00104 | +0.00000 (+0.0%) | 4.00±0.00 | +0.00 (+0.0%) | 0.000±0.000 | 1.0000±0.0000 | 0.0±0.0 | 0.0±0.0 | 0.2322±0.0129 | +0 (+0.0%) | 50.9±0.0 | +0 (+0.0%) |
| mackeyglass | interval[2,64] | 0,1,2 | 0.00707±0.00111 | +0.00001 (+0.1%) | 2.25±0.43 | -1.75 (-43.8%) | 0.000±0.000 | 1.0000±0.0000 | 0.3±0.6 | 0.0±0.0 | 0.2763±0.0510 | +0.04413 (+19.0%) | 50.7±0.2 | -0.1745 (-0.3%) |

### B-1b Does rank reduction ever fire?

**Rank reduction does fire: of the 150 OAT runs, 8 reduce their rank, 8 times in total.** Each one is listed, with `c_t` the control signal at that moment:

| run file | task | τ_low | M | K | step | r change | c_t | τ_c | τ_r |
|---|---|---|---|---|---|---|---|---|---|
| `anbn_adaptive-eta_s1_taulow0.006.json` | anbn | 0.006 | 3 | 100 | 11700 | 8→4 | 0.003365 | 0.003349 | 1.624e-05 |
| `henon_adaptive-eta_s0_taulow0.006.json` | henon | 0.006 | 3 | 100 | 2800 | 8→4 | 0 | 0 | 0 |
| `henon_adaptive-eta_s1_tauhigh0.01.json` | henon | 0.003 | 3 | 100 | 600 | 8→4 | 0 | 0 | 0 |
| `henon_adaptive-eta_s1_taulow0.006.json` | henon | 0.006 | 3 | 100 | 2800 | 8→4 | 0 | 0 | 0 |
| `henon_adaptive-eta_s2_taulow0.006.json` | henon | 0.006 | 3 | 100 | 2800 | 8→4 | 0 | 0 | 0 |
| `mackeyglass_adaptive-eta_s1_K50.json` | mackeyglass | 0.003 | 3 | 50 | 300 | 8→4 | 0.002768 | 0.00276 | 7.878e-06 |
| `mackeyglass_adaptive-eta_s2_K50.json` | mackeyglass | 0.003 | 3 | 50 | 500 | 8→4 | 0.001788 | 0.001744 | 4.315e-05 |
| `mackeyglass_adaptive-eta_s2_tauhigh0.01.json` | mackeyglass | 0.003 | 3 | 100 | 800 | 8→4 | 0 | 0 | 0 |

Scale of the control signal (recorded only at rank changes; see the table above and the trigger columns of Table A): whether a rank reduction can happen depends on how the default τ_low = 0.003 compares with the observed `c_t`.


## Key numbers (auto-generated, numbers only)

- rotation / n=64, r_max=32 / c(r) rule: frac_at_cap 97.5%, mean rank 31.43 (cap 32), tail error 0.12119, n_up/n_down 3.0/0.0, peak memory 58.8 MB
- rotation / n=64, r_max=32 / c(r) rule: 9 rank changes in total, τ_c dominant in 100.0% (among rank increases 100.0%), normalised τ_c mean 0.1029, τ_r mean 0.0243
- rotation / n=64, r_max=32 / fixed c=8: frac_at_cap 97.5%, mean rank 31.43 (cap 32), tail error 0.12119, n_up/n_down 3.0/0.0, peak memory 58.8 MB
- rotation / n=64, r_max=32 / fixed c=8: 9 rank changes in total, τ_c dominant in 100.0% (among rank increases 100.0%), normalised τ_c mean 0.1029, τ_r mean 0.0243
- rotation / n=64, r_max=32 / pre-projection off: frac_at_cap 97.5%, mean rank 31.43 (cap 32), tail error 0.12121, n_up/n_down 3.0/0.0, peak memory 87.6 MB
- rotation / n=64, r_max=32 / pre-projection off: 9 rank changes in total, τ_c dominant in 0.0% (among rank increases 0.0%), normalised τ_c mean 0.0000, τ_r mean 0.1043
- rotation / n=64, r_max=64 (force_pp) / fixed c=8: frac_at_cap 96.7%, mean rank 62.36 (cap 64), tail error 0.12119, n_up/n_down 4.0/0.0, peak memory 69.8 MB
- rotation / n=64, r_max=64 (force_pp) / fixed c=8: 12 rank changes in total, τ_c dominant in 100.0% (among rank increases 100.0%), normalised τ_c mean 0.0885, τ_r mean 0.0182
- rotation / n=128, r_max=64 / c(r) rule: frac_at_cap 0.0%, mean rank 15.81 (cap 64), tail error 0.12219, n_up/n_down 2.0/0.0, peak memory 307.7 MB
- rotation / n=128, r_max=64 / c(r) rule: 6 rank changes in total, τ_c dominant in 66.7% (among rank increases 66.7%), normalised τ_c mean 0.0634, τ_r mean 0.0429
- anbn / n=64, r_max=32 / c(r) rule: frac_at_cap 0.0%, mean rank 5.19 (cap 32), tail error 0.10996, n_up/n_down 0.3/0.0, peak memory 51.7 MB
- anbn / n=64, r_max=32 / c(r) rule: 1 rank changes in total, τ_c dominant in 0.0% (among rank increases 0.0%), normalised τ_c mean 0.0086, τ_r mean 0.0122
- anbn / n=64, r_max=32 / fixed c=8: frac_at_cap 0.0%, mean rank 5.19 (cap 32), tail error 0.10996, n_up/n_down 0.3/0.0, peak memory 51.7 MB
- anbn / n=64, r_max=32 / fixed c=8: 1 rank changes in total, τ_c dominant in 0.0% (among rank increases 0.0%), normalised τ_c mean 0.0086, τ_r mean 0.0122
- anbn / n=64, r_max=32 / pre-projection off: frac_at_cap 0.0%, mean rank 4.00 (cap 32), tail error 0.10996, n_up/n_down 0.0/0.0, peak memory 70.3 MB
- anbn / n=64, r_max=64 (force_pp) / fixed c=8: frac_at_cap 0.0%, mean rank 5.19 (cap 64), tail error 0.10996, n_up/n_down 0.3/0.0, peak memory 51.5 MB
- anbn / n=64, r_max=64 (force_pp) / fixed c=8: 1 rank changes in total, τ_c dominant in 0.0% (among rank increases 0.0%), normalised τ_c mean 0.0086, τ_r mean 0.0122
- anbn / n=128, r_max=64 / c(r) rule: frac_at_cap 0.0%, mean rank 4.31 (cap 64), tail error 0.10996, n_up/n_down 0.3/0.0, peak memory 285.2 MB
- anbn / n=128, r_max=64 / c(r) rule: 1 rank changes in total, τ_c dominant in 0.0% (among rank increases 0.0%), normalised τ_c mean 0.0064, τ_r mean 0.0152
- anbn, n=64: in the fixed-rank sweep the smallest r reaching grad_cos ≥ 0.95 = 4; smallest r with tail error ≤ (1+0.05)×best (0.10996) = 4; completed ranks [4, 8, 16, 24, 32, 48, 64]; controller mean rank for the same task and width 4.00–5.19
- rotation, n=64: in the fixed-rank sweep the smallest r reaching grad_cos ≥ 0.95 = 4; smallest r with tail error ≤ (1+0.05)×best (0.12119) = 4; completed ranks [4, 8, 16, 24, 32, 48, 64]; controller mean rank for the same task and width 31.43–62.36
- OAT rotation: tail error of the 10 variants goes from 0.12119 (K=50) to 0.12120 (interval[2,64]), range 0.00001
- OAT rotation: mean rank goes from 26.78 (K=200) to 58.47 (interval[2,64]), range 31.69
- OAT henon: tail error of the 10 variants goes from 0.00477 (K=200) to 0.00486 (M=2), range 0.00009
- OAT henon: mean rank goes from 4.00 (K=200) to 28.80 (tau_high=0.01), range 24.80
- OAT sunspot: the tail errors of all 10 variants agree to 5 decimals (0.14293), range 0.00000
- OAT sunspot: mean rank goes from 2.00 (interval[2,64]) to 4.00 (K=200), range 2.00
- OAT anbn: the tail errors of all 10 variants agree to 5 decimals (0.10996), range 0.00000
- OAT anbn: mean rank goes from 4.00 (tau_high=0.04) to 16.39 (tau_high=0.01), range 12.39
- OAT mackeyglass: tail error of the 10 variants goes from 0.00706 (K=200) to 0.00708 (K=50), range 0.00002
- OAT mackeyglass: mean rank goes from 2.25 (interval[2,64]) to 7.71 (tau_high=0.01), range 5.46

## Field tolerance

All runs carry the fields added for D3 (frac_at_cap / n_up / n_down / n_ctrl_checks / rank_changes / --c / --force_preproject / --preproject); no fallback was used.

How the fallbacks are computed:

- `frac_at_cap`: the top-level field is counted per step. The fallback can only use the `rank` checkpoints in `records`, one every `log_every` steps. Its resolution is much lower and it can over- or underestimate brief contact with the cap, so treat it as an order of magnitude only.
- `n_up`/`n_down`: the controller can change the rank every `K` steps, but the log records it only every `log_every` steps, so the fallback count is a lower bound (an up and a down move within one logging interval cancel).
- The trigger (τ_c dominant / τ_r dominant) and the normalised τ_c, τ_r exist only in `rank_changes`. There is no substitute, so when they are missing they are reported as `n/a` and not inferred.
- `peak_MB` is written as null by run_adaptive.py under `--device cpu`; this is expected.
- The kernel variant is decided from the `--preproject` / `--c` arguments first. If `c_effective` / `preproject_effective` (written back by run_adaptive.py) are present, they are added as a note on the value the kernel actually used.

## Figures

- `results/figures/fig_r2_adaptive_diag.pdf` — written
- `results/figures/fig_r2_adaptive_oat.pdf` — written

