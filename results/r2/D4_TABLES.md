# D4 unified-protocol result tables (Tables S6-S8)

Generated 2026-10-15 12:31:23 by `code/make_r2_d4_tables.py`. Every number here comes from the run json files alone; nothing is copied from the submitted paper.

- input directories: `results/r2/d4_eval`
- protocol asserted per cell: 10 seeds 0-9 x 20000 steps, shadow on, the stage-2 D4 learning rate
- per-run value: mean over records with `step >= 0.8 * args.steps` (last 20% window, the same rule that scored the LR selection)
- across seeds: mean +/- population standard deviation (ddof=0), seed = unit of aggregation
- metric direction: accuracy (higher better) for anbn, copy; MSE/NMSE (lower better) for every other task; `grad_cos` higher better everywhere
- LR selection: results/r2/D4_SELECT_stage2.json (81 of 81 pairs have a stage-2 winner, generated 2026-10-15 12:31:20)
- LaTeX written: `tab_r2_fidelity.tex`, `tab_r2_timeseries.tex`, `tab_r2_realts.tex`, `tab_r2_lr.tex`, `tab_r2_stats.tex`

A cell marked `†` fails at least one protocol assertion; the LaTeX tables carry the same dagger and every deviation is itemised below.

## Assertions

| check | result |
|---|---|
| metric-direction table vs `skrtrl/tasks.py` | OK |
| runs parsed | 810 |
| cells used (task, method, seed) | 810 |
| cells with n_seeds = 10 and steps = 20000 | 144 of 144 |
| cells flagged `†` (present but off-protocol) | 0 (all compliant) |
| cells with no usable run at all | 0 |
| missing runs | 0 of 810 expected |
| LR equals the stage-2 selection | OK for all 810 runs |
| window mean equals `make_r2_d4_select`'s score | OK for all 810 runs |
| runs present but unused (duplicate / off-config) | 0 |
| unreadable / malformed inputs | 0 |

## Table S6 -- gradient fidelity on the diagnostic tasks

### grad_cos, last-window mean per run, mean +/- std over seeds

LaTeX: `tab_r2_fidelity.tex` (`tab:r2fidelity`). Exact RTRL and TBPTT are omitted: the cosine is undefined for them, not missing.

| Method | copy | adding | rotation | anbn |
|---|---|---|---|---|
| SK-RTRL r=32 | 0.995 +/- 0.008 (n=10) | **0.894 +/- 0.030 (n=10)** | **0.930 +/- 0.040 (n=10)** | **0.993 +/- 0.002 (n=10)** |
| SK-RTRL r=16 | **0.999 +/- 0.002 (n=10)** | 0.853 +/- 0.040 (n=10) | 0.841 +/- 0.056 (n=10) | 0.983 +/- 0.006 (n=10) |
| SK-RTRL r=4 | 0.997 +/- 0.005 (n=10) | 0.844 +/- 0.042 (n=10) | 0.679 +/- 0.051 (n=10) | 0.972 +/- 0.008 (n=10) |
| SnAp-1 | 0.820 +/- 0.006 (n=10) | 0.524 +/- 0.064 (n=10) | 0.411 +/- 0.014 (n=10) | 0.470 +/- 0.115 (n=10) |
| KF-RTRL | 0.606 +/- 0.095 (n=10) | 0.560 +/- 0.067 (n=10) | 0.379 +/- 0.069 (n=10) | 0.441 +/- 0.056 (n=10) |
| RFLO | 0.034 +/- 0.007 (n=10) | 0.531 +/- 0.059 (n=10) | 0.144 +/- 0.038 (n=10) | 0.516 +/- 0.051 (n=10) |
| UORO | 0.048 +/- 0.022 (n=10) | 0.119 +/- 0.039 (n=10) | 0.034 +/- 0.013 (n=10) | 0.091 +/- 0.014 (n=10) |

### Diagnostic task metric (supplementary, not a paper table)

copy and $a^nb^n$ are accuracies (higher better); adding and rotation are MSE (lower better). Provided because the statistics table reports paired tests on this metric as well as on the cosine.

| Method | copy | adding | rotation | anbn |
|---|---|---|---|---|
| exact RTRL | 0.1370 +/- 0.0144 (n=10) | **0.2067 +/- 0.0085 (n=10)** | 0.1072 +/- 0.0057 (n=10) | **0.9347 +/- 0.0072 (n=10)** |
| SK-RTRL r=32 | 0.1711 +/- 0.0194 (n=10) | 0.2098 +/- 0.0095 (n=10) | 0.1039 +/- 0.0089 (n=10) | 0.9274 +/- 0.0106 (n=10) |
| SK-RTRL r=16 | 0.1542 +/- 0.0267 (n=10) | 0.2118 +/- 0.0105 (n=10) | 0.1129 +/- 0.0043 (n=10) | 0.9237 +/- 0.0082 (n=10) |
| SK-RTRL r=4 | 0.1392 +/- 0.0151 (n=10) | 0.2116 +/- 0.0104 (n=10) | 0.1098 +/- 0.0057 (n=10) | 0.9156 +/- 0.0112 (n=10) |
| SnAp-1 | 0.1242 +/- 0.0041 (n=10) | 0.2131 +/- 0.0200 (n=10) | 0.0509 +/- 0.0044 (n=10) | 0.9018 +/- 0.0023 (n=10) |
| KF-RTRL | 0.1467 +/- 0.0129 (n=10) | 0.2172 +/- 0.0079 (n=10) | 0.1137 +/- 0.0081 (n=10) | 0.8961 +/- 0.0054 (n=10) |
| RFLO | **0.1824 +/- 0.0242 (n=10)** | 0.2268 +/- 0.0145 (n=10) | 0.0977 +/- 0.0129 (n=10) | 0.9093 +/- 0.0036 (n=10) |
| UORO | 0.1264 +/- 0.0057 (n=10) | 0.2613 +/- 0.0824 (n=10) | 0.1186 +/- 0.0034 (n=10) | 0.8855 +/- 0.0040 (n=10) |
| TBPTT | 0.1256 +/- 0.0050 (n=10) | **0.2067 +/- 0.0085 (n=10)** | **0.0332 +/- 0.0080 (n=10)** | 0.9050 +/- 0.0071 (n=10) |

## Table S7 -- chaotic systems

### NMSE (lower better)

LaTeX: `tab_r2_timeseries.tex` (`tab:r2timeseries`).

| Method | henon | mackeyglass | lorenz |
|---|---|---|---|
| exact RTRL | **0.0045 +/- 0.0005 (n=10)** | 0.0024 +/- 0.0002 (n=10) | 0.0013 +/- 0.0002 (n=10) |
| SK-RTRL r=32 | 0.0050 +/- 0.0004 (n=10) | **0.0023 +/- 0.0003 (n=10)** | **0.0012 +/- 0.0001 (n=10)** |
| SK-RTRL r=16 | 0.0051 +/- 0.0003 (n=10) | 0.0025 +/- 0.0003 (n=10) | 0.0014 +/- 0.0003 (n=10) |
| SK-RTRL r=4 | 0.0052 +/- 0.0003 (n=10) | 0.0031 +/- 0.0006 (n=10) | 0.0014 +/- 0.0002 (n=10) |
| SnAp-1 | 0.0046 +/- 0.0006 (n=10) | 0.0035 +/- 0.0004 (n=10) | 0.0025 +/- 0.0007 (n=10) |
| KF-RTRL | 0.0205 +/- 0.0262 (n=10) | 0.0043 +/- 0.0005 (n=10) | 0.0021 +/- 0.0002 (n=10) |
| RFLO | 0.0565 +/- 0.0195 (n=10) | 0.0071 +/- 0.0019 (n=10) | 0.0042 +/- 0.0009 (n=10) |
| UORO | 0.6558 +/- 0.0319 (n=10) | 0.0054 +/- 0.0009 (n=10) | 0.0043 +/- 0.0006 (n=10) |
| TBPTT | **0.0045 +/- 0.0008 (n=10)** | 0.0127 +/- 0.0323 (n=10) | 0.0211 +/- 0.0322 (n=10) |

### grad_cos per system (the LaTeX table pools these into one column)

| Method | henon | mackeyglass | lorenz |
|---|---|---|---|
| SK-RTRL r=32 | **0.985 +/- 0.003 (n=10)** | **0.903 +/- 0.007 (n=10)** | **0.901 +/- 0.017 (n=10)** |
| SK-RTRL r=16 | 0.977 +/- 0.003 (n=10) | 0.846 +/- 0.019 (n=10) | 0.839 +/- 0.028 (n=10) |
| SK-RTRL r=4 | 0.976 +/- 0.004 (n=10) | 0.844 +/- 0.027 (n=10) | 0.839 +/- 0.032 (n=10) |
| SnAp-1 | 0.781 +/- 0.038 (n=10) | 0.353 +/- 0.091 (n=10) | 0.333 +/- 0.041 (n=10) |
| KF-RTRL | 0.629 +/- 0.070 (n=10) | 0.378 +/- 0.042 (n=10) | 0.459 +/- 0.027 (n=10) |
| RFLO | 0.392 +/- 0.053 (n=10) | 0.162 +/- 0.055 (n=10) | 0.200 +/- 0.052 (n=10) |
| UORO | 0.043 +/- 0.003 (n=10) | 0.043 +/- 0.007 (n=10) | 0.054 +/- 0.005 (n=10) |

### grad cosine pooled over the three systems and all seeds

| Method | grad cosine (pooled) | n runs | notes |
|---|---|---|---|
| exact RTRL | -- | -- | not defined for this method |
| SK-RTRL r=32 | 0.930 +/- 0.041 | 30 | -- |
| SK-RTRL r=16 | 0.887 +/- 0.066 | 30 | -- |
| SK-RTRL r=4 | 0.886 +/- 0.068 | 30 | -- |
| SnAp-1 | 0.489 +/- 0.216 | 30 | -- |
| KF-RTRL | 0.489 +/- 0.116 | 30 | -- |
| RFLO | 0.251 +/- 0.114 | 30 | -- |
| UORO | 0.047 +/- 0.007 | 30 | -- |
| TBPTT | -- | -- | not defined for this method |

## Table S8 -- Sunspot and Santa Fe laser

### NMSE (lower better)

LaTeX: `tab_r2_realts.tex` (`tab:r2realts`).

| Method | sunspot | laser |
|---|---|---|
| exact RTRL | 0.1335 +/- 0.0020 (n=10) | **0.0160 +/- 0.0017 (n=10)** |
| SK-RTRL r=32 | **0.1334 +/- 0.0019 (n=10)** | 0.0179 +/- 0.0015 (n=10) |
| SK-RTRL r=16 | 0.1335 +/- 0.0020 (n=10) | 0.0197 +/- 0.0036 (n=10) |
| SK-RTRL r=4 | 0.1335 +/- 0.0020 (n=10) | 0.0213 +/- 0.0040 (n=10) |
| SnAp-1 | 0.1398 +/- 0.0021 (n=10) | 0.0200 +/- 0.0028 (n=10) |
| KF-RTRL | 0.1365 +/- 0.0020 (n=10) | 0.0560 +/- 0.0308 (n=10) |
| RFLO | 0.1423 +/- 0.0020 (n=10) | 0.2001 +/- 0.0970 (n=10) |
| UORO | 0.1388 +/- 0.0021 (n=10) | 0.1509 +/- 0.0166 (n=10) |
| TBPTT | 0.1342 +/- 0.0020 (n=10) | 0.0167 +/- 0.0019 (n=10) |

### grad_cos per series

| Method | sunspot | laser |
|---|---|---|
| SK-RTRL r=32 | **0.931 +/- 0.015 (n=10)** | **0.960 +/- 0.010 (n=10)** |
| SK-RTRL r=16 | 0.899 +/- 0.014 (n=10) | 0.932 +/- 0.013 (n=10) |
| SK-RTRL r=4 | 0.898 +/- 0.015 (n=10) | 0.931 +/- 0.007 (n=10) |
| SnAp-1 | 0.622 +/- 0.032 (n=10) | 0.378 +/- 0.027 (n=10) |
| KF-RTRL | 0.442 +/- 0.025 (n=10) | 0.474 +/- 0.048 (n=10) |
| RFLO | 0.337 +/- 0.064 (n=10) | 0.046 +/- 0.045 (n=10) |
| UORO | 0.063 +/- 0.009 (n=10) | 0.033 +/- 0.003 (n=10) |

### grad cosine pooled over both series and all seeds

| Method | grad cosine (pooled) | n runs | notes |
|---|---|---|---|
| exact RTRL | -- | -- | not defined for this method |
| SK-RTRL r=32 | 0.945 +/- 0.019 | 20 | -- |
| SK-RTRL r=16 | 0.915 +/- 0.021 | 20 | -- |
| SK-RTRL r=4 | 0.914 +/- 0.021 | 20 | -- |
| SnAp-1 | 0.500 +/- 0.126 | 20 | -- |
| KF-RTRL | 0.458 +/- 0.041 | 20 | -- |
| RFLO | 0.192 +/- 0.155 | 20 | -- |
| UORO | 0.048 +/- 0.017 | 20 | -- |
| TBPTT | -- | -- | not defined for this method |

## Selected learning rate per (task, estimator)

LaTeX: `tab_r2_lr.tex` (`tab:r2lr`). Source: results/r2/D4_SELECT_stage2.json (81 of 81 pairs have a stage-2 winner, generated 2026-10-15 12:31:20)

| Method | adding | copy | anbn | rotation | henon | mackeyglass | lorenz | sunspot | laser |
|---|---|---|---|---|---|---|---|---|---|
| exact RTRL | 3e-3 | 1e-2 | 1e-3 | 1e-4 | 1e-3 | 3e-4 | 3e-4 | 1e-4 | 1e-3 |
| SK-RTRL r=32 | 3e-3 | 3e-3 | 1e-3 | 1e-4 | 1e-3 | 3e-4 | 3e-4 | 1e-4 | 1e-3 |
| SK-RTRL r=16 | 3e-3 | 1e-2 | 1e-3 | 3e-4 | 1e-3 | 3e-4 | 3e-4 | 1e-4 | 1e-3 |
| SK-RTRL r=4 | 3e-3 | 1e-2 | 1e-3 | 1e-4 | 1e-3 | 3e-4 | 3e-4 | 1e-4 | 1e-3 |
| SnAp-1 | 1e-2 | 1e-4 | 1e-3 | 3e-4 | 1e-3 | 3e-4 | 3e-4 | 1e-4 | 1e-3 |
| KF-RTRL | 3e-3 | 1e-2 | 1e-3 | 3e-4 | 1e-3 | 3e-4 | 3e-4 | 1e-4 | 1e-3 |
| RFLO | 3e-3 | 3e-3 | 1e-3 | 1e-3 | 3e-3 | 1e-4 | 1e-4 | 1e-4 | 3e-4 |
| UORO | 1e-2 | 1e-3 | 1e-3 | 1e-4 | 3e-4 | 3e-4 | 3e-4 | 1e-4 | 3e-4 |
| TBPTT | 3e-3/k=25 | 1e-4/k=25 | 3e-3/k=50 | 1e-3/k=25 | 1e-2/k=25 | 1e-2/k=25 | 1e-2/k=25 | 3e-3/k=50 | 1e-2/k=25 |

