# D4 all-pairs paired statistics

Generated 2026-10-15 12:31:23 by `code/make_r2_d4_tables.py`. Confidence intervals and effect sizes for **all** principal comparisons, not only those against SnAp-1.

- comparisons, fixed in advance: `skrtrl-r16` against each of the eight other estimators, plus `skrtrl-r4` and `skrtrl-r32` against `snap1` -- 10 per (task, metric)
- metrics: the task metric (`metric`) and the gradient cosine (`grad_cos`); `grad_cos` is undefined for exact, tbptt, so those comparisons are structurally absent and do not enter the family
- pairing unit: the seed. `delta` is signed so that a positive value means the first-named method is better, using the direction of that metric
- `p`: two-sided Wilcoxon signed-rank (exact for small n); `holm_p`: Holm step-down inside the family of that (task, metric), family size fixed in advance, so a comparison whose runs are missing still costs family size (conservative)
- `CI95`: 10000-resample paired bootstrap over seeds (rng seed 0); `d_z`: paired Cohen effect size
- a comparison needs at least 3 paired seeds; with 10 the two-sided Wilcoxon floor is p=0.002, with 5 it is p=0.0625

Computed 162 of 180 planned comparisons; 108 are Holm-significant at 0.05.

## adding (task metric: MSE, lower better)

### adding -- task metric (Holm family size 10)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 10 | 0.2118 | 0.2067 | -0.0051 | [-0.0072, -0.0034] | 0.0020 | 0.0195 | -1.55 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.2118 | 0.2098 | -0.0021 | [-0.0033, -0.0009] | 0.0098 | 0.0586 | -1.00 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.2118 | 0.2116 | -0.0002 | [-0.0004, -0.0001] | 0.0371 | 0.1855 | -0.81 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.2118 | 0.2131 | +0.0013 | [-0.0056, +0.0114] | 0.7695 | 1.0000 | +0.09 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.2118 | 0.2172 | +0.0054 | [+0.0007, +0.0102] | 0.1055 | 0.4219 | +0.66 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.2118 | 0.2268 | +0.0150 | [+0.0097, +0.0205] | 0.0020 | 0.0195 | +1.61 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.2118 | 0.2613 | +0.0495 | [+0.0170, +0.1063] | 0.0020 | 0.0195 | +0.57 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 10 | 0.2118 | 0.2067 | -0.0051 | [-0.0072, -0.0034] | 0.0020 | 0.0195 | -1.55 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.2116 | 0.2131 | +0.0015 | [-0.0054, +0.0116] | 0.7695 | 1.0000 | +0.10 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.2098 | 0.2131 | +0.0034 | [-0.0043, +0.0146] | 0.9219 | 1.0000 | +0.20 | seeds 0,1,2,3,4,5,6,7,8,9 |

### adding -- gradient cosine (Holm family size 8)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.853 | 0.894 | -0.042 | [-0.056, -0.029] | 0.0020 | 0.0156 | -1.79 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.853 | 0.844 | +0.009 | [+0.004, +0.015] | 0.0098 | 0.0156 | +0.94 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.853 | 0.524 | +0.328 | [+0.277, +0.385] | 0.0020 | 0.0156 | +3.53 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.853 | 0.560 | +0.293 | [+0.246, +0.336] | 0.0020 | 0.0156 | +3.78 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.853 | 0.531 | +0.321 | [+0.275, +0.378] | 0.0020 | 0.0156 | +3.62 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.853 | 0.119 | +0.734 | [+0.703, +0.765] | 0.0020 | 0.0156 | +14.21 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.844 | 0.524 | +0.319 | [+0.266, +0.379] | 0.0020 | 0.0156 | +3.28 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.894 | 0.524 | +0.370 | [+0.325, +0.421] | 0.0020 | 0.0156 | +4.50 | seeds 0,1,2,3,4,5,6,7,8,9 |

## copy (task metric: accuracy, higher better)

### copy -- task metric (Holm family size 10)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 10 | 0.1542 | 0.1370 | +0.0172 | [+0.0024, +0.0336] | 0.0840 | 0.3359 | +0.65 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.1542 | 0.1711 | -0.0169 | [-0.0347, +0.0041] | 0.1309 | 0.3359 | -0.51 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.1542 | 0.1392 | +0.0150 | [+0.0024, +0.0293] | 0.1055 | 0.3359 | +0.65 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.1542 | 0.1242 | +0.0300 | [+0.0144, +0.0478] | 0.0039 | 0.0352 | +1.05 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.1542 | 0.1467 | +0.0075 | [-0.0045, +0.0218] | 0.4316 | 0.4316 | +0.34 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.1542 | 0.1824 | -0.0282 | [-0.0536, +0.0015] | 0.0645 | 0.3223 | -0.61 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.1542 | 0.1264 | +0.0278 | [+0.0111, +0.0473] | 0.0098 | 0.0781 | +0.90 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 10 | 0.1542 | 0.1256 | +0.0287 | [+0.0126, +0.0471] | 0.0098 | 0.0781 | +0.97 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.1392 | 0.1242 | +0.0149 | [+0.0053, +0.0255] | 0.0273 | 0.1641 | +0.84 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.1711 | 0.1242 | +0.0469 | [+0.0361, +0.0603] | 0.0020 | 0.0195 | +2.23 | seeds 0,1,2,3,4,5,6,7,8,9 |

### copy -- gradient cosine (Holm family size 8)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.999 | 0.995 | +0.004 | [-0.000, +0.009] | 0.0371 | 0.0742 | +0.46 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.999 | 0.997 | +0.001 | [-0.002, +0.006] | 0.8457 | 0.8457 | +0.22 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.999 | 0.820 | +0.179 | [+0.175, +0.182] | 0.0020 | 0.0156 | +28.78 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.999 | 0.606 | +0.393 | [+0.336, +0.456] | 0.0020 | 0.0156 | +3.87 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.999 | 0.034 | +0.964 | [+0.960, +0.969] | 0.0020 | 0.0156 | +126.24 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.999 | 0.048 | +0.951 | [+0.937, +0.964] | 0.0020 | 0.0156 | +39.92 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.997 | 0.820 | +0.177 | [+0.172, +0.182] | 0.0020 | 0.0156 | +20.09 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.995 | 0.820 | +0.175 | [+0.169, +0.181] | 0.0020 | 0.0156 | +17.26 | seeds 0,1,2,3,4,5,6,7,8,9 |

## anbn (task metric: accuracy, higher better)

### anbn -- task metric (Holm family size 10)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 10 | 0.9237 | 0.9347 | -0.0110 | [-0.0159, -0.0058] | 0.0098 | 0.0293 | -1.26 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.9237 | 0.9274 | -0.0037 | [-0.0122, +0.0035] | 0.6953 | 0.6953 | -0.28 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.9237 | 0.9156 | +0.0081 | [-0.0021, +0.0174] | 0.1934 | 0.3867 | +0.49 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.9237 | 0.9018 | +0.0219 | [+0.0171, +0.0266] | 0.0020 | 0.0195 | +2.70 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.9237 | 0.8961 | +0.0276 | [+0.0237, +0.0314] | 0.0020 | 0.0195 | +4.22 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.9237 | 0.9093 | +0.0143 | [+0.0094, +0.0188] | 0.0039 | 0.0195 | +1.78 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.9237 | 0.8855 | +0.0382 | [+0.0329, +0.0437] | 0.0020 | 0.0195 | +4.06 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 10 | 0.9237 | 0.9050 | +0.0187 | [+0.0139, +0.0234] | 0.0020 | 0.0195 | +2.34 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.9156 | 0.9018 | +0.0138 | [+0.0071, +0.0207] | 0.0059 | 0.0234 | +1.18 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.9274 | 0.9018 | +0.0256 | [+0.0196, +0.0315] | 0.0020 | 0.0195 | +2.50 | seeds 0,1,2,3,4,5,6,7,8,9 |

### anbn -- gradient cosine (Holm family size 8)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.983 | 0.993 | -0.010 | [-0.015, -0.006] | 0.0039 | 0.0156 | -1.35 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.983 | 0.972 | +0.011 | [+0.005, +0.018] | 0.0137 | 0.0156 | +0.95 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.983 | 0.470 | +0.512 | [+0.439, +0.583] | 0.0020 | 0.0156 | +4.17 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.983 | 0.441 | +0.542 | [+0.508, +0.575] | 0.0020 | 0.0156 | +9.51 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.983 | 0.516 | +0.467 | [+0.432, +0.497] | 0.0020 | 0.0156 | +8.26 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.983 | 0.091 | +0.891 | [+0.881, +0.903] | 0.0020 | 0.0156 | +48.40 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.972 | 0.470 | +0.501 | [+0.430, +0.570] | 0.0020 | 0.0156 | +4.20 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.993 | 0.470 | +0.523 | [+0.451, +0.592] | 0.0020 | 0.0156 | +4.33 | seeds 0,1,2,3,4,5,6,7,8,9 |

## rotation (task metric: MSE, lower better)

### rotation -- task metric (Holm family size 10)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 10 | 0.1129 | 0.1072 | -0.0057 | [-0.0110, -0.0005] | 0.1055 | 0.4219 | -0.64 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.1129 | 0.1039 | -0.0091 | [-0.0168, -0.0012] | 0.1055 | 0.4219 | -0.68 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.1129 | 0.1098 | -0.0031 | [-0.0070, +0.0010] | 0.1934 | 0.4219 | -0.45 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.1129 | 0.0509 | -0.0620 | [-0.0651, -0.0582] | 0.0020 | 0.0195 | -10.56 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.1129 | 0.1137 | +0.0008 | [-0.0042, +0.0054] | 0.8457 | 0.8457 | +0.10 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.1129 | 0.0977 | -0.0152 | [-0.0226, -0.0067] | 0.0137 | 0.0820 | -1.11 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.1129 | 0.1186 | +0.0057 | [+0.0022, +0.0091] | 0.0137 | 0.0820 | +0.98 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 10 | 0.1129 | 0.0332 | -0.0797 | [-0.0852, -0.0741] | 0.0020 | 0.0195 | -8.37 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.1098 | 0.0509 | -0.0589 | [-0.0634, -0.0537] | 0.0020 | 0.0195 | -7.01 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.1039 | 0.0509 | -0.0530 | [-0.0599, -0.0462] | 0.0020 | 0.0195 | -4.52 | seeds 0,1,2,3,4,5,6,7,8,9 |

### rotation -- gradient cosine (Holm family size 8)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.841 | 0.930 | -0.089 | [-0.137, -0.044] | 0.0098 | 0.0156 | -1.12 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.841 | 0.679 | +0.163 | [+0.111, +0.215] | 0.0020 | 0.0156 | +1.84 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.841 | 0.411 | +0.430 | [+0.396, +0.462] | 0.0020 | 0.0156 | +7.56 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.841 | 0.379 | +0.462 | [+0.390, +0.528] | 0.0020 | 0.0156 | +3.94 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.841 | 0.144 | +0.697 | [+0.667, +0.727] | 0.0020 | 0.0156 | +13.62 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.841 | 0.034 | +0.807 | [+0.770, +0.845] | 0.0020 | 0.0156 | +12.51 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.679 | 0.411 | +0.268 | [+0.234, +0.299] | 0.0020 | 0.0156 | +4.84 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.930 | 0.411 | +0.519 | [+0.493, +0.542] | 0.0020 | 0.0156 | +12.32 | seeds 0,1,2,3,4,5,6,7,8,9 |

## henon (task metric: NMSE, lower better)

### henon -- task metric (Holm family size 10)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 10 | 0.0051 | 0.0045 | -0.0006 | [-0.0010, -0.0003] | 0.0273 | 0.1914 | -0.99 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.0051 | 0.0050 | -0.0001 | [-0.0004, +0.0002] | 0.6250 | 0.6445 | -0.17 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.0051 | 0.0052 | +0.0002 | [-0.0001, +0.0004] | 0.3223 | 0.6445 | +0.34 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.0051 | 0.0046 | -0.0005 | [-0.0008, -0.0001] | 0.0273 | 0.1914 | -0.83 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.0051 | 0.0205 | +0.0154 | [+0.0057, +0.0331] | 0.0020 | 0.0195 | +0.56 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.0051 | 0.0565 | +0.0514 | [+0.0407, +0.0643] | 0.0020 | 0.0195 | +2.50 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.0051 | 0.6558 | +0.6508 | [+0.6320, +0.6712] | 0.0020 | 0.0195 | +19.43 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 10 | 0.0051 | 0.0045 | -0.0006 | [-0.0012, -0.0001] | 0.1055 | 0.3359 | -0.63 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.0052 | 0.0046 | -0.0006 | [-0.0010, -0.0002] | 0.0371 | 0.1914 | -0.80 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.0050 | 0.0046 | -0.0004 | [-0.0008, +0.0001] | 0.0840 | 0.3359 | -0.48 | seeds 0,1,2,3,4,5,6,7,8,9 |

### henon -- gradient cosine (Holm family size 8)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.977 | 0.985 | -0.009 | [-0.010, -0.007] | 0.0020 | 0.0156 | -2.71 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.977 | 0.976 | +0.001 | [-0.002, +0.003] | 0.6250 | 0.6250 | +0.16 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.977 | 0.781 | +0.196 | [+0.172, +0.221] | 0.0020 | 0.0156 | +4.73 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.977 | 0.629 | +0.348 | [+0.300, +0.384] | 0.0020 | 0.0156 | +4.78 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.977 | 0.392 | +0.585 | [+0.552, +0.619] | 0.0020 | 0.0156 | +10.27 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.977 | 0.043 | +0.934 | [+0.931, +0.937] | 0.0020 | 0.0156 | +190.80 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.976 | 0.781 | +0.196 | [+0.173, +0.219] | 0.0020 | 0.0156 | +4.93 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.985 | 0.781 | +0.205 | [+0.181, +0.229] | 0.0020 | 0.0156 | +5.01 | seeds 0,1,2,3,4,5,6,7,8,9 |

## mackeyglass (task metric: NMSE, lower better)

### mackeyglass -- task metric (Holm family size 10)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 10 | 0.0025 | 0.0024 | -0.0001 | [-0.0003, +0.0000] | 0.1934 | 0.3867 | -0.50 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.0025 | 0.0023 | -0.0002 | [-0.0004, +0.0001] | 0.2324 | 0.3867 | -0.38 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.0025 | 0.0031 | +0.0006 | [+0.0002, +0.0010] | 0.0195 | 0.0977 | +0.95 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.0025 | 0.0035 | +0.0010 | [+0.0008, +0.0013] | 0.0020 | 0.0195 | +2.37 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.0025 | 0.0043 | +0.0018 | [+0.0015, +0.0021] | 0.0020 | 0.0195 | +3.69 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.0025 | 0.0071 | +0.0046 | [+0.0035, +0.0060] | 0.0020 | 0.0195 | +2.16 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.0025 | 0.0054 | +0.0029 | [+0.0023, +0.0034] | 0.0020 | 0.0195 | +3.12 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 10 | 0.0025 | 0.0127 | +0.0102 | [-0.0007, +0.0318] | 0.1055 | 0.3164 | +0.30 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.0031 | 0.0035 | +0.0004 | [+0.0000, +0.0008] | 0.0645 | 0.2578 | +0.65 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.0023 | 0.0035 | +0.0012 | [+0.0008, +0.0016] | 0.0020 | 0.0195 | +1.91 | seeds 0,1,2,3,4,5,6,7,8,9 |

### mackeyglass -- gradient cosine (Holm family size 8)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.846 | 0.903 | -0.058 | [-0.070, -0.046] | 0.0020 | 0.0156 | -2.71 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.846 | 0.844 | +0.002 | [-0.018, +0.024] | 0.9219 | 0.9219 | +0.05 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.846 | 0.353 | +0.493 | [+0.440, +0.543] | 0.0020 | 0.0156 | +5.53 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.846 | 0.378 | +0.468 | [+0.439, +0.499] | 0.0020 | 0.0156 | +9.10 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.846 | 0.162 | +0.684 | [+0.649, +0.715] | 0.0020 | 0.0156 | +12.12 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.846 | 0.043 | +0.802 | [+0.789, +0.816] | 0.0020 | 0.0156 | +34.76 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.844 | 0.353 | +0.491 | [+0.430, +0.553] | 0.0020 | 0.0156 | +4.65 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.903 | 0.353 | +0.551 | [+0.496, +0.606] | 0.0020 | 0.0156 | +5.88 | seeds 0,1,2,3,4,5,6,7,8,9 |

## lorenz (task metric: NMSE, lower better)

### lorenz -- task metric (Holm family size 10)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 10 | 0.0014 | 0.0013 | -0.0001 | [-0.0003, +0.0000] | 0.3223 | 0.9668 | -0.45 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.0014 | 0.0012 | -0.0001 | [-0.0003, -0.0000] | 0.0645 | 0.2578 | -0.61 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.0014 | 0.0014 | -0.0000 | [-0.0003, +0.0002] | 1.0000 | 1.0000 | -0.08 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.0014 | 0.0025 | +0.0011 | [+0.0007, +0.0015] | 0.0020 | 0.0195 | +1.60 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.0014 | 0.0021 | +0.0007 | [+0.0005, +0.0010] | 0.0039 | 0.0195 | +1.76 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.0014 | 0.0042 | +0.0028 | [+0.0021, +0.0034] | 0.0020 | 0.0195 | +2.61 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.0014 | 0.0043 | +0.0029 | [+0.0026, +0.0033] | 0.0020 | 0.0195 | +5.36 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 10 | 0.0014 | 0.0211 | +0.0197 | [+0.0001, +0.0418] | 0.5566 | 1.0000 | +0.58 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.0014 | 0.0025 | +0.0011 | [+0.0007, +0.0016] | 0.0020 | 0.0195 | +1.51 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.0012 | 0.0025 | +0.0012 | [+0.0008, +0.0017] | 0.0020 | 0.0195 | +1.60 | seeds 0,1,2,3,4,5,6,7,8,9 |

### lorenz -- gradient cosine (Holm family size 8)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.839 | 0.901 | -0.063 | [-0.081, -0.043] | 0.0020 | 0.0156 | -1.95 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.839 | 0.839 | -0.000 | [-0.026, +0.026] | 0.9219 | 0.9219 | -0.00 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.839 | 0.333 | +0.506 | [+0.475, +0.536] | 0.0020 | 0.0156 | +9.83 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.839 | 0.459 | +0.380 | [+0.350, +0.411] | 0.0020 | 0.0156 | +7.29 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.839 | 0.200 | +0.639 | [+0.609, +0.668] | 0.0020 | 0.0156 | +12.84 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.839 | 0.054 | +0.785 | [+0.767, +0.803] | 0.0020 | 0.0156 | +25.25 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.839 | 0.333 | +0.506 | [+0.472, +0.543] | 0.0020 | 0.0156 | +8.35 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.901 | 0.333 | +0.569 | [+0.546, +0.596] | 0.0020 | 0.0156 | +13.56 | seeds 0,1,2,3,4,5,6,7,8,9 |

## sunspot (task metric: NMSE, lower better)

### sunspot -- task metric (Holm family size 10)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 10 | 0.1335 | 0.1335 | -0.0000 | [-0.0001, +0.0001] | 0.7695 | 1.0000 | -0.03 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.1335 | 0.1334 | -0.0002 | [-0.0003, -0.0000] | 0.0645 | 0.1934 | -0.67 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.1335 | 0.1335 | -0.0000 | [-0.0001, +0.0000] | 0.8457 | 1.0000 | -0.09 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.1335 | 0.1398 | +0.0062 | [+0.0060, +0.0064] | 0.0020 | 0.0195 | +18.83 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.1335 | 0.1365 | +0.0029 | [+0.0023, +0.0037] | 0.0020 | 0.0195 | +2.33 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.1335 | 0.1423 | +0.0088 | [+0.0081, +0.0097] | 0.0020 | 0.0195 | +6.13 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.1335 | 0.1388 | +0.0053 | [+0.0045, +0.0059] | 0.0020 | 0.0195 | +4.54 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 10 | 0.1335 | 0.1342 | +0.0007 | [+0.0003, +0.0012] | 0.0098 | 0.0391 | +0.90 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.1335 | 0.1398 | +0.0062 | [+0.0060, +0.0064] | 0.0020 | 0.0195 | +18.32 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.1334 | 0.1398 | +0.0064 | [+0.0062, +0.0066] | 0.0020 | 0.0195 | +21.39 | seeds 0,1,2,3,4,5,6,7,8,9 |

### sunspot -- gradient cosine (Holm family size 8)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.899 | 0.931 | -0.033 | [-0.038, -0.028] | 0.0020 | 0.0156 | -3.57 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.899 | 0.898 | +0.001 | [-0.001, +0.003] | 0.4316 | 0.4316 | +0.34 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.899 | 0.622 | +0.276 | [+0.252, +0.300] | 0.0020 | 0.0156 | +6.68 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.899 | 0.442 | +0.457 | [+0.438, +0.476] | 0.0020 | 0.0156 | +14.10 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.899 | 0.337 | +0.562 | [+0.523, +0.601] | 0.0020 | 0.0156 | +8.43 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.899 | 0.063 | +0.836 | [+0.826, +0.845] | 0.0020 | 0.0156 | +50.42 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.898 | 0.622 | +0.275 | [+0.251, +0.299] | 0.0020 | 0.0156 | +6.72 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.931 | 0.622 | +0.309 | [+0.286, +0.330] | 0.0020 | 0.0156 | +8.15 | seeds 0,1,2,3,4,5,6,7,8,9 |

## laser (task metric: NMSE, lower better)

### laser -- task metric (Holm family size 10)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 10 | 0.0197 | 0.0160 | -0.0037 | [-0.0055, -0.0024] | 0.0020 | 0.0195 | -1.38 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.0197 | 0.0179 | -0.0018 | [-0.0045, +0.0002] | 0.3223 | 0.9668 | -0.43 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.0197 | 0.0213 | +0.0016 | [+0.0001, +0.0038] | 0.1309 | 0.5273 | +0.51 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.0197 | 0.0200 | +0.0003 | [-0.0015, +0.0020] | 0.6953 | 0.9844 | +0.11 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.0197 | 0.0560 | +0.0363 | [+0.0218, +0.0577] | 0.0020 | 0.0195 | +1.13 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.0197 | 0.2001 | +0.1804 | [+0.1266, +0.2422] | 0.0020 | 0.0195 | +1.81 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.0197 | 0.1509 | +0.1312 | [+0.1231, +0.1399] | 0.0020 | 0.0195 | +9.05 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 10 | 0.0197 | 0.0167 | -0.0030 | [-0.0065, -0.0004] | 0.0840 | 0.5039 | -0.56 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.0213 | 0.0200 | -0.0013 | [-0.0042, +0.0009] | 0.4922 | 0.9844 | -0.30 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.0179 | 0.0200 | +0.0021 | [+0.0000, +0.0041] | 0.1055 | 0.5273 | +0.62 | seeds 0,1,2,3,4,5,6,7,8,9 |

### laser -- gradient cosine (Holm family size 8)

| comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p | d_z | note |
|---|---|---|---|---|---|---|---|---|---|
| SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| SK-RTRL r=16 vs SK-RTRL r=32 | 10 | 0.932 | 0.960 | -0.028 | [-0.038, -0.019] | 0.0020 | 0.0156 | -1.81 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.932 | 0.931 | +0.001 | [-0.011, +0.013] | 0.8457 | 0.8457 | +0.03 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs SnAp-1 | 10 | 0.932 | 0.378 | +0.554 | [+0.537, +0.573] | 0.0020 | 0.0156 | +18.65 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs KF-RTRL | 10 | 0.932 | 0.474 | +0.457 | [+0.427, +0.486] | 0.0020 | 0.0156 | +9.00 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs RFLO | 10 | 0.932 | 0.046 | +0.885 | [+0.859, +0.911] | 0.0020 | 0.0156 | +19.80 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs UORO | 10 | 0.932 | 0.033 | +0.899 | [+0.891, +0.906] | 0.0020 | 0.0156 | +70.72 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| SK-RTRL r=4 vs SnAp-1 | 10 | 0.931 | 0.378 | +0.553 | [+0.535, +0.575] | 0.0020 | 0.0156 | +16.42 | seeds 0,1,2,3,4,5,6,7,8,9 |
| SK-RTRL r=32 vs SnAp-1 | 10 | 0.960 | 0.378 | +0.582 | [+0.568, +0.599] | 0.0020 | 0.0156 | +21.95 | seeds 0,1,2,3,4,5,6,7,8,9 |


---

# Full matrix: three ranks x six opponents

Every SK-RTRL rank in SK-RTRL r=4, SK-RTRL r=16, SK-RTRL r=32 against every one of exact RTRL, SnAp-1, KF-RTRL, RFLO, UORO, TBPTT, on all 9 tasks and both metrics. This is the comparison set the *body* of the paper quotes its records over; the pre-registered set of the previous section (r=16 vs all, r=4 / r=32 vs SnAp-1 only) cannot produce them.

| count | value | definition |
|---|---|---|
| planned, full matrix | 324 | 3 ranks x 6 opponents x 9 tasks x 2 metrics |
| structurally applicable, full matrix | **270** | the above minus the gradient cosine for exact RTRL and TBPTT (3 x 2 x 9 = 54 undefined cells) |
| computed (>= 3 paired seeds) | 270 | |
| undefined by the protocol | 54 | grad_cos for exact / TBPTT |
| too few paired seeds | 0 | |
| planned, pre-registered set | 180 | 10 per (task, metric) x 9 tasks x 2 metrics |
| applicable, pre-registered set | 162 | |

**Holm families.** Two definitions are reported for every comparison and both are fixed before the data are seen; `holm_p` in D4_STATS.json carries the one `--holm-family` selected (this run: **task-metric**).

| family definition | size on the task metric | size on the cosine | smallest attainable holm_p at n=10 (Wilcoxon floor p=0.001953) |
|---|---|---|---|
| per (task, metric) | 18 (3 ranks x 6 opponents) | 12 (3 ranks x 4 opponents) | 0.0352 on the metric, 0.0234 on the cosine |
| per (rank, metric) | 54 (6 opponents x 9 tasks) | 36 (4 opponents x 9 tasks) | 0.1055 on the metric, 0.0703 on the cosine |

With ten seeds the exact two-sided Wilcoxon floor is p = 2/2^10 = 0.001953, so in a (rank, metric) family NO comparison can reach holm_p < 0.05 however large the effect: the floor is 0.0703 on the cosine and 0.1055 on the task metric. The (task, metric) family keeps every floor below 0.05 (0.0234 and 0.0352). This is a reporting choice, not a data problem, and the manuscript must state which one it uses.

`delta` is signed so a positive value favours the first-named method; `Cliff delta` is the dominance of the first over the second across all 10 x 10 cross pairs of seed values (+1.00 = no seed of the second reaches any seed of the first).

## SK-RTRL r=4 -- full opponent sweep

### SK-RTRL r=4 -- task metric (Holm family: 18 per (task, metric), 54 per (rank, metric))

| task | comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p (task fam) | holm_p (rank fam) | d_z | Cliff delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adding | SK-RTRL r=4 vs exact RTRL | 10 | 0.2116 | 0.2067 | -0.0049 | [-0.0069, -0.0033] | 0.0020 | 0.0352 | 0.1055 | -1.56 | -0.28 |
| adding | SK-RTRL r=4 vs SnAp-1 | 10 | 0.2116 | 0.2131 | +0.0015 | [-0.0054, +0.0116] | 0.7695 | 1.0000 | 1.0000 | +0.10 | -0.06 |
| adding | SK-RTRL r=4 vs KF-RTRL | 10 | 0.2116 | 0.2172 | +0.0056 | [+0.0010, +0.0103] | 0.0488 | 0.2441 | 0.7812 | +0.70 | +0.24 |
| adding | SK-RTRL r=4 vs RFLO | 10 | 0.2116 | 0.2268 | +0.0152 | [+0.0099, +0.0207] | 0.0020 | 0.0352 | 0.1055 | +1.64 | +0.60 |
| adding | SK-RTRL r=4 vs UORO | 10 | 0.2116 | 0.2613 | +0.0497 | [+0.0173, +0.1065] | 0.0020 | 0.0352 | 0.1055 | +0.57 | +0.94 |
| adding | SK-RTRL r=4 vs TBPTT | 10 | 0.2116 | 0.2067 | -0.0049 | [-0.0069, -0.0033] | 0.0020 | 0.0352 | 0.1055 | -1.56 | -0.28 |
| copy | SK-RTRL r=4 vs exact RTRL | 10 | 0.1392 | 0.1370 | +0.0022 | [-0.0090, +0.0127] | 0.7695 | 1.0000 | 1.0000 | +0.12 | +0.12 |
| copy | SK-RTRL r=4 vs SnAp-1 | 10 | 0.1392 | 0.1242 | +0.0149 | [+0.0053, +0.0255] | 0.0273 | 0.2461 | 0.5195 | +0.84 | +0.74 |
| copy | SK-RTRL r=4 vs KF-RTRL | 10 | 0.1392 | 0.1467 | -0.0075 | [-0.0141, -0.0012] | 0.0273 | 0.2461 | 0.5195 | -0.68 | -0.38 |
| copy | SK-RTRL r=4 vs RFLO | 10 | 0.1392 | 0.1824 | -0.0432 | [-0.0605, -0.0232] | 0.0039 | 0.0586 | 0.1133 | -1.37 | -0.86 |
| copy | SK-RTRL r=4 vs UORO | 10 | 0.1392 | 0.1264 | +0.0128 | [+0.0032, +0.0241] | 0.0645 | 0.4512 | 0.9023 | +0.71 | +0.52 |
| copy | SK-RTRL r=4 vs TBPTT | 10 | 0.1392 | 0.1256 | +0.0136 | [+0.0034, +0.0253] | 0.1055 | 0.4512 | 1.0000 | +0.71 | +0.64 |
| anbn | SK-RTRL r=4 vs exact RTRL | 10 | 0.9156 | 0.9347 | -0.0191 | [-0.0277, -0.0104] | 0.0059 | 0.0352 | 0.1406 | -1.27 | -0.82 |
| anbn | SK-RTRL r=4 vs SnAp-1 | 10 | 0.9156 | 0.9018 | +0.0138 | [+0.0071, +0.0207] | 0.0059 | 0.0352 | 0.1406 | +1.18 | +0.72 |
| anbn | SK-RTRL r=4 vs KF-RTRL | 10 | 0.9156 | 0.8961 | +0.0195 | [+0.0110, +0.0286] | 0.0020 | 0.0352 | 0.1055 | +1.30 | +0.88 |
| anbn | SK-RTRL r=4 vs RFLO | 10 | 0.9156 | 0.9093 | +0.0062 | [-0.0015, +0.0141] | 0.1602 | 0.3203 | 1.0000 | +0.47 | +0.36 |
| anbn | SK-RTRL r=4 vs UORO | 10 | 0.9156 | 0.8855 | +0.0301 | [+0.0224, +0.0383] | 0.0020 | 0.0352 | 0.1055 | +2.25 | +1.00 |
| anbn | SK-RTRL r=4 vs TBPTT | 10 | 0.9156 | 0.9050 | +0.0106 | [+0.0002, +0.0216] | 0.1602 | 0.3203 | 1.0000 | +0.57 | +0.48 |
| rotation | SK-RTRL r=4 vs exact RTRL | 10 | 0.1098 | 0.1072 | -0.0026 | [-0.0075, +0.0027] | 0.3750 | 1.0000 | 1.0000 | -0.30 | -0.26 |
| rotation | SK-RTRL r=4 vs SnAp-1 | 10 | 0.1098 | 0.0509 | -0.0589 | [-0.0634, -0.0537] | 0.0020 | 0.0352 | 0.1055 | -7.01 | -1.00 |
| rotation | SK-RTRL r=4 vs KF-RTRL | 10 | 0.1098 | 0.1137 | +0.0039 | [-0.0019, +0.0093] | 0.2754 | 1.0000 | 1.0000 | +0.41 | +0.28 |
| rotation | SK-RTRL r=4 vs RFLO | 10 | 0.1098 | 0.0977 | -0.0121 | [-0.0193, -0.0033] | 0.0488 | 0.3418 | 0.7812 | -0.90 | -0.58 |
| rotation | SK-RTRL r=4 vs UORO | 10 | 0.1098 | 0.1186 | +0.0088 | [+0.0054, +0.0120] | 0.0039 | 0.0430 | 0.1133 | +1.59 | +0.84 |
| rotation | SK-RTRL r=4 vs TBPTT | 10 | 0.1098 | 0.0332 | -0.0766 | [-0.0824, -0.0713] | 0.0020 | 0.0352 | 0.1055 | -8.16 | -1.00 |
| henon | SK-RTRL r=4 vs exact RTRL | 10 | 0.0052 | 0.0045 | -0.0008 | [-0.0011, -0.0004] | 0.0039 | 0.0352 | 0.1133 | -1.43 | -0.82 |
| henon | SK-RTRL r=4 vs SnAp-1 | 10 | 0.0052 | 0.0046 | -0.0006 | [-0.0010, -0.0002] | 0.0371 | 0.1914 | 0.6309 | -0.80 | -0.62 |
| henon | SK-RTRL r=4 vs KF-RTRL | 10 | 0.0052 | 0.0205 | +0.0152 | [+0.0055, +0.0331] | 0.0020 | 0.0352 | 0.1055 | +0.55 | +1.00 |
| henon | SK-RTRL r=4 vs RFLO | 10 | 0.0052 | 0.0565 | +0.0512 | [+0.0407, +0.0642] | 0.0020 | 0.0352 | 0.1055 | +2.50 | +1.00 |
| henon | SK-RTRL r=4 vs UORO | 10 | 0.0052 | 0.6558 | +0.6506 | [+0.6317, +0.6713] | 0.0020 | 0.0352 | 0.1055 | +19.26 | +1.00 |
| henon | SK-RTRL r=4 vs TBPTT | 10 | 0.0052 | 0.0045 | -0.0007 | [-0.0012, -0.0003] | 0.0098 | 0.0781 | 0.2148 | -1.04 | -0.62 |
| mackeyglass | SK-RTRL r=4 vs exact RTRL | 10 | 0.0031 | 0.0024 | -0.0007 | [-0.0010, -0.0005] | 0.0020 | 0.0352 | 0.1055 | -1.50 | -0.86 |
| mackeyglass | SK-RTRL r=4 vs SnAp-1 | 10 | 0.0031 | 0.0035 | +0.0004 | [+0.0000, +0.0008] | 0.0645 | 0.3867 | 0.9023 | +0.65 | +0.50 |
| mackeyglass | SK-RTRL r=4 vs KF-RTRL | 10 | 0.0031 | 0.0043 | +0.0012 | [+0.0008, +0.0015] | 0.0020 | 0.0352 | 0.1055 | +1.97 | +0.84 |
| mackeyglass | SK-RTRL r=4 vs RFLO | 10 | 0.0031 | 0.0071 | +0.0040 | [+0.0030, +0.0052] | 0.0020 | 0.0352 | 0.1055 | +2.19 | +1.00 |
| mackeyglass | SK-RTRL r=4 vs UORO | 10 | 0.0031 | 0.0054 | +0.0023 | [+0.0016, +0.0028] | 0.0039 | 0.0352 | 0.1133 | +2.16 | +0.98 |
| mackeyglass | SK-RTRL r=4 vs TBPTT | 10 | 0.0031 | 0.0127 | +0.0096 | [-0.0014, +0.0312] | 0.0840 | 0.4199 | 1.0000 | +0.28 | -0.78 |
| lorenz | SK-RTRL r=4 vs exact RTRL | 10 | 0.0014 | 0.0013 | -0.0001 | [-0.0003, +0.0000] | 0.2324 | 1.0000 | 1.0000 | -0.37 | -0.46 |
| lorenz | SK-RTRL r=4 vs SnAp-1 | 10 | 0.0014 | 0.0025 | +0.0011 | [+0.0007, +0.0016] | 0.0020 | 0.0352 | 0.1055 | +1.51 | +0.96 |
| lorenz | SK-RTRL r=4 vs KF-RTRL | 10 | 0.0014 | 0.0021 | +0.0008 | [+0.0005, +0.0009] | 0.0039 | 0.0352 | 0.1133 | +2.10 | +0.98 |
| lorenz | SK-RTRL r=4 vs RFLO | 10 | 0.0014 | 0.0042 | +0.0029 | [+0.0023, +0.0034] | 0.0020 | 0.0352 | 0.1055 | +3.15 | +1.00 |
| lorenz | SK-RTRL r=4 vs UORO | 10 | 0.0014 | 0.0043 | +0.0030 | [+0.0026, +0.0033] | 0.0020 | 0.0352 | 0.1055 | +5.11 | +1.00 |
| lorenz | SK-RTRL r=4 vs TBPTT | 10 | 0.0014 | 0.0211 | +0.0197 | [+0.0001, +0.0417] | 0.5566 | 1.0000 | 1.0000 | +0.58 | -0.10 |
| sunspot | SK-RTRL r=4 vs exact RTRL | 10 | 0.1335 | 0.1335 | +0.0000 | [-0.0001, +0.0001] | 0.9219 | 1.0000 | 1.0000 | +0.01 | +0.02 |
| sunspot | SK-RTRL r=4 vs SnAp-1 | 10 | 0.1335 | 0.1398 | +0.0062 | [+0.0060, +0.0064] | 0.0020 | 0.0352 | 0.1055 | +18.32 | +0.96 |
| sunspot | SK-RTRL r=4 vs KF-RTRL | 10 | 0.1335 | 0.1365 | +0.0029 | [+0.0023, +0.0037] | 0.0020 | 0.0352 | 0.1055 | +2.40 | +0.70 |
| sunspot | SK-RTRL r=4 vs RFLO | 10 | 0.1335 | 0.1423 | +0.0088 | [+0.0081, +0.0097] | 0.0020 | 0.0352 | 0.1055 | +6.30 | +1.00 |
| sunspot | SK-RTRL r=4 vs UORO | 10 | 0.1335 | 0.1388 | +0.0053 | [+0.0046, +0.0059] | 0.0020 | 0.0352 | 0.1055 | +4.70 | +0.92 |
| sunspot | SK-RTRL r=4 vs TBPTT | 10 | 0.1335 | 0.1342 | +0.0007 | [+0.0003, +0.0012] | 0.0098 | 0.0488 | 0.2148 | +0.93 | +0.32 |
| laser | SK-RTRL r=4 vs exact RTRL | 10 | 0.0213 | 0.0160 | -0.0053 | [-0.0074, -0.0036] | 0.0020 | 0.0352 | 0.1055 | -1.66 | -0.88 |
| laser | SK-RTRL r=4 vs SnAp-1 | 10 | 0.0213 | 0.0200 | -0.0013 | [-0.0042, +0.0009] | 0.4922 | 0.9844 | 1.0000 | -0.30 | -0.12 |
| laser | SK-RTRL r=4 vs KF-RTRL | 10 | 0.0213 | 0.0560 | +0.0347 | [+0.0214, +0.0555] | 0.0020 | 0.0352 | 0.1055 | +1.10 | +1.00 |
| laser | SK-RTRL r=4 vs RFLO | 10 | 0.0213 | 0.2001 | +0.1788 | [+0.1245, +0.2418] | 0.0020 | 0.0352 | 0.1055 | +1.78 | +1.00 |
| laser | SK-RTRL r=4 vs UORO | 10 | 0.0213 | 0.1509 | +0.1296 | [+0.1214, +0.1388] | 0.0020 | 0.0352 | 0.1055 | +8.71 | +1.00 |
| laser | SK-RTRL r=4 vs TBPTT | 10 | 0.0213 | 0.0167 | -0.0046 | [-0.0078, -0.0016] | 0.0195 | 0.1172 | 0.3906 | -0.86 | -0.72 |

### SK-RTRL r=4 -- gradient cosine (Holm family: 12 per (task, metric), 36 per (rank, metric))

| task | comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p (task fam) | holm_p (rank fam) | d_z | Cliff delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adding | SK-RTRL r=4 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| adding | SK-RTRL r=4 vs SnAp-1 | 10 | 0.844 | 0.524 | +0.319 | [+0.266, +0.379] | 0.0020 | 0.0234 | 0.0703 | +3.28 | +1.00 |
| adding | SK-RTRL r=4 vs KF-RTRL | 10 | 0.844 | 0.560 | +0.284 | [+0.236, +0.329] | 0.0020 | 0.0234 | 0.0703 | +3.54 | +1.00 |
| adding | SK-RTRL r=4 vs RFLO | 10 | 0.844 | 0.531 | +0.313 | [+0.263, +0.372] | 0.0020 | 0.0234 | 0.0703 | +3.30 | +1.00 |
| adding | SK-RTRL r=4 vs UORO | 10 | 0.844 | 0.119 | +0.725 | [+0.692, +0.759] | 0.0020 | 0.0234 | 0.0703 | +12.85 | +1.00 |
| adding | SK-RTRL r=4 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| copy | SK-RTRL r=4 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| copy | SK-RTRL r=4 vs SnAp-1 | 10 | 0.997 | 0.820 | +0.177 | [+0.172, +0.182] | 0.0020 | 0.0234 | 0.0703 | +20.09 | +1.00 |
| copy | SK-RTRL r=4 vs KF-RTRL | 10 | 0.997 | 0.606 | +0.392 | [+0.337, +0.453] | 0.0020 | 0.0234 | 0.0703 | +3.97 | +1.00 |
| copy | SK-RTRL r=4 vs RFLO | 10 | 0.997 | 0.034 | +0.963 | [+0.957, +0.969] | 0.0020 | 0.0234 | 0.0703 | +96.43 | +1.00 |
| copy | SK-RTRL r=4 vs UORO | 10 | 0.997 | 0.048 | +0.950 | [+0.936, +0.963] | 0.0020 | 0.0234 | 0.0703 | +40.85 | +1.00 |
| copy | SK-RTRL r=4 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| anbn | SK-RTRL r=4 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| anbn | SK-RTRL r=4 vs SnAp-1 | 10 | 0.972 | 0.470 | +0.501 | [+0.430, +0.570] | 0.0020 | 0.0234 | 0.0703 | +4.20 | +1.00 |
| anbn | SK-RTRL r=4 vs KF-RTRL | 10 | 0.972 | 0.441 | +0.531 | [+0.498, +0.565] | 0.0020 | 0.0234 | 0.0703 | +9.34 | +1.00 |
| anbn | SK-RTRL r=4 vs RFLO | 10 | 0.972 | 0.516 | +0.456 | [+0.422, +0.486] | 0.0020 | 0.0234 | 0.0703 | +8.25 | +1.00 |
| anbn | SK-RTRL r=4 vs UORO | 10 | 0.972 | 0.091 | +0.880 | [+0.872, +0.890] | 0.0020 | 0.0234 | 0.0703 | +55.67 | +1.00 |
| anbn | SK-RTRL r=4 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| rotation | SK-RTRL r=4 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| rotation | SK-RTRL r=4 vs SnAp-1 | 10 | 0.679 | 0.411 | +0.268 | [+0.234, +0.299] | 0.0020 | 0.0234 | 0.0703 | +4.84 | +1.00 |
| rotation | SK-RTRL r=4 vs KF-RTRL | 10 | 0.679 | 0.379 | +0.300 | [+0.262, +0.339] | 0.0020 | 0.0234 | 0.0703 | +4.56 | +1.00 |
| rotation | SK-RTRL r=4 vs RFLO | 10 | 0.679 | 0.144 | +0.535 | [+0.503, +0.567] | 0.0020 | 0.0234 | 0.0703 | +9.74 | +1.00 |
| rotation | SK-RTRL r=4 vs UORO | 10 | 0.679 | 0.034 | +0.645 | [+0.611, +0.675] | 0.0020 | 0.0234 | 0.0703 | +11.78 | +1.00 |
| rotation | SK-RTRL r=4 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| henon | SK-RTRL r=4 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| henon | SK-RTRL r=4 vs SnAp-1 | 10 | 0.976 | 0.781 | +0.196 | [+0.173, +0.219] | 0.0020 | 0.0234 | 0.0703 | +4.93 | +1.00 |
| henon | SK-RTRL r=4 vs KF-RTRL | 10 | 0.976 | 0.629 | +0.347 | [+0.300, +0.384] | 0.0020 | 0.0234 | 0.0703 | +4.81 | +1.00 |
| henon | SK-RTRL r=4 vs RFLO | 10 | 0.976 | 0.392 | +0.584 | [+0.552, +0.617] | 0.0020 | 0.0234 | 0.0703 | +10.61 | +1.00 |
| henon | SK-RTRL r=4 vs UORO | 10 | 0.976 | 0.043 | +0.933 | [+0.930, +0.936] | 0.0020 | 0.0234 | 0.0703 | +180.36 | +1.00 |
| henon | SK-RTRL r=4 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| mackeyglass | SK-RTRL r=4 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| mackeyglass | SK-RTRL r=4 vs SnAp-1 | 10 | 0.844 | 0.353 | +0.491 | [+0.430, +0.553] | 0.0020 | 0.0234 | 0.0703 | +4.65 | +1.00 |
| mackeyglass | SK-RTRL r=4 vs KF-RTRL | 10 | 0.844 | 0.378 | +0.466 | [+0.435, +0.500] | 0.0020 | 0.0234 | 0.0703 | +8.20 | +1.00 |
| mackeyglass | SK-RTRL r=4 vs RFLO | 10 | 0.844 | 0.162 | +0.682 | [+0.643, +0.717] | 0.0020 | 0.0234 | 0.0703 | +10.81 | +1.00 |
| mackeyglass | SK-RTRL r=4 vs UORO | 10 | 0.844 | 0.043 | +0.801 | [+0.780, +0.818] | 0.0020 | 0.0234 | 0.0703 | +25.11 | +1.00 |
| mackeyglass | SK-RTRL r=4 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| lorenz | SK-RTRL r=4 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| lorenz | SK-RTRL r=4 vs SnAp-1 | 10 | 0.839 | 0.333 | +0.506 | [+0.472, +0.543] | 0.0020 | 0.0234 | 0.0703 | +8.35 | +1.00 |
| lorenz | SK-RTRL r=4 vs KF-RTRL | 10 | 0.839 | 0.459 | +0.380 | [+0.354, +0.403] | 0.0020 | 0.0234 | 0.0703 | +9.16 | +1.00 |
| lorenz | SK-RTRL r=4 vs RFLO | 10 | 0.839 | 0.200 | +0.639 | [+0.597, +0.676] | 0.0020 | 0.0234 | 0.0703 | +9.46 | +1.00 |
| lorenz | SK-RTRL r=4 vs UORO | 10 | 0.839 | 0.054 | +0.785 | [+0.763, +0.802] | 0.0020 | 0.0234 | 0.0703 | +23.22 | +1.00 |
| lorenz | SK-RTRL r=4 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| sunspot | SK-RTRL r=4 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| sunspot | SK-RTRL r=4 vs SnAp-1 | 10 | 0.898 | 0.622 | +0.275 | [+0.251, +0.299] | 0.0020 | 0.0234 | 0.0703 | +6.72 | +1.00 |
| sunspot | SK-RTRL r=4 vs KF-RTRL | 10 | 0.898 | 0.442 | +0.456 | [+0.437, +0.475] | 0.0020 | 0.0234 | 0.0703 | +14.19 | +1.00 |
| sunspot | SK-RTRL r=4 vs RFLO | 10 | 0.898 | 0.337 | +0.561 | [+0.522, +0.600] | 0.0020 | 0.0234 | 0.0703 | +8.37 | +1.00 |
| sunspot | SK-RTRL r=4 vs UORO | 10 | 0.898 | 0.063 | +0.835 | [+0.825, +0.844] | 0.0020 | 0.0234 | 0.0703 | +51.01 | +1.00 |
| sunspot | SK-RTRL r=4 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| laser | SK-RTRL r=4 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| laser | SK-RTRL r=4 vs SnAp-1 | 10 | 0.931 | 0.378 | +0.553 | [+0.535, +0.575] | 0.0020 | 0.0234 | 0.0703 | +16.42 | +1.00 |
| laser | SK-RTRL r=4 vs KF-RTRL | 10 | 0.931 | 0.474 | +0.456 | [+0.427, +0.487] | 0.0020 | 0.0234 | 0.0703 | +8.93 | +1.00 |
| laser | SK-RTRL r=4 vs RFLO | 10 | 0.931 | 0.046 | +0.884 | [+0.857, +0.916] | 0.0020 | 0.0234 | 0.0703 | +17.43 | +1.00 |
| laser | SK-RTRL r=4 vs UORO | 10 | 0.931 | 0.033 | +0.898 | [+0.893, +0.904] | 0.0020 | 0.0234 | 0.0703 | +100.97 | +1.00 |
| laser | SK-RTRL r=4 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |

## SK-RTRL r=16 -- full opponent sweep

### SK-RTRL r=16 -- task metric (Holm family: 18 per (task, metric), 54 per (rank, metric))

| task | comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p (task fam) | holm_p (rank fam) | d_z | Cliff delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adding | SK-RTRL r=16 vs exact RTRL | 10 | 0.2118 | 0.2067 | -0.0051 | [-0.0072, -0.0034] | 0.0020 | 0.0352 | 0.1055 | -1.55 | -0.28 |
| adding | SK-RTRL r=16 vs SnAp-1 | 10 | 0.2118 | 0.2131 | +0.0013 | [-0.0056, +0.0114] | 0.7695 | 1.0000 | 1.0000 | +0.09 | -0.06 |
| adding | SK-RTRL r=16 vs KF-RTRL | 10 | 0.2118 | 0.2172 | +0.0054 | [+0.0007, +0.0102] | 0.1055 | 0.4219 | 1.0000 | +0.66 | +0.24 |
| adding | SK-RTRL r=16 vs RFLO | 10 | 0.2118 | 0.2268 | +0.0150 | [+0.0097, +0.0205] | 0.0020 | 0.0352 | 0.1055 | +1.61 | +0.60 |
| adding | SK-RTRL r=16 vs UORO | 10 | 0.2118 | 0.2613 | +0.0495 | [+0.0170, +0.1063] | 0.0020 | 0.0352 | 0.1055 | +0.57 | +0.92 |
| adding | SK-RTRL r=16 vs TBPTT | 10 | 0.2118 | 0.2067 | -0.0051 | [-0.0072, -0.0034] | 0.0020 | 0.0352 | 0.1055 | -1.55 | -0.28 |
| copy | SK-RTRL r=16 vs exact RTRL | 10 | 0.1542 | 0.1370 | +0.0172 | [+0.0024, +0.0336] | 0.0840 | 0.4512 | 1.0000 | +0.65 | +0.42 |
| copy | SK-RTRL r=16 vs SnAp-1 | 10 | 0.1542 | 0.1242 | +0.0300 | [+0.0144, +0.0478] | 0.0039 | 0.0586 | 0.1055 | +1.05 | +0.82 |
| copy | SK-RTRL r=16 vs KF-RTRL | 10 | 0.1542 | 0.1467 | +0.0075 | [-0.0045, +0.0218] | 0.4316 | 1.0000 | 1.0000 | +0.34 | +0.06 |
| copy | SK-RTRL r=16 vs RFLO | 10 | 0.1542 | 0.1824 | -0.0282 | [-0.0536, +0.0015] | 0.0645 | 0.4512 | 0.9668 | -0.61 | -0.58 |
| copy | SK-RTRL r=16 vs UORO | 10 | 0.1542 | 0.1264 | +0.0278 | [+0.0111, +0.0473] | 0.0098 | 0.1172 | 0.2246 | +0.90 | +0.76 |
| copy | SK-RTRL r=16 vs TBPTT | 10 | 0.1542 | 0.1256 | +0.0287 | [+0.0126, +0.0471] | 0.0098 | 0.1172 | 0.2246 | +0.97 | +0.78 |
| anbn | SK-RTRL r=16 vs exact RTRL | 10 | 0.9237 | 0.9347 | -0.0110 | [-0.0159, -0.0058] | 0.0098 | 0.0391 | 0.2246 | -1.26 | -0.60 |
| anbn | SK-RTRL r=16 vs SnAp-1 | 10 | 0.9237 | 0.9018 | +0.0219 | [+0.0171, +0.0266] | 0.0020 | 0.0352 | 0.1055 | +2.70 | +1.00 |
| anbn | SK-RTRL r=16 vs KF-RTRL | 10 | 0.9237 | 0.8961 | +0.0276 | [+0.0237, +0.0314] | 0.0020 | 0.0352 | 0.1055 | +4.22 | +1.00 |
| anbn | SK-RTRL r=16 vs RFLO | 10 | 0.9237 | 0.9093 | +0.0143 | [+0.0094, +0.0188] | 0.0039 | 0.0352 | 0.1055 | +1.78 | +0.86 |
| anbn | SK-RTRL r=16 vs UORO | 10 | 0.9237 | 0.8855 | +0.0382 | [+0.0329, +0.0437] | 0.0020 | 0.0352 | 0.1055 | +4.06 | +1.00 |
| anbn | SK-RTRL r=16 vs TBPTT | 10 | 0.9237 | 0.9050 | +0.0187 | [+0.0139, +0.0234] | 0.0020 | 0.0352 | 0.1055 | +2.34 | +0.92 |
| rotation | SK-RTRL r=16 vs exact RTRL | 10 | 0.1129 | 0.1072 | -0.0057 | [-0.0110, -0.0005] | 0.1055 | 0.6328 | 1.0000 | -0.64 | -0.58 |
| rotation | SK-RTRL r=16 vs SnAp-1 | 10 | 0.1129 | 0.0509 | -0.0620 | [-0.0651, -0.0582] | 0.0020 | 0.0352 | 0.1055 | -10.56 | -1.00 |
| rotation | SK-RTRL r=16 vs KF-RTRL | 10 | 0.1129 | 0.1137 | +0.0008 | [-0.0042, +0.0054] | 0.8457 | 1.0000 | 1.0000 | +0.10 | +0.04 |
| rotation | SK-RTRL r=16 vs RFLO | 10 | 0.1129 | 0.0977 | -0.0152 | [-0.0226, -0.0067] | 0.0137 | 0.1367 | 0.2598 | -1.11 | -0.64 |
| rotation | SK-RTRL r=16 vs UORO | 10 | 0.1129 | 0.1186 | +0.0057 | [+0.0022, +0.0091] | 0.0137 | 0.1367 | 0.2598 | +0.98 | +0.70 |
| rotation | SK-RTRL r=16 vs TBPTT | 10 | 0.1129 | 0.0332 | -0.0797 | [-0.0852, -0.0741] | 0.0020 | 0.0352 | 0.1055 | -8.37 | -1.00 |
| henon | SK-RTRL r=16 vs exact RTRL | 10 | 0.0051 | 0.0045 | -0.0006 | [-0.0010, -0.0003] | 0.0273 | 0.1914 | 0.4648 | -0.99 | -0.70 |
| henon | SK-RTRL r=16 vs SnAp-1 | 10 | 0.0051 | 0.0046 | -0.0005 | [-0.0008, -0.0001] | 0.0273 | 0.1914 | 0.4648 | -0.83 | -0.54 |
| henon | SK-RTRL r=16 vs KF-RTRL | 10 | 0.0051 | 0.0205 | +0.0154 | [+0.0057, +0.0331] | 0.0020 | 0.0352 | 0.1055 | +0.56 | +1.00 |
| henon | SK-RTRL r=16 vs RFLO | 10 | 0.0051 | 0.0565 | +0.0514 | [+0.0407, +0.0643] | 0.0020 | 0.0352 | 0.1055 | +2.50 | +1.00 |
| henon | SK-RTRL r=16 vs UORO | 10 | 0.0051 | 0.6558 | +0.6508 | [+0.6320, +0.6712] | 0.0020 | 0.0352 | 0.1055 | +19.43 | +1.00 |
| henon | SK-RTRL r=16 vs TBPTT | 10 | 0.0051 | 0.0045 | -0.0006 | [-0.0012, -0.0001] | 0.1055 | 0.2520 | 1.0000 | -0.63 | -0.48 |
| mackeyglass | SK-RTRL r=16 vs exact RTRL | 10 | 0.0025 | 0.0024 | -0.0001 | [-0.0003, +0.0000] | 0.1934 | 0.4219 | 1.0000 | -0.50 | -0.32 |
| mackeyglass | SK-RTRL r=16 vs SnAp-1 | 10 | 0.0025 | 0.0035 | +0.0010 | [+0.0008, +0.0013] | 0.0020 | 0.0352 | 0.1055 | +2.37 | +1.00 |
| mackeyglass | SK-RTRL r=16 vs KF-RTRL | 10 | 0.0025 | 0.0043 | +0.0018 | [+0.0015, +0.0021] | 0.0020 | 0.0352 | 0.1055 | +3.69 | +1.00 |
| mackeyglass | SK-RTRL r=16 vs RFLO | 10 | 0.0025 | 0.0071 | +0.0046 | [+0.0035, +0.0060] | 0.0020 | 0.0352 | 0.1055 | +2.16 | +1.00 |
| mackeyglass | SK-RTRL r=16 vs UORO | 10 | 0.0025 | 0.0054 | +0.0029 | [+0.0023, +0.0034] | 0.0020 | 0.0352 | 0.1055 | +3.12 | +1.00 |
| mackeyglass | SK-RTRL r=16 vs TBPTT | 10 | 0.0025 | 0.0127 | +0.0102 | [-0.0007, +0.0318] | 0.1055 | 0.4219 | 1.0000 | +0.30 | -0.64 |
| lorenz | SK-RTRL r=16 vs exact RTRL | 10 | 0.0014 | 0.0013 | -0.0001 | [-0.0003, +0.0000] | 0.3223 | 1.0000 | 1.0000 | -0.45 | -0.36 |
| lorenz | SK-RTRL r=16 vs SnAp-1 | 10 | 0.0014 | 0.0025 | +0.0011 | [+0.0007, +0.0015] | 0.0020 | 0.0352 | 0.1055 | +1.60 | +0.88 |
| lorenz | SK-RTRL r=16 vs KF-RTRL | 10 | 0.0014 | 0.0021 | +0.0007 | [+0.0005, +0.0010] | 0.0039 | 0.0352 | 0.1055 | +1.76 | +0.92 |
| lorenz | SK-RTRL r=16 vs RFLO | 10 | 0.0014 | 0.0042 | +0.0028 | [+0.0021, +0.0034] | 0.0020 | 0.0352 | 0.1055 | +2.61 | +1.00 |
| lorenz | SK-RTRL r=16 vs UORO | 10 | 0.0014 | 0.0043 | +0.0029 | [+0.0026, +0.0033] | 0.0020 | 0.0352 | 0.1055 | +5.36 | +1.00 |
| lorenz | SK-RTRL r=16 vs TBPTT | 10 | 0.0014 | 0.0211 | +0.0197 | [+0.0001, +0.0418] | 0.5566 | 1.0000 | 1.0000 | +0.58 | -0.06 |
| sunspot | SK-RTRL r=16 vs exact RTRL | 10 | 0.1335 | 0.1335 | -0.0000 | [-0.0001, +0.0001] | 0.7695 | 1.0000 | 1.0000 | -0.03 | +0.04 |
| sunspot | SK-RTRL r=16 vs SnAp-1 | 10 | 0.1335 | 0.1398 | +0.0062 | [+0.0060, +0.0064] | 0.0020 | 0.0352 | 0.1055 | +18.83 | +0.94 |
| sunspot | SK-RTRL r=16 vs KF-RTRL | 10 | 0.1335 | 0.1365 | +0.0029 | [+0.0023, +0.0037] | 0.0020 | 0.0352 | 0.1055 | +2.33 | +0.68 |
| sunspot | SK-RTRL r=16 vs RFLO | 10 | 0.1335 | 0.1423 | +0.0088 | [+0.0081, +0.0097] | 0.0020 | 0.0352 | 0.1055 | +6.13 | +1.00 |
| sunspot | SK-RTRL r=16 vs UORO | 10 | 0.1335 | 0.1388 | +0.0053 | [+0.0045, +0.0059] | 0.0020 | 0.0352 | 0.1055 | +4.54 | +0.92 |
| sunspot | SK-RTRL r=16 vs TBPTT | 10 | 0.1335 | 0.1342 | +0.0007 | [+0.0003, +0.0012] | 0.0098 | 0.0488 | 0.2246 | +0.90 | +0.30 |
| laser | SK-RTRL r=16 vs exact RTRL | 10 | 0.0197 | 0.0160 | -0.0037 | [-0.0055, -0.0024] | 0.0020 | 0.0352 | 0.1055 | -1.38 | -0.80 |
| laser | SK-RTRL r=16 vs SnAp-1 | 10 | 0.0197 | 0.0200 | +0.0003 | [-0.0015, +0.0020] | 0.6953 | 0.9844 | 1.0000 | +0.11 | +0.28 |
| laser | SK-RTRL r=16 vs KF-RTRL | 10 | 0.0197 | 0.0560 | +0.0363 | [+0.0218, +0.0577] | 0.0020 | 0.0352 | 0.1055 | +1.13 | +1.00 |
| laser | SK-RTRL r=16 vs RFLO | 10 | 0.0197 | 0.2001 | +0.1804 | [+0.1266, +0.2422] | 0.0020 | 0.0352 | 0.1055 | +1.81 | +1.00 |
| laser | SK-RTRL r=16 vs UORO | 10 | 0.0197 | 0.1509 | +0.1312 | [+0.1231, +0.1399] | 0.0020 | 0.0352 | 0.1055 | +9.05 | +1.00 |
| laser | SK-RTRL r=16 vs TBPTT | 10 | 0.0197 | 0.0167 | -0.0030 | [-0.0065, -0.0004] | 0.0840 | 0.4199 | 1.0000 | -0.56 | -0.56 |

### SK-RTRL r=16 -- gradient cosine (Holm family: 12 per (task, metric), 36 per (rank, metric))

| task | comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p (task fam) | holm_p (rank fam) | d_z | Cliff delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adding | SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| adding | SK-RTRL r=16 vs SnAp-1 | 10 | 0.853 | 0.524 | +0.328 | [+0.277, +0.385] | 0.0020 | 0.0234 | 0.0703 | +3.53 | +1.00 |
| adding | SK-RTRL r=16 vs KF-RTRL | 10 | 0.853 | 0.560 | +0.293 | [+0.246, +0.336] | 0.0020 | 0.0234 | 0.0703 | +3.78 | +1.00 |
| adding | SK-RTRL r=16 vs RFLO | 10 | 0.853 | 0.531 | +0.321 | [+0.275, +0.378] | 0.0020 | 0.0234 | 0.0703 | +3.62 | +1.00 |
| adding | SK-RTRL r=16 vs UORO | 10 | 0.853 | 0.119 | +0.734 | [+0.703, +0.765] | 0.0020 | 0.0234 | 0.0703 | +14.21 | +1.00 |
| adding | SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| copy | SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| copy | SK-RTRL r=16 vs SnAp-1 | 10 | 0.999 | 0.820 | +0.179 | [+0.175, +0.182] | 0.0020 | 0.0234 | 0.0703 | +28.78 | +1.00 |
| copy | SK-RTRL r=16 vs KF-RTRL | 10 | 0.999 | 0.606 | +0.393 | [+0.336, +0.456] | 0.0020 | 0.0234 | 0.0703 | +3.87 | +1.00 |
| copy | SK-RTRL r=16 vs RFLO | 10 | 0.999 | 0.034 | +0.964 | [+0.960, +0.969] | 0.0020 | 0.0234 | 0.0703 | +126.24 | +1.00 |
| copy | SK-RTRL r=16 vs UORO | 10 | 0.999 | 0.048 | +0.951 | [+0.937, +0.964] | 0.0020 | 0.0234 | 0.0703 | +39.92 | +1.00 |
| copy | SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| anbn | SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| anbn | SK-RTRL r=16 vs SnAp-1 | 10 | 0.983 | 0.470 | +0.512 | [+0.439, +0.583] | 0.0020 | 0.0234 | 0.0703 | +4.17 | +1.00 |
| anbn | SK-RTRL r=16 vs KF-RTRL | 10 | 0.983 | 0.441 | +0.542 | [+0.508, +0.575] | 0.0020 | 0.0234 | 0.0703 | +9.51 | +1.00 |
| anbn | SK-RTRL r=16 vs RFLO | 10 | 0.983 | 0.516 | +0.467 | [+0.432, +0.497] | 0.0020 | 0.0234 | 0.0703 | +8.26 | +1.00 |
| anbn | SK-RTRL r=16 vs UORO | 10 | 0.983 | 0.091 | +0.891 | [+0.881, +0.903] | 0.0020 | 0.0234 | 0.0703 | +48.40 | +1.00 |
| anbn | SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| rotation | SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| rotation | SK-RTRL r=16 vs SnAp-1 | 10 | 0.841 | 0.411 | +0.430 | [+0.396, +0.462] | 0.0020 | 0.0234 | 0.0703 | +7.56 | +1.00 |
| rotation | SK-RTRL r=16 vs KF-RTRL | 10 | 0.841 | 0.379 | +0.462 | [+0.390, +0.528] | 0.0020 | 0.0234 | 0.0703 | +3.94 | +1.00 |
| rotation | SK-RTRL r=16 vs RFLO | 10 | 0.841 | 0.144 | +0.697 | [+0.667, +0.727] | 0.0020 | 0.0234 | 0.0703 | +13.62 | +1.00 |
| rotation | SK-RTRL r=16 vs UORO | 10 | 0.841 | 0.034 | +0.807 | [+0.770, +0.845] | 0.0020 | 0.0234 | 0.0703 | +12.51 | +1.00 |
| rotation | SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| henon | SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| henon | SK-RTRL r=16 vs SnAp-1 | 10 | 0.977 | 0.781 | +0.196 | [+0.172, +0.221] | 0.0020 | 0.0234 | 0.0703 | +4.73 | +1.00 |
| henon | SK-RTRL r=16 vs KF-RTRL | 10 | 0.977 | 0.629 | +0.348 | [+0.300, +0.384] | 0.0020 | 0.0234 | 0.0703 | +4.78 | +1.00 |
| henon | SK-RTRL r=16 vs RFLO | 10 | 0.977 | 0.392 | +0.585 | [+0.552, +0.619] | 0.0020 | 0.0234 | 0.0703 | +10.27 | +1.00 |
| henon | SK-RTRL r=16 vs UORO | 10 | 0.977 | 0.043 | +0.934 | [+0.931, +0.937] | 0.0020 | 0.0234 | 0.0703 | +190.80 | +1.00 |
| henon | SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| mackeyglass | SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| mackeyglass | SK-RTRL r=16 vs SnAp-1 | 10 | 0.846 | 0.353 | +0.493 | [+0.440, +0.543] | 0.0020 | 0.0234 | 0.0703 | +5.53 | +1.00 |
| mackeyglass | SK-RTRL r=16 vs KF-RTRL | 10 | 0.846 | 0.378 | +0.468 | [+0.439, +0.499] | 0.0020 | 0.0234 | 0.0703 | +9.10 | +1.00 |
| mackeyglass | SK-RTRL r=16 vs RFLO | 10 | 0.846 | 0.162 | +0.684 | [+0.649, +0.715] | 0.0020 | 0.0234 | 0.0703 | +12.12 | +1.00 |
| mackeyglass | SK-RTRL r=16 vs UORO | 10 | 0.846 | 0.043 | +0.802 | [+0.789, +0.816] | 0.0020 | 0.0234 | 0.0703 | +34.76 | +1.00 |
| mackeyglass | SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| lorenz | SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| lorenz | SK-RTRL r=16 vs SnAp-1 | 10 | 0.839 | 0.333 | +0.506 | [+0.475, +0.536] | 0.0020 | 0.0234 | 0.0703 | +9.83 | +1.00 |
| lorenz | SK-RTRL r=16 vs KF-RTRL | 10 | 0.839 | 0.459 | +0.380 | [+0.350, +0.411] | 0.0020 | 0.0234 | 0.0703 | +7.29 | +1.00 |
| lorenz | SK-RTRL r=16 vs RFLO | 10 | 0.839 | 0.200 | +0.639 | [+0.609, +0.668] | 0.0020 | 0.0234 | 0.0703 | +12.84 | +1.00 |
| lorenz | SK-RTRL r=16 vs UORO | 10 | 0.839 | 0.054 | +0.785 | [+0.767, +0.803] | 0.0020 | 0.0234 | 0.0703 | +25.25 | +1.00 |
| lorenz | SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| sunspot | SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| sunspot | SK-RTRL r=16 vs SnAp-1 | 10 | 0.899 | 0.622 | +0.276 | [+0.252, +0.300] | 0.0020 | 0.0234 | 0.0703 | +6.68 | +1.00 |
| sunspot | SK-RTRL r=16 vs KF-RTRL | 10 | 0.899 | 0.442 | +0.457 | [+0.438, +0.476] | 0.0020 | 0.0234 | 0.0703 | +14.10 | +1.00 |
| sunspot | SK-RTRL r=16 vs RFLO | 10 | 0.899 | 0.337 | +0.562 | [+0.523, +0.601] | 0.0020 | 0.0234 | 0.0703 | +8.43 | +1.00 |
| sunspot | SK-RTRL r=16 vs UORO | 10 | 0.899 | 0.063 | +0.836 | [+0.826, +0.845] | 0.0020 | 0.0234 | 0.0703 | +50.42 | +1.00 |
| sunspot | SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| laser | SK-RTRL r=16 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| laser | SK-RTRL r=16 vs SnAp-1 | 10 | 0.932 | 0.378 | +0.554 | [+0.537, +0.573] | 0.0020 | 0.0234 | 0.0703 | +18.65 | +1.00 |
| laser | SK-RTRL r=16 vs KF-RTRL | 10 | 0.932 | 0.474 | +0.457 | [+0.427, +0.486] | 0.0020 | 0.0234 | 0.0703 | +9.00 | +1.00 |
| laser | SK-RTRL r=16 vs RFLO | 10 | 0.932 | 0.046 | +0.885 | [+0.859, +0.911] | 0.0020 | 0.0234 | 0.0703 | +19.80 | +1.00 |
| laser | SK-RTRL r=16 vs UORO | 10 | 0.932 | 0.033 | +0.899 | [+0.891, +0.906] | 0.0020 | 0.0234 | 0.0703 | +70.72 | +1.00 |
| laser | SK-RTRL r=16 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |

## SK-RTRL r=32 -- full opponent sweep

### SK-RTRL r=32 -- task metric (Holm family: 18 per (task, metric), 54 per (rank, metric))

| task | comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p (task fam) | holm_p (rank fam) | d_z | Cliff delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adding | SK-RTRL r=32 vs exact RTRL | 10 | 0.2098 | 0.2067 | -0.0031 | [-0.0047, -0.0019] | 0.0039 | 0.0352 | 0.1055 | -1.27 | -0.18 |
| adding | SK-RTRL r=32 vs SnAp-1 | 10 | 0.2098 | 0.2131 | +0.0034 | [-0.0043, +0.0146] | 0.9219 | 1.0000 | 1.0000 | +0.20 | +0.04 |
| adding | SK-RTRL r=32 vs KF-RTRL | 10 | 0.2098 | 0.2172 | +0.0074 | [+0.0030, +0.0116] | 0.0195 | 0.1172 | 0.3516 | +1.01 | +0.44 |
| adding | SK-RTRL r=32 vs RFLO | 10 | 0.2098 | 0.2268 | +0.0171 | [+0.0114, +0.0230] | 0.0020 | 0.0352 | 0.1055 | +1.71 | +0.64 |
| adding | SK-RTRL r=32 vs UORO | 10 | 0.2098 | 0.2613 | +0.0516 | [+0.0198, +0.1078] | 0.0020 | 0.0352 | 0.1055 | +0.60 | +1.00 |
| adding | SK-RTRL r=32 vs TBPTT | 10 | 0.2098 | 0.2067 | -0.0031 | [-0.0047, -0.0019] | 0.0039 | 0.0352 | 0.1055 | -1.27 | -0.18 |
| copy | SK-RTRL r=32 vs exact RTRL | 10 | 0.1711 | 0.1370 | +0.0341 | [+0.0211, +0.0463] | 0.0039 | 0.0586 | 0.1055 | +1.56 | +0.84 |
| copy | SK-RTRL r=32 vs SnAp-1 | 10 | 0.1711 | 0.1242 | +0.0469 | [+0.0361, +0.0603] | 0.0020 | 0.0352 | 0.1055 | +2.23 | +1.00 |
| copy | SK-RTRL r=32 vs KF-RTRL | 10 | 0.1711 | 0.1467 | +0.0244 | [+0.0101, +0.0385] | 0.0195 | 0.1953 | 0.3516 | +1.00 | +0.76 |
| copy | SK-RTRL r=32 vs RFLO | 10 | 0.1711 | 0.1824 | -0.0113 | [-0.0312, +0.0077] | 0.3750 | 1.0000 | 1.0000 | -0.34 | -0.30 |
| copy | SK-RTRL r=32 vs UORO | 10 | 0.1711 | 0.1264 | +0.0447 | [+0.0336, +0.0586] | 0.0020 | 0.0352 | 0.1055 | +2.05 | +1.00 |
| copy | SK-RTRL r=32 vs TBPTT | 10 | 0.1711 | 0.1256 | +0.0455 | [+0.0359, +0.0572] | 0.0020 | 0.0352 | 0.1055 | +2.46 | +1.00 |
| anbn | SK-RTRL r=32 vs exact RTRL | 10 | 0.9274 | 0.9347 | -0.0073 | [-0.0139, -0.0011] | 0.0645 | 0.1934 | 0.8379 | -0.66 | -0.42 |
| anbn | SK-RTRL r=32 vs SnAp-1 | 10 | 0.9274 | 0.9018 | +0.0256 | [+0.0196, +0.0315] | 0.0020 | 0.0352 | 0.1055 | +2.50 | +1.00 |
| anbn | SK-RTRL r=32 vs KF-RTRL | 10 | 0.9274 | 0.8961 | +0.0313 | [+0.0229, +0.0401] | 0.0020 | 0.0352 | 0.1055 | +2.12 | +1.00 |
| anbn | SK-RTRL r=32 vs RFLO | 10 | 0.9274 | 0.9093 | +0.0180 | [+0.0114, +0.0249] | 0.0039 | 0.0352 | 0.1055 | +1.58 | +0.88 |
| anbn | SK-RTRL r=32 vs UORO | 10 | 0.9274 | 0.8855 | +0.0419 | [+0.0346, +0.0493] | 0.0020 | 0.0352 | 0.1055 | +3.33 | +1.00 |
| anbn | SK-RTRL r=32 vs TBPTT | 10 | 0.9274 | 0.9050 | +0.0223 | [+0.0143, +0.0312] | 0.0020 | 0.0352 | 0.1055 | +1.54 | +0.92 |
| rotation | SK-RTRL r=32 vs exact RTRL | 10 | 0.1039 | 0.1072 | +0.0034 | [-0.0021, +0.0091] | 0.4922 | 1.0000 | 1.0000 | +0.35 | +0.22 |
| rotation | SK-RTRL r=32 vs SnAp-1 | 10 | 0.1039 | 0.0509 | -0.0530 | [-0.0599, -0.0462] | 0.0020 | 0.0352 | 0.1055 | -4.52 | -1.00 |
| rotation | SK-RTRL r=32 vs KF-RTRL | 10 | 0.1039 | 0.1137 | +0.0098 | [+0.0018, +0.0184] | 0.0273 | 0.2188 | 0.4102 | +0.69 | +0.56 |
| rotation | SK-RTRL r=32 vs RFLO | 10 | 0.1039 | 0.0977 | -0.0062 | [-0.0153, +0.0038] | 0.2754 | 1.0000 | 1.0000 | -0.38 | -0.40 |
| rotation | SK-RTRL r=32 vs UORO | 10 | 0.1039 | 0.1186 | +0.0148 | [+0.0099, +0.0196] | 0.0020 | 0.0352 | 0.1055 | +1.78 | +0.86 |
| rotation | SK-RTRL r=32 vs TBPTT | 10 | 0.1039 | 0.0332 | -0.0706 | [-0.0777, -0.0642] | 0.0020 | 0.0352 | 0.1055 | -6.04 | -1.00 |
| henon | SK-RTRL r=32 vs exact RTRL | 10 | 0.0050 | 0.0045 | -0.0005 | [-0.0010, -0.0000] | 0.0488 | 0.1953 | 0.6836 | -0.65 | -0.66 |
| henon | SK-RTRL r=32 vs SnAp-1 | 10 | 0.0050 | 0.0046 | -0.0004 | [-0.0008, +0.0001] | 0.0840 | 0.2520 | 1.0000 | -0.48 | -0.46 |
| henon | SK-RTRL r=32 vs KF-RTRL | 10 | 0.0050 | 0.0205 | +0.0155 | [+0.0057, +0.0335] | 0.0020 | 0.0352 | 0.1055 | +0.56 | +1.00 |
| henon | SK-RTRL r=32 vs RFLO | 10 | 0.0050 | 0.0565 | +0.0515 | [+0.0409, +0.0644] | 0.0020 | 0.0352 | 0.1055 | +2.52 | +1.00 |
| henon | SK-RTRL r=32 vs UORO | 10 | 0.0050 | 0.6558 | +0.6508 | [+0.6320, +0.6715] | 0.0020 | 0.0352 | 0.1055 | +19.33 | +1.00 |
| henon | SK-RTRL r=32 vs TBPTT | 10 | 0.0050 | 0.0045 | -0.0005 | [-0.0010, -0.0000] | 0.3223 | 0.3223 | 1.0000 | -0.57 | -0.34 |
| mackeyglass | SK-RTRL r=32 vs exact RTRL | 10 | 0.0023 | 0.0024 | +0.0000 | [-0.0002, +0.0002] | 0.4316 | 0.4316 | 1.0000 | +0.09 | +0.04 |
| mackeyglass | SK-RTRL r=32 vs SnAp-1 | 10 | 0.0023 | 0.0035 | +0.0012 | [+0.0008, +0.0016] | 0.0020 | 0.0352 | 0.1055 | +1.91 | +1.00 |
| mackeyglass | SK-RTRL r=32 vs KF-RTRL | 10 | 0.0023 | 0.0043 | +0.0019 | [+0.0015, +0.0023] | 0.0020 | 0.0352 | 0.1055 | +3.05 | +1.00 |
| mackeyglass | SK-RTRL r=32 vs RFLO | 10 | 0.0023 | 0.0071 | +0.0048 | [+0.0038, +0.0061] | 0.0020 | 0.0352 | 0.1055 | +2.44 | +1.00 |
| mackeyglass | SK-RTRL r=32 vs UORO | 10 | 0.0023 | 0.0054 | +0.0031 | [+0.0025, +0.0036] | 0.0020 | 0.0352 | 0.1055 | +3.03 | +1.00 |
| mackeyglass | SK-RTRL r=32 vs TBPTT | 10 | 0.0023 | 0.0127 | +0.0103 | [-0.0006, +0.0319] | 0.1309 | 0.4219 | 1.0000 | +0.30 | -0.48 |
| lorenz | SK-RTRL r=32 vs exact RTRL | 10 | 0.0012 | 0.0013 | +0.0000 | [-0.0000, +0.0001] | 0.7695 | 1.0000 | 1.0000 | +0.11 | -0.02 |
| lorenz | SK-RTRL r=32 vs SnAp-1 | 10 | 0.0012 | 0.0025 | +0.0012 | [+0.0008, +0.0017] | 0.0020 | 0.0352 | 0.1055 | +1.60 | +1.00 |
| lorenz | SK-RTRL r=32 vs KF-RTRL | 10 | 0.0012 | 0.0021 | +0.0009 | [+0.0008, +0.0010] | 0.0020 | 0.0352 | 0.1055 | +4.16 | +1.00 |
| lorenz | SK-RTRL r=32 vs RFLO | 10 | 0.0012 | 0.0042 | +0.0030 | [+0.0024, +0.0035] | 0.0020 | 0.0352 | 0.1055 | +3.23 | +1.00 |
| lorenz | SK-RTRL r=32 vs UORO | 10 | 0.0012 | 0.0043 | +0.0031 | [+0.0028, +0.0034] | 0.0020 | 0.0352 | 0.1055 | +5.97 | +1.00 |
| lorenz | SK-RTRL r=32 vs TBPTT | 10 | 0.0012 | 0.0211 | +0.0198 | [+0.0002, +0.0418] | 0.5566 | 1.0000 | 1.0000 | +0.58 | -0.02 |
| sunspot | SK-RTRL r=32 vs exact RTRL | 10 | 0.1334 | 0.1335 | +0.0001 | [+0.0001, +0.0002] | 0.0195 | 0.0586 | 0.3516 | +0.97 | +0.10 |
| sunspot | SK-RTRL r=32 vs SnAp-1 | 10 | 0.1334 | 0.1398 | +0.0064 | [+0.0062, +0.0066] | 0.0020 | 0.0352 | 0.1055 | +21.39 | +0.96 |
| sunspot | SK-RTRL r=32 vs KF-RTRL | 10 | 0.1334 | 0.1365 | +0.0031 | [+0.0024, +0.0039] | 0.0020 | 0.0352 | 0.1055 | +2.39 | +0.76 |
| sunspot | SK-RTRL r=32 vs RFLO | 10 | 0.1334 | 0.1423 | +0.0090 | [+0.0083, +0.0099] | 0.0020 | 0.0352 | 0.1055 | +6.37 | +1.00 |
| sunspot | SK-RTRL r=32 vs UORO | 10 | 0.1334 | 0.1388 | +0.0054 | [+0.0048, +0.0059] | 0.0020 | 0.0352 | 0.1055 | +5.41 | +0.92 |
| sunspot | SK-RTRL r=32 vs TBPTT | 10 | 0.1334 | 0.1342 | +0.0008 | [+0.0005, +0.0013] | 0.0020 | 0.0352 | 0.1055 | +1.27 | +0.38 |
| laser | SK-RTRL r=32 vs exact RTRL | 10 | 0.0179 | 0.0160 | -0.0019 | [-0.0029, -0.0008] | 0.0137 | 0.0957 | 0.2598 | -1.09 | -0.62 |
| laser | SK-RTRL r=32 vs SnAp-1 | 10 | 0.0179 | 0.0200 | +0.0021 | [+0.0000, +0.0041] | 0.1055 | 0.4219 | 1.0000 | +0.62 | +0.48 |
| laser | SK-RTRL r=32 vs KF-RTRL | 10 | 0.0179 | 0.0560 | +0.0381 | [+0.0242, +0.0596] | 0.0020 | 0.0352 | 0.1055 | +1.17 | +1.00 |
| laser | SK-RTRL r=32 vs RFLO | 10 | 0.0179 | 0.2001 | +0.1822 | [+0.1274, +0.2465] | 0.0020 | 0.0352 | 0.1055 | +1.78 | +1.00 |
| laser | SK-RTRL r=32 vs UORO | 10 | 0.0179 | 0.1509 | +0.1330 | [+0.1236, +0.1439] | 0.0020 | 0.0352 | 0.1055 | +7.61 | +1.00 |
| laser | SK-RTRL r=32 vs TBPTT | 10 | 0.0179 | 0.0167 | -0.0012 | [-0.0026, +0.0004] | 0.1309 | 0.4219 | 1.0000 | -0.48 | -0.34 |

### SK-RTRL r=32 -- gradient cosine (Holm family: 12 per (task, metric), 36 per (rank, metric))

| task | comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p (task fam) | holm_p (rank fam) | d_z | Cliff delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adding | SK-RTRL r=32 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| adding | SK-RTRL r=32 vs SnAp-1 | 10 | 0.894 | 0.524 | +0.370 | [+0.325, +0.421] | 0.0020 | 0.0234 | 0.0703 | +4.50 | +1.00 |
| adding | SK-RTRL r=32 vs KF-RTRL | 10 | 0.894 | 0.560 | +0.334 | [+0.284, +0.379] | 0.0020 | 0.0234 | 0.0703 | +4.10 | +1.00 |
| adding | SK-RTRL r=32 vs RFLO | 10 | 0.894 | 0.531 | +0.363 | [+0.320, +0.417] | 0.0020 | 0.0234 | 0.0703 | +4.32 | +1.00 |
| adding | SK-RTRL r=32 vs UORO | 10 | 0.894 | 0.119 | +0.776 | [+0.756, +0.800] | 0.0020 | 0.0234 | 0.0703 | +20.43 | +1.00 |
| adding | SK-RTRL r=32 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| copy | SK-RTRL r=32 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| copy | SK-RTRL r=32 vs SnAp-1 | 10 | 0.995 | 0.820 | +0.175 | [+0.169, +0.181] | 0.0020 | 0.0234 | 0.0703 | +17.26 | +1.00 |
| copy | SK-RTRL r=32 vs KF-RTRL | 10 | 0.995 | 0.606 | +0.389 | [+0.331, +0.454] | 0.0020 | 0.0234 | 0.0703 | +3.72 | +1.00 |
| copy | SK-RTRL r=32 vs RFLO | 10 | 0.995 | 0.034 | +0.961 | [+0.953, +0.966] | 0.0020 | 0.0234 | 0.0703 | +86.51 | +1.00 |
| copy | SK-RTRL r=32 vs UORO | 10 | 0.995 | 0.048 | +0.947 | [+0.932, +0.962] | 0.0020 | 0.0234 | 0.0703 | +37.33 | +1.00 |
| copy | SK-RTRL r=32 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| anbn | SK-RTRL r=32 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| anbn | SK-RTRL r=32 vs SnAp-1 | 10 | 0.993 | 0.470 | +0.523 | [+0.451, +0.592] | 0.0020 | 0.0234 | 0.0703 | +4.33 | +1.00 |
| anbn | SK-RTRL r=32 vs KF-RTRL | 10 | 0.993 | 0.441 | +0.552 | [+0.517, +0.587] | 0.0020 | 0.0234 | 0.0703 | +9.27 | +1.00 |
| anbn | SK-RTRL r=32 vs RFLO | 10 | 0.993 | 0.516 | +0.477 | [+0.444, +0.506] | 0.0020 | 0.0234 | 0.0703 | +8.92 | +1.00 |
| anbn | SK-RTRL r=32 vs UORO | 10 | 0.993 | 0.091 | +0.902 | [+0.894, +0.910] | 0.0020 | 0.0234 | 0.0703 | +69.35 | +1.00 |
| anbn | SK-RTRL r=32 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| rotation | SK-RTRL r=32 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| rotation | SK-RTRL r=32 vs SnAp-1 | 10 | 0.930 | 0.411 | +0.519 | [+0.493, +0.542] | 0.0020 | 0.0234 | 0.0703 | +12.32 | +1.00 |
| rotation | SK-RTRL r=32 vs KF-RTRL | 10 | 0.930 | 0.379 | +0.551 | [+0.507, +0.593] | 0.0020 | 0.0234 | 0.0703 | +7.54 | +1.00 |
| rotation | SK-RTRL r=32 vs RFLO | 10 | 0.930 | 0.144 | +0.786 | [+0.758, +0.815] | 0.0020 | 0.0234 | 0.0703 | +15.84 | +1.00 |
| rotation | SK-RTRL r=32 vs UORO | 10 | 0.930 | 0.034 | +0.896 | [+0.865, +0.924] | 0.0020 | 0.0234 | 0.0703 | +17.82 | +1.00 |
| rotation | SK-RTRL r=32 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| henon | SK-RTRL r=32 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| henon | SK-RTRL r=32 vs SnAp-1 | 10 | 0.985 | 0.781 | +0.205 | [+0.181, +0.229] | 0.0020 | 0.0234 | 0.0703 | +5.01 | +1.00 |
| henon | SK-RTRL r=32 vs KF-RTRL | 10 | 0.985 | 0.629 | +0.356 | [+0.309, +0.393] | 0.0020 | 0.0234 | 0.0703 | +4.94 | +1.00 |
| henon | SK-RTRL r=32 vs RFLO | 10 | 0.985 | 0.392 | +0.594 | [+0.561, +0.627] | 0.0020 | 0.0234 | 0.0703 | +10.51 | +1.00 |
| henon | SK-RTRL r=32 vs UORO | 10 | 0.985 | 0.043 | +0.943 | [+0.940, +0.945] | 0.0020 | 0.0234 | 0.0703 | +240.76 | +1.00 |
| henon | SK-RTRL r=32 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| mackeyglass | SK-RTRL r=32 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| mackeyglass | SK-RTRL r=32 vs SnAp-1 | 10 | 0.903 | 0.353 | +0.551 | [+0.496, +0.606] | 0.0020 | 0.0234 | 0.0703 | +5.88 | +1.00 |
| mackeyglass | SK-RTRL r=32 vs KF-RTRL | 10 | 0.903 | 0.378 | +0.525 | [+0.501, +0.554] | 0.0020 | 0.0234 | 0.0703 | +11.36 | +1.00 |
| mackeyglass | SK-RTRL r=32 vs RFLO | 10 | 0.903 | 0.162 | +0.742 | [+0.704, +0.776] | 0.0020 | 0.0234 | 0.0703 | +12.30 | +1.00 |
| mackeyglass | SK-RTRL r=32 vs UORO | 10 | 0.903 | 0.043 | +0.860 | [+0.854, +0.866] | 0.0020 | 0.0234 | 0.0703 | +81.18 | +1.00 |
| mackeyglass | SK-RTRL r=32 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| lorenz | SK-RTRL r=32 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| lorenz | SK-RTRL r=32 vs SnAp-1 | 10 | 0.901 | 0.333 | +0.569 | [+0.546, +0.596] | 0.0020 | 0.0234 | 0.0703 | +13.56 | +1.00 |
| lorenz | SK-RTRL r=32 vs KF-RTRL | 10 | 0.901 | 0.459 | +0.443 | [+0.424, +0.464] | 0.0020 | 0.0234 | 0.0703 | +12.76 | +1.00 |
| lorenz | SK-RTRL r=32 vs RFLO | 10 | 0.901 | 0.200 | +0.702 | [+0.665, +0.735] | 0.0020 | 0.0234 | 0.0703 | +11.89 | +1.00 |
| lorenz | SK-RTRL r=32 vs UORO | 10 | 0.901 | 0.054 | +0.847 | [+0.836, +0.858] | 0.0020 | 0.0234 | 0.0703 | +44.65 | +1.00 |
| lorenz | SK-RTRL r=32 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| sunspot | SK-RTRL r=32 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| sunspot | SK-RTRL r=32 vs SnAp-1 | 10 | 0.931 | 0.622 | +0.309 | [+0.286, +0.330] | 0.0020 | 0.0234 | 0.0703 | +8.15 | +1.00 |
| sunspot | SK-RTRL r=32 vs KF-RTRL | 10 | 0.931 | 0.442 | +0.489 | [+0.470, +0.507] | 0.0020 | 0.0234 | 0.0703 | +15.36 | +1.00 |
| sunspot | SK-RTRL r=32 vs RFLO | 10 | 0.931 | 0.337 | +0.594 | [+0.558, +0.630] | 0.0020 | 0.0234 | 0.0703 | +9.58 | +1.00 |
| sunspot | SK-RTRL r=32 vs UORO | 10 | 0.931 | 0.063 | +0.868 | [+0.857, +0.876] | 0.0020 | 0.0234 | 0.0703 | +52.63 | +1.00 |
| sunspot | SK-RTRL r=32 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |
| laser | SK-RTRL r=32 vs exact RTRL | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for exact |
| laser | SK-RTRL r=32 vs SnAp-1 | 10 | 0.960 | 0.378 | +0.582 | [+0.568, +0.599] | 0.0020 | 0.0234 | 0.0703 | +21.95 | +1.00 |
| laser | SK-RTRL r=32 vs KF-RTRL | 10 | 0.960 | 0.474 | +0.485 | [+0.459, +0.511] | 0.0020 | 0.0234 | 0.0703 | +10.91 | +1.00 |
| laser | SK-RTRL r=32 vs RFLO | 10 | 0.960 | 0.046 | +0.913 | [+0.889, +0.940] | 0.0020 | 0.0234 | 0.0703 | +21.20 | +1.00 |
| laser | SK-RTRL r=32 vs UORO | 10 | 0.960 | 0.033 | +0.927 | [+0.921, +0.933] | 0.0020 | 0.0234 | 0.0703 | +93.86 | +1.00 |
| laser | SK-RTRL r=32 vs TBPTT | 0 | -- | -- | -- | -- | -- | -- | -- | -- | not_applicable: grad_cos is not defined for tbptt |

## Auxiliary: rank against rank

Not part of the 270 (these compare the method with itself at another rank, not with a competitor); Holm-corrected inside their own (task, metric) family of 3.

### rank vs rank -- task metric

| task | comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p (task fam) | holm_p (rank fam) | d_z | Cliff delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adding | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.2118 | 0.2116 | -0.0002 | [-0.0004, -0.0001] | 0.0371 | 0.0371 | 0.0371 | -0.81 | -0.06 |
| adding | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.2098 | 0.2118 | +0.0021 | [+0.0009, +0.0033] | 0.0098 | 0.0293 | 0.0293 | +1.00 | +0.14 |
| adding | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.2098 | 0.2116 | +0.0018 | [+0.0008, +0.0031] | 0.0137 | 0.0293 | 0.0293 | +0.91 | +0.14 |
| copy | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.1542 | 0.1392 | +0.0150 | [+0.0024, +0.0293] | 0.1055 | 0.2109 | 0.2109 | +0.65 | +0.36 |
| copy | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.1711 | 0.1542 | +0.0169 | [-0.0041, +0.0347] | 0.1309 | 0.2109 | 0.2109 | +0.51 | +0.46 |
| copy | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.1711 | 0.1392 | +0.0319 | [+0.0139, +0.0506] | 0.0137 | 0.0410 | 0.0410 | +1.02 | +0.80 |
| anbn | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.9237 | 0.9156 | +0.0081 | [-0.0021, +0.0174] | 0.1934 | 0.3867 | 0.3867 | +0.49 | +0.50 |
| anbn | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.9274 | 0.9237 | +0.0037 | [-0.0035, +0.0122] | 0.6953 | 0.6953 | 0.6953 | +0.28 | +0.14 |
| anbn | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.9274 | 0.9156 | +0.0118 | [+0.0026, +0.0207] | 0.0488 | 0.1465 | 0.1465 | +0.76 | +0.52 |
| rotation | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.1129 | 0.1098 | -0.0031 | [-0.0070, +0.0010] | 0.1934 | 0.3164 | 0.3164 | -0.45 | -0.32 |
| rotation | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.1039 | 0.1129 | +0.0091 | [+0.0012, +0.0168] | 0.1055 | 0.3164 | 0.3164 | +0.68 | +0.58 |
| rotation | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.1039 | 0.1098 | +0.0060 | [+0.0000, +0.0118] | 0.1055 | 0.3164 | 0.3164 | +0.60 | +0.38 |
| henon | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.0051 | 0.0052 | +0.0002 | [-0.0001, +0.0004] | 0.3223 | 0.6445 | 0.6445 | +0.34 | +0.18 |
| henon | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.0050 | 0.0051 | +0.0001 | [-0.0002, +0.0004] | 0.6250 | 0.6445 | 0.6445 | +0.17 | +0.20 |
| henon | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.0050 | 0.0052 | +0.0002 | [-0.0001, +0.0005] | 0.1602 | 0.4805 | 0.4805 | +0.50 | +0.38 |
| mackeyglass | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.0025 | 0.0031 | +0.0006 | [+0.0002, +0.0010] | 0.0195 | 0.0391 | 0.0391 | +0.95 | +0.70 |
| mackeyglass | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.0023 | 0.0025 | +0.0002 | [-0.0001, +0.0004] | 0.2324 | 0.2324 | 0.2324 | +0.38 | +0.38 |
| mackeyglass | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.0023 | 0.0031 | +0.0008 | [+0.0004, +0.0011] | 0.0039 | 0.0117 | 0.0117 | +1.26 | +0.88 |
| lorenz | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.0014 | 0.0014 | -0.0000 | [-0.0003, +0.0002] | 1.0000 | 1.0000 | 1.0000 | -0.08 | -0.08 |
| lorenz | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.0012 | 0.0014 | +0.0001 | [+0.0000, +0.0003] | 0.0645 | 0.1934 | 0.1934 | +0.61 | +0.32 |
| lorenz | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.0012 | 0.0014 | +0.0001 | [-0.0000, +0.0003] | 0.3223 | 0.6445 | 0.6445 | +0.43 | +0.32 |
| sunspot | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.1335 | 0.1335 | -0.0000 | [-0.0001, +0.0000] | 0.8457 | 0.8457 | 0.8457 | -0.09 | +0.00 |
| sunspot | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.1334 | 0.1335 | +0.0002 | [+0.0000, +0.0003] | 0.0645 | 0.1465 | 0.1465 | +0.67 | +0.04 |
| sunspot | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.1334 | 0.1335 | +0.0001 | [+0.0000, +0.0003] | 0.0488 | 0.1465 | 0.1465 | +0.74 | +0.08 |
| laser | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.0197 | 0.0213 | +0.0016 | [+0.0001, +0.0038] | 0.1309 | 0.2617 | 0.2617 | +0.51 | +0.22 |
| laser | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.0179 | 0.0197 | +0.0018 | [-0.0002, +0.0045] | 0.3223 | 0.3223 | 0.3223 | +0.43 | +0.32 |
| laser | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.0179 | 0.0213 | +0.0034 | [+0.0013, +0.0057] | 0.0273 | 0.0820 | 0.0820 | +0.90 | +0.58 |

### rank vs rank -- gradient cosine

| task | comparison | n | mean A | mean B | delta (A better) | 95% CI | p | holm_p (task fam) | holm_p (rank fam) | d_z | Cliff delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adding | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.853 | 0.844 | +0.009 | [+0.004, +0.015] | 0.0098 | 0.0098 | 0.0098 | +0.94 | +0.16 |
| adding | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.894 | 0.853 | +0.042 | [+0.029, +0.056] | 0.0020 | 0.0059 | 0.0059 | +1.79 | +0.60 |
| adding | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.894 | 0.844 | +0.050 | [+0.037, +0.066] | 0.0020 | 0.0059 | 0.0059 | +2.07 | +0.64 |
| copy | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.999 | 0.997 | +0.001 | [-0.002, +0.006] | 0.8457 | 1.0000 | 1.0000 | +0.22 | +0.20 |
| copy | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.995 | 0.999 | -0.004 | [-0.009, +0.000] | 0.0371 | 0.1113 | 0.1113 | -0.46 | -0.42 |
| copy | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.995 | 0.997 | -0.003 | [-0.009, +0.004] | 0.7695 | 1.0000 | 1.0000 | -0.23 | -0.12 |
| anbn | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.983 | 0.972 | +0.011 | [+0.005, +0.018] | 0.0137 | 0.0137 | 0.0137 | +0.95 | +0.72 |
| anbn | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.993 | 0.983 | +0.010 | [+0.006, +0.015] | 0.0039 | 0.0078 | 0.0078 | +1.35 | +0.96 |
| anbn | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.993 | 0.972 | +0.021 | [+0.017, +0.026] | 0.0020 | 0.0059 | 0.0059 | +2.71 | +1.00 |
| rotation | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.841 | 0.679 | +0.163 | [+0.111, +0.215] | 0.0020 | 0.0059 | 0.0059 | +1.84 | +1.00 |
| rotation | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.930 | 0.841 | +0.089 | [+0.044, +0.137] | 0.0098 | 0.0098 | 0.0098 | +1.12 | +0.74 |
| rotation | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.930 | 0.679 | +0.251 | [+0.214, +0.293] | 0.0020 | 0.0059 | 0.0059 | +3.73 | +1.00 |
| henon | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.977 | 0.976 | +0.001 | [-0.002, +0.003] | 0.6250 | 0.6250 | 0.6250 | +0.16 | +0.08 |
| henon | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.985 | 0.977 | +0.009 | [+0.007, +0.010] | 0.0020 | 0.0059 | 0.0059 | +2.71 | +0.92 |
| henon | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.985 | 0.976 | +0.009 | [+0.007, +0.011] | 0.0020 | 0.0059 | 0.0059 | +2.73 | +0.98 |
| mackeyglass | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.846 | 0.844 | +0.002 | [-0.018, +0.024] | 0.9219 | 0.9219 | 0.9219 | +0.05 | -0.12 |
| mackeyglass | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.903 | 0.846 | +0.058 | [+0.046, +0.070] | 0.0020 | 0.0059 | 0.0059 | +2.71 | +1.00 |
| mackeyglass | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.903 | 0.844 | +0.060 | [+0.046, +0.076] | 0.0020 | 0.0059 | 0.0059 | +2.39 | +1.00 |
| lorenz | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.839 | 0.839 | -0.000 | [-0.026, +0.026] | 0.9219 | 0.9219 | 0.9219 | -0.00 | -0.16 |
| lorenz | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.901 | 0.839 | +0.063 | [+0.043, +0.081] | 0.0020 | 0.0059 | 0.0059 | +1.95 | +0.92 |
| lorenz | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.901 | 0.839 | +0.063 | [+0.040, +0.084] | 0.0039 | 0.0078 | 0.0078 | +1.69 | +0.96 |
| sunspot | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.899 | 0.898 | +0.001 | [-0.001, +0.003] | 0.4316 | 0.4316 | 0.4316 | +0.34 | +0.06 |
| sunspot | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.931 | 0.899 | +0.033 | [+0.028, +0.038] | 0.0020 | 0.0059 | 0.0059 | +3.57 | +0.86 |
| sunspot | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.931 | 0.898 | +0.034 | [+0.028, +0.040] | 0.0020 | 0.0059 | 0.0059 | +3.33 | +0.84 |
| laser | SK-RTRL r=16 vs SK-RTRL r=4 | 10 | 0.932 | 0.931 | +0.001 | [-0.011, +0.013] | 0.8457 | 0.8457 | 0.8457 | +0.03 | +0.12 |
| laser | SK-RTRL r=32 vs SK-RTRL r=16 | 10 | 0.960 | 0.932 | +0.028 | [+0.019, +0.038] | 0.0020 | 0.0059 | 0.0059 | +1.81 | +0.92 |
| laser | SK-RTRL r=32 vs SK-RTRL r=4 | 10 | 0.960 | 0.931 | +0.029 | [+0.020, +0.038] | 0.0020 | 0.0059 | 0.0059 | +1.96 | +1.00 |

# Claim audit

Every count the body of the paper quotes, re-derived from the full matrix above. W-T-L is written win-tie-loss.

## Claim 1 -- the gradient-cosine sweep (paper: "SK-RTRL wins all 108, Cliff's delta exactly 1.00, p_H = 0.0156 throughout")

Set: 3 ranks x 4 approximate estimators (SnAp-1, KF-RTRL, RFLO, UORO) x 9 tasks = 108 comparisons on the gradient cosine; 108 computed.

| criterion | W-T-L over all 108 | SK-RTRL r=4 | SK-RTRL r=16 | SK-RTRL r=32 |
|---|---|---|---|---|
| sign of delta | 108-0-0 | 36-0-0 | 36-0-0 | 36-0-0 |
| bootstrap 95% CI excludes 0 | 108-0-0 | 36-0-0 | 36-0-0 | 36-0-0 |
| holm_p < 0.05, (task, metric) family | 108-0-0 | 36-0-0 | 36-0-0 | 36-0-0 |
| holm_p < 0.05, (rank, metric) family | 0-108-0 | 0-36-0 | 0-36-0 | 0-36-0 |

| quantity | value |
|---|---|
| comparisons with Cliff delta exactly +1.00 | 108 of 108 |
| Cliff delta range | +1.00 to +1.00 |
| holm_p range, (task, metric) family | 0.0234 to 0.0234 |
| holm_p range, (rank, metric) family | 0.0703 to 0.0703 |

## Claim 2 -- win-tie-loss on the task metric (paper: 31-5-9 at r=4, 32-6-7 at r=16, 35-6-4 at r=32)

Set, per rank: the 5 approximate estimators (SnAp-1, KF-RTRL, RFLO, UORO, TBPTT) x 9 tasks = 45 comparisons on the task metric. The paper's stated rule is the bootstrap CI one.

| criterion | SK-RTRL r=4 | SK-RTRL r=16 | SK-RTRL r=32 |
|---|---|---|---|
| sign of delta | 35-0-10 | 37-0-8 | 37-0-8 |
| bootstrap 95% CI excludes 0 | 31-5-9 | 32-6-7 | 35-6-4 |
| holm_p < 0.05, (task, metric) family | 24-18-3 | 26-16-3 | 30-12-3 |
| holm_p < 0.05, (rank, metric) family | 0-45-0 | 0-45-0 | 0-45-0 |

## Claim 3 -- against truncated BPTT alone (paper: 2 wins, 5 ties, 2 losses for r=16, Holm criterion)

| criterion | SK-RTRL r=4 | SK-RTRL r=16 | SK-RTRL r=32 |
|---|---|---|---|
| sign of delta | 5-0-4 | 5-0-4 | 5-0-4 |
| bootstrap 95% CI excludes 0 | 4-1-4 | 4-1-4 | 4-2-3 |
| holm_p < 0.05, (task, metric) family | 1-6-2 | 2-5-2 | 3-4-2 |
| holm_p < 0.05, (rank, metric) family | 0-9-0 | 0-9-0 | 0-9-0 |

Per task, SK-RTRL r=16 vs TBPTT on the task metric:

| task | delta (r=16 better) | holm_p (task fam) | verdict (Holm) | verdict (CI) |
|---|---|---|---|---|
| adding | -0.0051 | 0.0352 | loss | loss |
| copy | +0.0287 | 0.1172 | tie | win |
| anbn | +0.0187 | 0.0352 | win | win |
| rotation | -0.0797 | 0.0352 | loss | loss |
| henon | -0.0006 | 0.2520 | tie | loss |
| mackeyglass | +0.0102 | 0.4219 | tie | tie |
| lorenz | +0.0197 | 1.0000 | tie | win |
| sunspot | +0.0007 | 0.0488 | win | win |
| laser | -0.0030 | 0.4199 | tie | loss |

Median d_z of r=16 vs TBPTT over the nine tasks: +0.30.

## Claim 4 -- against exact RTRL, median d_z over the nine tasks (task metric)

| rank | median d_z |
|---|---|
| SK-RTRL r=4 | -1.27 |
| SK-RTRL r=16 | -0.64 |
| SK-RTRL r=32 | +0.09 |

