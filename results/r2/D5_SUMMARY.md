# D5 -- gated cells (GRU / LSTM): SK-RTRL applicability

Generated 2026-10-15 12:28 by `make_r2_d5_report.py` (reads `results/r2/d5_gated` and `results/r2/d4_eval`; writes nothing else).

- aggregation: window20 (window from step >= 80% of 20000)
- across seeds: mean ± population std (ddof=0), 95% CI = Student-t on the seed mean (sample std, t_{.975,n-1}); seeds [0, 1, 2]
- effect size: paired on the same seed, Cohen's d_z (+ = the row is better under the task's own direction) and a paired-t p. The task metric is paired against **exact RTRL**; `grad_cos` is paired against **SnAp-1**, because the cosine is not a defined quantity for exact RTRL.
- `e_t` is reported as a log10 median over the window (the bound spans >20 decades inside a single run; an arithmetic mean of it is meaningless)
- cells present in `results/r2/d5_gated`: gru, lstm

## 0. Headline

- **GRU**: 84 runs aggregated, of which 48 carry a certificate summary (the certificate is only defined for the sketched/SnAp estimators); total certificate violations across every such run and every step: **0**.
- **GRU**: fraction of steps with rho_bar<1 (Corollary 1 plateau regime), SK-RTRL r=16: rotation 0.000, anbn 0.000, henon 0.000, sunspot 0.000 -- so the plateau corollary is vacuous on this grid.
- **LSTM**: 84 runs aggregated, of which 48 carry a certificate summary (the certificate is only defined for the sketched/SnAp estimators); total certificate violations across every such run and every step: **0**.
- **LSTM**: fraction of steps with rho_bar<1 (Corollary 1 plateau regime), SK-RTRL r=16: rotation 0.000, anbn 0.000, henon 0.000, sunspot 0.000 -- so the plateau corollary is vacuous on this grid.
- KF-RTRL is absent by derivation on every gated row (see section 1); KF-RTRL json files in `results/r2/d5_gated` for the cell(s) present: **0** -- confirmed absent.
- Inventory complete against the job files for the cell(s) present.

## 1. Inventory and anomalies

Job files read: `jobs/d5_gated.txt`

- expected (`# out=` annotations): **168**
- present: **168**
- missing: **0**
- unexpected extra json in `results/r2/d5_gated`: **0**

No missing runs against the job-file expectation.

Every json parsed cleanly.

### NaN / divergence audit (raw record stream, not the window)

No NaN, no non-finite window value, no certificate violation, no certificate invalidation, no SVD fallback, correct step count -- in every run present.

### Divergence screen

Criterion: the last-20 % window mean against the best value the run reached **after step 0** -- accuracy tasks fire below 0.80x their own peak, error tasks above 2.0x their own best. (The step-0 record is degenerate on the accuracy tasks: an untrained net scores 1.000 by emitting the majority symbol, so a naive "worse than step 0" test fires on all 84 runs and on the tanh reference too, and says nothing.)

| cell | task     | estimator    | seed | step-0 | window | note                                    |
|------|----------|--------------|------|--------|--------|-----------------------------------------|
| gru  | henon    | exact RTRL   | 0    | 0.8787 | 0.0008 | window 0.0008212 > 2.0 x best 0.0004011 |
| gru  | henon    | exact RTRL   | 1    | 0.9049 | 0.0009 | window 0.0009132 > 2.0 x best 0.0004434 |
| gru  | henon    | SK-RTRL r=16 | 0    | 0.8787 | 0.0009 | window 0.0009074 > 2.0 x best 0.0004347 |
| gru  | henon    | SK-RTRL r=16 | 1    | 0.9049 | 0.0010 | window 0.000992 > 2.0 x best 0.000437   |
| gru  | henon    | SK-RTRL r=32 | 0    | 0.8787 | 0.0009 | window 0.0008828 > 2.0 x best 0.0003852 |
| gru  | henon    | SK-RTRL r=32 | 1    | 0.9049 | 0.0010 | window 0.001049 > 2.0 x best 0.0004455  |
| gru  | henon    | SK-RTRL r=32 | 2    | 0.9010 | 0.0009 | window 0.0009001 > 2.0 x best 0.0004244 |
| gru  | henon    | SK-RTRL r=4  | 0    | 0.8787 | 0.0009 | window 0.0009303 > 2.0 x best 0.0004256 |
| gru  | henon    | SK-RTRL r=4  | 2    | 0.9010 | 0.0010 | window 0.0009546 > 2.0 x best 0.0004476 |
| gru  | henon    | UORO         | 0    | 0.8787 | 0.3534 | window 0.3534 > 2.0 x best 0.01387      |
| gru  | henon    | UORO         | 1    | 0.9049 | 0.0446 | window 0.04464 > 2.0 x best 0.01727     |
| gru  | henon    | UORO         | 2    | 0.9010 | 0.0747 | window 0.07469 > 2.0 x best 0.01758     |
| gru  | rotation | SnAp-1       | 0    | 0.1282 | 0.0605 | window 0.06051 > 2.0 x best 0.02855     |
| gru  | sunspot  | UORO         | 0    | 1.8280 | 0.7099 | window 0.7099 > 2.0 x best 0.1156       |
| gru  | sunspot  | UORO         | 1    | 0.7497 | 0.3806 | window 0.3806 > 2.0 x best 0.1131       |
| gru  | sunspot  | UORO         | 2    | 0.8689 | 0.5012 | window 0.5012 > 2.0 x best 0.1142       |
| lstm | henon    | exact RTRL   | 0    | 0.8871 | 0.0007 | window 0.0006593 > 2.0 x best 0.0002521 |
| lstm | henon    | exact RTRL   | 2    | 0.8988 | 0.0007 | window 0.0006938 > 2.0 x best 0.0002844 |
| lstm | henon    | SK-RTRL r=16 | 0    | 0.8871 | 0.0007 | window 0.0006542 > 2.0 x best 0.0002588 |
| lstm | henon    | SK-RTRL r=32 | 0    | 0.8871 | 0.0006 | window 0.0006398 > 2.0 x best 0.0002409 |
| lstm | henon    | SK-RTRL r=32 | 1    | 0.9065 | 0.0006 | window 0.0006472 > 2.0 x best 0.0002589 |
| lstm | henon    | SK-RTRL r=4  | 0    | 0.8871 | 0.0007 | window 0.0007147 > 2.0 x best 0.0002989 |
| lstm | henon    | SK-RTRL r=4  | 1    | 0.9065 | 0.0007 | window 0.0007096 > 2.0 x best 0.0002806 |
| lstm | henon    | SK-RTRL r=4  | 2    | 0.8988 | 0.0007 | window 0.0006975 > 2.0 x best 0.0002675 |
| lstm | henon    | UORO         | 0    | 0.8871 | 0.0350 | window 0.03496 > 2.0 x best 0.01491     |
| lstm | henon    | UORO         | 1    | 0.9065 | 0.0687 | window 0.06871 > 2.0 x best 0.03306     |

Structurally absent estimators (absent by derivation, NOT missing data):

- **KF-RTRL**: Kronecker factorisation needs A_t = D_t W and a rank-1, block-diagonal immediate Jacobian; a GRU's I_t is not block diagonal (reset-gate path through U_h). run_m3.py refuses --cell gru|lstm --algo kfrtrl.
- **TBPTT**: D4 fairness baseline, not part of the D5 estimator comparison; --algo tbptt_nnrnn is tanh-only by construction.

## 2. GRU -- task metric and gradient fidelity

**rotation** (metric = MSE, lower is better; lr 0.001)

| estimator    | MSE (mean ± sd [95% CI])         | Δ vs exact [95% CI]       | d_z   | grad_cos (mean ± sd [CI])     | d_z(cos vs SnAp-1) | median e_t (log10, finite part) | e_t +inf frac | cert viol. | peak MB     | wall s    |
|--------------|----------------------------------|---------------------------|-------|-------------------------------|--------------------|---------------------------------|---------------|------------|-------------|-----------|
| exact RTRL   | 0.1066 ± 0.0107 [0.0739, 0.1393] | ref                       | ref   | n/a                           | n/a                | --                              | --            | --         | 73.7 ± 0.0  | 184 ± 3   |
| SK-RTRL r=32 | 0.1083 ± 0.0110 [0.0748, 0.1419] | -0.0017 [-0.0039, 0.0004] | -1.98 | 0.940 ± 0.048 [0.794, 1.085]  | +8.87              | 1e+21.4 [1e+20.3, 1e+24.7]      | 0.06          | 0          | 160.7 ± 0.0 | 1296 ± 7  |
| SK-RTRL r=16 | 0.1085 ± 0.0094 [0.0799, 0.1370] | -0.0018 [-0.0062, 0.0025] | -1.05 | 0.856 ± 0.087 [0.592, 1.121]  | +4.80              | 1e+24.4 [1e+21.0, 1e+25.5]      | 0.10          | 0          | 137.3 ± 0.0 | 1110 ± 11 |
| SK-RTRL r=4  | 0.1061 ± 0.0094 [0.0775, 0.1347] | 0.0005 [-0.0044, 0.0053]  | +0.25 | 0.731 ± 0.122 [0.359, 1.103]  | +3.00              | 1e+22.8 [1e+21.9, 1e+23.9]      | 0.06          | 0          | 132.4 ± 0.0 | 1040 ± 44 |
| SnAp-1       | 0.0605 ± 0.0064 [0.0411, 0.0799] | 0.0461 [0.0270, 0.0652]   | +5.99 | 0.186 ± 0.027 [0.103, 0.269]  | ref                | 1e+24.2 [1e+23.6, 1e+24.7]      | 0.21          | 0          | 129.7 ± 0.0 | 383 ± 14  |
| RFLO         | 0.1090 ± 0.0113 [0.0745, 0.1435] | -0.0024 [-0.0059, 0.0012] | -1.65 | 0.640 ± 0.067 [0.436, 0.843]  | +8.16              | --                              | --            | --         | 74.2 ± 0.0  | 177 ± 2   |
| UORO         | 0.1233 ± 0.0026 [0.1155, 0.1312] | -0.0167 [-0.0425, 0.0091] | -1.61 | 0.031 ± 0.037 [-0.082, 0.144] | -2.00              | --                              | --            | --         | 74.2 ± 0.0  | 220 ± 20  |

**anbn** (metric = accuracy, higher is better; lr 0.001)

| estimator    | accuracy (mean ± sd [95% CI])    | Δ vs exact [95% CI]        | d_z    | grad_cos (mean ± sd [CI])    | d_z(cos vs SnAp-1) | median e_t (log10, finite part) | e_t +inf frac | cert viol. | peak MB     | wall s    |
|--------------|----------------------------------|----------------------------|--------|------------------------------|--------------------|---------------------------------|---------------|------------|-------------|-----------|
| exact RTRL   | 0.9314 ± 0.0006 [0.9295, 0.9333] | ref                        | ref    | n/a                          | n/a                | --                              | --            | --         | 69.6 ± 0.0  | 194 ± 11  |
| SK-RTRL r=32 | 0.9258 ± 0.0020 [0.9198, 0.9318] | -0.0056 [-0.0109, -0.0004] | -2.66  | 0.937 ± 0.012 [0.901, 0.973] | +5.05              | 1e+11.6 [1e+10.9, 1e+11.7]      | 0.00          | 0          | 149.6 ± 0.0 | 1265 ± 32 |
| SK-RTRL r=16 | 0.9269 ± 0.0008 [0.9245, 0.9293] | -0.0045 [-0.0074, -0.0016] | -3.87  | 0.881 ± 0.019 [0.821, 0.940] | +3.57              | 1e+11.7 [1e+10.0, 1e+12.0]      | 0.00          | 0          | 128.9 ± 0.0 | 1076 ± 43 |
| SK-RTRL r=4  | 0.9284 ± 0.0017 [0.9231, 0.9337] | -0.0030 [-0.0068, 0.0008]  | -1.99  | 0.784 ± 0.043 [0.654, 0.914] | +2.75              | 1e+12.3 [1e+11.9, 1e+12.6]      | 0.00          | 0          | 124.0 ± 0.0 | 1068 ± 31 |
| SnAp-1       | 0.9198 ± 0.0003 [0.9190, 0.9207] | -0.0116 [-0.0142, -0.0090] | -10.99 | 0.457 ± 0.089 [0.186, 0.727] | ref                | 1e+15.0 [1e+15.0, 1e+16.0]      | 0.00          | 0          | 121.6 ± 0.0 | 329 ± 16  |
| RFLO         | 0.9213 ± 0.0068 [0.9006, 0.9420] | -0.0101 [-0.0293, 0.0090]  | -1.31  | 0.423 ± 0.062 [0.233, 0.612] | -0.18              | --                              | --            | --         | 70.0 ± 0.0  | 198 ± 12  |
| UORO         | 0.8957 ± 0.0042 [0.8830, 0.9085] | -0.0357 [-0.0471, -0.0242] | -7.74  | 0.031 ± 0.002 [0.025, 0.036] | -3.89              | --                              | --            | --         | 70.0 ± 0.0  | 285 ± 3   |

**henon** (metric = NMSE, lower is better; lr 0.001)

| estimator    | NMSE (mean ± sd [95% CI])          | Δ vs exact [95% CI]            | d_z   | grad_cos (mean ± sd [CI])    | d_z(cos vs SnAp-1) | median e_t (log10, finite part) | e_t +inf frac | cert viol. | peak MB     | wall s    |
|--------------|------------------------------------|--------------------------------|-------|------------------------------|--------------------|---------------------------------|---------------|------------|-------------|-----------|
| exact RTRL   | 0.0009 ± 3.80e-05 [0.0008, 0.0010] | ref                            | ref   | n/a                          | n/a                | --                              | --            | --         | 67.7 ± 0.0  | 163 ± 14  |
| SK-RTRL r=32 | 0.0009 ± 7.47e-05 [0.0007, 0.0012] | -7.27e-05 [-0.0002, 7.28e-05]  | -1.24 | 0.990 ± 0.002 [0.985, 0.995] | +24.57             | 1e+24.1 [1e+22.8, 1e+24.5]      | 0.50          | 0          | 145.4 ± 0.0 | 1259 ± 10 |
| SK-RTRL r=16 | 0.0009 ± 3.49e-05 [0.0008, 0.0011] | -7.52e-05 [-0.0001, -4.23e-05] | -5.67 | 0.974 ± 0.004 [0.963, 0.985] | +26.50             | 1e+23.4 [1e+22.4, 1e+23.6]      | 0.50          | 0          | 123.4 ± 0.0 | 1136 ± 35 |
| SK-RTRL r=4  | 0.0010 ± 7.49e-05 [0.0008, 0.0012] | -0.0001 [-0.0003, 1.87e-05]    | -2.16 | 0.958 ± 0.001 [0.954, 0.961] | +21.40             | 1e+25.2 [1e+24.3, 1e+25.4]      | 0.50          | 0          | 118.8 ± 0.0 | 1095 ± 18 |
| SnAp-1       | 0.0019 ± 0.0001 [0.0014, 0.0023]   | -0.0010 [-0.0015, -0.0006]     | -5.50 | 0.487 ± 0.018 [0.431, 0.543] | ref                | 1e+29.3 [1e+28.8, 1e+30.6]      | 0.50          | 0          | 117.2 ± 0.0 | 321 ± 45  |
| RFLO         | 0.0049 ± 0.0009 [0.0021, 0.0076]   | -0.0040 [-0.0069, -0.0011]     | -3.41 | 0.419 ± 0.082 [0.169, 0.670] | -0.56              | --                              | --            | --         | 68.1 ± 0.0  | 203 ± 6   |
| UORO         | 0.1576 ± 0.1390 [-0.2653, 0.5805]  | -0.1567 [-0.5797, 0.2663]      | -0.92 | 0.030 ± 0.006 [0.013, 0.047] | -27.99             | --                              | --            | --         | 68.1 ± 0.0  | 231 ± 2   |

**sunspot** (metric = NMSE, lower is better; lr 0.001)

| estimator    | NMSE (mean ± sd [95% CI])        | Δ vs exact [95% CI]        | d_z    | grad_cos (mean ± sd [CI])      | d_z(cos vs SnAp-1) | median e_t (log10, finite part) | e_t +inf frac | cert viol. | peak MB     | wall s    |
|--------------|----------------------------------|----------------------------|--------|--------------------------------|--------------------|---------------------------------|---------------|------------|-------------|-----------|
| exact RTRL   | 0.1287 ± 0.0033 [0.1188, 0.1387] | ref                        | ref    | n/a                            | n/a                | --                              | --            | --         | 67.1 ± 0.0  | 174 ± 4   |
| SK-RTRL r=32 | 0.1297 ± 0.0024 [0.1222, 0.1371] | -0.0009 [-0.0040, 0.0021]  | -0.75  | 0.893 ± 0.012 [0.856, 0.930]   | +10.58             | 1e+26.7 [1e+23.5, 1e+27.2]      | 0.50          | 0          | 144.9 ± 0.0 | 1247 ± 11 |
| SK-RTRL r=16 | 0.1299 ± 0.0024 [0.1227, 0.1372] | -0.0012 [-0.0039, 0.0016]  | -1.06  | 0.747 ± 0.009 [0.720, 0.774]   | +7.50              | 1e+24.1 [1e+23.2, 1e+26.6]      | 0.50          | 0          | 122.9 ± 0.0 | 1153 ± 24 |
| SK-RTRL r=4  | 0.1297 ± 0.0024 [0.1225, 0.1369] | -0.0010 [-0.0046, 0.0026]  | -0.68  | 0.727 ± 0.014 [0.685, 0.770]   | +4.39              | 1e+26.9 [1e+24.9, 1e+27.2]      | 0.50          | 0          | 118.2 ± 0.0 | 1045 ± 55 |
| SnAp-1       | 0.1342 ± 0.0024 [0.1269, 0.1414] | -0.0054 [-0.0082, -0.0027] | -4.96  | 0.431 ± 0.043 [0.301, 0.562]   | ref                | 1e+34.0 [1e+32.8, 1e+34.3]      | 0.50          | 0          | 116.7 ± 0.0 | 343 ± 18  |
| RFLO         | 0.1452 ± 0.0029 [0.1363, 0.1540] | -0.0164 [-0.0175, -0.0154] | -38.74 | 0.260 ± 0.056 [0.091, 0.429]   | -1.96              | --                              | --            | --         | 67.6 ± 0.0  | 217 ± 10  |
| UORO         | 0.5306 ± 0.1360 [0.1166, 0.9445] | -0.4018 [-0.8229, 0.0193]  | -2.37  | -0.006 ± 0.032 [-0.105, 0.093] | -5.28              | --                              | --            | --         | 67.6 ± 0.0  | 233 ± 23  |

Paired-t p values against exact (metric / grad_cos), n=3 seeds:

| task     | estimator    | n | p(metric vs exact) | p(grad_cos vs SnAp-1) |
|----------|--------------|---|--------------------|-----------------------|
| rotation | SK-RTRL r=32 | 3 | 0.076              | 0.004                 |
| rotation | SK-RTRL r=16 | 3 | 0.210              | 0.014                 |
| rotation | SK-RTRL r=4  | 3 | 0.711              | 0.035                 |
| rotation | SnAp-1       | 3 | 0.009              | n/a                   |
| rotation | RFLO         | 3 | 0.104              | 0.005                 |
| rotation | UORO         | 3 | 0.108              | 0.074                 |
| anbn     | SK-RTRL r=32 | 3 | 0.044              | 0.013                 |
| anbn     | SK-RTRL r=16 | 3 | 0.022              | 0.025                 |
| anbn     | SK-RTRL r=4  | 3 | 0.075              | 0.041                 |
| anbn     | SnAp-1       | 3 | 0.003              | n/a                   |
| anbn     | RFLO         | 3 | 0.151              | 0.780                 |
| anbn     | UORO         | 3 | 0.006              | 0.021                 |
| henon    | SK-RTRL r=32 | 3 | 0.165              | <0.001                |
| henon    | SK-RTRL r=16 | 3 | 0.010              | <0.001                |
| henon    | SK-RTRL r=4  | 3 | 0.065              | <0.001                |
| henon    | SnAp-1       | 3 | 0.011              | n/a                   |
| henon    | RFLO         | 3 | 0.028              | 0.432                 |
| henon    | UORO         | 3 | 0.252              | <0.001                |
| sunspot  | SK-RTRL r=32 | 3 | 0.325              | 0.003                 |
| sunspot  | SK-RTRL r=16 | 3 | 0.208              | 0.006                 |
| sunspot  | SK-RTRL r=4  | 3 | 0.363              | 0.017                 |
| sunspot  | SnAp-1       | 3 | 0.013              | n/a                   |
| sunspot  | RFLO         | 3 | <0.001             | 0.077                 |
| sunspot  | UORO         | 3 | 0.055              | 0.012                 |

> With 3 seeds the 95 % CI half-width is 4.30 s/sqrt(3) = 2.48 s and the paired-t has 2 df: read these intervals as a spread indicator, not as the 10-seed D4 inference. The D5 grid was sized for a structural applicability claim, not for a powered head-to-head.

## 2'. LSTM -- task metric and gradient fidelity

**rotation** (metric = MSE, lower is better; lr 0.001)

| estimator    | MSE (mean ± sd [95% CI])         | Δ vs exact [95% CI]       | d_z   | grad_cos (mean ± sd [CI])    | d_z(cos vs SnAp-1) | median e_t (log10, finite part) | e_t +inf frac | cert viol. | peak MB     | wall s         |
|--------------|----------------------------------|---------------------------|-------|------------------------------|--------------------|---------------------------------|---------------|------------|-------------|----------------|
| exact RTRL   | 0.1083 ± 0.0100 [0.0780, 0.1386] | ref                       | ref   | n/a                          | n/a                | --                              | --            | --         | 166.3 ± 0.0 | 53 ± 8.49e-01  |
| SK-RTRL r=32 | 0.1087 ± 0.0107 [0.0763, 0.1412] | -0.0004 [-0.0042, 0.0034] | -0.27 | 0.979 ± 0.005 [0.963, 0.995] | +16.17             | 1e+21.8 [1e+19.6, 1e+22.3]      | 0.42          | 0          | 334.8 ± 0.0 | 600 ± 2        |
| SK-RTRL r=16 | 0.1158 ± 0.0018 [0.1103, 0.1213] | -0.0075 [-0.0352, 0.0201] | -0.68 | 0.960 ± 0.016 [0.911, 1.010] | +14.42             | 1e+20.5 [1e+19.1, 1e+21.1]      | 0.46          | 0          | 325.2 ± 0.0 | 498 ± 7        |
| SK-RTRL r=4  | 0.1087 ± 0.0073 [0.0863, 0.1310] | -0.0004 [-0.0124, 0.0117] | -0.08 | 0.873 ± 0.031 [0.778, 0.967] | +22.77             | 1e+21.7 [1e+18.8, 1e+24.7]      | 0.38          | 0          | 318.5 ± 0.0 | 457 ± 4        |
| SnAp-1       | 0.0616 ± 0.0027 [0.0533, 0.0699] | 0.0467 [0.0188, 0.0747]   | +4.15 | 0.334 ± 0.027 [0.251, 0.418] | ref                | 1e+19.7 [1e+19.6, 1e+19.8]      | 0.44          | 0          | 315.5 ± 0.0 | 100 ± 5.51e-01 |
| RFLO         | 0.1115 ± 0.0097 [0.0819, 0.1410] | -0.0032 [-0.0070, 0.0007] | -2.03 | 0.545 ± 0.060 [0.363, 0.728] | +3.42              | --                              | --            | --         | 167.5 ± 0.0 | 59 ± 6.46e-01  |
| UORO         | 0.1239 ± 0.0015 [0.1194, 0.1285] | -0.0156 [-0.0414, 0.0102] | -1.51 | 0.022 ± 0.004 [0.009, 0.034] | -8.21              | --                              | --            | --         | 167.0 ± 0.0 | 67 ± 2.86e-01  |

**anbn** (metric = accuracy, higher is better; lr 0.001)

| estimator    | accuracy (mean ± sd [95% CI])    | Δ vs exact [95% CI]        | d_z   | grad_cos (mean ± sd [CI])       | d_z(cos vs SnAp-1) | median e_t (log10, finite part) | e_t +inf frac | cert viol. | peak MB     | wall s        |
|--------------|----------------------------------|----------------------------|-------|---------------------------------|--------------------|---------------------------------|---------------|------------|-------------|---------------|
| exact RTRL   | 0.9462 ± 0.0018 [0.9407, 0.9516] | ref                        | ref   | n/a                             | n/a                | --                              | --            | --         | 154.2 ± 0.0 | 58 ± 2.95e-01 |
| SK-RTRL r=32 | 0.9459 ± 0.0004 [0.9447, 0.9470] | -0.0003 [-0.0046, 0.0040]  | -0.17 | 0.991 ± 7.73e-04 [0.989, 0.993] | +2.80              | 1e+17.3 [1e+16.9, 1e+17.4]      | 0.00          | 0          | 310.3 ± 0.0 | 579 ± 3       |
| SK-RTRL r=16 | 0.9458 ± 0.0007 [0.9438, 0.9479] | -0.0003 [-0.0041, 0.0035]  | -0.20 | 0.967 ± 0.009 [0.941, 0.993]    | +1.51              | 1e+17.4 [1e+17.1, 1e+17.8]      | 0.00          | 0          | 301.3 ± 0.0 | 477 ± 2       |
| SK-RTRL r=4  | 0.9443 ± 0.0011 [0.9411, 0.9476] | -0.0018 [-0.0051, 0.0015]  | -1.36 | 0.969 ± 0.011 [0.936, 1.001]    | +3.44              | 1e+16.4 [1e+16.1, 1e+16.7]      | 0.00          | 0          | 294.7 ± 0.0 | 448 ± 2       |
| SnAp-1       | 0.9455 ± 0.0006 [0.9438, 0.9473] | -0.0006 [-0.0047, 0.0034]  | -0.39 | 0.909 ± 0.023 [0.837, 0.980]    | ref                | 1e+17.2 [1e+16.8, 1e+17.6]      | 0.00          | 0          | 291.3 ± 0.0 | 97 ± 1        |
| RFLO         | 0.9368 ± 0.0006 [0.9350, 0.9387] | -0.0093 [-0.0166, -0.0020] | -3.18 | 0.432 ± 0.050 [0.281, 0.583]    | -8.72              | --                              | --            | --         | 155.3 ± 0.0 | 64 ± 1        |
| UORO         | 0.8951 ± 0.0033 [0.8850, 0.9051] | -0.0511 [-0.0660, -0.0362] | -8.52 | 0.041 ± 0.003 [0.031, 0.050]    | -27.16             | --                              | --            | --         | 154.8 ± 0.0 | 71 ± 1.26e-01 |

**henon** (metric = NMSE, lower is better; lr 0.001)

| estimator    | NMSE (mean ± sd [95% CI])          | Δ vs exact [95% CI]             | d_z   | grad_cos (mean ± sd [CI])    | d_z(cos vs SnAp-1) | median e_t (log10, finite part) | e_t +inf frac | cert viol. | peak MB     | wall s         |
|--------------|------------------------------------|---------------------------------|-------|------------------------------|--------------------|---------------------------------|---------------|------------|-------------|----------------|
| exact RTRL   | 0.0007 ± 3.10e-05 [0.0006, 0.0008] | ref                             | ref   | n/a                          | n/a                | --                              | --            | --         | 150.7 ± 0.0 | 52 ± 2.05e-01  |
| SK-RTRL r=32 | 0.0007 ± 1.94e-05 [0.0006, 0.0007] | -2.88e-08 [-6.43e-05, 6.42e-05] | -0.00 | 0.993 ± 0.001 [0.990, 0.997] | +9.77              | 1e+34.4 [1e+34.1, 1e+34.6]      | 0.50          | 0          | 301.1 ± 0.0 | 611 ± 8.88e-01 |
| SK-RTRL r=16 | 0.0007 ± 1.80e-05 [0.0006, 0.0007] | -2.18e-05 [-0.0001, 7.72e-05]   | -0.55 | 0.986 ± 0.004 [0.972, 0.999] | +9.91              | 1e+34.6 [1e+33.8, 1e+34.7]      | 0.50          | 0          | 293.1 ± 0.0 | 495 ± 2        |
| SK-RTRL r=4  | 0.0007 ± 7.20e-06 [0.0007, 0.0007] | -5.03e-05 [-0.0002, 5.96e-05]   | -1.14 | 0.982 ± 0.002 [0.975, 0.989] | +9.72              | 1e+34.2 [1e+33.8, 1e+34.5]      | 0.50          | 0          | 286.6 ± 0.0 | 462 ± 6.38e-01 |
| SnAp-1       | 0.0009 ± 5.46e-05 [0.0007, 0.0011] | -0.0002 [-0.0005, -3.63e-06]    | -2.52 | 0.667 ± 0.027 [0.584, 0.751] | ref                | 1e+36.4 (n=1)                   | 0.67          | 0          | 283.9 ± 0.0 | 99 ± 3.35e-01  |
| RFLO         | 0.0034 ± 0.0002 [0.0027, 0.0040]   | -0.0027 [-0.0034, -0.0020]      | -9.61 | 0.397 ± 0.022 [0.330, 0.463] | -7.04              | --                              | --            | --         | 151.8 ± 0.0 | 58 ± 3.20e-01  |
| UORO         | 0.0519 ± 0.0138 [0.0100, 0.0938]   | -0.0512 [-0.0932, -0.0093]      | -3.03 | 0.036 ± 0.002 [0.028, 0.043] | -19.37             | --                              | --            | --         | 151.3 ± 0.0 | 65 ± 2.14e-01  |

**sunspot** (metric = NMSE, lower is better; lr 0.001)

| estimator    | NMSE (mean ± sd [95% CI])        | Δ vs exact [95% CI]        | d_z   | grad_cos (mean ± sd [CI])    | d_z(cos vs SnAp-1) | median e_t (log10, finite part)  | e_t +inf frac | cert viol. | peak MB     | wall s        |
|--------------|----------------------------------|----------------------------|-------|------------------------------|--------------------|----------------------------------|---------------|------------|-------------|---------------|
| exact RTRL   | 0.1325 ± 0.0023 [0.1254, 0.1395] | ref                        | ref   | n/a                          | n/a                | --                               | --            | --         | 150.2 ± 0.0 | 51 ± 4.55e-01 |
| SK-RTRL r=32 | 0.1320 ± 0.0019 [0.1261, 0.1379] | 0.0004 [-0.0009, 0.0018]   | +0.83 | 0.991 ± 0.001 [0.986, 0.995] | +25.65             | 1e+35.1 [1e+34.7, 1e+35.2]       | 0.50          | 0          | 300.6 ± 0.0 | 594 ± 1       |
| SK-RTRL r=16 | 0.1328 ± 0.0032 [0.1231, 0.1425] | -0.0003 [-0.0032, 0.0026]  | -0.26 | 0.969 ± 0.005 [0.954, 0.984] | +17.58             | 1e+35.6 [1e+34.5, 1e+36.8]       | 0.50          | 0          | 292.6 ± 0.0 | 488 ± 2       |
| SK-RTRL r=4  | 0.1322 ± 0.0024 [0.1248, 0.1396] | 0.0003 [-0.0005, 0.0011]   | +0.84 | 0.959 ± 0.015 [0.913, 1.006] | +10.80             | 1e+35.4 [1e+34.7, 1e+36.0] (n=2) | 0.58          | 0          | 286.1 ± 0.0 | 452 ± 3       |
| SnAp-1       | 0.1359 ± 0.0035 [0.1253, 0.1465] | -0.0034 [-0.0071, 0.0003]  | -2.29 | 0.635 ± 0.011 [0.602, 0.668] | ref                | 0 (all finite = 0)               | 0.75          | 0          | 283.3 ± 0.0 | 96 ± 2        |
| RFLO         | 0.1428 ± 0.0032 [0.1331, 0.1525] | -0.0103 [-0.0140, -0.0066] | -6.90 | 0.334 ± 0.070 [0.120, 0.548] | -3.03              | --                               | --            | --         | 151.3 ± 0.0 | 58 ± 5.81e-01 |
| UORO         | 0.1580 ± 0.0046 [0.1440, 0.1721] | -0.0255 [-0.0333, -0.0177] | -8.14 | 0.022 ± 0.001 [0.019, 0.025] | -42.65             | --                               | --            | --         | 150.8 ± 0.0 | 65 ± 2.20e-01 |

Paired-t p values against exact (metric / grad_cos), n=3 seeds:

| task     | estimator    | n | p(metric vs exact) | p(grad_cos vs SnAp-1) |
|----------|--------------|---|--------------------|-----------------------|
| rotation | SK-RTRL r=32 | 3 | 0.689              | 0.001                 |
| rotation | SK-RTRL r=16 | 3 | 0.363              | 0.002                 |
| rotation | SK-RTRL r=4  | 3 | 0.907              | <0.001                |
| rotation | SnAp-1       | 3 | 0.019              | n/a                   |
| rotation | RFLO         | 3 | 0.072              | 0.027                 |
| rotation | UORO         | 3 | 0.121              | 0.005                 |
| anbn     | SK-RTRL r=32 | 3 | 0.791              | 0.040                 |
| anbn     | SK-RTRL r=16 | 3 | 0.757              | 0.120                 |
| anbn     | SK-RTRL r=4  | 3 | 0.143              | 0.027                 |
| anbn     | SnAp-1       | 3 | 0.568              | n/a                   |
| anbn     | RFLO         | 3 | 0.031              | 0.004                 |
| anbn     | UORO         | 3 | 0.005              | <0.001                |
| henon    | SK-RTRL r=32 | 3 | 0.999              | 0.003                 |
| henon    | SK-RTRL r=16 | 3 | 0.444              | 0.003                 |
| henon    | SK-RTRL r=4  | 3 | 0.188              | 0.004                 |
| henon    | SnAp-1       | 3 | 0.049              | n/a                   |
| henon    | RFLO         | 3 | 0.004              | 0.007                 |
| henon    | UORO         | 3 | 0.034              | <0.001                |
| sunspot  | SK-RTRL r=32 | 3 | 0.288              | <0.001                |
| sunspot  | SK-RTRL r=16 | 3 | 0.692              | 0.001                 |
| sunspot  | SK-RTRL r=4  | 3 | 0.284              | 0.003                 |
| sunspot  | SnAp-1       | 3 | 0.058              | n/a                   |
| sunspot  | RFLO         | 3 | 0.007              | 0.035                 |
| sunspot  | UORO         | 3 | 0.005              | <0.001                |

> With 3 seeds the 95 % CI half-width is 4.30 s/sqrt(3) = 2.48 s and the paired-t has 2 df: read these intervals as a spread indicator, not as the 10-seed D4 inference. The D5 grid was sized for a structural applicability claim, not for a powered head-to-head.

## 3. Certificate on a gated cell vs the tanh cell

`cert_violations` = `cert_summary.n_bound_violation`: steps where the certified bound fell below the measured gradient error, i.e. Theorem 1 failed. `frac rho_bar<1` is the fraction of steps where the contraction condition of Corollary 1 held (the plateau regime). `frac rel<1` is the fraction where the bound was below the gradient norm, i.e. where the certificate was non-vacuous as a relative statement. `e_t +inf frac` is the fraction of window steps where the bound overflowed float64: the bound accumulates per-step factors rho_bar > 1 over a 20 000-step trajectory, so on the long-memory tasks it leaves the representable range. **This is a property of the bound, not of the cell** -- the tanh rows overflow at the same rate, which is why both cells are printed in the same table. An overflowed bound cannot be violated, so it is counted as vacuous-but-valid, never as a pass.

**rotation**

| estimator    | cell     | n  | cert viol. | median e_t (log10, finite part)   | e_t +inf frac | mean true_E | mean rho_bar | frac rho_bar<1 | frac rel<1 | frac tight<=10 |
|--------------|----------|----|------------|-----------------------------------|---------------|-------------|--------------|----------------|------------|----------------|
| SK-RTRL r=32 | tanh RNN | 10 | 0          | 1e+23.9 [1e+23.4, 1e+24.5] (n=10) | 0.11          | 15.759      | 4.51         | 0.000          | 0.000      | 0.032          |
| SK-RTRL r=32 | GRU      | 3  | 0          | 1e+21.4 [1e+20.3, 1e+24.7]        | 0.06          | 2.461       | 4.08         | 0.000          | 0.000      | 0.036          |
| SK-RTRL r=32 | LSTM     | 3  | 0          | 1e+21.8 [1e+19.6, 1e+22.3]        | 0.42          | 5.559       | 8.02         | 0.000          | 0.000      | 0.032          |
| SK-RTRL r=16 | tanh RNN | 10 | 0          | 1e+23.3 [1e+20.9, 1e+26.1] (n=10) | 0.10          | 17.710      | 4.38         | 0.000          | 0.000      | 0.032          |
| SK-RTRL r=16 | GRU      | 3  | 0          | 1e+24.4 [1e+21.0, 1e+25.5]        | 0.10          | 5.954       | 4.54         | 0.000          | 0.000      | 0.037          |
| SK-RTRL r=16 | LSTM     | 3  | 0          | 1e+20.5 [1e+19.1, 1e+21.1]        | 0.46          | 25.068      | 8.78         | 0.000          | 0.000      | 0.032          |
| SK-RTRL r=4  | tanh RNN | 10 | 0          | 1e+24.2 [1e+24.1, 1e+24.6] (n=10) | 0.18          | 13.050      | 5.05         | 0.000          | 0.000      | 0.032          |
| SK-RTRL r=4  | GRU      | 3  | 0          | 1e+22.8 [1e+21.9, 1e+23.9]        | 0.06          | 5.678       | 4.23         | 0.000          | 0.000      | 0.037          |
| SK-RTRL r=4  | LSTM     | 3  | 0          | 1e+21.7 [1e+18.8, 1e+24.7]        | 0.38          | 12.513      | 7.73         | 0.000          | 0.000      | 0.032          |
| SnAp-1       | tanh RNN | 10 | 0          | 1e+25.1 [1e+20.9, 1e+25.3] (n=10) | 0.26          | 52.160      | 5.48         | 0.000          | 0.000      | 0.032          |
| SnAp-1       | GRU      | 3  | 0          | 1e+24.2 [1e+23.6, 1e+24.7]        | 0.21          | 11.761      | 5.21         | 0.000          | 0.000      | 0.035          |
| SnAp-1       | LSTM     | 3  | 0          | 1e+19.7 [1e+19.6, 1e+19.8]        | 0.44          | 13.432      | 8.22         | 0.000          | 0.000      | 0.032          |

**anbn**

| estimator    | cell     | n  | cert viol. | median e_t (log10, finite part)  | e_t +inf frac | mean true_E | mean rho_bar | frac rho_bar<1 | frac rel<1 | frac tight<=10 |
|--------------|----------|----|------------|----------------------------------|---------------|-------------|--------------|----------------|------------|----------------|
| SK-RTRL r=32 | tanh RNN | 10 | 0          | 1e+11.3 [1e+9.0, 1e+13.5] (n=10) | 0.00          | 2.644       | 3.82         | 0.000          | 0.000      | 0.000          |
| SK-RTRL r=32 | GRU      | 3  | 0          | 1e+11.6 [1e+10.9, 1e+11.7]       | 0.00          | 2.228       | 4.20         | 0.000          | 0.000      | 0.000          |
| SK-RTRL r=32 | LSTM     | 3  | 0          | 1e+17.3 [1e+16.9, 1e+17.4]       | 0.00          | 3.989       | 7.08         | 0.000          | 0.000      | 0.000          |
| SK-RTRL r=16 | tanh RNN | 10 | 0          | 1e+10.8 [1e+9.7, 1e+12.1] (n=10) | 0.00          | 4.184       | 3.71         | 0.000          | 0.000      | 0.000          |
| SK-RTRL r=16 | GRU      | 3  | 0          | 1e+11.7 [1e+10.0, 1e+12.0]       | 0.00          | 3.075       | 4.06         | 0.000          | 0.000      | 0.000          |
| SK-RTRL r=16 | LSTM     | 3  | 0          | 1e+17.4 [1e+17.1, 1e+17.8]       | 0.00          | 7.034       | 7.32         | 0.000          | 0.000      | 0.000          |
| SK-RTRL r=4  | tanh RNN | 10 | 0          | 1e+10.9 [1e+9.4, 1e+12.9] (n=10) | 0.00          | 5.107       | 3.66         | 0.000          | 0.000      | 0.000          |
| SK-RTRL r=4  | GRU      | 3  | 0          | 1e+12.3 [1e+11.9, 1e+12.6]       | 0.00          | 4.448       | 4.33         | 0.000          | 0.000      | 0.000          |
| SK-RTRL r=4  | LSTM     | 3  | 0          | 1e+16.4 [1e+16.1, 1e+16.7]       | 0.00          | 6.340       | 6.82         | 0.000          | 0.000      | 0.000          |
| SnAp-1       | tanh RNN | 10 | 0          | 1e+13.3 [1e+9.9, 1e+15.6] (n=10) | 0.00          | 28.782      | 4.19         | 0.000          | 0.000      | 0.000          |
| SnAp-1       | GRU      | 3  | 0          | 1e+15.0 [1e+15.0, 1e+16.0]       | 0.00          | 11.451      | 5.30         | 0.000          | 0.000      | 0.000          |
| SnAp-1       | LSTM     | 3  | 0          | 1e+17.2 [1e+16.8, 1e+17.6]       | 0.00          | 19.289      | 7.44         | 0.000          | 0.000      | 0.000          |

**henon**

| estimator    | cell     | n  | cert viol. | median e_t (log10, finite part)   | e_t +inf frac | mean true_E | mean rho_bar | frac rho_bar<1 | frac rel<1 | frac tight<=10 |
|--------------|----------|----|------------|-----------------------------------|---------------|-------------|--------------|----------------|------------|----------------|
| SK-RTRL r=32 | tanh RNN | 10 | 0          | 1e+25.9 [1e+24.9, 1e+26.5] (n=10) | 0.50          | 3.852       | 4.08         | 0.000          | 0.000      | 0.014          |
| SK-RTRL r=32 | GRU      | 3  | 0          | 1e+24.1 [1e+22.8, 1e+24.5]        | 0.50          | 1.169       | 3.24         | 0.000          | 6.26e-04   | 0.015          |
| SK-RTRL r=32 | LSTM     | 3  | 0          | 1e+34.4 [1e+34.1, 1e+34.6]        | 0.50          | 0.839       | 5.42         | 0.000          | 3.38e-05   | 0.010          |
| SK-RTRL r=16 | tanh RNN | 10 | 0          | 1e+25.1 [1e+24.4, 1e+25.5] (n=10) | 0.50          | 4.642       | 4.07         | 0.000          | 0.000      | 0.014          |
| SK-RTRL r=16 | GRU      | 3  | 0          | 1e+23.4 [1e+22.4, 1e+23.6]        | 0.50          | 1.495       | 3.17         | 0.000          | 1.02e-04   | 0.015          |
| SK-RTRL r=16 | LSTM     | 3  | 0          | 1e+34.6 [1e+33.8, 1e+34.7]        | 0.50          | 1.089       | 5.39         | 0.000          | 0.000      | 0.010          |
| SK-RTRL r=4  | tanh RNN | 10 | 0          | 1e+24.6 [1e+23.8, 1e+25.8] (n=10) | 0.50          | 4.695       | 3.98         | 0.000          | 0.000      | 0.014          |
| SK-RTRL r=4  | GRU      | 3  | 0          | 1e+25.2 [1e+24.3, 1e+25.4]        | 0.50          | 1.720       | 3.42         | 0.000          | 6.77e-05   | 0.015          |
| SK-RTRL r=4  | LSTM     | 3  | 0          | 1e+34.2 [1e+33.8, 1e+34.5]        | 0.50          | 1.130       | 5.35         | 0.000          | 0.000      | 0.010          |
| SnAp-1       | tanh RNN | 10 | 0          | 1e+27.6 [1e+21.6, 1e+30.0] (n=10) | 0.50          | 12.227      | 4.29         | 0.000          | 0.000      | 0.013          |
| SnAp-1       | GRU      | 3  | 0          | 1e+29.3 [1e+28.8, 1e+30.6]        | 0.50          | 3.176       | 4.02         | 0.000          | 0.000      | 0.012          |
| SnAp-1       | LSTM     | 3  | 0          | 1e+36.4 (n=1)                     | 0.67          | 3.087       | 5.87         | 0.000          | 0.000      | 0.010          |

**sunspot**

| estimator    | cell     | n  | cert viol. | median e_t (log10, finite part)   | e_t +inf frac | mean true_E | mean rho_bar | frac rho_bar<1 | frac rel<1 | frac tight<=10 |
|--------------|----------|----|------------|-----------------------------------|---------------|-------------|--------------|----------------|------------|----------------|
| SK-RTRL r=32 | tanh RNN | 10 | 0          | 1e+32.8 [1e+32.5, 1e+32.9] (n=10) | 0.50          | 10.992      | 4.56         | 0.000          | 0.000      | 0.010          |
| SK-RTRL r=32 | GRU      | 3  | 0          | 1e+26.7 [1e+23.5, 1e+27.2]        | 0.50          | 2.997       | 3.40         | 0.000          | 0.001      | 0.017          |
| SK-RTRL r=32 | LSTM     | 3  | 0          | 1e+35.1 [1e+34.7, 1e+35.2]        | 0.50          | 2.200       | 5.28         | 0.000          | 0.000      | 0.011          |
| SK-RTRL r=16 | tanh RNN | 10 | 0          | 1e+32.8 [1e+32.5, 1e+33.0] (n=10) | 0.50          | 13.627      | 4.56         | 0.000          | 0.000      | 0.010          |
| SK-RTRL r=16 | GRU      | 3  | 0          | 1e+24.1 [1e+23.2, 1e+26.6]        | 0.50          | 4.254       | 3.31         | 0.000          | 3.55e-04   | 0.017          |
| SK-RTRL r=16 | LSTM     | 3  | 0          | 1e+35.6 [1e+34.5, 1e+36.8]        | 0.50          | 4.049       | 5.36         | 0.000          | 0.000      | 0.011          |
| SK-RTRL r=4  | tanh RNN | 10 | 0          | 1e+32.8 [1e+32.5, 1e+33.0] (n=10) | 0.50          | 14.024      | 4.56         | 0.000          | 0.000      | 0.010          |
| SK-RTRL r=4  | GRU      | 3  | 0          | 1e+26.9 [1e+24.9, 1e+27.2]        | 0.50          | 4.037       | 3.36         | 0.000          | 4.06e-04   | 0.016          |
| SK-RTRL r=4  | LSTM     | 3  | 0          | 1e+35.4 [1e+34.7, 1e+36.0] (n=2)  | 0.58          | 3.706       | 5.43         | 0.000          | 0.000      | 0.011          |
| SnAp-1       | tanh RNN | 10 | 0          | 1e+33.1 [1e+32.7, 1e+33.4] (n=10) | 0.50          | 7.575       | 4.61         | 0.000          | 0.000      | 0.010          |
| SnAp-1       | GRU      | 3  | 0          | 1e+34.0 [1e+32.8, 1e+34.3]        | 0.50          | 6.228       | 4.79         | 0.000          | 0.000      | 0.013          |
| SnAp-1       | LSTM     | 3  | 0          | 0 (all finite = 0)                | 0.75          | 14.474      | 6.14         | 0.000          | 0.000      | 0.011          |

## 4. Sketch spectrum: does absorbing I^perp_t raise the effective rank?

Read at the LAST residual checkpoint (`frac`=1.0) of each run. `stable_rank` = (||E||_F/||E||_2)^2 of the residual, `sketch_eff` = ||E||_F normalised by the sketch's own factor norm, `mass_topk` = fraction of the residual's squared spectrum in its top k singular values (so a SMALLER mass_top4 means a flatter, harder spectrum), `best_rank_err` = the rank-r truncation error the sketch is chasing.

**SK-RTRL r=4**

| task     | cell     | n  | stable_rank          | sketch_eff           | mass_top4            | mass_top8            | mass_top16           | mass_top32              | best_rank_err        | ||E||_F               |
|----------|----------|----|----------------------|----------------------|----------------------|----------------------|----------------------|-------------------------|----------------------|-----------------------|
| rotation | tanh RNN | 10 | 2.615 ± 1.313 (n=10) | 2.062 ± 0.851 (n=10) | 0.770 ± 0.173 (n=10) | 0.858 ± 0.112 (n=10) | 0.929 ± 0.055 (n=10) | 0.981 ± 0.015 (n=10)    | 7.397 ± 2.412 (n=10) | 16.248 ± 9.955 (n=10) |
| rotation | GRU      | 3  | 1.834 ± 0.367        | 1.642 ± 0.209        | 0.809 ± 0.077        | 0.878 ± 0.062        | 0.940 ± 0.034        | 0.986 ± 0.008           | 3.472 ± 0.696        | 5.623 ± 0.944         |
| rotation | LSTM     | 3  | 2.035 ± 0.607        | 1.783 ± 0.545        | 0.875 ± 0.066        | 0.956 ± 0.035        | 0.987 ± 0.014        | 0.999 ± 0.002           | 5.674 ± 6.281        | 9.691 ± 10.689        |
| anbn     | tanh RNN | 10 | 1.822 ± 0.388 (n=10) | 1.390 ± 0.205 (n=10) | 0.841 ± 0.078 (n=10) | 0.900 ± 0.068 (n=10) | 0.927 ± 0.065 (n=10) | 0.936 ± 0.063 (n=10)    | 3.485 ± 1.017 (n=10) | 4.825 ± 1.517 (n=10)  |
| anbn     | GRU      | 3  | 1.751 ± 0.128        | 2.099 ± 0.074        | 0.819 ± 0.035        | 0.887 ± 0.046        | 0.931 ± 0.054        | 0.954 ± 0.058           | 2.203 ± 0.067        | 4.623 ± 0.222         |
| anbn     | LSTM     | 3  | 1.836 ± 0.031        | 1.287 ± 0.056        | 0.805 ± 0.034        | 0.885 ± 0.044        | 0.932 ± 0.054        | 0.954 ± 0.058           | 4.772 ± 0.712        | 6.178 ± 1.190         |
| henon    | tanh RNN | 10 | 2.127 ± 0.197 (n=10) | 1.106 ± 0.021 (n=10) | 0.800 ± 0.033 (n=10) | 0.913 ± 0.018 (n=10) | 0.985 ± 0.005 (n=10) | 1.000 ± 6.77e-05 (n=10) | 6.171 ± 0.509 (n=10) | 6.819 ± 0.531 (n=10)  |
| henon    | GRU      | 3  | 3.226 ± 0.628        | 1.139 ± 0.059        | 0.580 ± 0.059        | 0.713 ± 0.049        | 0.855 ± 0.028        | 0.966 ± 0.007           | 1.978 ± 0.142        | 2.245 ± 0.061         |
| henon    | LSTM     | 3  | 3.992 ± 0.548        | 1.062 ± 0.002        | 0.566 ± 0.043        | 0.728 ± 0.031        | 0.880 ± 0.017        | 0.976 ± 0.005           | 1.339 ± 0.049        | 1.422 ± 0.051         |
| sunspot  | tanh RNN | 10 | 1.163 ± 0.049 (n=10) | 2.210 ± 0.249 (n=10) | 0.937 ± 0.015 (n=10) | 0.957 ± 0.011 (n=10) | 0.977 ± 0.006 (n=10) | 0.994 ± 0.002 (n=10)    | 8.760 ± 0.508 (n=10) | 19.391 ± 2.638 (n=10) |
| sunspot  | GRU      | 3  | 2.132 ± 0.154        | 1.780 ± 0.215        | 0.758 ± 0.057        | 0.846 ± 0.038        | 0.923 ± 0.020        | 0.981 ± 0.005           | 3.148 ± 0.252        | 5.627 ± 0.945         |
| sunspot  | LSTM     | 3  | 1.420 ± 0.185        | 1.689 ± 0.193        | 0.935 ± 0.012        | 0.963 ± 0.006        | 0.984 ± 0.002        | 0.997 ± 4.41e-04        | 2.608 ± 0.196        | 4.437 ± 0.825         |

**SK-RTRL r=16**

| task     | cell     | n  | stable_rank          | sketch_eff           | mass_top4            | mass_top8            | mass_top16           | mass_top32              | best_rank_err        | ||E||_F                |
|----------|----------|----|----------------------|----------------------|----------------------|----------------------|----------------------|-------------------------|----------------------|------------------------|
| rotation | tanh RNN | 10 | 2.726 ± 1.959 (n=10) | 3.830 ± 1.677 (n=10) | 0.773 ± 0.214 (n=10) | 0.858 ± 0.134 (n=10) | 0.937 ± 0.058 (n=10) | 0.988 ± 0.011 (n=10)    | 5.088 ± 1.653 (n=10) | 19.610 ± 10.767 (n=10) |
| rotation | GRU      | 3  | 2.532 ± 0.603        | 3.190 ± 0.836        | 0.744 ± 0.121        | 0.858 ± 0.088        | 0.940 ± 0.046        | 0.988 ± 0.013           | 1.453 ± 0.039        | 4.666 ± 1.350          |
| rotation | LSTM     | 3  | 1.825 ± 0.649        | 10.237 ± 6.946       | 0.908 ± 0.057        | 0.972 ± 0.016        | 0.994 ± 0.002        | 1.000 ± 2.35e-04        | 5.570 ± 4.302        | 83.591 ± 98.244        |
| anbn     | tanh RNN | 10 | 1.865 ± 0.286 (n=10) | 7.844 ± 2.728 (n=10) | 0.842 ± 0.070 (n=10) | 0.902 ± 0.065 (n=10) | 0.928 ± 0.063 (n=10) | 0.936 ± 0.063 (n=10)    | 0.548 ± 0.192 (n=10) | 3.898 ± 0.998 (n=10)   |
| anbn     | GRU      | 3  | 1.759 ± 0.051        | 3.390 ± 0.090        | 0.815 ± 0.051        | 0.887 ± 0.055        | 0.931 ± 0.058        | 0.953 ± 0.059           | 0.938 ± 0.008        | 3.179 ± 0.097          |
| anbn     | LSTM     | 3  | 2.131 ± 0.126        | 2.789 ± 0.288        | 0.786 ± 0.064        | 0.878 ± 0.057        | 0.930 ± 0.059        | 0.953 ± 0.059           | 2.561 ± 0.177        | 7.163 ± 1.033          |
| henon    | tanh RNN | 10 | 2.014 ± 0.273 (n=10) | 3.758 ± 0.348 (n=10) | 0.803 ± 0.038 (n=10) | 0.911 ± 0.021 (n=10) | 0.984 ± 0.005 (n=10) | 1.000 ± 6.24e-05 (n=10) | 1.801 ± 0.312 (n=10) | 6.675 ± 0.675 (n=10)   |
| henon    | GRU      | 3  | 3.139 ± 0.752        | 1.726 ± 0.013        | 0.567 ± 0.023        | 0.703 ± 0.014        | 0.845 ± 0.009        | 0.962 ± 0.003           | 1.147 ± 0.142        | 1.981 ± 0.254          |
| henon    | LSTM     | 3  | 3.879 ± 0.452        | 2.008 ± 0.163        | 0.578 ± 0.062        | 0.742 ± 0.048        | 0.887 ± 0.032        | 0.978 ± 0.009           | 0.686 ± 0.068        | 1.374 ± 0.133          |
| sunspot  | tanh RNN | 10 | 1.172 ± 0.048 (n=10) | 3.571 ± 0.325 (n=10) | 0.935 ± 0.014 (n=10) | 0.956 ± 0.010 (n=10) | 0.977 ± 0.005 (n=10) | 0.993 ± 0.001 (n=10)    | 5.275 ± 0.277 (n=10) | 18.826 ± 1.824 (n=10)  |
| sunspot  | GRU      | 3  | 1.872 ± 0.253        | 3.625 ± 0.356        | 0.808 ± 0.038        | 0.876 ± 0.026        | 0.937 ± 0.012        | 0.985 ± 0.003           | 1.816 ± 0.048        | 6.584 ± 0.684          |
| sunspot  | LSTM     | 3  | 1.315 ± 0.124        | 3.492 ± 0.615        | 0.941 ± 0.020        | 0.966 ± 0.011        | 0.985 ± 0.006        | 0.997 ± 0.001           | 1.376 ± 0.176        | 4.911 ± 1.533          |

**SK-RTRL r=32**

| task     | cell     | n  | stable_rank          | sketch_eff             | mass_top4            | mass_top8            | mass_top16           | mass_top32              | best_rank_err        | ||E||_F               |
|----------|----------|----|----------------------|------------------------|----------------------|----------------------|----------------------|-------------------------|----------------------|-----------------------|
| rotation | tanh RNN | 10 | 1.598 ± 0.610 (n=10) | 6.062 ± 1.799 (n=10)   | 0.893 ± 0.104 (n=10) | 0.938 ± 0.060 (n=10) | 0.970 ± 0.029 (n=10) | 0.993 ± 0.008 (n=10)    | 2.560 ± 0.660 (n=10) | 14.718 ± 3.328 (n=10) |
| rotation | GRU      | 3  | 2.634 ± 1.687        | 4.868 ± 1.197          | 0.752 ± 0.180        | 0.849 ± 0.113        | 0.932 ± 0.052        | 0.984 ± 0.013           | 0.727 ± 0.250        | 3.616 ± 1.855         |
| rotation | LSTM     | 3  | 1.575 ± 0.263        | 12.385 ± 5.784         | 0.942 ± 0.021        | 0.982 ± 0.005        | 0.995 ± 0.002        | 0.999 ± 4.50e-04        | 0.285 ± 0.124        | 4.199 ± 3.281         |
| anbn     | tanh RNN | 10 | 1.939 ± 0.374 (n=10) | 42.666 ± 41.834 (n=10) | 0.826 ± 0.077 (n=10) | 0.896 ± 0.067 (n=10) | 0.927 ± 0.063 (n=10) | 0.936 ± 0.062 (n=10)    | 0.112 ± 0.086 (n=10) | 2.305 ± 0.641 (n=10)  |
| anbn     | GRU      | 3  | 1.770 ± 0.099        | 5.092 ± 0.504          | 0.806 ± 0.045        | 0.881 ± 0.052        | 0.928 ± 0.056        | 0.952 ± 0.058           | 0.476 ± 0.066        | 2.451 ± 0.565         |
| anbn     | LSTM     | 3  | 1.920 ± 0.193        | 4.932 ± 0.291          | 0.802 ± 0.033        | 0.882 ± 0.045        | 0.931 ± 0.054        | 0.954 ± 0.058           | 0.899 ± 0.219        | 4.369 ± 0.811         |
| henon    | tanh RNN | 10 | 2.570 ± 0.302 (n=10) | 19.305 ± 5.767 (n=10)  | 0.707 ± 0.042 (n=10) | 0.852 ± 0.027 (n=10) | 0.966 ± 0.010 (n=10) | 0.999 ± 3.83e-04 (n=10) | 0.336 ± 0.128 (n=10) | 5.823 ± 0.694 (n=10)  |
| henon    | GRU      | 3  | 3.937 ± 1.041        | 2.772 ± 0.043          | 0.530 ± 0.056        | 0.682 ± 0.038        | 0.834 ± 0.019        | 0.959 ± 0.005           | 0.559 ± 0.013        | 1.550 ± 0.046         |
| henon    | LSTM     | 3  | 4.138 ± 0.469        | 3.466 ± 0.358          | 0.558 ± 0.047        | 0.727 ± 0.040        | 0.883 ± 0.026        | 0.976 ± 0.007           | 0.314 ± 0.023        | 1.085 ± 0.088         |
| sunspot  | tanh RNN | 10 | 1.198 ± 0.053 (n=10) | 5.320 ± 0.597 (n=10)   | 0.926 ± 0.015 (n=10) | 0.950 ± 0.010 (n=10) | 0.973 ± 0.005 (n=10) | 0.992 ± 0.002 (n=10)    | 2.805 ± 0.160 (n=10) | 14.874 ± 1.388 (n=10) |
| sunspot  | GRU      | 3  | 2.164 ± 0.314        | 4.985 ± 0.626          | 0.769 ± 0.030        | 0.859 ± 0.019        | 0.931 ± 0.011        | 0.984 ± 0.003           | 0.815 ± 0.011        | 4.071 ± 0.562         |
| sunspot  | LSTM     | 3  | 1.428 ± 0.027        | 4.657 ± 0.036          | 0.922 ± 0.013        | 0.956 ± 0.007        | 0.981 ± 0.003        | 0.996 ± 6.90e-04        | 0.560 ± 0.039        | 2.605 ± 0.167         |

### r -> fidelity, both cells side by side

Mean window `grad_cos` at each r, and the gain from r=4 to r=32.

| task     | cell     | r=4   | r=16  | r=32  | SnAp-1 | UORO   | RFLO  | r32-r4 | monotone in r |
|----------|----------|-------|-------|-------|--------|--------|-------|--------|---------------|
| rotation | tanh RNN | 0.679 | 0.841 | 0.930 | 0.411  | 0.034  | 0.144 | 0.251  | yes           |
| rotation | GRU      | 0.731 | 0.856 | 0.940 | 0.186  | 0.031  | 0.640 | 0.209  | yes           |
| rotation | LSTM     | 0.873 | 0.960 | 0.979 | 0.334  | 0.022  | 0.545 | 0.106  | yes           |
| anbn     | tanh RNN | 0.972 | 0.983 | 0.993 | 0.470  | 0.091  | 0.516 | 0.021  | yes           |
| anbn     | GRU      | 0.784 | 0.881 | 0.937 | 0.457  | 0.031  | 0.423 | 0.153  | yes           |
| anbn     | LSTM     | 0.969 | 0.967 | 0.991 | 0.909  | 0.041  | 0.432 | 0.022  | no            |
| henon    | tanh RNN | 0.976 | 0.977 | 0.985 | 0.781  | 0.043  | 0.392 | 0.009  | yes           |
| henon    | GRU      | 0.958 | 0.974 | 0.990 | 0.487  | 0.030  | 0.419 | 0.032  | yes           |
| henon    | LSTM     | 0.982 | 0.986 | 0.993 | 0.667  | 0.036  | 0.397 | 0.011  | yes           |
| sunspot  | tanh RNN | 0.898 | 0.899 | 0.931 | 0.622  | 0.063  | 0.337 | 0.034  | yes           |
| sunspot  | GRU      | 0.727 | 0.747 | 0.893 | 0.431  | -0.006 | 0.260 | 0.166  | yes           |
| sunspot  | LSTM     | 0.959 | 0.969 | 0.991 | 0.635  | 0.022  | 0.334 | 0.031  | yes           |

### Table 11 feed: the two lower-block rows, verbatim

`fraction rho^g_t<1` is `cert_summary.frac_rel_lt1` at SK-RTRL r=16 -- the online direction-certifiable criterion ||delta_t||_2 e_t / ||ghat_t||_2 < 1. `r_eps/n` (eps=0.1) is interpolated in k on the logged `mass_topk` grid k in {4,8,16,32} and divided by n; a value pinned at 0.062 = 4/64 means the top-4 mass already cleared 1-eps, so the true value sits at or below that bound.

| cell     | row                              | rotation             | anbn                 | henon                | sunspot              |
|----------|----------------------------------|----------------------|----------------------|----------------------|----------------------|
| tanh RNN | fraction rho^g_t<1 (r=16)        | 0.0000               | 0.0000               | 0.0000               | 0.0000               |
| tanh RNN | r_eps/n (eps=0.1, r=16 residual) | 0.175 ± 0.119 (n=10) | 0.285 ± 0.215 (n=10) | 0.120 ± 0.019 (n=10) | 0.062 ± 0.000 (n=10) |
| GRU      | fraction rho^g_t<1 (r=16)        | 0.0000               | 0.0000               | 0.0001               | 0.0004               |
| GRU      | r_eps/n (eps=0.1, r=16 residual) | 0.175 ± 0.099        | 0.235 ± 0.187        | 0.367 ± 0.014        | 0.166 ± 0.040        |
| LSTM     | fraction rho^g_t<1 (r=16)        | 0.0000               | 0.0000               | 0.0000               | 0.0000               |
| LSTM     | r_eps/n (eps=0.1, r=16 residual) | 0.075 ± 0.017        | 0.241 ± 0.183        | 0.289 ± 0.055        | 0.062 ± 0.000        |

Same row at the other ranks, as a robustness check (the residual rank is a property of the task and the cell, so it should not move much with r):

| estimator    | cell     | rotation             | anbn                 | henon                | sunspot              |
|--------------|----------|----------------------|----------------------|----------------------|----------------------|
| SK-RTRL r=4  | tanh RNN | 0.183 ± 0.126 (n=10) | 0.287 ± 0.213 (n=10) | 0.120 ± 0.016 (n=10) | 0.062 ± 0.000 (n=10) |
| SK-RTRL r=4  | GRU      | 0.164 ± 0.078        | 0.240 ± 0.184        | 0.341 ± 0.053        | 0.201 ± 0.053        |
| SK-RTRL r=4  | LSTM     | 0.082 ± 0.028        | 0.242 ± 0.182        | 0.297 ± 0.035        | 0.062 ± 0.000        |
| SK-RTRL r=16 | tanh RNN | 0.175 ± 0.119 (n=10) | 0.285 ± 0.215 (n=10) | 0.120 ± 0.019 (n=10) | 0.062 ± 0.000 (n=10) |
| SK-RTRL r=16 | GRU      | 0.175 ± 0.099        | 0.235 ± 0.187        | 0.367 ± 0.014        | 0.166 ± 0.040        |
| SK-RTRL r=16 | LSTM     | 0.075 ± 0.017        | 0.241 ± 0.183        | 0.289 ± 0.055        | 0.062 ± 0.000        |
| SK-RTRL r=32 | tanh RNN | 0.101 ± 0.066 (n=10) | 0.289 ± 0.212 (n=10) | 0.175 ± 0.024 (n=10) | 0.062 ± 0.000 (n=10) |
| SK-RTRL r=32 | GRU      | 0.189 ± 0.116        | 0.241 ± 0.183        | 0.379 ± 0.025        | 0.193 ± 0.023        |
| SK-RTRL r=32 | LSTM     | 0.062 ± 0.000        | 0.245 ± 0.181        | 0.297 ± 0.047        | 0.062 ± 0.000        |

## 5. Cost of the gated cell relative to tanh

Ratio of across-seed means (seeds do not pair across cells -- different grids -- so this is a ratio of means, not a paired statistic).

**GRU / tanh**

| task     | estimator    | tanh peak MB | gru peak MB | MB ratio | tanh wall s | gru wall s | wall ratio |
|----------|--------------|--------------|-------------|----------|-------------|------------|------------|
| rotation | exact RTRL   | 35.1         | 73.7        | 2.10x    | 47          | 184        | 3.93x      |
| rotation | SK-RTRL r=32 | 58.8         | 160.7       | 2.73x    | 701         | 1296       | 1.85x      |
| rotation | SK-RTRL r=16 | 56.5         | 137.3       | 2.43x    | 602         | 1110       | 1.84x      |
| rotation | SK-RTRL r=4  | 54.4         | 132.4       | 2.43x    | 550         | 1040       | 1.89x      |
| rotation | SnAp-1       | 53.8         | 129.7       | 2.41x    | 108         | 383        | 3.56x      |
| rotation | RFLO         | 35.3         | 74.2        | 2.10x    | 59          | 177        | 3.02x      |
| rotation | UORO         | 35.3         | 74.2        | 2.10x    | 69          | 220        | 3.19x      |
| anbn     | exact RTRL   | 34.0         | 69.6        | 2.05x    | 55          | 194        | 3.52x      |
| anbn     | SK-RTRL r=32 | 55.1         | 149.6       | 2.72x    | 803         | 1265       | 1.58x      |
| anbn     | SK-RTRL r=16 | 54.2         | 128.9       | 2.38x    | 684         | 1076       | 1.57x      |
| anbn     | SK-RTRL r=4  | 51.7         | 124.0       | 2.40x    | 630         | 1068       | 1.70x      |
| anbn     | SnAp-1       | 51.1         | 121.6       | 2.38x    | 105         | 329        | 3.13x      |
| anbn     | RFLO         | 34.1         | 70.0        | 2.05x    | 67          | 198        | 2.95x      |
| anbn     | UORO         | 34.1         | 70.0        | 2.05x    | 76          | 285        | 3.73x      |
| henon    | exact RTRL   | 33.8         | 67.7        | 2.00x    | 46          | 163        | 3.54x      |
| henon    | SK-RTRL r=32 | 55.1         | 145.4       | 2.64x    | 822         | 1259       | 1.53x      |
| henon    | SK-RTRL r=16 | 52.6         | 123.4       | 2.35x    | 704         | 1136       | 1.61x      |
| henon    | SK-RTRL r=4  | 51.0         | 118.8       | 2.33x    | 647         | 1095       | 1.69x      |
| henon    | SnAp-1       | 50.5         | 117.2       | 2.32x    | 108         | 321        | 2.96x      |
| henon    | RFLO         | 34.0         | 68.1        | 2.00x    | 63          | 203        | 3.20x      |
| henon    | UORO         | 34.0         | 68.1        | 2.00x    | 70          | 231        | 3.29x      |
| sunspot  | exact RTRL   | 33.3         | 67.1        | 2.01x    | 45          | 174        | 3.84x      |
| sunspot  | SK-RTRL r=32 | 54.6         | 144.9       | 2.66x    | 723         | 1247       | 1.73x      |
| sunspot  | SK-RTRL r=16 | 52.1         | 122.9       | 2.36x    | 601         | 1153       | 1.92x      |
| sunspot  | SK-RTRL r=4  | 50.5         | 118.2       | 2.34x    | 549         | 1045       | 1.90x      |
| sunspot  | SnAp-1       | 50.0         | 116.7       | 2.34x    | 107         | 343        | 3.20x      |
| sunspot  | RFLO         | 33.5         | 67.6        | 2.02x    | 63          | 217        | 3.46x      |
| sunspot  | UORO         | 33.5         | 67.6        | 2.02x    | 70          | 233        | 3.35x      |

**LSTM / tanh**

| task     | estimator    | tanh peak MB | lstm peak MB | MB ratio | tanh wall s | lstm wall s | wall ratio |
|----------|--------------|--------------|--------------|----------|-------------|-------------|------------|
| rotation | exact RTRL   | 35.1         | 166.3        | 4.73x    | 47          | 53          | 1.13x      |
| rotation | SK-RTRL r=32 | 58.8         | 334.8        | 5.69x    | 701         | 600         | 0.86x      |
| rotation | SK-RTRL r=16 | 56.5         | 325.2        | 5.76x    | 602         | 498         | 0.83x      |
| rotation | SK-RTRL r=4  | 54.4         | 318.5        | 5.86x    | 550         | 457         | 0.83x      |
| rotation | SnAp-1       | 53.8         | 315.5        | 5.86x    | 108         | 100         | 0.93x      |
| rotation | RFLO         | 35.3         | 167.5        | 4.75x    | 59          | 59          | 1.01x      |
| rotation | UORO         | 35.3         | 167.0        | 4.73x    | 69          | 67          | 0.97x      |
| anbn     | exact RTRL   | 34.0         | 154.2        | 4.54x    | 55          | 58          | 1.06x      |
| anbn     | SK-RTRL r=32 | 55.1         | 310.3        | 5.63x    | 803         | 579         | 0.72x      |
| anbn     | SK-RTRL r=16 | 54.2         | 301.3        | 5.56x    | 684         | 477         | 0.70x      |
| anbn     | SK-RTRL r=4  | 51.7         | 294.7        | 5.70x    | 630         | 448         | 0.71x      |
| anbn     | SnAp-1       | 51.1         | 291.3        | 5.70x    | 105         | 97          | 0.92x      |
| anbn     | RFLO         | 34.1         | 155.3        | 4.55x    | 67          | 64          | 0.95x      |
| anbn     | UORO         | 34.1         | 154.8        | 4.53x    | 76          | 71          | 0.92x      |
| henon    | exact RTRL   | 33.8         | 150.7        | 4.46x    | 46          | 52          | 1.13x      |
| henon    | SK-RTRL r=32 | 55.1         | 301.1        | 5.47x    | 822         | 611         | 0.74x      |
| henon    | SK-RTRL r=16 | 52.6         | 293.1        | 5.57x    | 704         | 495         | 0.70x      |
| henon    | SK-RTRL r=4  | 51.0         | 286.6        | 5.62x    | 647         | 462         | 0.71x      |
| henon    | SnAp-1       | 50.5         | 283.9        | 5.62x    | 108         | 99          | 0.91x      |
| henon    | RFLO         | 34.0         | 151.8        | 4.47x    | 63          | 58          | 0.92x      |
| henon    | UORO         | 34.0         | 151.3        | 4.45x    | 70          | 65          | 0.93x      |
| sunspot  | exact RTRL   | 33.3         | 150.2        | 4.51x    | 45          | 51          | 1.13x      |
| sunspot  | SK-RTRL r=32 | 54.6         | 300.6        | 5.51x    | 723         | 594         | 0.82x      |
| sunspot  | SK-RTRL r=16 | 52.1         | 292.6        | 5.62x    | 601         | 488         | 0.81x      |
| sunspot  | SK-RTRL r=4  | 50.5         | 286.1        | 5.67x    | 549         | 452         | 0.82x      |
| sunspot  | SnAp-1       | 50.0         | 283.3        | 5.67x    | 107         | 96          | 0.90x      |
| sunspot  | RFLO         | 33.5         | 151.3        | 4.52x    | 63          | 58          | 0.92x      |
| sunspot  | UORO         | 33.5         | 150.8        | 4.51x    | 70          | 65          | 0.93x      |

> `wall_s` was measured on a shared GPU in both stages and the two stages were not contended alike, so the wall-clock ratio is an upper bound on the real per-step cost of the gated cell, not a clean measurement. `peak_MB` is allocator high-water mark and is not contention-sensitive.

## 6. Protocol deviations and confounds

- learning rate: D5 used 0.001 for every row (the job-file header says `D4_SELECT_stage2.json` was absent at generation time, so `--default-lr` applied); the tanh reference in `results/r2/d4_eval` used the D4 stage-2 selected rates 0.0001, 0.0003, 0.001, 0.003, 0.01. **Task-metric comparisons across cells are therefore not like-for-like** -- the certificate, spectrum and memory comparisons are the load-bearing ones.
- seeds: D5 has 3 seeds per cell, the tanh reference has up to 10. Every across-cell row prints its own n.
- `wall_s` came off a shared GPU in both stages; treat the ratio as an upper bound.
- width/batch: n in [64], batch in [8] across both stages -- matched.
- shadow: [1] (the exact-gradient reference that `grad_cos`, `true_E` and the certificate check need).

No cell carries a seed-count, step-count or non-finite flag.

