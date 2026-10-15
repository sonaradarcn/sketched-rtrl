# Temporal hold-out on the measured series (sunspot, laser), Table 9

Generated 2026-10-15 12:28 by `make_r2_holdout_report.py` (reads `results/r2/d4_holdout` and `results/r2/d4_eval`; writes nothing else).

- hold-out value: `metric_holdout` from the run json -- the mean one-step-ahead error over a SINGLE online pass through the held-out tail (`--holdout_frac 0.2`), no parameter updates, hidden state from zero, same `--washout` reset period as training (`run_m3.holdout_pass`).
- training value: the D4 window on the SAME run -- arithmetic mean of `metric` over every logged record with `step >= 80% of 20000`, via `make_r2_d4_tables.agg_field`.
- full-sequence reference: the same D4 window over `results/r2/d4_eval`, 10 seeds, at the D4 stage-2 selected learning rate / TBPTT window; those runs cycled the whole series.
- across seeds: mean ± population std (ddof=0), 95% CI = Student-t on the seed mean (sample std, t_{.975,n-1}); hold-out seeds [0, 1, 2, 3, 4].
- paired statistics: Wilcoxon signed-rank (two-sided) + bootstrap 95% CI on the mean paired difference (10000 resamples) + Cohen's d_z, on the shared hold-out seeds. Sign convention is the D4 one: **positive = SK-RTRL better**.
- collapse criterion, fixed before any number was read: hold-out / training-tail ratio > **2.0** on an error task.
- both tasks are error tasks (`laser, sunspot`), so lower is better throughout.

## 0. Headline

- **sunspot**: hold-out ordering vs published full-sequence ordering -- Spearman rho = 1.000, Kendall tau = 1.000, 0/10 discordant pairs; the ordering is IDENTICAL. On the hold-out segment **9 of 10** pairwise gaps have a bootstrap CI that excludes zero, so part of the ordering is resolved.
- **laser**: hold-out ordering vs published full-sequence ordering -- Spearman rho = -0.300, Kendall tau = -0.200, 6/10 discordant pairs; the ordering CHANGES. On the hold-out segment **0 of 10** pairwise gaps have a bootstrap CI that excludes zero, so the reordering is within hold-out noise and NO ordering claim is supportable on this task.
- SK-RTRL vs exact RTRL, paired on the same 5 seeds: the absolute gap is larger on the hold-out segment in **1 of 4** (task, rank) cells; SK-RTRL is behind exact on the hold-out segment in **1 of 4**; the hold-out bootstrap CI excludes zero in **1 of 4**. The largest hold-out gap to exact RTRL is **1.73 %** of exact's own level.
- collapse screen (hold-out / training-tail ratio > 2.0 on an error task): **0 of 10** (task, estimator) cells collapse on the ratio of means; **0** cells contain at least one individual seed above the threshold.
- inventory complete: all 50 hold-out runs in `jobs/d4_holdout.txt` are present and parsed.
- anomalies (NaN, non-finite, wrong step budget, config drift): **0**.

## 1. Inventory and anomalies

- expected (`# out=` annotations in `jobs/d4_holdout.txt`): **50**
- present: **50**
- missing: **0**
- unexpected extra json in `results/r2/d4_holdout`: **0**
- hold-out runs aggregated: **50** ; full-sequence reference runs aggregated: **100**

No missing runs against the job-file expectation.

Every json parsed cleanly and every one carries `metric_holdout`.

### Hold-out split, as recorded by `run_m3.holdout_pass`

| task    | holdout_steps | split_step | note                                                      |
|---------|---------------|------------|-----------------------------------------------------------|
| sunspot | 600           | 2464       | single pass, no parameter updates, hidden state from zero |
| laser   | 200           | 864        | single pass, no parameter updates, hidden state from zero |

### Anomaly audit (raw record stream + args, not the window)

No NaN, no non-finite window value, no wrong step budget, no config drift against `jobs/d4_eval.txt`, no SVD fallback.

## 2. Hold-out vs training tail vs published full sequence

One row = one (task, estimator). Columns 2-3 come from the SAME five hold-out runs and differ only in which data the metric was measured on; column 5 is the published 10-seed number from `results/r2/d4_eval`, whose runs cycled the whole series (see the confounds).

**sunspot** (NMSE, lower is better)

| estimator    | hold-out (mean±sd [CI95])        | training tail (same runs)        | ratio of means | per-seed ratio (mean±sd) | full sequence, 10 seeds (`results/r2/d4_eval`) | hold-out last200 | flags |
|--------------|----------------------------------|----------------------------------|----------------|--------------------------|------------------------------------------------|------------------|-------|
| exact RTRL   | 0.1318 ± 0.0032 [0.1274, 0.1363] | 0.1343 ± 0.0016 [0.1321, 0.1365] | 0.982          | 0.982 ± 0.034            | 0.1335 ± 0.0020 [0.1321, 0.1350]               | 0.1314 ± 0.0075  |       |
| SK-RTRL r=32 | 0.1318 ± 0.0030 [0.1276, 0.1360] | 0.1341 ± 0.0016 [0.1319, 0.1363] | 0.983          | 0.983 ± 0.033            | 0.1334 ± 0.0019 [0.1319, 0.1348]               | 0.1314 ± 0.0072  |       |
| SK-RTRL r=16 | 0.1320 ± 0.0032 [0.1276, 0.1365] | 0.1342 ± 0.0015 [0.1322, 0.1363] | 0.984          | 0.984 ± 0.034            | 0.1335 ± 0.0020 [0.1320, 0.1351]               | 0.1316 ± 0.0074  |       |
| SnAp-1       | 0.1385 ± 0.0034 [0.1338, 0.1432] | 0.1405 ± 0.0016 [0.1383, 0.1428] | 0.985          | 0.986 ± 0.035            | 0.1398 ± 0.0021 [0.1382, 0.1413]               | 0.1380 ± 0.0082  |       |
| TBPTT        | 0.1332 ± 0.0028 [0.1293, 0.1371] | 0.1348 ± 0.0017 [0.1325, 0.1372] | 0.988          | 0.988 ± 0.033            | 0.1342 ± 0.0020 [0.1327, 0.1358]               | 0.1331 ± 0.0073  |       |

**laser** (NMSE, lower is better)

| estimator    | hold-out (mean±sd [CI95])        | training tail (same runs)        | ratio of means | per-seed ratio (mean±sd) | full sequence, 10 seeds (`results/r2/d4_eval`) | hold-out last200 | flags |
|--------------|----------------------------------|----------------------------------|----------------|--------------------------|------------------------------------------------|------------------|-------|
| exact RTRL   | 0.0169 ± 0.0048 [0.0102, 0.0236] | 0.0161 ± 0.0012 [0.0144, 0.0179] | 1.049          | 1.040 ± 0.250            | 0.0160 ± 0.0017 [0.0147, 0.0172]               | 0.0169 ± 0.0048  |       |
| SK-RTRL r=32 | 0.0166 ± 0.0068 [0.0072, 0.0260] | 0.0174 ± 0.0011 [0.0160, 0.0189] | 0.952          | 0.960 ± 0.409            | 0.0179 ± 0.0015 [0.0168, 0.0191]               | 0.0166 ± 0.0068  |       |
| SK-RTRL r=16 | 0.0168 ± 0.0044 [0.0106, 0.0229] | 0.0199 ± 0.0004 [0.0193, 0.0205] | 0.843          | 0.840 ± 0.204            | 0.0197 ± 0.0036 [0.0170, 0.0224]               | 0.0168 ± 0.0044  |       |
| SnAp-1       | 0.0164 ± 0.0043 [0.0105, 0.0223] | 0.0178 ± 0.0012 [0.0161, 0.0195] | 0.920          | 0.933 ± 0.283            | 0.0200 ± 0.0028 [0.0179, 0.0221]               | 0.0164 ± 0.0043  |       |
| TBPTT        | 0.0163 ± 0.0047 [0.0097, 0.0228] | 0.0164 ± 0.0014 [0.0145, 0.0184] | 0.989          | 0.998 ± 0.302            | 0.0167 ± 0.0019 [0.0153, 0.0182]               | 0.0163 ± 0.0047  |       |

### Per-seed hold-out values

| sunspot: estimator | ho s0  | ho s1  | ho s2  | ho s3  | ho s4  | tr s0  | tr s1  | tr s2  | tr s3  | tr s4  |
|--------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| exact RTRL         | 0.1307 | 0.1377 | 0.1325 | 0.1291 | 0.1291 | 0.1334 | 0.1323 | 0.1334 | 0.1362 | 0.1361 |
| SK-RTRL r=32       | 0.1307 | 0.1372 | 0.1326 | 0.1293 | 0.1291 | 0.1330 | 0.1321 | 0.1335 | 0.1360 | 0.1360 |
| SK-RTRL r=16       | 0.1309 | 0.1378 | 0.1330 | 0.1292 | 0.1292 | 0.1334 | 0.1323 | 0.1335 | 0.1358 | 0.1362 |
| SnAp-1             | 0.1385 | 0.1446 | 0.1386 | 0.1354 | 0.1352 | 0.1395 | 0.1386 | 0.1396 | 0.1426 | 0.1424 |
| TBPTT              | 0.1342 | 0.1381 | 0.1327 | 0.1305 | 0.1306 | 0.1339 | 0.1327 | 0.1338 | 0.1370 | 0.1367 |

| laser: estimator | ho s0  | ho s1  | ho s2  | ho s3  | ho s4  | tr s0  | tr s1  | tr s2  | tr s3  | tr s4  |
|------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| exact RTRL       | 0.0119 | 0.0192 | 0.0251 | 0.0127 | 0.0157 | 0.0152 | 0.0180 | 0.0170 | 0.0158 | 0.0146 |
| SK-RTRL r=32     | 0.0214 | 0.0174 | 0.0259 | 0.0115 | 0.0069 | 0.0191 | 0.0175 | 0.0162 | 0.0164 | 0.0181 |
| SK-RTRL r=16     | 0.0252 | 0.0137 | 0.0135 | 0.0169 | 0.0146 | 0.0204 | 0.0199 | 0.0192 | 0.0201 | 0.0197 |
| SnAp-1           | 0.0128 | 0.0200 | 0.0219 | 0.0168 | 0.0104 | 0.0169 | 0.0176 | 0.0162 | 0.0191 | 0.0193 |
| TBPTT            | 0.0110 | 0.0221 | 0.0196 | 0.0182 | 0.0104 | 0.0165 | 0.0186 | 0.0143 | 0.0158 | 0.0170 |

## 3. Does the ordering survive out of sample?

**sunspot**

- hold-out:        SK-RTRL r=32 0.1318  <  exact RTRL 0.1318  <  SK-RTRL r=16 0.1320  <  TBPTT 0.1332  <  SnAp-1 0.1385
- training tail:   SK-RTRL r=32 0.1341  <  SK-RTRL r=16 0.1342  <  exact RTRL 0.1343  <  TBPTT 0.1348  <  SnAp-1 0.1405
- full sequence:   SK-RTRL r=32 0.1334  <  exact RTRL 0.1335  <  SK-RTRL r=16 0.1335  <  TBPTT 0.1342  <  SnAp-1 0.1398
- hold-out vs full sequence: Spearman rho = 1.000, Kendall tau = 1.000, discordant pairs 0/10, identical ordering.
- hold-out vs its own training tail: Spearman rho = 0.900, Kendall tau = 0.800, discordant pairs 1/10, ordering changes.
- hold-out: **9 of 10** pairwise gaps have a bootstrap 95% CI excluding zero.
- training tail: **8 of 10** pairwise gaps have a bootstrap 95% CI excluding zero.

**laser**

- hold-out:        TBPTT 0.0163  <  SnAp-1 0.0164  <  SK-RTRL r=32 0.0166  <  SK-RTRL r=16 0.0168  <  exact RTRL 0.0169
- training tail:   exact RTRL 0.0161  <  TBPTT 0.0164  <  SK-RTRL r=32 0.0174  <  SnAp-1 0.0178  <  SK-RTRL r=16 0.0199
- full sequence:   exact RTRL 0.0160  <  TBPTT 0.0167  <  SK-RTRL r=32 0.0179  <  SK-RTRL r=16 0.0197  <  SnAp-1 0.0200
- hold-out vs full sequence: Spearman rho = -0.300, Kendall tau = -0.200, discordant pairs 6/10, ordering changes.
- hold-out vs its own training tail: Spearman rho = -0.100, Kendall tau = 0.000, discordant pairs 5/10, ordering changes.
- hold-out: **0 of 10** pairwise gaps have a bootstrap 95% CI excluding zero.
- training tail: **4 of 10** pairwise gaps have a bootstrap 95% CI excluding zero.
- **the hold-out ordering on this task is not resolved by the data**: not one of the 10 pairwise gaps separates, so any reordering against the full-sequence table is noise and no ordering claim may be made from it.

### Which hold-out gaps actually separate

Same paired bootstrap as section 4, run over every unordered pair of estimators. `sep` marks a pair whose 95 % CI excludes zero; those are the only gaps the hold-out segment licenses a statement about.

| task    | A            | B            | delta A-B [CI95 boot]       | d_z   | sep |
|---------|--------------|--------------|-----------------------------|-------|-----|
| sunspot | exact RTRL   | SK-RTRL r=32 | -5.18e-05 [-0.0003, 0.0001] | -0.20 |     |
| sunspot | exact RTRL   | SK-RTRL r=16 | +0.0002 [0.0001, 0.0003]    | +1.29 | sep |
| sunspot | exact RTRL   | SnAp-1       | +0.0066 [0.0062, 0.0072]    | +9.67 | sep |
| sunspot | exact RTRL   | TBPTT        | +0.0014 [0.0005, 0.0025]    | +1.07 | sep |
| sunspot | SK-RTRL r=32 | SK-RTRL r=16 | +0.0003 [6.36e-05, 0.0004]  | +1.08 | sep |
| sunspot | SK-RTRL r=32 | SnAp-1       | +0.0067 [0.0061, 0.0074]    | +8.00 | sep |
| sunspot | SK-RTRL r=32 | TBPTT        | +0.0014 [0.0006, 0.0026]    | +1.12 | sep |
| sunspot | SK-RTRL r=16 | SnAp-1       | +0.0064 [0.0059, 0.0071]    | +8.45 | sep |
| sunspot | SK-RTRL r=16 | TBPTT        | +0.0012 [0.0003, 0.0023]    | +0.87 | sep |
| sunspot | SnAp-1       | TBPTT        | -0.0052 [-0.0060, -0.0046]  | -5.71 | sep |

| task  | A            | B            | delta A-B [CI95 boot]     | d_z   | sep |
|-------|--------------|--------------|---------------------------|-------|-----|
| laser | exact RTRL   | SK-RTRL r=32 | -0.0003 [-0.0053, 0.0053] | -0.04 |     |
| laser | exact RTRL   | SK-RTRL r=16 | -0.0001 [-0.0073, 0.0078] | -0.01 |     |
| laser | exact RTRL   | SnAp-1       | -0.0005 [-0.0034, 0.0022] | -0.14 |     |
| laser | exact RTRL   | TBPTT        | -0.0007 [-0.0045, 0.0032] | -0.13 |     |
| laser | SK-RTRL r=32 | SK-RTRL r=16 | +0.0002 [-0.0071, 0.0060] | +0.02 |     |
| laser | SK-RTRL r=32 | SnAp-1       | -0.0002 [-0.0053, 0.0041] | -0.04 |     |
| laser | SK-RTRL r=32 | TBPTT        | -0.0004 [-0.0065, 0.0052] | -0.05 |     |
| laser | SK-RTRL r=16 | SnAp-1       | -0.0004 [-0.0074, 0.0059] | -0.04 |     |
| laser | SK-RTRL r=16 | TBPTT        | -0.0005 [-0.0081, 0.0061] | -0.06 |     |
| laser | SnAp-1       | TBPTT        | -0.0001 [-0.0016, 0.0014] | -0.07 |     |

## 4. SK-RTRL against the baselines, paired on the same seeds

Positive delta = SK-RTRL better (the task's own direction). `delta [CI95]` is the mean paired difference with a 10000-resample bootstrap CI; `p` is the two-sided Wilcoxon signed-rank p, whose attainable floor at n=5 is 0.0625 -- **no cell here can reach p<0.05**, by construction and not by outcome.

**hold-out segment** (`metric_holdout`)

| task    | SK-RTRL      | baseline   | n | SK mean | base mean | delta [CI95 boot]           | d_z   | Wilcoxon p | Cliff | note |
|---------|--------------|------------|---|---------|-----------|-----------------------------|-------|------------|-------|------|
| sunspot | SK-RTRL r=16 | exact RTRL | 5 | 0.1320  | 0.1318    | -0.0002 [-0.0003, -0.0001]  | -1.29 | 0.0625     | -0.28 |      |
| sunspot | SK-RTRL r=16 | SnAp-1     | 5 | 0.1320  | 0.1385    | +0.0064 [0.0059, 0.0071]    | +8.45 | 0.0625     | 0.84  |      |
| sunspot | SK-RTRL r=16 | TBPTT      | 5 | 0.1320  | 0.1332    | +0.0012 [0.0003, 0.0023]    | +0.87 | 0.1250     | 0.28  |      |
| sunspot | SK-RTRL r=32 | exact RTRL | 5 | 0.1318  | 0.1318    | +5.18e-05 [-0.0001, 0.0003] | +0.20 | 1.0000     | -0.04 |      |
| sunspot | SK-RTRL r=32 | SnAp-1     | 5 | 0.1318  | 0.1385    | +0.0067 [0.0061, 0.0074]    | +8.00 | 0.0625     | 0.84  |      |
| sunspot | SK-RTRL r=32 | TBPTT      | 5 | 0.1318  | 0.1332    | +0.0014 [0.0006, 0.0026]    | +1.12 | 0.0625     | 0.36  |      |
| laser   | SK-RTRL r=16 | exact RTRL | 5 | 0.0168  | 0.0169    | +0.0001 [-0.0078, 0.0073]   | +0.01 | 1.0000     | -0.12 |      |
| laser   | SK-RTRL r=16 | SnAp-1     | 5 | 0.0168  | 0.0164    | -0.0004 [-0.0074, 0.0059]   | -0.04 | 1.0000     | -0.12 |      |
| laser   | SK-RTRL r=16 | TBPTT      | 5 | 0.0168  | 0.0163    | -0.0005 [-0.0081, 0.0061]   | -0.06 | 1.0000     | -0.04 |      |
| laser   | SK-RTRL r=32 | exact RTRL | 5 | 0.0166  | 0.0169    | +0.0003 [-0.0053, 0.0053]   | +0.04 | 0.8125     | 0.04  |      |
| laser   | SK-RTRL r=32 | SnAp-1     | 5 | 0.0166  | 0.0164    | -0.0002 [-0.0053, 0.0041]   | -0.04 | 1.0000     | -0.04 |      |
| laser   | SK-RTRL r=32 | TBPTT      | 5 | 0.0166  | 0.0163    | -0.0004 [-0.0065, 0.0052]   | -0.05 | 1.0000     | -0.04 |      |

**training tail** (`metric_train_tail`)

| task    | SK-RTRL      | baseline   | n | SK mean | base mean | delta [CI95 boot]             | d_z    | Wilcoxon p | Cliff | note |
|---------|--------------|------------|---|---------|-----------|-------------------------------|--------|------------|-------|------|
| sunspot | SK-RTRL r=16 | exact RTRL | 5 | 0.1342  | 0.1343    | +4.36e-05 [-8.65e-05, 0.0002] | +0.19  | 0.8125     | 0.04  |      |
| sunspot | SK-RTRL r=16 | SnAp-1     | 5 | 0.1342  | 0.1405    | +0.0063 [0.0061, 0.0066]      | +20.17 | 0.0625     | 1.00  |      |
| sunspot | SK-RTRL r=16 | TBPTT      | 5 | 0.1342  | 0.1348    | +0.0006 [0.0004, 0.0009]      | +1.61  | 0.0625     | 0.36  |      |
| sunspot | SK-RTRL r=32 | exact RTRL | 5 | 0.1341  | 0.1343    | +0.0002 [6.96e-05, 0.0003]    | +1.29  | 0.1250     | 0.20  |      |
| sunspot | SK-RTRL r=32 | SnAp-1     | 5 | 0.1341  | 0.1405    | +0.0064 [0.0063, 0.0065]      | +39.08 | 0.0625     | 1.00  |      |
| sunspot | SK-RTRL r=32 | TBPTT      | 5 | 0.1341  | 0.1348    | +0.0007 [0.0005, 0.0009]      | +2.99  | 0.0625     | 0.36  |      |
| laser   | SK-RTRL r=16 | exact RTRL | 5 | 0.0199  | 0.0161    | -0.0038 [-0.0050, -0.0025]    | -2.37  | 0.0625     | -1.00 |      |
| laser   | SK-RTRL r=16 | SnAp-1     | 5 | 0.0199  | 0.0178    | -0.0021 [-0.0031, -0.0010]    | -1.54  | 0.0625     | -0.92 |      |
| laser   | SK-RTRL r=16 | TBPTT      | 5 | 0.0199  | 0.0164    | -0.0035 [-0.0045, -0.0024]    | -2.42  | 0.0625     | -1.00 |      |
| laser   | SK-RTRL r=32 | exact RTRL | 5 | 0.0174  | 0.0161    | -0.0013 [-0.0031, 0.0004]     | -0.59  | 0.4375     | -0.60 |      |
| laser   | SK-RTRL r=32 | SnAp-1     | 5 | 0.0174  | 0.0178    | +0.0004 [-0.0011, 0.0018]     | +0.20  | 0.4375     | 0.28  |      |
| laser   | SK-RTRL r=32 | TBPTT      | 5 | 0.0174  | 0.0164    | -0.0010 [-0.0020, 6.65e-05]   | -0.72  | 0.3125     | -0.36 |      |

### Does the gap to exact RTRL widen out of sample?

Both columns are the same paired statistic on the same five seeds, differing only in which data the metric was measured on. `gap` is |delta|; a negative delta means SK-RTRL is behind exact RTRL.

| task    | SK-RTRL      | delta train | delta hold-out | gap train | gap hold-out | gap/exact train | gap/exact hold-out | gap ratio | direction   | flip      | hold-out CI   |
|---------|--------------|-------------|----------------|-----------|--------------|-----------------|--------------------|-----------|-------------|-----------|---------------|
| sunspot | SK-RTRL r=16 | +4.358e-05  | -0.0002034     | 4.36e-05  | 0.0002       | 0.03 %          | 0.15 %             | 4.67      | gap grows   | SIGN FLIP | CI excludes 0 |
| sunspot | SK-RTRL r=32 | +0.0001773  | +5.182e-05     | 0.0002    | 5.18e-05     | 0.13 %          | 0.04 %             | 0.29      | gap shrinks |           | CI spans 0    |
| laser   | SK-RTRL r=16 | -0.003758   | +0.0001403     | 0.0038    | 0.0001       | 23.31 %         | 0.83 %             | 0.04      | gap shrinks | SIGN FLIP | CI spans 0    |
| laser   | SK-RTRL r=32 | -0.001321   | +0.0002932     | 0.0013    | 0.0003       | 8.20 %          | 1.73 %             | 0.22      | gap shrinks | SIGN FLIP | CI spans 0    |

## 5. Collapse screen

Criterion fixed in advance: an estimator collapses out of sample if its hold-out metric is more than **2.0 x** its own training-tail metric. The ratio of means is the headline; the per-seed maximum is printed so a single bad seed cannot hide behind an average.

| task    | estimator    | ratio of means | mean per-seed ratio | max per-seed ratio | seeds > 2.0 | verdict |
|---------|--------------|----------------|---------------------|--------------------|-------------|---------|
| sunspot | exact RTRL   | 0.982          | 0.982               | 1.041              | --          | ok      |
| sunspot | SK-RTRL r=32 | 0.983          | 0.983               | 1.039              | --          | ok      |
| sunspot | SK-RTRL r=16 | 0.984          | 0.984               | 1.041              | --          | ok      |
| sunspot | SnAp-1       | 0.985          | 0.986               | 1.043              | --          | ok      |
| sunspot | TBPTT        | 0.988          | 0.988               | 1.041              | --          | ok      |
| laser   | exact RTRL   | 1.049          | 1.040               | 1.475              | --          | ok      |
| laser   | SK-RTRL r=32 | 0.952          | 0.960               | 1.598              | --          | ok      |
| laser   | SK-RTRL r=16 | 0.843          | 0.840               | 1.234              | --          | ok      |
| laser   | SnAp-1       | 0.920          | 0.933               | 1.349              | --          | ok      |
| laser   | TBPTT        | 0.989          | 0.998               | 1.372              | --          | ok      |

## 6. Confounds and what this check does NOT show

- a `--holdout_frac 0.2` run trains on the first 80 % of the series only; the `results/r2/d4_eval` reference trained on 100 % of it. The training-tail column of a hold-out run is therefore the same optimiser on a SHORTER stream, not a replica of the published column -- the load-bearing comparison in this report is the WITHIN-run hold-out/training one.
- the hold-out pass is a single online pass of [600] / [200] samples (sunspot / laser) against a 20000-step training stream, so a per-seed hold-out value is noisier by construction.
- `metric_holdout` starts the hidden state from zero inside the tail, so the first samples of the pass carry a state-estimation transient; `metric_holdout_last200` is printed beside it as the sensitivity column.
- `metric_holdout_last200` is DEGENERATE on laser: the whole hold-out pass is 200 samples, so the last-200 column is the same number as `metric_holdout` there and provides no sensitivity at all. It is informative on sunspot only.
- the hold-out segments are SHORT (sunspot [600], laser [200]). The resulting per-seed spread is what makes the hold-out CIs several times wider than the training-tail CIs in section 2, and it is why section 3 reports how many pairwise gaps actually separate rather than only a rank correlation.
- seeds: the hold-out grid has 5, the `results/r2/d4_eval` reference up to 10. Every cell prints its own n; the paired statistics use only the 5 shared hold-out seeds.
- at n=5 the two-sided Wilcoxon signed-rank p cannot go below 0.0625, so NO comparison in section 4 can reach p<0.05. The p column is reported for completeness; the bootstrap CI and d_z carry the evidence, and no Holm correction is applied to a family whose attainable floor already exceeds 0.05.
- width / batch / washout across both stages: n in [64], batch in [8], washout in [200] -- matched.

- this is a hold-out in TIME on two measured series at n=64, 20 000 steps. It says nothing about the diagnostic or chaotic tasks, which are generated on the fly and have no in-sample problem to begin with.
- the hold-out pass uses no estimator at all: it is a forward pass of the trained cell and readout. It therefore measures what the estimator's gradients LEARNED, not how well the estimator tracks a Jacobian.

