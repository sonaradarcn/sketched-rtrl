# RL residual spectrum (T-maze) — aggregated report

Data directories: `results/r2/rl` (clip 0) + `results/r2/rl_clip05` (clip 0.5)  |  roster done: 27/27 (0 missing)  |  ignored files: 0  |  unreadable/no spectrum: 0

## Conventions

The measured object is the same as in the supervised D1 tasks: the residual $R_t = J_t - S_t$, where $J_t$ is the influence matrix carried by the exact-RTRL shadow and $S_t$ is the SnAp-1 trace embedded block-diagonally (`skrtrl/rl.py` `TanhCore.spectrum`, the same definition as `checkpoint_spectrum` in `run_e1_spectrum.py`). The two arms differ only in the policy trajectory on which the residual is measured, not in the definition of the residual. The snap1 arm trains with SnAp-1 and discards exactly the whole $J_t-S_t$, so it is the "no sketch" reference.

The RL runs use iso-width $n=64$, $P=4416$ (= parameter count $n(n+n_{\mathrm{obs}}+1)=64\times69$), with 16 lanes per checkpoint. The checkpoints are the six fracs of `--spectrum_ckpts`. `--spectrum_min_age 4` and `--spectrum_defer 200` slide each checkpoint later, away from the T-maze steps at which all lanes reset together (there the residual is exactly zero and would read as a spurious low rank). If the age threshold is still not reached when `--spectrum_defer` runs out, the checkpoint is taken anyway, and each such checkpoint is listed in the anomaly list below.

One convention difference has to be handled explicitly. `run_m5.py` stores only the lane-mean singular value vector `sv`, whereas the D1 report's `r_eps` is the lane mean of the per-lane `r_eps`. So in this report `r_eps` can only be read off the lane-mean spectrum (the "mean spectrum" convention). To measure this offset rather than assume it, D1's `sv` was recomputed the same way (`run_e1_spectrum.py` also stores only the lane-mean spectrum); the difference between the two conventions is in Table F(i). `mass_top{k}`, `stable_rank` and `res_frac_of_J` are lane means of per-lane quantities in both experiments, so they are not affected and can be compared directly with Table 4 of the paper.

Grouping follows the rule of Section 6.4: final-stage $r_{0.1}/n<0.3$ low-rank, $\le 0.5$ intermediate, $>0.6$ not low-rank.

## Missing files and anomalies

The roster is complete (27/27); nothing is missing.

Per-checkpoint anomalies: none. All 27 runs × 6 checkpoints are usable, with no NaN, and no checkpoint was forced by the deadline before `--spectrum_min_age` was reached.

### Table A — final stage (frac=1.0), corridor × arm, 3 seeds

mean±sd over seeds (each (corridor, arm, seed) has a single checkpoint at the final stage, so n=3). `r_eps10/n` is computed from the stored lane-mean spectrum; see the conventions above. `mass_top16`, `stable_rank` and `res_frac_of_J` are lane means, with the same convention as D1.

| corridor | arm | n_seeds | r_eps10/n | r_eps05/n | r_eps10 | mass_top16 | stable rank | res_frac_of_J | group |
|---|---|---|---|---|---|---|---|---|---|
| 10 | skrtrl-r16 / clip 0 | 3/3 | 0.1927±0.0502 | 0.2812±0.0950 | 12.33±3.21 | 0.9845±0.0178 | 1.95±0.41 | 0.5632±0.1404 | low-rank |
| 10 | snap1 / clip 0 | 3/3 | 0.1771±0.1273 | 0.3125±0.2148 | 11.33±8.14 | 0.9799±0.0036 | 1.52±0.32 | 0.9052±0.1035 | low-rank |
| 10 | skrtrl-r16 / clip 0.5 | 3/3 | 0.0625±0.0156 | 0.2344±0.0271 | 4.00±1.00 | 0.8523±0.1570 | 1.03±0.02 | 0.0913±0.0104 | low-rank |
| 20 | skrtrl-r16 / clip 0 | 3/3 | 0.1615±0.0705 | 0.2500±0.0716 | 10.33±4.51 | 0.9479±0.0709 | 1.66±0.10 | 0.6792±0.2550 | low-rank |
| 20 | snap1 / clip 0 | 3/3 | 0.2344±0.0156 | 0.3385±0.0180 | 15.00±1.00 | 0.9881±0.0012 | 1.73±0.12 | 0.6702±0.1637 | low-rank |
| 20 | skrtrl-r16 / clip 0.5 | 3/3 | 0.0521±0.0090 | 0.3542±0.0451 | 3.33±0.58 | 0.9754±0.0363 | 1.07±0.02 | 0.1030±0.0238 | low-rank |
| 40 | skrtrl-r16 / clip 0 | 3/3 | 0.1667±0.1542 | 0.2135±0.1671 | 10.67±9.87 | 0.9872±0.0220 | 1.93±1.31 | 0.2795±0.1315 | low-rank |
| 40 | snap1 / clip 0 | 3/3 | 0.3854±0.0861 | 0.4792±0.1005 | 24.67±5.51 | 0.8869±0.0403 | 2.49±0.46 | 0.4721±0.1483 | intermediate |
| 40 | skrtrl-r16 / clip 0.5 | 3/3 | 0.0781±0.0156 | 0.1250±0.0156 | 5.00±1.00 | 0.9789±0.0359 | 1.07±0.05 | 0.0932±0.0039 | low-rank |
| | | | | | | | | | |
| 10/20/40 | skrtrl-r16 / clip 0 | 3/3 | 0.1736±0.0896 | 0.2483±0.1067 | 11.11±5.73 | 0.9732±0.0426 | 1.85±0.70 | 0.5073±0.2392 | low-rank |
| 10/20/40 | snap1 / clip 0 | 3/3 | 0.2656±0.1210 | 0.3767±0.1420 | 17.00±7.75 | 0.9516±0.0527 | 1.91±0.53 | 0.6825±0.2239 | low-rank |
| 10/20/40 | skrtrl-r16 / clip 0.5 | 3/3 | 0.0642±0.0165 | 0.2378±0.1030 | 4.11±1.05 | 0.9355±0.1035 | 1.06±0.04 | 0.0958±0.0142 | low-rank |

### Table B — full checkpoint grid: r_eps10/n by training stage

Columns are the six fracs of `--spectrum_ckpts` (fractions of `--steps`); each cell is mean±sd over 3 seeds.

| corridor | arm | frac=0.01 | frac=0.05 | frac=0.25 | frac=0.5 | frac=0.75 | frac=1 |
|---|---|---|---|---|---|---|---|
| 10 | skrtrl-r16 / clip 0 | 0.7240±0.0090 | 0.7135±0.0180 | 0.3385±0.1152 | 0.2604±0.0802 | 0.2083±0.0549 | 0.1927±0.0502 |
| 10 | snap1 / clip 0 | 0.7240±0.0090 | 0.7188±0.0156 | 0.6354±0.0451 | 0.4531±0.0976 | 0.3281±0.0312 | 0.1771±0.1273 |
| 10 | skrtrl-r16 / clip 0.5 | 0.0521±0.0239 | 0.0833±0.0180 | 0.0781 | 0.0677±0.0090 | 0.0833±0.0239 | 0.0625±0.0156 |
| 20 | skrtrl-r16 / clip 0 | 0.3281±0.0781 | 0.3490±0.1310 | 0.3229±0.2351 | 0.3021±0.1097 | 0.2188±0.0716 | 0.1615±0.0705 |
| 20 | snap1 / clip 0 | 0.5104±0.0705 | 0.7083±0.0180 | 0.4896±0.0239 | 0.4375±0.0563 | 0.2917±0.0502 | 0.2344±0.0156 |
| 20 | skrtrl-r16 / clip 0.5 | 0.0156 | 0.0677±0.0239 | 0.0625±0.0156 | 0.0625 | 0.0521±0.0090 | 0.0521±0.0090 |
| 40 | skrtrl-r16 / clip 0 | 0.2188±0.0563 | 0.1510±0.0325 | 0.0469±0.0271 | 0.0573±0.0451 | 0.0781±0.0469 | 0.1667±0.1542 |
| 40 | snap1 / clip 0 | 0.2760±0.0393 | 0.1771±0.0802 | 0.2500±0.3384 | 0.3854±0.3071 | 0.3438±0.2515 | 0.3854±0.0861 |
| 40 | skrtrl-r16 / clip 0.5 | 0.0156 | 0.0365±0.0090 | 0.1042±0.0180 | 0.0677±0.0180 | 0.0781 | 0.0781±0.0156 |

### Table C — clip 0.5 relative to clip 0 (skrtrl-r16, paired by (corridor, seed))

Each cell is the paired difference `clip 0.5 − clip 0`, mean±sd; final-stage checkpoints.

| quantity | corridor 10 | corridor 20 | corridor 40 | all (9 pairs) |
|---|---|---|---|---|
| Δ r_eps10/n | -0.1302±0.0631 | -0.1094±0.0781 | -0.0885±0.1534 | -0.1094±0.0934 |
| Δ r_eps05/n | -0.0469±0.1094 | 0.1042±0.0325 | -0.0885±0.1804 | -0.0104±0.1382 |
| Δ mass_top16 | -0.1322±0.1412 | 0.0276±0.0943 | -0.0083±0.0507 | -0.0377±0.1145 |
| Δ stable rank | -0.92±0.43 | -0.59±0.08 | -0.86±1.33 | -0.79±0.72 |
| Δ res_frac_of_J | -0.4719±0.1400 | -0.5762±0.2723 | -0.1862±0.1343 | -0.4115±0.2419 |

### Table D — snap1 relative to skrtrl-r16 (both clip 0, paired by (corridor, seed))

The snap1 arm's policy is trained by SnAp-1 itself, and the residual $J_t-S_t$ is the whole off-block-diagonal part it discards, so it serves as the "no sketch" reference. Each cell is the paired difference `snap1 − skrtrl-r16`, mean±sd; final-stage checkpoints.

| quantity | corridor 10 | corridor 20 | corridor 40 | all (9 pairs) |
|---|---|---|---|---|
| Δ r_eps10/n | -0.0156±0.1094 | 0.0729±0.0651 | 0.2188±0.0716 | 0.0920±0.1258 |
| Δ r_eps05/n | 0.0312±0.2102 | 0.0885±0.0549 | 0.2656±0.0681 | 0.1285±0.1554 |
| Δ mass_top16 | -0.0046±0.0174 | 0.0402±0.0715 | -0.1003±0.0570 | -0.0216±0.0776 |
| Δ stable rank | -0.43±0.44 | 0.07±0.22 | 0.56±1.71 | 0.07±0.99 |
| Δ res_frac_of_J | 0.3420±0.1910 | -0.0090±0.1546 | 0.1926±0.2690 | 0.1752±0.2376 |

### Table E — side by side with the supervised D1 tasks (n=64, final stage), grouped by the Section 6.4 rule

D1 rows take, from `results/r2/d1_spectrum/<task>_n64_s{0,1,2}.json`, the frac=1.0 checkpoints (pooled over 3 seeds × reset ages), with the same convention as Table 4 of the paper. `r_eps10/n (official)` is the lane mean of the per-lane r_eps used by the D1 report. `r_eps10/n (mean spectrum)` is recomputed the same way as for the T-maze (read off the stored lane-mean spectrum), and the difference between the two columns is the convention offset. T-maze rows have only the mean-spectrum column. Grouping follows Section 6.4: `<0.3` low-rank, `≤0.5` intermediate, `>0.6` not low-rank.

| task / arm | source | r_eps10/n (official) | r_eps10/n (mean spectrum) | mass_top16 | stable rank | res_frac_of_J | group (mean spectrum) |
|---|---|---|---|---|---|---|---|
| rotation | D1 | 0.0893±0.0455 | 0.0760±0.0534 | 0.9999±0.0001 | 1.72±0.68 | 0.6299±0.2629 | low-rank |
| a^n b^n | D1 | 0.2733±0.0766 | 0.1958±0.0363 | 0.9151±0.0909 | 2.05±0.41 | 0.5240±0.0977 | low-rank |
| lorenz | D1 | 0.2837±0.1135 | 0.2844±0.1139 | 0.9748±0.0234 | 2.12±0.66 | 0.5341±0.1152 | low-rank |
| mackeyglass | D1 | 0.4027±0.2233 | 0.4031±0.2220 | 0.9382±0.0776 | 2.63±0.59 | 0.5078±0.0770 | intermediate |
| laser | D1 | 0.4823±0.0385 | 0.4823±0.0368 | 0.9119±0.0227 | 3.91±0.63 | 0.6380±0.0218 | intermediate |
| henon | D1 | 0.4845±0.0340 | 0.4865±0.0344 | 0.8773±0.0234 | 4.79±0.57 | 0.4686±0.0190 | intermediate |
| sunspot | D1 | 0.6568±0.0924 | 0.6594±0.0945 | 0.8239±0.0389 | 4.83±0.94 | 0.6351±0.0514 | not low-rank |
| adding | D1 | 0.6974±0.0068 | 0.6979±0.0076 | 0.8000±0.0071 | 5.65±0.87 | 0.4897±0.0090 | not low-rank |
| copy | D1 | 0.7542±0.0189 | 0.7542±0.0182 | 0.7312±0.0388 | 8.03±3.59 | 0.5700±0.0383 | not low-rank |
| | | | | | | | |
| T-maze skrtrl-r16 / clip 0 | RL | — | 0.1736±0.0896 | 0.9732±0.0426 | 1.85±0.70 | 0.5073±0.2392 | low-rank |
| T-maze snap1 / clip 0 | RL | — | 0.2656±0.1210 | 0.9516±0.0527 | 1.91±0.53 | 0.6825±0.2239 | low-rank |
| T-maze skrtrl-r16 / clip 0.5 | RL | — | 0.0642±0.0165 | 0.9355±0.1035 | 1.06±0.04 | 0.0958±0.0142 | low-rank |

### Table F — convention cross-checks

**(i) Difference between the two r_eps conventions on D1** (n=64, final stage; `mean spectrum − official`). This is the offset to subtract when the T-maze mean-spectrum readings are set beside Table 4 of the paper. A negative value means the mean-spectrum convention is optimistic (it reads a smaller rank).

| task | r_eps10/n (official) | r_eps10/n (mean spectrum) | diff | r_eps05/n diff |
|---|---|---|---|---|
| rotation | 0.0893±0.0455 | 0.0760±0.0534 | -0.0133 | -0.0184 |
| a^n b^n | 0.2733±0.0766 | 0.1958±0.0363 | -0.0775 | -0.0648 |
| lorenz | 0.2837±0.1135 | 0.2844±0.1139 | +0.0007 | +0.0020 |
| mackeyglass | 0.4027±0.2233 | 0.4031±0.2220 | +0.0004 | +0.0009 |
| laser | 0.4823±0.0385 | 0.4823±0.0368 | +0.0000 | +0.0001 |
| henon | 0.4845±0.0340 | 0.4865±0.0344 | +0.0020 | +0.0046 |
| sunspot | 0.6568±0.0924 | 0.6594±0.0945 | +0.0026 | -0.0001 |
| adding | 0.6974±0.0068 | 0.6979±0.0076 | +0.0005 | -0.0012 |
| copy | 0.7542±0.0189 | 0.7542±0.0182 | +0.0000 | +0.0017 |
| **all 9 tasks** | | | mean -0.0094, max abs offset 0.0775 | mean -0.0084 |

**(ii) Agreement of the T-maze mean spectrum with the stored lane means** (all 27 runs × 6 checkpoints). The difference between the two ways of computing `mass_top16` and `stable_rank` at the same checkpoint shows how heterogeneous the spectra are across lanes: the larger it is, the more concentrated the mean spectrum is than a "typical lane", and the more the offset in (i) should be read as a lower bound.

| quantity | stored (lane mean) | recomputed from mean spectrum | diff (mean spectrum − stored) |
|---|---|---|---|
| mass_top16 | 0.9236±0.0895 | 0.9625±0.0656 | 0.0389±0.0547 |
| stable rank | 2.23±1.44 | 2.08±1.36 | -0.15±0.31 |

## Key numbers (auto-generated, numbers only)

- Final stage skrtrl-r16 / clip 0 (3 corridors × 3 seeds = 9 checkpoints): r_eps10/n = 0.1736±0.0896, mass_top16 = 0.9732±0.0426, stable rank = 1.85±0.70, res_frac_of_J = 0.5073±0.2392, group low-rank.
- Final stage snap1 / clip 0 (3 corridors × 3 seeds = 9 checkpoints): r_eps10/n = 0.2656±0.1210, mass_top16 = 0.9516±0.0527, stable rank = 1.91±0.53, res_frac_of_J = 0.6825±0.2239, group low-rank.
- Final stage skrtrl-r16 / clip 0.5 (3 corridors × 3 seeds = 9 checkpoints): r_eps10/n = 0.0642±0.0165, mass_top16 = 0.9355±0.1035, stable rank = 1.06±0.04, res_frac_of_J = 0.0958±0.0142, group low-rank.
- clip 0.5 − clip 0 (skrtrl-r16, paired by (corridor, seed), 9 pairs): Δ r_eps10/n = -0.1094±0.0934.
- snap1 − skrtrl-r16 (both clip 0, 9 pairs): Δ r_eps10/n = 0.0920±0.1258.
- skrtrl-r16 / clip 0: r_eps10/n at frac=0.01 is 0.4236±0.2352; at the final stage it is 0.1736±0.0896 (Δ -0.2500).
- snap1 / clip 0: r_eps10/n at frac=0.01 is 0.5035±0.1982; at the final stage it is 0.2656±0.1210 (Δ -0.2378).
- skrtrl-r16 / clip 0.5: r_eps10/n at frac=0.01 is 0.0278±0.0218; at the final stage it is 0.0642±0.0165 (Δ +0.0365).
- Corridor 10, both clip-0 arms, at frac=0.01: 0.7240±0.0081, inside the D1 early "unstructured residual" band 0.686–0.746; here too the compression comes from training.
- D1 low-rank group (mean-spectrum r_eps10/n < 0.3): rotation, a^n b^n, lorenz.
- skrtrl-r16 / clip 0: mean-spectrum reading 0.1736 is in group "low-rank"; adding the largest downward convention offset measured on D1, 0.0775, gives 0.2511, group unchanged.
- snap1 / clip 0: mean-spectrum reading 0.2656 is in group "low-rank"; adding the largest downward convention offset measured on D1, 0.0775, gives 0.3431, group changes to "intermediate".
- skrtrl-r16 / clip 0.5: mean-spectrum reading 0.0642 is in group "low-rank"; adding the largest downward convention offset measured on D1, 0.0775, gives 0.1417, group unchanged.

## Figures

- `results\figures\fig_r2_spectrum_rl.pdf` — written
