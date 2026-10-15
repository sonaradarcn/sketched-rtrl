# D1 residual-spectrum experiment — aggregated report

Data directory: `results/r2/d1_spectrum`  |  roster done: 63/63 (0 missing)  |  tagged files (not aggregated): 7  |  unreadable/no records: 0

## Missing files

The roster is complete; nothing is missing.

Tagged files (file name with a suffix such as `_r1cmp`; left out of the aggregation and kept for reference only; per-file numbers in Table B' below):
- `adding_n512_s0_long.json`
- `adding_n512_s2_long.json`
- `adding_n512_s3_extra.json`
- `adding_n512_s4_extra.json`
- `adding_n512_s5_extra.json`
- `adding_n512_s6_extra.json`
- `adding_n512_s7_extra.json`

### Table A — n=64, 9 tasks, final stage (frac=1.0) and pooled over all stages

mean±std over seeds x ages (final-stage columns) or over seeds x ages x fracs (all-stage columns). `n_seeds` = number of completed seeds for the task (/3).

| task | family | n_seeds | mass_top16 (final/all) | stable_rank (final/all) | r_eps10/n (final/all) | r_eps05/n (final/all) | res_frac_of_J (final/all) |
|---|---|---|---|---|---|---|---|
| adding | diagnostic | 3/3 | 0.8000±0.0071 / 0.8032±0.0150 | 5.6472±0.8707 / 4.8627±1.3082 | 0.6974±0.0068 / 0.7130±0.0180 | 0.8033±0.0094 / 0.8185±0.0168 | 0.4897±0.0090 / 0.5364±0.0440 |
| copy | diagnostic | 3/3 | 0.7312±0.0388 / 0.7471±0.0439 | 8.0318±3.5913 / 8.0313±2.8661 | 0.7542±0.0189 / 0.7465±0.0278 | 0.8473±0.0136 / 0.8428±0.0239 | 0.5700±0.0383 / 0.5675±0.0354 |
| anbn | diagnostic | 3/3 | 0.9151±0.0909 / 0.8548±0.1092 | 2.0475±0.4127 / 2.9595±1.2763 | 0.2733±0.0766 / 0.4767±0.1959 | 0.3409±0.0722 / 0.5690±0.2064 | 0.5240±0.0977 / 0.5185±0.0766 |
| rotation | diagnostic | 3/3 | 0.9999±0.0001 / 0.9397±0.0818 | 1.7210±0.6750 / 3.0815±2.0251 | 0.0893±0.0455 / 0.3445±0.2689 | 0.1215±0.0520 / 0.4172±0.3028 | 0.6299±0.2629 / 0.5468±0.2051 |
| henon | chaotic | 3/3 | 0.8773±0.0234 / 0.8433±0.0610 | 4.7918±0.5724 / 4.5032±1.1206 | 0.4845±0.0340 / 0.5642±0.1133 | 0.5673±0.0367 / 0.6578±0.1211 | 0.4686±0.0190 / 0.5200±0.0717 |
| mackeyglass | chaotic | 3/3 | 0.9382±0.0776 / 0.8342±0.0831 | 2.6260±0.5920 / 5.0203±2.5424 | 0.4027±0.2233 / 0.6427±0.1798 | 0.4793±0.2521 / 0.7403±0.1972 | 0.5078±0.0770 / 0.5632±0.0869 |
| lorenz | chaotic | 3/3 | 0.9748±0.0234 / 0.8264±0.0782 | 2.1192±0.6569 / 4.5131±2.3589 | 0.2837±0.1135 / 0.6434±0.1918 | 0.3564±0.1286 / 0.7386±0.2010 | 0.5341±0.1152 / 0.5757±0.0800 |
| sunspot | real | 3/3 | 0.8239±0.0389 / 0.7953±0.0410 | 4.8325±0.9444 / 5.6166±1.6599 | 0.6568±0.0924 / 0.7095±0.0720 | 0.7668±0.0956 / 0.8144±0.0682 | 0.6351±0.0514 / 0.5935±0.0459 |
| laser | real | 3/3 | 0.9119±0.0227 / 0.8808±0.0460 | 3.9116±0.6258 / 3.9274±0.8189 | 0.4823±0.0385 / 0.5730±0.0961 | 0.5842±0.0380 / 0.6840±0.0985 | 0.6380±0.0218 / 0.6360±0.0334 |

Early (frac<=0.05) / mid (0.25-0.5) / late (>=0.75) stage bins of r_eps10/n:

| task | early | mid | late |
|---|---|---|---|
| adding | 0.7360±0.0085 | 0.7056±0.0037 | 0.6973±0.0070 |
| copy | 0.7460±0.0126 | 0.7484±0.0256 | 0.7451±0.0394 |
| anbn | 0.7195±0.0413 | 0.4238±0.0947 | 0.2869±0.0776 |
| rotation | 0.6857±0.1081 | 0.2500±0.1233 | 0.0977±0.0482 |
| henon | 0.7160±0.0293 | 0.4957±0.0376 | 0.4809±0.0359 |
| mackeyglass | 0.7317±0.0237 | 0.7180±0.0526 | 0.4785±0.2322 |
| lorenz | 0.7419±0.0060 | 0.7327±0.0158 | 0.4555±0.2406 |
| sunspot | 0.7432±0.0064 | 0.7374±0.0224 | 0.6479±0.0972 |
| laser | 0.6978±0.0284 | 0.5343±0.0291 | 0.4869±0.0370 |

### Table B — width sweep, 4 diagnostic tasks x n∈{64,128,256,512}, final stage (frac=1.0)

| task | n | n_seeds | mass_top16 | mass_topn4 | stable_rank | r_eps10 | r_eps10/n | r_eps05/n |
|---|---|---|---|---|---|---|---|---|
| adding | 64 | 3/3 | 0.8000±0.0071 | 0.8000±0.0071 | 5.65±0.87 | 44.63±0.43 | 0.6974±0.0068 | 0.8033±0.0094 |
| adding | 128 | 3/3 | 0.6150±0.0039 | 0.7955±0.0020 | 8.48±1.23 | 91.08±1.22 | 0.7115±0.0095 | 0.8159±0.0085 |
| adding | 256 | 3/3 | 0.4220±0.0042 | 0.7861±0.0039 | 13.45±1.02 | 185.78±1.93 | 0.7257±0.0075 | 0.8275±0.0068 |
| adding | 512 | 3/3 | 0.5942±0.3075 | 0.8876±0.0918 | 14.46±18.14 | 234.75±157.72 | 0.4585±0.3080 | 0.5372±0.3478 |
| copy | 64 | 3/3 | 0.7312±0.0388 | 0.7312±0.0388 | 8.03±3.59 | 48.27±1.21 | 0.7542±0.0189 | 0.8473±0.0136 |
| copy | 128 | 3/3 | 0.9900±0.0149 | 0.9983±0.0051 | 2.15±0.78 | 13.57±10.61 | 0.1061±0.0829 | 0.1534±0.1299 |
| copy | 256 | 3/3 | 0.9145±0.0830 | 0.9996±0.0007 | 4.65±3.05 | 24.36±13.24 | 0.0951±0.0517 | 0.1318±0.0712 |
| copy | 512 | 3/3 | 0.9683±0.0247 | 0.9999±0.0002 | 3.87±1.77 | 22.90±9.02 | 0.0447±0.0176 | 0.0729±0.0334 |
| anbn | 64 | 3/3 | 0.9151±0.0909 | 0.9151±0.0909 | 2.05±0.41 | 17.49±4.90 | 0.2733±0.0766 | 0.3409±0.0722 |
| anbn | 128 | 3/3 | 0.9120±0.0928 | 0.9227±0.0925 | 2.30±0.56 | 23.47±11.67 | 0.1833±0.0912 | 0.2221±0.0913 |
| anbn | 256 | 3/3 | 0.9176±0.0939 | 0.9244±0.0924 | 1.61±0.41 | 28.48±24.61 | 0.1112±0.0961 | 0.1252±0.0980 |
| anbn | 512 | 3/3 | 0.9295±0.0998 | 0.9481±0.1026 | 1.52±0.47 | 42.37±51.06 | 0.0827±0.0997 | 0.0942±0.1039 |
| rotation | 64 | 3/3 | 0.9999±0.0001 | 0.9999±0.0001 | 1.72±0.68 | 5.72±2.91 | 0.0893±0.0455 | 0.1215±0.0520 |
| rotation | 128 | 3/3 | 0.9994±0.0010 | 1.0000±0.0000 | 2.17±0.36 | 9.27±2.52 | 0.0724±0.0197 | 0.0912±0.0244 |
| rotation | 256 | 3/3 | 0.9989±0.0011 | 1.0000±0.0000 | 1.91±0.50 | 8.20±3.34 | 0.0320±0.0130 | 0.0427±0.0169 |
| rotation | 512 | 3/3 | 1.0000±0.0000 | 1.0000±0.0000 | 1.33±0.31 | 3.28±1.74 | 0.0064±0.0034 | 0.0086±0.0047 |

Before training (first stage reached, frac=stage_lo) vs at convergence (frac=1.0), r_eps10/n — shows directly whether the required rank grows linearly with width and whether training compresses it:

| task | n | frac (pre) | r_eps10/n (pre-training) | r_eps10/n (converged) | Δ (converged - pre) |
|---|---|---|---|---|---|
| adding | 64 | 0.01 | 0.7431±0.0028 | 0.6974±0.0068 | -0.0457 |
| adding | 128 | 0.01 | 0.7394±0.0041 | 0.7115±0.0095 | -0.0279 |
| adding | 256 | 0.01 | 0.7382±0.0026 | 0.7257±0.0075 | -0.0125 |
| adding | 512 | 0.01 | 0.7354±0.0046 | 0.4585±0.3080 | -0.2769 |
| copy | 64 | 0.01 | 0.7441±0.0138 | 0.7542±0.0189 | +0.0100 |
| copy | 128 | 0.01 | 0.7426±0.0117 | 0.1061±0.0829 | -0.6366 |
| copy | 256 | 0.01 | 0.7405±0.0106 | 0.0951±0.0517 | -0.6454 |
| copy | 512 | 0.01 | 0.5944±0.2070 | 0.0447±0.0176 | -0.5497 |
| anbn | 64 | 0.01 | 0.7323±0.0382 | 0.2733±0.0766 | -0.4590 |
| anbn | 128 | 0.01 | 0.7542±0.0251 | 0.1833±0.0912 | -0.5709 |
| anbn | 256 | 0.01 | 0.7528±0.0286 | 0.1112±0.0961 | -0.6415 |
| anbn | 512 | 0.01 | 0.4416±0.2086 | 0.0827±0.0997 | -0.3588 |
| rotation | 64 | 0.01 | 0.6952±0.1447 | 0.0893±0.0455 | -0.6059 |
| rotation | 128 | 0.01 | 0.3322±0.2045 | 0.0724±0.0197 | -0.2598 |
| rotation | 256 | 0.01 | 0.2040±0.1715 | 0.0320±0.0130 | -0.1720 |
| rotation | 512 | 0.01 | 0.1234±0.0824 | 0.0064±0.0034 | -0.1170 |

### Table B' — separate accounting for tagged files (these enter no mean of Table A/B/C)

Tables A/B/C read only the untagged roster files (3 seeds per task and width, one file per seed). This section lists the files with a suffix separately, for use when the text quotes the extra seeds. Stability is judged from each run's own training curve: final/best > 1.50 counts as destabilised. `turn` is the last step at which the curve reached its best value (where it turned), `cross` is the first later step above the threshold, and the value in brackets is the fraction of the run's own step budget. The criterion only tells which runs within the same (task, n) cell are comparable with each other. It is not a general cross-task health check (on tasks whose training metric converges to a very small value, relative jitter at the end can exceed the threshold anyway), so it does not appear as a column of Table A/B.

**adding, n=512** — per file (final stage frac=1.0, pooled over the file's 5 age checkpoints). Rows tagged `roster` are the untagged files that Table B pools. They are listed here only so that the stability split below can be checked, and their numbers match Table B exactly:

| file | seed | tag | steps | r_eps10/n | r_eps05/n | r_eps10 | stable_rank | mass_top16 | best | final | final/best | turn | cross | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `adding_n512_s0.json` | 0 | roster | 20000 | 0.6111±0.0092 | 0.7254±0.0068 | 312.90±4.72 | 2.51±0.24 | 0.5539±0.0335 | 0.3002 | 0.3002 | 1.000 | 19501 (0.98) | — | completed |
| `adding_n512_s1.json` | 1 | roster | 20000 | 0.7218±0.0021 | 0.8209±0.0019 | 369.55±1.10 | 39.21±1.61 | 0.2528±0.0058 | 0.2686 | 0.2689 | 1.001 | 19001 (0.95) | — | completed |
| `adding_n512_s2.json` | 2 | roster | 20000 | 0.0426±0.0050 | 0.0653±0.0075 | 21.80±2.57 | 1.67±0.21 | 0.9759±0.0058 | 0.2828 | 0.6633 | 2.345 | 13501 (0.68) | 17501 (0.88) | destabilised |
| `adding_n512_s0_long.json` | 0 | long | 40000 | 0.6683±0.0011 | 0.7677±0.0009 | 342.15±0.55 | 37.21±1.92 | 0.2658±0.0003 | 0.1873 | 0.1889 | 1.009 | 38501 (0.96) | — | completed |
| `adding_n512_s2_long.json` | 2 | long | 40000 | 0.7284±0.0009 | 0.8279±0.0003 | 372.95±0.48 | 42.98±0.27 | 0.2581±0.0016 | 0.1687 | 0.1711 | 1.014 | 29001 (0.73) | — | completed |
| `adding_n512_s3_extra.json` | 3 | extra | 20000 | 0.7361±0.0031 | 0.8349±0.0017 | 376.90±1.61 | 41.23±1.03 | 0.2593±0.0063 | 0.2467 | 0.2467 | 1.000 | 19501 (0.98) | — | completed |
| `adding_n512_s4_extra.json` | 4 | extra | 20000 | 0.7159±0.0023 | 0.8160±0.0020 | 366.55±1.18 | 44.02±1.55 | 0.2528±0.0064 | 0.2811 | 0.2811 | 1.000 | 19501 (0.98) | — | completed |
| `adding_n512_s5_extra.json` | 5 | extra | 20000 | 0.7416±0.0013 | 0.8387±0.0007 | 379.70±0.65 | 43.86±1.16 | 0.2473±0.0050 | 0.2497 | 0.2497 | 1.000 | 19501 (0.98) | — | completed |
| `adding_n512_s6_extra.json` | 6 | extra | 20000 | 0.7083±0.0037 | 0.8103±0.0033 | 362.65±1.89 | 43.03±2.00 | 0.2512±0.0073 | 0.2583 | 0.2583 | 1.000 | 19501 (0.98) | — | completed |
| `adding_n512_s7_extra.json` | 7 | extra | 20000 | 0.0137 | 0.0176 | 7.00 | 1.83±0.01 | 1.0000±0.0000 | 0.3077 | 0.7676 | 2.494 | 14001 (0.70) | 15001 (0.75) | destabilised |

**adding, n=512** — pooled readings side by side (final stage frac=1.0):

| set | n_files | n_seeds | r_eps10/n | r_eps05/n | r_eps10 | stable_rank | mass_top16 | note |
|---|---|---|---|---|---|---|---|---|
| roster (untagged, = this row of Table B) | 3 | 3 | 0.4585±0.3080 | 0.5372±0.3478 | 234.75±157.72 | 14.46±18.14 | 0.5942±0.3075 |  |
| tagged set | 7 | 7 | 0.6160±0.2506 | 0.7019±0.2843 | 315.41±128.29 | 36.31±14.50 | 0.3621±0.2643 |  |
| union (all files) | 10 | 8 | 0.5688±0.2757 | 0.6525±0.3106 | 291.21±141.14 | 29.75±18.50 | 0.4317±0.2951 | seed 0,2 appears in both untagged and tagged files; both are counted |
| union, trained to completion (final/best <= 1.50) | 8 | 7 | 0.7039±0.0416 | 0.8052±0.0371 | 360.42±21.31 | 36.76±13.36 | 0.2927±0.1008 | includes adding_n512_s0, adding_n512_s0_long, adding_n512_s1, adding_n512_s2_long, adding_n512_s3_extra, adding_n512_s4_extra, adding_n512_s5_extra, adding_n512_s6_extra |
| union, destabilised (final/best > 1.50) | 2 | 2 | 0.0281±0.0156 | 0.0415±0.0257 | 14.40±7.99 | 1.75±0.17 | 0.9880±0.0133 | includes adding_n512_s2, adding_n512_s7_extra |


### Table C — main measure (J−S_t) vs bd measure (J−blockdiag(J)) on r_eps10, n=64, pooled over all stages and ages

diff = r_eps10 − bd_r_eps10, paired per checkpoint before taking mean±std; pct = mean diff as a percentage of the bd mean.

| task family | n_records | r_eps10 (main) | r_eps10 (bd) | diff (main−bd) | diff / bd |
|---|---|---|---|---|---|
| diagnostic | 360 | 36.49 | 37.05 | -0.56±1.05 | -1.5% |
| chaotic | 270 | 39.47 | 39.51 | -0.03±0.84 | -0.1% |
| real | 180 | 41.04 | 41.76 | -0.72±0.67 | -1.7% |
| all | 810 | 38.50 | 38.92 | -0.42±0.95 | -1.1% |

## Key numbers (auto-generated, numbers only)

- n=64 final-stage mass_top16: highest rotation (0.9999), lowest copy (0.7312), data for 9/9 tasks.
- Late (>=0.75) vs early (<=0.05) stage, mean change of r_eps10/n: -0.2379 (9 tasks, per task in Table A).
- adding: r_eps10/n over n=64→512 goes from 0.6974 to 0.4585 (data for 4/4 widths).
- copy: r_eps10/n over n=64→512 goes from 0.7542 to 0.0447 (data for 4/4 widths).
- anbn: r_eps10/n over n=64→512 goes from 0.2733 to 0.0827 (data for 4/4 widths).
- rotation: r_eps10/n over n=64→512 goes from 0.0893 to 0.0064 (data for 4/4 widths).
- diagnostic task family (n=64, pooled over all stages, 360 records): r_eps10 main-measure mean 36.49, bd-measure mean 37.05, difference -0.56 (-1.5%).
- chaotic task family (n=64, pooled over all stages, 270 records): r_eps10 main-measure mean 39.47, bd-measure mean 39.51, difference -0.03 (-0.1%).
- real task family (n=64, pooled over all stages, 180 records): r_eps10 main-measure mean 41.04, bd-measure mean 41.76, difference -0.72 (-1.7%).
- All tasks (n=64, 810 records): r_eps10 main-measure mean 38.50 vs bd-measure mean 38.92, difference -0.42 (-1.1%).

## Figures

- `results/figures/fig_r2_spectrum_stage.pdf` — written
- `results/figures/fig_r2_spectrum_width.pdf` — written
- `results/figures/fig_r2_spectrum_age.pdf` — written

