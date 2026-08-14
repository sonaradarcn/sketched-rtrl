# Time-series tables rebuilt from the released raw runs

Per run: mean of the last five logged records. Across seeds: mean +/- population
standard deviation; the seed count is in parentheses. The grad-cosine column applies
the same per-run value over all seeds and all systems of the block at once.

## Table 6 - online next-step prediction on chaotic systems

| Method | Henon NMSE | Mackey-Glass NMSE | Lorenz NMSE | grad cosine |
|---|---|---|---|---|
| Exact RTRL | 0.0052 +/- 0.0010 (10) | 0.0180 +/- 0.0218 (10) | 0.0051 +/- 0.0040 (10) | -- |
| SK-RTRL r=32 | 0.0069 +/- 0.0016 (5) | 0.0070 +/- 0.0030 (5) | 0.0130 +/- 0.0184 (5) | 0.954 |
| SK-RTRL r=16 | 0.0064 +/- 0.0007 (10) | 0.0105 +/- 0.0091 (10) | 0.0128 +/- 0.0296 (10) | 0.938 |
| SK-RTRL r=4 | 0.0059 +/- 0.0007 (10) | 0.0065 +/- 0.0053 (10) | 0.0038 +/- 0.0021 (10) | 0.927 |
| SnAp-1 | 0.0052 +/- 0.0012 (10) | 0.0050 +/- 0.0010 (10) | 0.0024 +/- 0.0008 (10) | 0.505 |
| KF-RTRL | 0.0384 +/- 0.0709 (10) | 0.0286 +/- 0.0181 (10) | 0.0126 +/- 0.0079 (10) | 0.512 |
| RFLO | 0.0704 +/- 0.0163 (10) | 0.0572 +/- 0.0331 (10) | 0.0442 +/- 0.0124 (10) | 0.288 |
| UORO | 0.6583 +/- 0.2966 (10) | 0.0497 +/- 0.0238 (10) | 0.0425 +/- 0.0431 (10) | 0.056 |
| TBPTT | 0.0311 +/- 0.0030 (5) | 0.0080 +/- 0.0010 (5) | 0.0054 +/- 0.0007 (5) | -- |

## Table 7 - Sunspot and Santa Fe laser

| Method | Sunspot NMSE | Laser NMSE | grad cosine |
|---|---|---|---|
| Exact RTRL | 0.1451 +/- 0.0061 (10) | 0.0199 +/- 0.0041 (10) | -- |
| SK-RTRL r=16 | 0.1456 +/- 0.0042 (10) | 0.0217 +/- 0.0021 (10) | 0.883 |
| SK-RTRL r=4 | 0.1460 +/- 0.0049 (10) | 0.0239 +/- 0.0037 (10) | 0.867 |
| SnAp-1 | 0.1437 +/- 0.0019 (10) | 0.0214 +/- 0.0012 (10) | 0.439 |
| KF-RTRL | 0.1613 +/- 0.0153 (5) | 0.1788 +/- 0.2339 (5) | 0.431 |
| RFLO | 0.1536 +/- 0.0038 (10) | 0.2759 +/- 0.2241 (10) | 0.186 |
| UORO | 0.2119 +/- 0.0461 (10) | 0.4145 +/- 0.1523 (10) | 0.046 |
