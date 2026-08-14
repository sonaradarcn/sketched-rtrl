# Time-series tables rebuilt from the released raw runs

Per run: mean of the last five logged records. Across seeds: mean ¡À population
standard deviation; the seed count is in parentheses. The grad-cosine column is the
mean over every logged cosine of every seed and system in the block.

## Table 6 ¡ª online next-step prediction on chaotic systems

| Method | Henon NMSE | Mackey-Glass NMSE | Lorenz NMSE | grad cosine |
|---|---|---|---|---|
| Exact RTRL | 0.0052 ¡À 0.0010 (10) | 0.0180 ¡À 0.0218 (10) | 0.0051 ¡À 0.0040 (10) | -- |
| SK-RTRL r=32 | 0.0069 ¡À 0.0016 (5) | 0.0070 ¡À 0.0030 (5) | 0.0130 ¡À 0.0184 (5) | 0.954 |
| SK-RTRL r=16 | 0.0064 ¡À 0.0007 (10) | 0.0105 ¡À 0.0091 (10) | 0.0128 ¡À 0.0296 (10) | 0.938 |
| SK-RTRL r=4 | 0.0059 ¡À 0.0007 (10) | 0.0065 ¡À 0.0053 (10) | 0.0038 ¡À 0.0021 (10) | 0.927 |
| SnAp-1 | 0.0052 ¡À 0.0012 (10) | 0.0050 ¡À 0.0010 (10) | 0.0024 ¡À 0.0008 (10) | 0.505 |
| KF-RTRL | 0.0384 ¡À 0.0709 (10) | 0.0286 ¡À 0.0181 (10) | 0.0126 ¡À 0.0079 (10) | 0.512 |
| RFLO | 0.0704 ¡À 0.0163 (10) | 0.0572 ¡À 0.0331 (10) | 0.0442 ¡À 0.0124 (10) | 0.288 |
| UORO | 0.6583 ¡À 0.2966 (10) | 0.0497 ¡À 0.0238 (10) | 0.0425 ¡À 0.0431 (10) | 0.056 |
| TBPTT | 0.0311 ¡À 0.0030 (5) | 0.0080 ¡À 0.0010 (5) | 0.0054 ¡À 0.0007 (5) | -- |

## Table 7 ¡ª Sunspot and Santa Fe laser

| Method | Sunspot NMSE | Laser NMSE | grad cosine |
|---|---|---|---|
| Exact RTRL | 0.1451 ¡À 0.0061 (10) | 0.0199 ¡À 0.0041 (10) | -- |
| SK-RTRL r=16 | 0.1456 ¡À 0.0042 (10) | 0.0217 ¡À 0.0021 (10) | 0.883 |
| SK-RTRL r=4 | 0.1460 ¡À 0.0049 (10) | 0.0239 ¡À 0.0037 (10) | 0.867 |
| SnAp-1 | 0.1437 ¡À 0.0019 (10) | 0.0214 ¡À 0.0012 (10) | 0.439 |
| KF-RTRL | 0.1613 ¡À 0.0153 (5) | 0.1788 ¡À 0.2339 (5) | 0.431 |
| RFLO | 0.1536 ¡À 0.0038 (10) | 0.2759 ¡À 0.2241 (10) | 0.186 |
| UORO | 0.2119 ¡À 0.0461 (10) | 0.4145 ¡À 0.1523 (10) | 0.046 |
