# Revision re-run vs. original raw data vs. manuscript

Original JSONs: `results/` (round-1 campaign). Re-runs: `results/revise_repro/` (local 2 x RTX 3080, Python 3.10.20, torch 2.6.0+cu124).


## E6 rank x spectral clip -- Lorenz, SK-RTRL r=8, n=64, batch 8, 15000 steps

Final NMSE = `metric` of the last logged record (step 14750).


| clip | seed | paper | original JSON | re-run | re-run - original | peak MB (orig / repro) |
|---|---|---|---|---|---|---|
| clip0 | 0 | 0.0110 | 0.0114 | 0.0033 | -0.0081 | 23.5 / 23.5 |
| clip0 | 1 | 0.0090 | 0.0089 | 0.0040 | -0.0049 | 23.5 / 23.5 |
| clip0 | 2 | 0.0220 | 0.0223 | 0.0042 | -0.0181 | 23.5 / 23.5 |
| clip0.9 | 0 | 0.0054 | 0.0054 | 0.0054 | -0.0000 | 23.5 / 23.5 |
| clip0.9 | 1 | 0.0057 | 0.0057 | 0.0059 | +0.0002 | 23.5 / 23.5 |
| clip0.9 | 2 | 0.0053 | 0.0053 | 0.0056 | +0.0003 | 23.5 / 23.5 |

## Fig. 9 memory benchmark -- anbn, seed 0, batch 8, 1500 steps, shadow OFF


| algo | n | paper (MB) | original (MB) | re-run (MB) | identical? | wall s (orig / repro) |
|---|---|---|---|---|---|---|
| exact | 128 | 150 | 149.610 | 149.610 | yes | 5.0 / 5.3 |
| exact | 256 | 1062 | 1061.622 | 1061.622 | yes | 11.3 / 11.6 |
| exact | 384 | 3520 | 3520.196 | 3520.196 | yes | 41.2 / 41.8 |
| exact | 512 | 8293 | 8293.489 | 8293.489 | yes | 119.6 / 119.8 |
| snap1 | 128 | -- | 19.670 | 19.670 | yes | 5.8 / 6.1 |
| snap1 | 256 | -- | 29.776 | 29.776 | yes | 5.7 / 6.1 |
| snap1 | 384 | -- | 46.426 | 46.426 | yes | 5.8 / 6.6 |
| snap1 | 512 | -- | 69.712 | 69.712 | yes | 5.9 / 6.0 |
| skrtrl-r4 | 128 | -- | 35.748 | 35.748 | yes | 168.5 / 186.5 |
| skrtrl-r4 | 256 | -- | 93.010 | 93.010 | yes | 381.0 / 396.5 |
| skrtrl-r4 | 384 | -- | 187.679 | 187.679 | yes | 582.2 / 630.9 |
| skrtrl-r4 | 512 | -- | 320.747 | 320.747 | yes | 920.7 / 995.0 |
| skrtrl-r16 | 128 | 55 | 54.682 | 54.682 | yes | 176.7 / 198.3 |
| skrtrl-r16 | 256 | 167 | 166.585 | 166.585 | yes | 392.7 / 431.1 |
| skrtrl-r16 | 384 | 352 | 352.082 | 352.082 | yes | 603.9 / 656.7 |
| skrtrl-r16 | 512 | 614 | 613.954 | 613.954 | yes | 893.9 / 958.3 |

## E6 stability statistics (post-washout = logged steps >= 1000)

The last-step NMSE is a poor summary of an *unstable* run. These aggregate
statistics separate the reproducible signal (the clip suppresses excursions)
from the irreproducible one (where the last excursion happens to land).


| clip | seed | source | final | max post-washout | # points > 0.02 | median post-washout |
|---|---|---|---|---|---|---|
| clip0 | 0 | original | 0.0114 | 0.1133 | 3 | 0.0042 |
| clip0 | 0 | re-run | 0.0033 | 0.0143 | 0 | 0.0040 |
| clip0 | 1 | original | 0.0089 | 0.0198 | 0 | 0.0042 |
| clip0 | 1 | re-run | 0.0040 | 0.1398 | 6 | 0.0055 |
| clip0 | 2 | original | 0.0223 | 0.2791 | 3 | 0.0030 |
| clip0 | 2 | re-run | 0.0042 | 0.0159 | 0 | 0.0035 |
| clip0.9 | 0 | original | 0.0054 | 0.0122 | 0 | 0.0053 |
| clip0.9 | 0 | re-run | 0.0054 | 0.0122 | 0 | 0.0054 |
| clip0.9 | 1 | original | 0.0057 | 0.0104 | 0 | 0.0053 |
| clip0.9 | 1 | re-run | 0.0059 | 0.0104 | 0 | 0.0053 |
| clip0.9 | 2 | original | 0.0053 | 0.0110 | 0 | 0.0053 |
| clip0.9 | 2 | re-run | 0.0056 | 0.0110 | 0 | 0.0053 |
