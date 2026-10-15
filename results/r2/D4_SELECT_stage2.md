# D4 learning-rate selection -- stage 2

Generated 2026-10-15 12:31:20 by `code/make_r2_d4_select.py select --stage 2`.

- input directory: `results/r2/d4_stage2`
- score per run: mean `metric` over records with `step >= 0.8 * args.steps` (last 20% window)
- score per config: unweighted mean over tuning seeds [100, 101] (the seed is the unit)
- metric direction: accuracy, higher is better for anbn, copy; MSE, lower is better for every other D4 task
- ties: broken towards the smaller LR, then the smaller TBPTT window (relative tolerance 1e-12)
- configs with only one tuning seed: not eligible
- runs with no record inside the last 20%: treated as unusable
- TBPTT window grid [25, 50], window flag `--tbptt_window`, bare tag means window 25

## Summary -- selected configuration per (task, estimator)

| task | exact | snap1 | skrtrl-r4 | skrtrl-r16 | skrtrl-r32 | uoro | kfrtrl | rflo | tbptt |
|---|---|---|---|---|---|---|---|---|---|
| adding | lr3e-03 | lr1e-02 | lr3e-03 | lr3e-03 | lr3e-03 | lr1e-02 | lr3e-03 | lr3e-03 | lr3e-03/w25 |
| copy | lr1e-02 | lr1e-04 | lr1e-02 | lr1e-02 | lr3e-03 | lr1e-03 | lr1e-02 | lr3e-03 | lr1e-04/w25 (=) |
| anbn | lr1e-03 | lr1e-03 | lr1e-03 | lr1e-03 | lr1e-03 | lr1e-03 | lr1e-03 | lr1e-03 | lr3e-03/w50 |
| rotation | lr1e-04 | lr3e-04 | lr1e-04 | lr3e-04 | lr1e-04 | lr1e-04 | lr3e-04 | lr1e-03 | lr1e-03/w25 |
| henon | lr1e-03 | lr1e-03 | lr1e-03 | lr1e-03 | lr1e-03 | lr3e-04 | lr1e-03 | lr3e-03 | lr1e-02/w25 |
| mackeyglass | lr3e-04 | lr3e-04 | lr3e-04 | lr3e-04 | lr3e-04 | lr3e-04 | lr3e-04 | lr1e-04 | lr1e-02/w25 |
| lorenz | lr3e-04 | lr3e-04 | lr3e-04 | lr3e-04 | lr3e-04 | lr3e-04 | lr3e-04 | lr1e-04 | lr1e-02/w25 |
| sunspot | lr1e-04 | lr1e-04 | lr1e-04 | lr1e-04 | lr1e-04 | lr1e-04 | lr1e-04 | lr1e-04 | lr3e-03/w50 |
| laser | lr1e-03 | lr1e-03 | lr1e-03 | lr1e-03 | lr1e-03 | lr3e-04 | lr1e-03 | lr3e-04 | lr1e-02/w25 |

`(B)` boundary, extension emitted; `(B!)` boundary but the extension cap is reached; `(=)` tied with another config.

## Counts

| quantity | value |
|---|---|
| (task, estimator) pairs | 81 |
| pairs with a selection | 81 |
| pairs pending (no eligible config) | 0 |
| pairs at a grid boundary, extension emitted | 0 |
| pairs at a grid boundary, cap reached | 0 |
| pairs with fewer than 2 eligible configs | 0 |
| pairs with a tie at the top | 1 |
| runs found | 324 |
| missing runs | 0 |
| unreadable / malformed files | 0 |

## Per-pair detail

### adding / exact

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 0.199193 | 0.224103 | **0.211648** | 2/2 | 1 |  |
| lr1e-02 | 0.01 | 0.274554 | 0.227904 | **0.251229** | 2/2 | 2 | ext-L1 |

- **selected**: `lr3e-03` (lr 0.003), seed-mean 0.211648
- top-2 for stage 2: `lr3e-03` (0.211648), `lr1e-02` (0.251229)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### adding / snap1

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 0.205701 | 0.239722 | **0.222712** | 2/2 | 2 |  |
| lr1e-02 | 0.01 | 0.20082 | 0.226108 | **0.213464** | 2/2 | 1 | ext-L1 |

- **selected**: `lr1e-02` (lr 0.01), seed-mean 0.213464
- top-2 for stage 2: `lr1e-02` (0.213464), `lr3e-03` (0.222712)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=1

### adding / skrtrl-r4

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 0.201047 | 0.230953 | **0.216** | 2/2 | 1 |  |
| lr1e-02 | 0.01 | 0.22789 | 0.242263 | **0.235077** | 2/2 | 2 | ext-L1 |

- **selected**: `lr3e-03` (lr 0.003), seed-mean 0.216
- top-2 for stage 2: `lr3e-03` (0.216), `lr1e-02` (0.235077)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### adding / skrtrl-r16

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 0.201193 | 0.229856 | **0.215524** | 2/2 | 1 |  |
| lr1e-02 | 0.01 | 0.225536 | 0.225141 | **0.225339** | 2/2 | 2 | ext-L1 |

- **selected**: `lr3e-03` (lr 0.003), seed-mean 0.215524
- top-2 for stage 2: `lr3e-03` (0.215524), `lr1e-02` (0.225339)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### adding / skrtrl-r32

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 0.197397 | 0.225534 | **0.211465** | 2/2 | 1 |  |
| lr1e-02 | 0.01 | 0.23046 | 0.223478 | **0.226969** | 2/2 | 2 | ext-L1 |

- **selected**: `lr3e-03` (lr 0.003), seed-mean 0.211465
- top-2 for stage 2: `lr3e-03` (0.211465), `lr1e-02` (0.226969)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### adding / uoro

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-02 | 0.01 | 0.218267 | 0.272334 | **0.2453** | 2/2 | 1 | ext-L1 |
| lr3e-02 | 0.03 | 0.261155 | 0.264373 | **0.262764** | 2/2 | 2 | ext-L2 |

- **selected**: `lr1e-02` (lr 0.01), seed-mean 0.2453
- top-2 for stage 2: `lr1e-02` (0.2453), `lr3e-02` (0.262764)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=1

### adding / kfrtrl

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 0.198835 | 0.232333 | **0.215584** | 2/2 | 1 |  |
| lr1e-02 | 0.01 | 0.218762 | 0.216645 | **0.217703** | 2/2 | 2 | ext-L1 |

- **selected**: `lr3e-03` (lr 0.003), seed-mean 0.215584
- top-2 for stage 2: `lr3e-03` (0.215584), `lr1e-02` (0.217703)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### adding / rflo

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 0.20822 | 0.240339 | **0.224279** | 2/2 | 1 |  |
| lr1e-02 | 0.01 | 0.21647 | 0.238631 | **0.227551** | 2/2 | 2 | ext-L1 |

- **selected**: `lr3e-03` (lr 0.003), seed-mean 0.224279
- top-2 for stage 2: `lr3e-03` (0.224279), `lr1e-02` (0.227551)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### adding / tbptt

direction: lower_is_better

| config | lr | window | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 25 | 0.199177 | 0.224105 | **0.211641** | 2/2 | 1 |  |
| lr3e-03_w50 | 0.003 | 50 | 0.199188 | 0.224104 | **0.211646** | 2/2 | 2 |  |

- **selected**: `lr3e-03` (lr 0.003, window 25), seed-mean 0.211641
- top-2 for stage 2: `lr3e-03` (0.211641), `lr3e-03_w50` (0.211646)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### copy / exact

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 0.145312 | 0.149766 | **0.147539** | 2/2 | 2 |  |
| lr1e-02 | 0.01 | 0.163008 | 0.156328 | **0.159668** | 2/2 | 1 | ext-L1 |

- **selected**: `lr1e-02` (lr 0.01), seed-mean 0.159668
- top-2 for stage 2: `lr1e-02` (0.159668), `lr3e-03` (0.147539)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=1

### copy / snap1

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-05 | 3e-05 | 0.111641 | 0.128789 | **0.120215** | 2/2 | 2 | ext-L1 |
| lr1e-04 | 0.0001 | 0.125078 | 0.126445 | **0.125762** | 2/2 | 1 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.125762
- top-2 for stage 2: `lr1e-04` (0.125762), `lr3e-05` (0.120215)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### copy / skrtrl-r4

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 0.1475 | 0.14832 | **0.14791** | 2/2 | 2 |  |
| lr1e-02 | 0.01 | 0.160273 | 0.162695 | **0.161484** | 2/2 | 1 | ext-L1 |

- **selected**: `lr1e-02` (lr 0.01), seed-mean 0.161484
- top-2 for stage 2: `lr1e-02` (0.161484), `lr3e-03` (0.14791)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=1

### copy / skrtrl-r16

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 0.143359 | 0.12125 | **0.132305** | 2/2 | 2 |  |
| lr1e-02 | 0.01 | 0.13375 | 0.137578 | **0.135664** | 2/2 | 1 | ext-L1 |

- **selected**: `lr1e-02` (lr 0.01), seed-mean 0.135664
- top-2 for stage 2: `lr1e-02` (0.135664), `lr3e-03` (0.132305)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=1

### copy / skrtrl-r32

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.124141 | 0.125664 | **0.124902** | 2/2 | 2 |  |
| lr3e-03 | 0.003 | 0.151914 | 0.15043 | **0.151172** | 2/2 | 1 |  |

- **selected**: `lr3e-03` (lr 0.003), seed-mean 0.151172
- top-2 for stage 2: `lr3e-03` (0.151172), `lr1e-04` (0.124902)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### copy / uoro

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.123125 | 0.130469 | **0.126797** | 2/2 | 2 |  |
| lr1e-03 | 0.001 | 0.128008 | 0.128477 | **0.128242** | 2/2 | 1 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.128242
- top-2 for stage 2: `lr1e-03` (0.128242), `lr1e-04` (0.126797)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### copy / kfrtrl

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-02 | 0.01 | 0.152188 | 0.196992 | **0.17459** | 2/2 | 1 | ext-L1 |
| lr3e-02 | 0.03 | 0.142734 | 0.127266 | **0.135** | 2/2 | 2 | ext-L2 |

- **selected**: `lr1e-02` (lr 0.01), seed-mean 0.17459
- top-2 for stage 2: `lr1e-02` (0.17459), `lr3e-02` (0.135)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=1

### copy / rflo

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 0.154102 | 0.166211 | **0.160156** | 2/2 | 1 |  |
| lr1e-02 | 0.01 | 0.166211 | 0.147695 | **0.156953** | 2/2 | 2 | ext-L1 |

- **selected**: `lr3e-03` (lr 0.003), seed-mean 0.160156
- top-2 for stage 2: `lr3e-03` (0.160156), `lr1e-02` (0.156953)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### copy / tbptt

direction: higher_is_better

| config | lr | window | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 25 | 0.125742 | 0.132227 | **0.128984** | 2/2 | 1 |  |
| lr1e-04_w50 | 0.0001 | 50 | 0.125742 | 0.132227 | **0.128984** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001, window 25), seed-mean 0.128984
- top-2 for stage 2: `lr1e-04` (0.128984), `lr1e-04_w50` (0.128984)
- tied with the winner: `lr1e-04_w50` (tie broken towards the smaller LR)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### anbn / exact

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.900977 | 0.899531 | **0.900254** | 2/2 | 2 |  |
| lr1e-03 | 0.001 | 0.911406 | 0.905703 | **0.908555** | 2/2 | 1 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.908555
- top-2 for stage 2: `lr1e-03` (0.908555), `lr1e-04` (0.900254)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### anbn / snap1

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.901797 | 0.908125 | **0.904961** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.898633 | 0.903281 | **0.900957** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.904961
- top-2 for stage 2: `lr1e-03` (0.904961), `lr3e-03` (0.900957)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### anbn / skrtrl-r4

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.916406 | 0.914688 | **0.915547** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.895898 | 0.894961 | **0.89543** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.915547
- top-2 for stage 2: `lr1e-03` (0.915547), `lr3e-03` (0.89543)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### anbn / skrtrl-r16

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.923359 | 0.917422 | **0.920391** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.894531 | 0.908945 | **0.901738** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.920391
- top-2 for stage 2: `lr1e-03` (0.920391), `lr3e-03` (0.901738)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### anbn / skrtrl-r32

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.901055 | 0.900586 | **0.90082** | 2/2 | 2 |  |
| lr1e-03 | 0.001 | 0.93043 | 0.934062 | **0.932246** | 2/2 | 1 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.932246
- top-2 for stage 2: `lr1e-03` (0.932246), `lr1e-04` (0.90082)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### anbn / uoro

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.885742 | 0.890977 | **0.888359** | 2/2 | 2 |  |
| lr1e-03 | 0.001 | 0.888437 | 0.888789 | **0.888613** | 2/2 | 1 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.888613
- top-2 for stage 2: `lr1e-03` (0.888613), `lr3e-04` (0.888359)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### anbn / kfrtrl

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.889648 | 0.902578 | **0.896113** | 2/2 | 2 |  |
| lr1e-03 | 0.001 | 0.9025 | 0.89457 | **0.898535** | 2/2 | 1 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.898535
- top-2 for stage 2: `lr1e-03` (0.898535), `lr3e-04` (0.896113)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### anbn / rflo

direction: higher_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.905117 | 0.904766 | **0.904941** | 2/2 | 2 |  |
| lr1e-03 | 0.001 | 0.903633 | 0.908945 | **0.906289** | 2/2 | 1 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.906289
- top-2 for stage 2: `lr1e-03` (0.906289), `lr3e-04` (0.904941)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### anbn / tbptt

direction: higher_is_better

| config | lr | window | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 25 | 0.88375 | 0.890234 | **0.886992** | 2/2 | 2 |  |
| lr3e-03_w50 | 0.003 | 50 | 0.907539 | 0.907617 | **0.907578** | 2/2 | 1 |  |

- **selected**: `lr3e-03_w50` (lr 0.003, window 50), seed-mean 0.907578
- top-2 for stage 2: `lr3e-03_w50` (0.907578), `lr1e-03` (0.886992)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### rotation / exact

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.102677 | 0.112581 | **0.107629** | 2/2 | 1 |  |
| lr3e-04 | 0.0003 | 0.110116 | 0.115512 | **0.112814** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.107629
- top-2 for stage 2: `lr1e-04` (0.107629), `lr3e-04` (0.112814)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### rotation / snap1

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.0494051 | 0.05713 | **0.0532675** | 2/2 | 1 |  |
| lr1e-03 | 0.001 | 0.0720014 | 0.0787185 | **0.07536** | 2/2 | 2 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.0532675
- top-2 for stage 2: `lr3e-04` (0.0532675), `lr1e-03` (0.07536)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### rotation / skrtrl-r4

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.106797 | 0.109539 | **0.108168** | 2/2 | 1 |  |
| lr3e-04 | 0.0003 | 0.115668 | 0.108219 | **0.111944** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.108168
- top-2 for stage 2: `lr1e-04` (0.108168), `lr3e-04` (0.111944)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### rotation / skrtrl-r16

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.113576 | 0.112143 | **0.112859** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.141219 | 0.136233 | **0.138726** | 2/2 | 2 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.112859
- top-2 for stage 2: `lr3e-04` (0.112859), `lr3e-03` (0.138726)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### rotation / skrtrl-r32

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.108304 | 0.099399 | **0.103852** | 2/2 | 1 |  |
| lr3e-04 | 0.0003 | 0.103623 | 0.117379 | **0.110501** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.103852
- top-2 for stage 2: `lr1e-04` (0.103852), `lr3e-04` (0.110501)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### rotation / uoro

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.104533 | 0.104531 | **0.104532** | 2/2 | 1 |  |
| lr3e-04 | 0.0003 | 0.122607 | 0.123124 | **0.122866** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.104532
- top-2 for stage 2: `lr1e-04` (0.104532), `lr3e-04` (0.122866)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### rotation / kfrtrl

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.103821 | 0.116537 | **0.110179** | 2/2 | 1 |  |
| lr1e-03 | 0.001 | 0.111528 | 0.121249 | **0.116389** | 2/2 | 2 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.110179
- top-2 for stage 2: `lr3e-04` (0.110179), `lr1e-03` (0.116389)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### rotation / rflo

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.108929 | 0.0895721 | **0.0992507** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.120602 | 0.12362 | **0.122111** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.0992507
- top-2 for stage 2: `lr1e-03` (0.0992507), `lr3e-03` (0.122111)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### rotation / tbptt

direction: lower_is_better

| config | lr | window | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 25 | 0.0280896 | 0.0353557 | **0.0317226** | 2/2 | 1 |  |
| lr1e-03_w50 | 0.001 | 50 | 0.0232636 | 0.044183 | **0.0337233** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001, window 25), seed-mean 0.0317226
- top-2 for stage 2: `lr1e-03` (0.0317226), `lr1e-03_w50` (0.0337233)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### henon / exact

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.00409647 | 0.00458204 | **0.00433925** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.00779518 | 0.00607444 | **0.00693481** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.00433925
- top-2 for stage 2: `lr1e-03` (0.00433925), `lr3e-03` (0.00693481)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### henon / snap1

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.00418031 | 0.00473844 | **0.00445938** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.00756628 | 0.00548953 | **0.00652791** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.00445938
- top-2 for stage 2: `lr1e-03` (0.00445938), `lr3e-03` (0.00652791)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### henon / skrtrl-r4

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.00443875 | 0.00463826 | **0.00453851** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.00763495 | 0.00975198 | **0.00869346** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.00453851
- top-2 for stage 2: `lr1e-03` (0.00453851), `lr3e-03` (0.00869346)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### henon / skrtrl-r16

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.00450937 | 0.00492142 | **0.0047154** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.00699855 | 0.00803847 | **0.00751851** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.0047154
- top-2 for stage 2: `lr1e-03` (0.0047154), `lr3e-03` (0.00751851)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### henon / skrtrl-r32

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.00408641 | 0.00513997 | **0.00461319** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.00843708 | 0.00772334 | **0.00808021** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.00461319
- top-2 for stage 2: `lr1e-03` (0.00461319), `lr3e-03` (0.00808021)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### henon / uoro

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.724951 | 0.723946 | **0.724449** | 2/2 | 2 |  |
| lr3e-04 | 0.0003 | 0.578381 | 0.660188 | **0.619285** | 2/2 | 1 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.619285
- top-2 for stage 2: `lr3e-04` (0.619285), `lr1e-04` (0.724449)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### henon / kfrtrl

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.0117997 | 0.00764844 | **0.00972408** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.0137902 | 0.0171742 | **0.0154822** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.00972408
- top-2 for stage 2: `lr1e-03` (0.00972408), `lr3e-03` (0.0154822)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### henon / rflo

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 0.0499418 | 0.0935515 | **0.0717466** | 2/2 | 1 |  |
| lr1e-02 | 0.01 | 0.170386 | 0.329617 | **0.250002** | 2/2 | 2 | ext-L1 |

- **selected**: `lr3e-03` (lr 0.003), seed-mean 0.0717466
- top-2 for stage 2: `lr3e-03` (0.0717466), `lr1e-02` (0.250002)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### henon / tbptt

direction: lower_is_better

| config | lr | window | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 25 | 0.00745444 | 0.00793052 | **0.00769248** | 2/2 | 2 |  |
| lr1e-02 | 0.01 | 25 | 0.00511485 | 0.00406964 | **0.00459225** | 2/2 | 1 | ext-L1 |

- **selected**: `lr1e-02` (lr 0.01, window 25), seed-mean 0.00459225
- top-2 for stage 2: `lr1e-02` (0.00459225), `lr3e-03` (0.00769248)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=1

### mackeyglass / exact

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.00204519 | 0.00224255 | **0.00214387** | 2/2 | 1 |  |
| lr1e-03 | 0.001 | 0.00389442 | 0.00659047 | **0.00524245** | 2/2 | 2 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.00214387
- top-2 for stage 2: `lr3e-04` (0.00214387), `lr1e-03` (0.00524245)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### mackeyglass / snap1

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.00298563 | 0.00374874 | **0.00336718** | 2/2 | 1 |  |
| lr1e-03 | 0.001 | 0.00305803 | 0.00417423 | **0.00361613** | 2/2 | 2 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.00336718
- top-2 for stage 2: `lr3e-04` (0.00336718), `lr1e-03` (0.00361613)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### mackeyglass / skrtrl-r4

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.00268141 | 0.00259192 | **0.00263666** | 2/2 | 2 |  |
| lr3e-04 | 0.0003 | 0.00216754 | 0.00259017 | **0.00237886** | 2/2 | 1 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.00237886
- top-2 for stage 2: `lr3e-04` (0.00237886), `lr1e-04` (0.00263666)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### mackeyglass / skrtrl-r16

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.00216052 | 0.00231858 | **0.00223955** | 2/2 | 1 |  |
| lr1e-03 | 0.001 | 0.00537909 | 0.00367706 | **0.00452807** | 2/2 | 2 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.00223955
- top-2 for stage 2: `lr3e-04` (0.00223955), `lr1e-03` (0.00452807)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### mackeyglass / skrtrl-r32

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.00288236 | 0.00208218 | **0.00248227** | 2/2 | 1 |  |
| lr1e-03 | 0.001 | 0.00434851 | 0.00275058 | **0.00354954** | 2/2 | 2 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.00248227
- top-2 for stage 2: `lr3e-04` (0.00248227), `lr1e-03` (0.00354954)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### mackeyglass / uoro

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.00614191 | 0.00523194 | **0.00568693** | 2/2 | 2 |  |
| lr3e-04 | 0.0003 | 0.00536948 | 0.00552077 | **0.00544513** | 2/2 | 1 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.00544513
- top-2 for stage 2: `lr3e-04` (0.00544513), `lr1e-04` (0.00568693)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### mackeyglass / kfrtrl

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.0034756 | 0.00371813 | **0.00359687** | 2/2 | 1 |  |
| lr1e-03 | 0.001 | 0.0236942 | 0.0446407 | **0.0341674** | 2/2 | 2 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.00359687
- top-2 for stage 2: `lr3e-04` (0.00359687), `lr1e-03` (0.0341674)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### mackeyglass / rflo

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.00570045 | 0.00533161 | **0.00551603** | 2/2 | 1 |  |
| lr3e-04 | 0.0003 | 0.0150153 | 0.0106507 | **0.012833** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.00551603
- top-2 for stage 2: `lr1e-04` (0.00551603), `lr3e-04` (0.012833)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### mackeyglass / tbptt

direction: lower_is_better

| config | lr | window | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|---|
| lr1e-02 | 0.01 | 25 | 0.00243554 | 0.00231798 | **0.00237676** | 2/2 | 1 | ext-L1 |
| lr3e-02 | 0.03 | 25 | 0.0121377 | 0.00218231 | **0.00716** | 2/2 | 2 | ext-L2 |

- **selected**: `lr1e-02` (lr 0.01, window 25), seed-mean 0.00237676
- top-2 for stage 2: `lr1e-02` (0.00237676), `lr3e-02` (0.00716)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=1

### lorenz / exact

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.00122787 | 0.00146728 | **0.00134758** | 2/2 | 1 |  |
| lr1e-03 | 0.001 | 0.00422669 | 0.0277274 | **0.015977** | 2/2 | 2 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.00134758
- top-2 for stage 2: `lr3e-04` (0.00134758), `lr1e-03` (0.015977)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### lorenz / snap1

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.00143028 | 0.00206831 | **0.0017493** | 2/2 | 1 |  |
| lr1e-03 | 0.001 | 0.0028348 | 0.00470469 | **0.00376975** | 2/2 | 2 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.0017493
- top-2 for stage 2: `lr3e-04` (0.0017493), `lr1e-03` (0.00376975)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### lorenz / skrtrl-r4

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.00113266 | 0.00143207 | **0.00128236** | 2/2 | 1 |  |
| lr1e-03 | 0.001 | 0.0038628 | 0.00663275 | **0.00524778** | 2/2 | 2 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.00128236
- top-2 for stage 2: `lr3e-04` (0.00128236), `lr1e-03` (0.00524778)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### lorenz / skrtrl-r16

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.0012008 | 0.00374702 | **0.00247391** | 2/2 | 1 |  |
| lr1e-03 | 0.001 | 0.00191531 | 0.0046962 | **0.00330576** | 2/2 | 2 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.00247391
- top-2 for stage 2: `lr3e-04` (0.00247391), `lr1e-03` (0.00330576)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### lorenz / skrtrl-r32

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.00126692 | 0.00141104 | **0.00133898** | 2/2 | 1 |  |
| lr1e-03 | 0.001 | 0.0103011 | 0.00732547 | **0.0088133** | 2/2 | 2 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.00133898
- top-2 for stage 2: `lr3e-04` (0.00133898), `lr1e-03` (0.0088133)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### lorenz / uoro

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.0052051 | 0.00415163 | **0.00467837** | 2/2 | 2 |  |
| lr3e-04 | 0.0003 | 0.00413057 | 0.00369142 | **0.003911** | 2/2 | 1 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.003911
- top-2 for stage 2: `lr3e-04` (0.003911), `lr1e-04` (0.00467837)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### lorenz / kfrtrl

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr3e-04 | 0.0003 | 0.00220249 | 0.00211184 | **0.00215717** | 2/2 | 1 |  |
| lr1e-03 | 0.001 | 0.00339892 | 0.0347566 | **0.0190777** | 2/2 | 2 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.00215717
- top-2 for stage 2: `lr3e-04` (0.00215717), `lr1e-03` (0.0190777)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### lorenz / rflo

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.00457828 | 0.00386138 | **0.00421983** | 2/2 | 1 |  |
| lr3e-04 | 0.0003 | 0.00217976 | 0.00696612 | **0.00457294** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.00421983
- top-2 for stage 2: `lr1e-04` (0.00421983), `lr3e-04` (0.00457294)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### lorenz / tbptt

direction: lower_is_better

| config | lr | window | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|---|
| lr1e-02 | 0.01 | 25 | 0.00560683 | 0.00149184 | **0.00354933** | 2/2 | 1 | ext-L1 |
| lr3e-02 | 0.03 | 25 | 0.0105721 | 0.00255918 | **0.00656566** | 2/2 | 2 | ext-L2 |

- **selected**: `lr1e-02` (lr 0.01, window 25), seed-mean 0.00354933
- top-2 for stage 2: `lr1e-02` (0.00354933), `lr3e-02` (0.00656566)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=1

### sunspot / exact

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.136537 | 0.132388 | **0.134462** | 2/2 | 1 |  |
| lr3e-04 | 0.0003 | 0.138694 | 0.133373 | **0.136034** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.134462
- top-2 for stage 2: `lr1e-04` (0.134462), `lr3e-04` (0.136034)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### sunspot / snap1

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.142352 | 0.138504 | **0.140428** | 2/2 | 1 |  |
| lr3e-04 | 0.0003 | 0.143449 | 0.139696 | **0.141572** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.140428
- top-2 for stage 2: `lr1e-04` (0.140428), `lr3e-04` (0.141572)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### sunspot / skrtrl-r4

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.136778 | 0.132825 | **0.134802** | 2/2 | 1 |  |
| lr3e-04 | 0.0003 | 0.141506 | 0.138065 | **0.139786** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.134802
- top-2 for stage 2: `lr1e-04` (0.134802), `lr3e-04` (0.139786)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### sunspot / skrtrl-r16

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.136275 | 0.132558 | **0.134416** | 2/2 | 1 |  |
| lr3e-04 | 0.0003 | 0.139303 | 0.134776 | **0.137039** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.134416
- top-2 for stage 2: `lr1e-04` (0.134416), `lr3e-04` (0.137039)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### sunspot / skrtrl-r32

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.136379 | 0.132312 | **0.134346** | 2/2 | 1 |  |
| lr3e-04 | 0.0003 | 0.139118 | 0.134798 | **0.136958** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.134346
- top-2 for stage 2: `lr1e-04` (0.134346), `lr3e-04` (0.136958)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### sunspot / uoro

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.14167 | 0.138073 | **0.139872** | 2/2 | 1 |  |
| lr3e-04 | 0.0003 | 0.145982 | 0.144219 | **0.145101** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.139872
- top-2 for stage 2: `lr1e-04` (0.139872), `lr3e-04` (0.145101)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### sunspot / kfrtrl

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.13888 | 0.13353 | **0.136205** | 2/2 | 1 |  |
| lr3e-04 | 0.0003 | 0.146938 | 0.138916 | **0.142927** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.136205
- top-2 for stage 2: `lr1e-04` (0.136205), `lr3e-04` (0.142927)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### sunspot / rflo

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.144849 | 0.142323 | **0.143586** | 2/2 | 1 |  |
| lr3e-04 | 0.0003 | 0.19378 | 0.150577 | **0.172178** | 2/2 | 2 |  |

- **selected**: `lr1e-04` (lr 0.0001), seed-mean 0.143586
- top-2 for stage 2: `lr1e-04` (0.143586), `lr3e-04` (0.172178)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### sunspot / tbptt

direction: lower_is_better

| config | lr | window | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|---|
| lr3e-03 | 0.003 | 25 | 0.140482 | 0.134588 | **0.137535** | 2/2 | 2 |  |
| lr3e-03_w50 | 0.003 | 50 | 0.136381 | 0.132636 | **0.134508** | 2/2 | 1 |  |

- **selected**: `lr3e-03_w50` (lr 0.003, window 50), seed-mean 0.134508
- top-2 for stage 2: `lr3e-03_w50` (0.134508), `lr3e-03` (0.137535)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### laser / exact

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.0134033 | 0.0157528 | **0.0145781** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.0332776 | 0.0355421 | **0.0344098** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.0145781
- top-2 for stage 2: `lr1e-03` (0.0145781), `lr3e-03` (0.0344098)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### laser / snap1

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.0229858 | 0.0205519 | **0.0217688** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.0310876 | 0.0298362 | **0.0304619** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.0217688
- top-2 for stage 2: `lr1e-03` (0.0217688), `lr3e-03` (0.0304619)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### laser / skrtrl-r4

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.0149842 | 0.018663 | **0.0168236** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.0413878 | 0.0300789 | **0.0357333** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.0168236
- top-2 for stage 2: `lr1e-03` (0.0168236), `lr3e-03` (0.0357333)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### laser / skrtrl-r16

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.0197992 | 0.0171886 | **0.0184939** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.0383155 | 0.0437294 | **0.0410225** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.0184939
- top-2 for stage 2: `lr1e-03` (0.0184939), `lr3e-03` (0.0410225)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### laser / skrtrl-r32

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.0161625 | 0.0182641 | **0.0172133** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.0550749 | 0.0396584 | **0.0473666** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.0172133
- top-2 for stage 2: `lr1e-03` (0.0172133), `lr3e-03` (0.0473666)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### laser / uoro

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.19568 | 0.196248 | **0.195964** | 2/2 | 2 |  |
| lr3e-04 | 0.0003 | 0.14754 | 0.1583 | **0.15292** | 2/2 | 1 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.15292
- top-2 for stage 2: `lr3e-04` (0.15292), `lr1e-04` (0.195964)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### laser / kfrtrl

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-03 | 0.001 | 0.0436658 | 0.0569984 | **0.0503321** | 2/2 | 1 |  |
| lr3e-03 | 0.003 | 0.0805905 | 0.109834 | **0.095212** | 2/2 | 2 |  |

- **selected**: `lr1e-03` (lr 0.001), seed-mean 0.0503321
- top-2 for stage 2: `lr1e-03` (0.0503321), `lr3e-03` (0.095212)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### laser / rflo

direction: lower_is_better

| config | lr | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|
| lr1e-04 | 0.0001 | 0.239941 | 0.188773 | **0.214357** | 2/2 | 2 |  |
| lr3e-04 | 0.0003 | 0.126706 | 0.098531 | **0.112619** | 2/2 | 1 |  |

- **selected**: `lr3e-04` (lr 0.0003), seed-mean 0.112619
- top-2 for stage 2: `lr3e-04` (0.112619), `lr1e-04` (0.214357)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=0

### laser / tbptt

direction: lower_is_better

| config | lr | window | s100 | s101 | seed-mean | n | rank | flags |
|---|---|---|---|---|---|---|---|---|
| lr1e-02 | 0.01 | 25 | 0.0179112 | 0.0195718 | **0.0187415** | 2/2 | 1 | ext-L1 |
| lr3e-02 | 0.03 | 25 | 0.157476 | 0.0108811 | **0.0841784** | 2/2 | 2 | ext-L2 |

- **selected**: `lr1e-02` (lr 0.01, window 25), seed-mean 0.0187415
- top-2 for stage 2: `lr1e-02` (0.0187415), `lr3e-02` (0.0841784)
- boundary: stage-2 winner is the final LR (no further extension); extend_level=1

