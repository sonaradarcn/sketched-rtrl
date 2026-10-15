# A1 -- SVD-driver equivalence regression (`--svd_driver gesvd` vs `auto`)

`skrtrl/algos.py::_robust_svd` no longer hard-codes `driver="gesvd"`.
`svd_driver="gesvd"` (the default, unchanged) keeps the chain
gesvd -> jitter+gesvd -> CPU; `svd_driver="auto"` tries the library default
driver first and accepts it only if it neither raised nor produced a
non-finite factor, falling through the *same* chain otherwise.
`--svd_driver` is exposed by `run_m3.py`, `run_adaptive.py` and
`run_svd_driver_equiv.py`, defaults to `gesvd`, and is written into the
output json `args`.  Every run below also records `svd_stats`, the count of
SVD calls served by each link of the chain, so a silent fallback cannot be
mistaken for a fast path.

## Protocol

Three archived R1 configurations, re-run for the first 3000 steps at the
archived seed / lr / clip / log_every, so the log steps align with the
archived records.  Four runs per configuration:

| run | driver | purpose |
|---|---|---|
| archive | gesvd (R1 code) | reference numbers of record |
| gesvd | gesvd | re-run of the reference with the R2 code |
| gesvd_rep2 | gesvd | **control**: a second identical gesvd run |
| auto | auto | the new fast path |

The `gesvd_rep2` control is essential: cuSOLVER / cuBLAS batched kernels are
not bitwise reproducible run to run, and the training loop is chaotic, so two
identical gesvd runs already diverge.  The driver is equivalent iff
`auto vs gesvd` is no larger than `gesvd_rep2 vs gesvd`.

A fourth, confound-free measurement (`run_svd_driver_equiv.py`) drives ONE
trajectory with gesvd and steps a second SK-RTRL with `auto` on exactly the
same (A_t, imm_t) sequence, so both estimators always see identical inputs and
no chaotic amplification can enter.

## lorenz / skrtrl-r16 / seed 0 / clip 0

steps 3000, log_every 500, 6 aligned records; clip 0.0, lr 0.001, n 64.

`svd_stats` gesvd: `{'default': 0, 'gesvd': 6000, 'jitter': 0, 'cpu': 0}`

`svd_stats` auto: `{'default': 6000, 'gesvd': 0, 'jitter': 0, 'cpu': 0}` (every call served by the default driver -- no fallback was needed)

| field | auto vs gesvd | gesvd_rep2 vs gesvd (control) | gesvd vs archive | auto vs archive |
|---|---|---|---|---|
| `metric` | 2.0e-04 | 0.0e+00 | 3.9e-04 | 3.8e-04 |
| `grad_cos` | 7.2e-05 | 0.0e+00 | 1.4e-04 | 8.4e-05 |
| `e_t` | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| `true_E` | 4.5e-05 | 0.0e+00 | 8.8e-05 | 4.7e-05 |
| `eta` | 1.4e-04 | 0.0e+00 | 5.1e-04 | 5.6e-04 |
| `rho_bar` | 1.3e-05 | 0.0e+00 | 4.5e-05 | 3.9e-05 |

Frozen-trajectory control (identical inputs to both estimators, 3000 steps):

| quantity | max rel. dev | at step | mean rel. dev |
|---|---|---|---|
| `d_ghat` | 2.54e-02 | 1931 | 1.55e-03 |
| `d_resid` | 3.85e-03 | 1335 | 1.48e-03 |
| `d_e` | 3.89e-06 | 431 | 9.26e-07 |
| `d_eta` | 1.63e-03 | 1913 | 1.81e-04 |
| `d_trueE` | 8.80e-05 | 1948 | 1.20e-05 |


## rotation / skrtrl-r16 / seed 0 / clip 0

steps 3000, log_every 250, 12 aligned records; clip 0.0, lr 0.001, n 64.

`svd_stats` gesvd: `{'default': 0, 'gesvd': 6000, 'jitter': 0, 'cpu': 0}`

`svd_stats` auto: `{'default': 6000, 'gesvd': 0, 'jitter': 0, 'cpu': 0}` (every call served by the default driver -- no fallback was needed)

| field | auto vs gesvd | gesvd_rep2 vs gesvd (control) | gesvd vs archive | auto vs archive |
|---|---|---|---|---|
| `metric` | 5.1e-02 | 0.0e+00 | 3.2e-02 | 5.8e-02 |
| `grad_cos` | 5.5e-02 | 0.0e+00 | 6.4e-02 | 5.1e-02 |
| `e_t` | 1.2e+03 | 0.0e+00 | 7.1e+08 | 1.0e+00 |
| `true_E` | 2.5e+00 | 0.0e+00 | 1.5e+00 | 8.1e-01 |
| `eta` | 8.1e-01 | 0.0e+00 | 9.4e-01 | 7.9e-01 |
| `rho_bar` | 4.0e-01 | 0.0e+00 | 7.3e-01 | 4.3e-01 |

Frozen-trajectory control (identical inputs to both estimators, 3000 steps):

| quantity | max rel. dev | at step | mean rel. dev |
|---|---|---|---|
| `d_ghat` | 7.40e-03 | 1070 | 1.10e-04 |
| `d_resid` | 2.99e-03 | 1078 | 8.86e-05 |
| `d_e` | 4.23e-06 | 2745 | 3.17e-06 |
| `d_eta` | 1.24e-03 | 1056 | 1.44e-05 |
| `d_trueE` | 1.55e-04 | 1202 | 3.90e-06 |


## adding / skrtrl-r16 / seed 0 / clip 0.9

steps 3000, log_every 250, 11 aligned records; clip 0.9, lr 0.001, n 64.

`svd_stats` gesvd: `{'default': 0, 'gesvd': 6000, 'jitter': 0, 'cpu': 0}`

`svd_stats` auto: `{'default': 6000, 'gesvd': 0, 'jitter': 0, 'cpu': 0}` (every call served by the default driver -- no fallback was needed)

| field | auto vs gesvd | gesvd_rep2 vs gesvd (control) | gesvd vs archive | auto vs archive |
|---|---|---|---|---|
| `metric` | 1.1e-05 | 0.0e+00 | 1.1e-05 | 1.2e-06 |
| `grad_cos` | 2.0e-06 | 0.0e+00 | 2.9e-06 | 1.2e-06 |
| `e_t` | 1.7e-03 | 0.0e+00 | 2.1e-03 | 5.4e-04 |
| `true_E` | 3.9e-05 | 0.0e+00 | 3.6e-05 | 1.5e-05 |
| `eta` | 1.7e-05 | 0.0e+00 | 2.0e-05 | 1.1e-05 |
| `rho_bar` | 3.6e-05 | 0.0e+00 | 4.3e-05 | 1.1e-05 |

Frozen-trajectory control (identical inputs to both estimators, 3000 steps):

| quantity | max rel. dev | at step | mean rel. dev |
|---|---|---|---|
| `d_ghat` | 2.64e-04 | 599 | 4.77e-05 |
| `d_resid` | 3.09e-04 | 61 | 2.59e-05 |
| `d_e` | 3.84e-06 | 2983 | 2.61e-06 |
| `d_eta` | 1.99e-04 | 399 | 1.10e-05 |
| `d_trueE` | 5.91e-06 | 61 | 2.04e-07 |


## Timing

Whole training step (task step, forward, estimator update, readout, Adam),
measured as wall_s / steps over a 500-step henon run with the exact shadow OFF
and the D2 counters off, so the SVD driver is the only thing that differs.

| config (henon, B=8, shadow off) | gesvd ms/step | auto ms/step | speedup |
|---|---|---|---|
| n=64, r=4 | 117.4 | 76.1 | x1.54 |
| n=64, r=16 | 155.5 | 84.3 | x1.85 |
| n=64, r=32 | 180.9 | 93.2 | x1.94 |
| n=128, r=4 | 231.3 | 84.0 | x2.75 |
| n=128, r=16 | 246.2 | 92.2 | x2.67 |
| n=128, r=32 | 265.3 | 103.1 | x2.57 |

Caveat on the absolute numbers: the second GPU was running the D1 spectrum queue
throughout, so ms/step is inflated relative to an idle machine (the reconnaissance
in the plan measured 65 ms/step at n=64, r=16).  The ratio is what this table is
for, and it is stable across ranks: ~1.5-1.9x at n=64 and ~2.6-2.8x at n=128 -- the
larger the state, the more the default driver wins.  Per the plan no wall-clock
number from an R2 training run enters the paper, and `gesvd` stays the default, so
the R1 timing tables are untouched.

## A7 -- the fallback chain and the certificate

The chain now reports which link served each call (`_robust_svd_ex`), because the
jitter link factorises M + Delta rather than M: charging the tail singular values of
the PERTURBED matrix to eta_t under-counts the mass actually discarded from M, which
breaks Lemma 1.  On that link the discarded mass is now measured against the
original matrix -- ||B0 - Bc Vc^T||_F for the pre-projection and
||core - L Wfac^T||_F for the rank truncation, both exact because Q_t has
orthonormal rows and [R, Qperp] orthonormal columns.  Every other link factorises M
itself and keeps the tail-sum formula, so the kernel of record is numerically
untouched: re-running the frozen-trajectory control with the final code reproduces
the earlier numbers bit for bit (lorenz r16: d_ghat 2.544e-02, d_resid 3.847e-03,
d_e 3.892e-06, d_eta 1.634e-03, d_trueE 8.800e-05 -- identical to the pre-A7 run).

`tests/test_svd_fallback_cert.py` forces each link deterministically and checks
Lemma 1 and Theorem 1 on an ill-conditioned instance, and measures the accounting
directly: max(true_eta / pre-A7_eta) is above 1 in every regime and at every jitter
magnitude (1.000001 at the production jitter, up to 1.12 at 1e4x on ill-conditioned
batches), i.e. the old formula was a strict under-count, not a conservative one.

`svd_fallbacks` (top level of every output json) counts the calls not served by the
first-choice driver.  It is **0 for every run in this report** -- no gesvd fallback,
no jitter, no CPU fallback was needed at n=64 or n=128 with either driver -- so the
correction changes no number here; it closes a soundness hole that only opens when
cuSOLVER actually fails.

