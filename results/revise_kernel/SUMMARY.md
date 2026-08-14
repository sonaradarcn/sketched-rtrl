# Amortised-rotation kernel: naive vs amortised

GPU NVIDIA GeForce RTX 3080 (20.0 GB), torch 2.6.0+cu124+cu12.4, fp32, batch 4, n_in 8, TF32 False.  CUDA-event timing, 20 warmup steps, 5 repeats of 30 steps, mean +/- std ms per step.


## 1. Pre-projection SVD in isolation (both kernels pay it)

| n | full SVD of the n x n append factor (ms/step) |
|---|---|
| 64 | 25.07 |
| 128 | 57.99 |
| 256 | 159.00 |
| 512 | 505.75 |

## 2. End-to-end step, mode="svd" (paper default)

| n | r | naive ms | amortised ms | speedup | naive MB | amort MB | rel.err ghat |
|---|---|---|---|---|---|---|---|
| 64 | 8 | 29.64+-0.34 | 32.25+-0.87 | 0.919 | 11.7 | 11.5 | 3.8e-07 |
| 64 | 16 | 32.85+-1.28 | 34.84+-1.03 | 0.943 | 13.5 | 14.3 | 3.4e-07 |
| 64 | 32 | 39.40+-1.58 | 40.62+-1.20 | 0.970 | 18.1 | 14.5 | 4.8e-07 |
| 128 | 8 | 64.73+-0.93 | 68.07+-0.65 | 0.951 | 21.6 | 20.6 | 3.7e-07 |
| 128 | 16 | 68.43+-0.87 | 69.26+-0.91 | 0.988 | 28.1 | 30.4 | 2.3e-07 |
| 128 | 32 | 75.65+-1.26 | 76.56+-1.04 | 0.988 | 45.4 | 26.9 | 3.3e-07 |
| 256 | 8 | 172.28+-5.71 | 176.39+-7.35 | 0.977 | 59.8 | 56.2 | 3.7e-07 |
| 256 | 16 | 172.94+-3.62 | 175.58+-4.41 | 0.985 | 85.8 | 93.9 | 2.3e-07 |
| 256 | 32 | 178.62+-1.52 | 178.10+-1.08 | 1.003 | 153.9 | 78.8 | 3.3e-07 |
| 512 | 8 | 520.28+-7.84 | 511.34+-2.17 | 1.017 | 212.1 | 196.9 | 3.7e-07 |
| 512 | 16 | 516.41+-4.32 | 515.78+-3.07 | 1.001 | 314.4 | 348.4 | 2.3e-07 |
| 512 | 32 | 534.47+-3.86 | 524.33+-6.78 | 1.019 | 573.1 | 280.4 | 3.3e-07 |

## 3. Step with the cheap pre-projection, mode="randproj" (isolates the rotation kernel)

| n | r | c | naive ms | amortised ms | speedup | naive MB | amort MB |
|---|---|---|---|---|---|---|---|
| 64 | 8 | 4 | 3.57+-0.10 | 5.41+-0.13 | 0.660 | 11.6 | 11.4 |
| 64 | 16 | 4 | 3.50+-0.03 | 5.42+-0.05 | 0.646 | 13.3 | 14.1 |
| 64 | 32 | 8 | 3.51+-0.05 | 5.45+-0.04 | 0.644 | 18.0 | 21.4 |
| 64 | 64 | 16 | 28.87+-0.88 | 36.99+-0.23 | 0.780 | 41.3 | 37.1 |
| 128 | 8 | 4 | 3.62+-0.09 | 5.37+-0.03 | 0.675 | 21.1 | 20.0 |
| 128 | 16 | 4 | 3.67+-0.12 | 5.47+-0.12 | 0.671 | 27.6 | 29.9 |
| 128 | 32 | 8 | 3.61+-0.07 | 5.59+-0.10 | 0.645 | 44.8 | 51.0 |
| 128 | 64 | 16 | 4.55+-0.11 | 6.75+-0.28 | 0.675 | 81.0 | 94.4 |
| 256 | 8 | 4 | 3.60+-0.07 | 5.37+-0.11 | 0.670 | 57.7 | 54.4 |
| 256 | 16 | 4 | 3.66+-0.09 | 5.43+-0.14 | 0.674 | 83.7 | 91.8 |
| 256 | 32 | 8 | 4.17+-0.03 | 5.57+-0.10 | 0.748 | 151.8 | 172.0 |
| 256 | 64 | 16 | 7.58+-0.11 | 6.64+-0.18 | 1.141 | 283.4 | 309.1 |
| 512 | 8 | 4 | 6.63+-0.12 | 5.48+-0.14 | 1.211 | 204.3 | 188.8 |
| 512 | 16 | 4 | 7.04+-0.08 | 5.60+-0.13 | 1.256 | 306.6 | 339.3 |
| 512 | 32 | 8 | 13.45+-0.04 | 5.81+-0.16 | 2.312 | 564.7 | 632.1 |
| 512 | 64 | 16 | 27.36+-0.04 | 6.73+-0.10 | 4.067 | 1089.8 | 1161.5 |

## 4. Collapse period K (n=512, r=32, mode=randproj)

| K | ms/step | peak MB |
|---|---|---|
| 1 | 5.40+-0.08 | 568.3 |
| 2 | 5.57+-0.07 | 573.5 |
| 4 | 5.68+-0.27 | 582.6 |
| 8 | 5.61+-0.12 | 599.3 |
| 16 | 5.72+-0.22 | 633.6 |
| 32 | 5.99+-0.19 | 718.0 |
| 64 | 7.17+-1.62 | 645.8 |

## 5. r-scaling at n=256 (mode=randproj)

`amortK1` = amortised representation but collapsing every step: it still costs O(n^2 r^2), so the gap between `naive` and `amortK1` measures the removal of the dense P x c append and the P x c thin QR, and the gap between `amortK1` and `amort` measures the deferral itself.

| r | naive ms | amort ms | amortK1 ms | naive/amort |
|---|---|---|---|---|
| 8 | 3.66 | 5.48 | 5.40 | 0.67 |
| 16 | 3.64 | 5.76 | 5.39 | 0.63 |
| 32 | 4.23 | 5.57 | 5.48 | 0.76 |
| 64 | 7.51 | 6.54 | 6.28 | 1.15 |
| 128 | 15.41 | 7.81 | 7.89 | 1.97 |
| 192 | 27.24 | 15.14 | 16.34 | 1.80 |
| 256 | 269.18 | 204.91 | 200.10 | 1.31 |

## 6. What the numbers say

1. **The two kernels agree.** `tests/test_amortised_kernel.py` reproduces L_t R_t^T,
   S_t + L_t R_t^T, ghat_t, eta_t and e_t step by step: worst relative deviation
   < 4e-14 in float64 (T = 300) and ~1e-6 in float32, with no drift over time.
   The certificate stays valid (0 violations) and the Cor.-3 endpoint (r = n) is
   still exactly RTRL. So the paper's "in exact arithmetic the two kernels return
   the same factors" is now backed by code.

2. **End-to-end, the change is invisible in the released configuration.** With the
   default `mode="svd"` the step is dominated by the *full* n x n SVD used for the
   rank-c pre-projection (table 1 vs table 2: 506 ms of a 520-534 ms step at
   n = 512, i.e. 95-99%). Speedups are 0.92-1.02x. This also explains the paper's
   own scaling data (results/scale_large.json: n = 512 gives 416 ms at r = 4 and
   410 ms at r = 16; n = 1024 gives 1929 vs 1933 ms) -- the measured step time was
   never rotation-bound, so no wall-clock number in the paper can move.

3. **The rotation itself does get cheaper, and the win grows with r.** With the
   cheap pre-projection (table 3), at n = 512 the amortised step is essentially
   flat in r (5.5 / 5.6 / 5.8 / 6.7 ms for r = 8/16/32/64) while the naive step
   grows (6.6 / 7.0 / 13.4 / 27.4 ms): speedup 1.21x, 1.26x, 2.31x, 4.07x. That is
   the O(n^2 r^2) -> O(n^2 r) signature.

4. **At small n the amortised kernel is slower** (0.64-0.78x for n <= 128): both
   kernels sit on a ~3.6 ms launch-bound floor there and the amortised one issues
   more small kernels.

5. **Most of the win is *restructuring*, not *deferral*.** `amortK1` (table 5)
   uses the same representation but collapses every step, so it is still
   O(n^2 r^2); it is already within noise of the fully deferred variant up to
   r = 192. The gain comes from never materialising the P x c dense append and
   never running the P x c thin QR (replaced by a c x c eigh in the Gram metric).
   Deferral only starts to pay where the collapse is the bottleneck.

6. **The collapse period has a ceiling.** With the paper's c = Theta(r), a period
   K = Theta(r) makes the deferred basis width w = r + K c = Theta(r^2) and the
   coefficient-space algebra O(w^2 c) = O(r^5) per step. The amortised O(n^2 r)
   therefore holds for r = O(sqrt(n)) (or for a constant append budget c = O(1)),
   not for arbitrary r. Table 4 shows the cost is flat for K <= 16 and rises
   afterwards; the kernel caps K so that w <= 5r.

7. **The r = n endpoint does not reach O(n^3).** At r = c = n the buffer width is
   Theta(K n) and the Gram bookkeeping is Theta(K^2 n^2) per step, so K = Theta(n)
   is not affordable. Measured: n = 64, r = 64 gives 0.78x; n = 256, r = 256 gives
   1.31x -- a constant-factor gain, not the O(n^4) -> O(n^3) of Table 2's last row.

8. **Memory is a wash** (0.90-1.19x of the naive kernel in table 3): the transient
   O(n^2 r) block buffer replaces the naive kernel's dense P x c append, P x c QR
   workspace and P x (r+c) concatenation.

