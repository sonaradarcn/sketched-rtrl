"""Render the amortised-kernel benchmark JSONs into a markdown summary table.

Usage (from the repository root):
  python make_kernel_table.py                       # reads results/revise_kernel
  python make_kernel_table.py --dir <other/dir>
"""
import argparse, json, os

# In the released repository the benchmark JSONs sit in results/revise_kernel; in the
# authors' working tree the repository is a `code/` subdirectory and they sit one level up.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_HERE, "results", "revise_kernel"),
    os.path.join(os.path.dirname(_HERE), "results", "revise_kernel"),
]
DEFAULT_DIR = next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[0])


def fmt(v, nd=2):
    return "--" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.dir
    out = args.out or os.path.join(d, "SUMMARY.md")

    def load(name):
        path = os.path.join(d, name)
        return json.load(open(path)) if os.path.exists(path) else None

    rp = load("kernel_bench_randproj.json")
    sv = load("kernel_bench_svd.json")
    rs = load("kernel_rscan_n256.json")

    L = []
    env = (rp or sv)["env"]
    L.append("# Amortised-rotation kernel: naive vs amortised\n")
    L.append(f"GPU {env['gpu']} ({env['gpu_total_GB']} GB), torch {env['torch']}+cu{env['cuda']}, "
             f"fp32, batch {env['batch']}, n_in {env['n_in']}, TF32 {env['tf32_matmul']}.  "
             f"CUDA-event timing, {env['warmup']} warmup steps, {env['repeats']} repeats "
             f"of {env['steps_per_measure']} steps, mean +/- std ms per step.\n")

    if rp:
        L.append("\n## 1. Pre-projection SVD in isolation (both kernels pay it)\n")
        L.append("| n | full SVD of the n x n append factor (ms/step) |")
        L.append("|---|---|")
        for r in rp["preproj_svd"]:
            L.append(f"| {r['n']} | {fmt(r.get('preproj_svd_ms'))} |")

    if sv:
        L.append("\n## 2. End-to-end step, mode=\"svd\" (paper default)\n")
        L.append("| n | r | naive ms | amortised ms | speedup | naive MB | amort MB | rel.err ghat |")
        L.append("|---|---|---|---|---|---|---|---|")
        for x in sv["rows"]:
            g = x.get("rel_err_ghat")
            gs = "--" if g is None else f"{g:.1e}"
            L.append(f"| {x['n']} | {x['r']} | {fmt(x.get('naive_ms'))}+-{fmt(x.get('naive_ms_std'))} "
                     f"| {fmt(x.get('amortised_ms'))}+-{fmt(x.get('amortised_ms_std'))} "
                     f"| {fmt(x.get('speedup'), 3)} | {fmt(x.get('naive_MB'), 1)} "
                     f"| {fmt(x.get('amortised_MB'), 1)} | {gs} |")

    if rp:
        L.append("\n## 3. Step with the cheap pre-projection, mode=\"randproj\" "
                 "(isolates the rotation kernel)\n")
        L.append("| n | r | c | naive ms | amortised ms | speedup | naive MB | amort MB |")
        L.append("|---|---|---|---|---|---|---|---|")
        for x in rp["rows"]:
            L.append(f"| {x['n']} | {x['r']} | {x['c']} | {fmt(x.get('naive_ms'))}+-{fmt(x.get('naive_ms_std'))} "
                     f"| {fmt(x.get('amortised_ms'))}+-{fmt(x.get('amortised_ms_std'))} "
                     f"| {fmt(x.get('speedup'), 3)} | {fmt(x.get('naive_MB'), 1)} "
                     f"| {fmt(x.get('amortised_MB'), 1)} |")

        if rp.get("K_sweep"):
            k0 = rp["K_sweep"][0]
            L.append(f"\n## 4. Collapse period K (n={k0['n']}, r={k0['r']}, mode=randproj)\n")
            L.append("| K | ms/step | peak MB |")
            L.append("|---|---|---|")
            for x in rp["K_sweep"]:
                L.append(f"| {x['K']} | {fmt(x.get('ms'))}+-{fmt(x.get('ms_std'))} | {fmt(x.get('MB'), 1)} |")

    scan = (rp or {}).get("r_scan") or (rs or {}).get("r_scan")
    if scan:
        L.append(f"\n## 5. r-scaling at n={scan[0]['n']} (mode=randproj)\n")
        L.append("`amortK1` = amortised representation but collapsing every step: it still costs "
                 "O(n^2 r^2), so the gap between `naive` and `amortK1` measures the removal of the "
                 "dense P x c append and the P x c thin QR, and the gap between `amortK1` and "
                 "`amort` measures the deferral itself.\n")
        L.append("| r | naive ms | amort ms | amortK1 ms | naive/amort |")
        L.append("|---|---|---|---|---|")
        for x in scan:
            sp = (x["naive_ms"] / x["amort_ms"]) if (x.get("naive_ms") and x.get("amort_ms")) else None
            L.append(f"| {x['r']} | {fmt(x.get('naive_ms'))} | {fmt(x.get('amort_ms'))} "
                     f"| {fmt(x.get('amortK1_ms'))} | {fmt(sp, 2)} |")

    L.append("""
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
""")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
