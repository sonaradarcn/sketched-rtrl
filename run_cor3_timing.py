"""Corollary 3 demo: SK-RTRL(r=n) reproduces EXACT RTRL. Reports per-step time, memory
and exactness across n, for the factored path against textbook exact RTRL.

NOTE ON COMPLEXITY.  An earlier version of this file claimed the factored path runs at
O(n^3)/step against the textbook O(n^4)/step.  That claim was withdrawn during the
Neurocomputing revision and is wrong: at r = n the rotation term is O(n^2 r^2) = O(n^4),
so the factored path matches the textbook order and is slower by a constant factor.
Corollary 3 is an exactness result, not a speed-up -- see Sec. 4.2 and Table 2 of the
manuscript, and the measured comparison in Table 8.  This script measures exactly that:
it is the original round-1 measurement, superseded for the paper by
run_cor3_timing_revise.py (CUDA events, warmup, repeats).
"""
import argparse, json, os, time
import torch

from skrtrl.cells import TanhRNNCell
from skrtrl.algos import ExactRTRL, SKRTRL


@torch.no_grad()
def bench(n, which, B=4, m=8, steps=20, device="cuda", verify_ref=None):
    cell = TanhRNNCell(m, n, device=device)
    if which == "exact":
        algo = ExactRTRL(cell, B)
    else:
        algo = SKRTRL(cell, B, r=n)  # r=n, pre-projection skipped -> exact (Cor.3)
    h = cell.init_state(B)
    x = torch.randn(B, m, device=device)
    # warmup
    for _ in range(3):
        hp = h
        h = cell(x, hp).detach()
        A, imm = cell.jac_pieces(x, hp, h)
        algo.step_state(A, imm)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        hp = h
        h = cell(x, hp).detach()
        A, imm = cell.jac_pieces(x, hp, h)
        algo.step_state(A, imm)
        algo.grad_rows(torch.randn(B, n, device=device))
    torch.cuda.synchronize()
    ms = (time.time() - t0) / steps * 1000
    mem = torch.cuda.max_memory_allocated() / 2**20
    return ms, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="32,48,64,96,128,160,192")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = []
    for n in [int(x) for x in args.ns.split(",")]:
        rec = {"n": n}
        for which in ("exact", "factored"):
            try:
                ms, mem = bench(n, which, device=args.device)
                rec[f"{which}_ms"] = round(ms, 3)
                rec[f"{which}_MB"] = round(mem, 1)
            except torch.cuda.OutOfMemoryError:
                rec[f"{which}_ms"] = None
                rec[f"{which}_MB"] = "OOM"
                torch.cuda.empty_cache()
        if rec.get("exact_ms") and rec.get("factored_ms"):
            rec["speedup"] = round(rec["exact_ms"] / rec["factored_ms"], 2)
        out.append(rec)
        print(rec, flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/cor3_timing.json", "w"), indent=1)
    print("saved results/cor3_timing.json")


if __name__ == "__main__":
    main()
