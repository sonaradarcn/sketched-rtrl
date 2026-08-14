"""Extended width sweep of Sec. 4.2: SK-RTRL-only per-step time and peak memory at large n.

This is the runner behind `results/scale_large.json`, the file the manuscript names when it
quotes 416 ms at r=4 against 410 ms at r=16 for n=512, and 1929 against 1933 ms for n=1024 --
the measurement showing that the step is dominated by the O(n^3) pre-projection rather than by
the rotation, and therefore essentially flat in r.

Only SK-RTRL is swept: exact RTRL runs out of memory well below these widths, which is the
point of the figure it supports. Protocol, matching the manuscript's protocol table: input
dimension m = 8, batch 4, 3 untimed warm-up steps, then 15 timed steps whose mean is reported;
peak memory is the CUDA allocator high-water mark taken after the warm-up. No optimiser step
runs and no seed is fixed -- the quantity measured is per-step cost, which does not depend on
the data. A width that runs out of memory is recorded as {"oom": true} rather than skipped.

Output schema is a flat list of rows, not the per-run record shape of the training scripts:
  [{"n": 384, "algo": "skrtrl-r4", "ms": ..., "MB": ...}, ...]

Usage:  python run_width_sweep.py [--ns 384,512,768,1024] [--ranks 4,16]
                                  [--out results/scale_large.json]
"""
import argparse
import json
import os
import time

import torch

from skrtrl.algos import SKRTRL
from skrtrl.cells import TanhRNNCell

WARMUP, TIMED, BATCH, M_IN = 3, 15, 4, 8


def measure(n, r, device="cuda"):
    cell = TanhRNNCell(M_IN, n, device=device)
    algo = SKRTRL(cell, BATCH, r=r)
    h = cell.init_state(BATCH)
    x = torch.randn(BATCH, M_IN, device=device)
    for _ in range(WARMUP):
        hp = h
        h = cell(x, hp).detach()
        A, imm = cell.jac_pieces(x, hp, h)
        algo.step_state(A, imm)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(TIMED):
        hp = h
        h = cell(x, hp).detach()
        A, imm = cell.jac_pieces(x, hp, h)
        algo.step_state(A, imm)
        algo.grad_rows(torch.randn(BATCH, n, device=device))
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / TIMED * 1000
    mb = torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else None
    return {"n": n, "algo": "skrtrl-r%d" % r, "ms": ms, "MB": mb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="384,512,768,1024")
    ap.add_argument("--ranks", default="4,16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join("results", "scale_large.json"))
    args = ap.parse_args()
    rows = []
    for n in [int(v) for v in args.ns.split(",")]:
        for r in [int(v) for v in args.ranks.split(",")]:
            try:
                rows.append(measure(n, r, args.device))
            except torch.cuda.OutOfMemoryError:
                rows.append({"n": n, "algo": "skrtrl-r%d" % r, "oom": True})
                torch.cuda.empty_cache()
            print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
