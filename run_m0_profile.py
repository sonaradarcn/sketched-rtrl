"""R004: wall-clock + memory profile of exact RTRL vs SK-RTRL(r) vs SnAp-1 across widths."""
import json, os, time
import torch

from skrtrl.cells import TanhRNNCell
from skrtrl.algos import ExactRTRL, SKRTRL


def profile(n, algo_name, r=0, B=8, m=10, steps=30, device="cuda"):
    cell = TanhRNNCell(m, n, device=device)
    if algo_name == "exact":
        if n > 256:
            return None
        algo = ExactRTRL(cell, B)
    else:
        algo = SKRTRL(cell, B, r=r)
    h = cell.init_state(B)
    x = torch.randn(B, m, device=device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        h_prev = h
        h = cell(x, h_prev).detach()
        A, imm = cell.jac_pieces(x, h_prev, h)
        algo.step_state(A, imm)
        algo.grad_rows(torch.randn(B, n, device=device))
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps * 1000
    mem = torch.cuda.max_memory_allocated() / 2**20
    return {"n": n, "algo": algo_name, "r": r, "ms_per_step": dt, "peak_MB": mem}


def main():
    out = []
    for n in (64, 128, 192, 256, 384, 512):
        for algo_name, r in [("snap1", 0), ("skrtrl", 4), ("skrtrl", 16), ("skrtrl", 64), ("exact", 0)]:
            try:
                rec = profile(n, algo_name, r)
            except torch.cuda.OutOfMemoryError:
                rec = {"n": n, "algo": algo_name, "r": r, "oom": True}
                torch.cuda.empty_cache()
            if rec:
                out.append(rec)
                print(rec, flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/m0_profile.json", "w"), indent=1)


if __name__ == "__main__":
    main()
