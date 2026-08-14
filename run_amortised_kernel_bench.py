"""Revision benchmark: naive (kernel of record) vs amortised-rotation kernel.

Protocol copied from run_cor3_timing_revise.py:
  * CUDA-event timing, 20 warmup steps, 5 independent repeats -> mean +/- std
  * peak memory measured in an isolated region (empty_cache +
    reset_peak_memory_stats) with the other kernel's state freed
  * numerical agreement of the two kernels checked at every (n, r)

Two pre-projection settings are reported, because they answer different questions:

  mode="svd"       the paper default: the rank-c pre-projection is taken from a
                   *full* SVD of the n x n matrix Ahat_t D_s, an O(n^3) term
                   that both kernels pay and that dominates the step at the n we
                   can run.  This is the end-to-end number.
  mode="randproj"  the matched-cost ablation already in the code: the
                   pre-projection is a random orthonormal O(n c^2) projection.
                   Every other P-sized operation is identical, so this isolates
                   the O(n^2 r^2) -> O(n^2 r) rotation change.

Writes results/revise_kernel/amortised_kernel_bench.json.
"""
import argparse, gc, json, os, platform, sys

import torch

from skrtrl.cells import TanhRNNCell
from skrtrl.algos import SKRTRL
from skrtrl.algos_amortised import SKRTRLAmortised


def make(n, kernel, r, B, m, device, mode, seed=0, collapse_every=None):
    torch.manual_seed(seed)
    cell = TanhRNNCell(m, n, device=device)
    if kernel == "naive":
        algo = SKRTRL(cell, B, r=r, mode=mode)
    else:
        algo = SKRTRLAmortised(cell, B, r=r, mode=mode, collapse_every=collapse_every)
    return cell, algo


def _run_steps(cell, algo, h, x, delta, steps):
    for _ in range(steps):
        hp = h
        h = cell(x, hp).detach()
        A, imm = cell.jac_pieces(x, hp, h)
        algo.step_state(A, imm)
        g = algo.grad_rows(delta)
    return h, g


@torch.no_grad()
def bench(n, kernel, r, mode, B=4, m=8, steps=30, warmup=20, repeats=5,
          device="cuda", seed=0, collapse_every=None):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    cell, algo = make(n, kernel, r, B, m, device, mode, seed, collapse_every)
    h = cell.init_state(B)
    x = torch.randn(B, m, device=device)
    delta = torch.randn(B, n, device=device)

    h, _ = _run_steps(cell, algo, h, x, delta, warmup)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    h, g = _run_steps(cell, algo, h, x, delta, steps)
    torch.cuda.synchronize()
    peak_MB = torch.cuda.max_memory_allocated() / 2 ** 20

    per_step = []
    for _ in range(repeats):
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        ev0.record()
        h, g = _run_steps(cell, algo, h, x, delta, steps)
        ev1.record()
        torch.cuda.synchronize()
        per_step.append(ev0.elapsed_time(ev1) / steps)

    t = torch.tensor(per_step)
    mean, std = t.mean().item(), (t.std().item() if repeats > 1 else 0.0)
    g = g.detach().clone()
    ncol = getattr(algo, "n_collapse", None)
    del algo, cell, h, x, delta
    gc.collect(); torch.cuda.empty_cache()
    return mean, std, peak_MB, g, ncol


@torch.no_grad()
def bench_preproj_svd(n, B=4, repeats=5, device="cuda"):
    """Isolated cost of the O(n^3) pre-projection SVD both kernels pay."""
    from skrtrl.algos import _robust_svd
    M = torch.randn(B, n, n, device=device)
    for _ in range(3):
        _robust_svd(M)
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record(); _robust_svd(M); ev1.record(); torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1))
    del M; gc.collect(); torch.cuda.empty_cache()
    return float(torch.tensor(ts).mean())


@torch.no_grad()
def agreement(n, r, mode, B=2, m=8, steps=25, device="cuda", seed=0):
    """Max relative deviation of ghat between the two kernels on one stream."""
    torch.manual_seed(seed)
    cell = TanhRNNCell(m, n, device=device)
    na = SKRTRL(cell, B, r=r, mode=mode)
    am = SKRTRLAmortised(cell, B, r=r, mode=mode)
    h = cell.init_state(B)
    xs = torch.randn(steps, B, m, device=device) * 0.8
    d = torch.randn(steps, B, n, device=device)
    worst_g = worst_e = 0.0
    for t in range(steps):
        hp = h
        h = cell(xs[t], hp).detach()
        A, imm = cell.jac_pieces(xs[t], hp, h)
        na.step_state(A, imm); am.step_state(A, imm)
        g1, g2 = na.grad_rows(d[t]), am.grad_rows(d[t])
        worst_g = max(worst_g, float((g1 - g2).norm() / g1.norm().clamp_min(1e-30)))
        worst_e = max(worst_e, float((na.e - am.e).norm() / na.e.norm().clamp_min(1e-30)))
    del na, am, cell; gc.collect(); torch.cuda.empty_cache()
    return worst_g, worst_e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="64,128,256,512")
    ap.add_argument("--rs", default="8,16,32")
    ap.add_argument("--modes", default="svd,randproj")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--sweep_K", default="128:32")     # "n:r" for the K sweep, "" to skip
    # r-scaling scan: naive vs amortised(K=Theta(r)) vs amortised(K=1).
    # K=1 still collapses every step (so it is O(n^2 r^2) like the naive kernel)
    # but already avoids the dense P x c append and the P x c thin QR, which
    # separates "deferring the rotation" from "not materialising the append".
    ap.add_argument("--rscan", default="")             # e.g. "256:8,16,32,64,128,192"
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/revise_kernel/amortised_kernel_bench.json")
    args = ap.parse_args()

    dev_idx = torch.cuda.current_device()
    env = {
        "gpu": torch.cuda.get_device_name(dev_idx),
        "gpu_total_GB": round(torch.cuda.get_device_properties(dev_idx).total_memory / 2 ** 30, 1),
        "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(dev_idx))),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "dtype": "float32", "batch": args.batch, "n_in": 8,
        "steps_per_measure": args.steps, "warmup": args.warmup, "repeats": args.repeats,
        "timer": "torch.cuda.Event",
        "platform": platform.platform(), "python": sys.version.split()[0],
    }
    print(json.dumps(env, indent=1), flush=True)

    ns = [int(v) for v in args.ns.split(",") if v]
    rs = [int(v) for v in args.rs.split(",") if v]
    modes = [v for v in args.modes.split(",") if v]

    rows, svd_rows = [], []
    for n in ns:
        try:
            svd_rows.append({"n": n, "preproj_svd_ms": round(bench_preproj_svd(n, args.batch), 3)})
            print(svd_rows[-1], flush=True)
        except Exception as exc:                                    # noqa: BLE001
            svd_rows.append({"n": n, "preproj_svd_ms": None, "err": str(exc)})

    for mode in modes:
        for n in ns:
            for r in rs:
                rec = {"mode": mode, "n": n, "r": r, "c": max(4, -(-r // 4))}
                gr = {}
                for kernel in ("naive", "amortised"):
                    try:
                        ms, sd, mem, g, ncol = bench(
                            n, kernel, r, mode, B=args.batch, steps=args.steps,
                            warmup=args.warmup, repeats=args.repeats, device=args.device)
                        rec[f"{kernel}_ms"] = round(ms, 4)
                        rec[f"{kernel}_ms_std"] = round(sd, 4)
                        rec[f"{kernel}_MB"] = round(mem, 1)
                        if ncol is not None:
                            rec["collapses"] = ncol
                        gr[kernel] = g
                    except torch.cuda.OutOfMemoryError:
                        rec[f"{kernel}_ms"] = None
                        rec[f"{kernel}_MB"] = "OOM"
                        gc.collect(); torch.cuda.empty_cache()
                if rec.get("naive_ms") and rec.get("amortised_ms"):
                    rec["speedup"] = round(rec["naive_ms"] / rec["amortised_ms"], 3)
                    rec["mem_ratio"] = round(rec["amortised_MB"] / rec["naive_MB"], 3)
                gr.clear(); gc.collect(); torch.cuda.empty_cache()
                if mode == "svd":
                    try:
                        wg, we = agreement(min(n, 128), r, mode, device=args.device)
                        rec["rel_err_ghat"] = wg
                        rec["rel_err_e"] = we
                    except Exception as exc:                        # noqa: BLE001
                        rec["rel_err_ghat"] = None
                        rec["agreement_err"] = str(exc)
                rows.append(rec)
                print(rec, flush=True)

    ksweep = []
    if args.sweep_K:
        n_k, r_k = (int(v) for v in args.sweep_K.split(":"))
        for K in (1, 2, 4, 8, 16, 32, 64):
            try:
                ms, sd, mem, _, ncol = bench(n_k, "amortised", r_k, "randproj",
                                             B=args.batch, steps=args.steps,
                                             warmup=args.warmup, repeats=args.repeats,
                                             device=args.device, collapse_every=K)
                ksweep.append({"n": n_k, "r": r_k, "K": K, "ms": round(ms, 4),
                               "ms_std": round(sd, 4), "MB": round(mem, 1)})
            except torch.cuda.OutOfMemoryError:
                ksweep.append({"n": n_k, "r": r_k, "K": K, "ms": None, "MB": "OOM"})
                gc.collect(); torch.cuda.empty_cache()
            print(ksweep[-1], flush=True)

    rscan = []
    if args.rscan:
        n_s, rlist = args.rscan.split(":")
        n_s = int(n_s)
        for r in [int(v) for v in rlist.split(",")]:
            rec = {"n": n_s, "r": r, "mode": "randproj"}
            for tag, kernel, K in (("naive", "naive", None),
                                   ("amort", "amortised", None),
                                   ("amortK1", "amortised", 1)):
                try:
                    ms, sd, mem, _, _ = bench(n_s, kernel, r, "randproj", B=args.batch,
                                              steps=args.steps, warmup=args.warmup,
                                              repeats=args.repeats, device=args.device,
                                              collapse_every=K)
                    rec[f"{tag}_ms"] = round(ms, 4)
                    rec[f"{tag}_ms_std"] = round(sd, 4)
                    rec[f"{tag}_MB"] = round(mem, 1)
                except torch.cuda.OutOfMemoryError:
                    rec[f"{tag}_ms"] = None
                    rec[f"{tag}_MB"] = "OOM"
                    gc.collect(); torch.cuda.empty_cache()
            rscan.append(rec)
            print(rec, flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"env": env, "rows": rows, "preproj_svd": svd_rows, "K_sweep": ksweep,
               "r_scan": rscan},
              open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
