"""Revision re-measurement for Table 8 (tab:cor3): factored-exact (Cor. 3, r=n)
vs textbook exact RTRL, per-step time and peak memory.

Hardened vs the original code/run_cor3_timing.py:
  * CUDA-event timing instead of time.time()
  * 20 warmup steps (was 3) + 5 independent repeats -> mean +/- std
  * peak memory measured in an isolated region (empty_cache + reset_peak_memory_stats)
    with the *other* algorithm's state freed, so no cross-contamination
  * numerical exactness check factored(r=n) vs textbook at every n
  * records GPU model / torch / TF32 flags for the paper footnote

Writes results/revise_timing/cor3_timing_revise.json. Never touches results/cor3_timing.json.
"""
import argparse, gc, json, os, platform, sys
import torch

from skrtrl.cells import TanhRNNCell
from skrtrl.algos import ExactRTRL, SKRTRL


def make(n, which, B, m, device, seed=0):
    torch.manual_seed(seed)
    cell = TanhRNNCell(m, n, device=device)
    algo = ExactRTRL(cell, B) if which == "exact" else SKRTRL(cell, B, r=n)
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
def bench(n, which, B=4, m=8, steps=50, warmup=20, repeats=5, device="cuda", seed=0):
    """Return (mean_ms_per_step, std_ms_per_step, peak_MB, final grad rows)."""
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    cell, algo = make(n, which, B, m, device, seed)
    h = cell.init_state(B)
    x = torch.randn(B, m, device=device)
    delta = torch.randn(B, n, device=device)

    h, _ = _run_steps(cell, algo, h, x, delta, warmup)
    torch.cuda.synchronize()

    # --- peak memory over a clean measured window ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    h, g = _run_steps(cell, algo, h, x, delta, steps)
    torch.cuda.synchronize()
    peak_MB = torch.cuda.max_memory_allocated() / 2 ** 20

    # --- timing: CUDA events, `repeats` independent measurements ---
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
    del algo, cell, h, x, delta
    gc.collect(); torch.cuda.empty_cache()
    return mean, std, peak_MB, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="32,48,64,96,128,160,192")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/revise_timing/cor3_timing_revise.json")
    args = ap.parse_args()

    dev_idx = torch.cuda.current_device()
    # The protocol is fp32 with TF32 matmuls disabled. Enforce it here rather than trusting
    # the ambient state -- the env record below then reports what was actually in force.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    env = {
        "gpu": torch.cuda.get_device_name(dev_idx),
        "gpu_total_GB": round(torch.cuda.get_device_properties(dev_idx).total_memory / 2 ** 30, 1),
        "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(dev_idx))),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "dtype": "float32", "batch": 4, "n_in": 8,
        "steps_per_measure": args.steps, "warmup": args.warmup, "repeats": args.repeats,
        "timer": "torch.cuda.Event",
        "platform": platform.platform(), "python": sys.version.split()[0],
    }
    print(json.dumps(env, indent=1), flush=True)

    out = []
    for n in [int(x) for x in args.ns.split(",")]:
        rec = {"n": n}
        grads = {}
        for which in ("exact", "factored"):
            try:
                ms, sd, mem, g = bench(n, which, steps=args.steps, warmup=args.warmup,
                                       repeats=args.repeats, device=args.device)
                rec[f"{which}_ms"] = round(ms, 3)
                rec[f"{which}_ms_std"] = round(sd, 4)
                rec[f"{which}_MB"] = round(mem, 1)
                grads[which] = g
            except torch.cuda.OutOfMemoryError:
                rec[f"{which}_ms"] = None
                rec[f"{which}_MB"] = "OOM"
                gc.collect(); torch.cuda.empty_cache()
        if rec.get("exact_ms") and rec.get("factored_ms"):
            rec["speedup"] = round(rec["exact_ms"] / rec["factored_ms"], 3)
        if len(grads) == 2:
            a, b = grads["exact"], grads["factored"]
            rec["rel_err_grad"] = float((a - b).norm() / (a.norm() + 1e-12))
        grads.clear(); gc.collect(); torch.cuda.empty_cache()
        out.append(rec)
        print(rec, flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"env": env, "rows": out}, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
