"""M1 (R005-R008, GATE): off-diagonal residual spectrum of the true influence matrix.

Trains a tanh-RNN (n=64) with exact RTRL on each task; every LOG_EVERY steps computes
singular values of the residual J - blockdiag(J) and logs top-k mass fractions.
Output: results/m1_spectrum_<task>.json
"""
import argparse, json, os, time
import torch

from skrtrl.tasks import TASKS
from skrtrl.train import OnlineLearner


def blockdiag_mask_residual(J, n, p):
    """Zero the SnAp-1 entries of J (B, n, P) -> residual."""
    R = J.clone()
    idx = torch.arange(n, device=J.device)
    Rv = R.view(J.shape[0], n, n, p)
    Rv[:, idx, idx, :] = 0.0
    return Rv.view(J.shape[0], n, n * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    task = TASKS[args.task](args.batch, args.device, seed=args.seed)
    learner = OnlineLearner(task, args.n, "exact", lr=args.lr, device=args.device, seed=args.seed)

    log = {"args": vars(args), "records": []}
    t0 = time.time()
    losses = []
    for step in range(args.steps):
        m = learner.step(update=True)
        if m:
            losses.append(m["loss"])
        if step % args.log_every == 0 or step == args.steps - 1:
            J = learner.algo.J
            res = blockdiag_mask_residual(J, learner.cell.n, learner.cell.p)
            sv = torch.linalg.svdvals(res)            # (B, n)
            tot = (sv ** 2).sum(dim=1).clamp_min(1e-30)
            fracs = {}
            for k in (1, 4, 8, 16, 32, 64):
                if k <= sv.shape[1]:
                    fracs[f"top{k}"] = ((sv[:, :k] ** 2).sum(dim=1) / tot).mean().item()
            res_norm = res.flatten(1).norm(dim=1).mean().item()
            j_norm = J.flatten(1).norm(dim=1).mean().item()
            rec = {"step": step, "loss": float(sum(losses[-50:]) / max(len(losses[-50:]), 1)),
                   "res_frac_of_J": res_norm / max(j_norm, 1e-30), **fracs}
            log["records"].append(rec)
            print(f"[{args.task}] step {step} loss {rec['loss']:.4f} resfrac {rec['res_frac_of_J']:.3f} "
                  f"top4 {fracs.get('top4', 0):.3f} top16 {fracs.get('top16', 0):.3f}", flush=True)

    log["wall_s"] = time.time() - t0
    os.makedirs("results", exist_ok=True)
    out = f"results/m1_spectrum_{args.task}.json"
    with open(out, "w") as f:
        json.dump(log, f, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
