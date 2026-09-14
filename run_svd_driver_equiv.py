"""A1 control: SVD-driver equivalence on a FROZEN trajectory.

The end-to-end regression (same seed, same steps, two drivers) is confounded: the
learning loop is chaotic, so any float difference in the first SVD is amplified by
thousands of subsequent parameter updates.  This script removes that confound by
driving ONE trajectory (the `gesvd` learner does the updates) and stepping a second
SK-RTRL instance with `svd_driver="auto"` on exactly the same (A_t, imm_t) sequence.
Both estimators therefore see identical inputs at every step, and the reported
deviations are the numerical effect of the driver alone.

Reported per step (relative, Frobenius):
  d_ghat     || ghat_auto - ghat_gesvd || / || ghat_gesvd ||
  d_resid    || (S+LR^T)_auto - (S+LR^T)_gesvd ||_F / || (S+LR^T)_gesvd ||_F
  d_e        | e_auto - e_gesvd | / e_gesvd          (certificate)
  d_eta      | eta_auto - eta_gesvd | / eta_gesvd    (per-step discarded mass)
  d_trueE    | ||E||_auto - ||E||_gesvd | / ||E||_gesvd   (vs the exact shadow)

Usage:
  python run_svd_driver_equiv.py --task lorenz --r 16 --steps 3000
"""
import argparse, json, math, os, time
import torch

from skrtrl.tasks import TASKS
from skrtrl.train import OnlineLearner
from skrtrl.algos import ExactRTRL, SKRTRL

TS_TASKS = {"henon", "mackeyglass", "lorenz", "sunspot", "laser"}


def reldev(a, b):
    den = max(abs(b), 1e-30)
    if math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0):
        return 0.0
    if math.isinf(a) or math.isinf(b):
        return float("inf")
    return abs(a - b) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="lorenz", choices=list(TASKS))
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--washout", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="results/r2/svd_equiv")
    ap.add_argument("--tag", default="frozen")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out = f"{args.outdir}/{args.task}_r{args.r}_clip{args.clip:g}_{args.tag}.json"
    if os.path.exists(out):
        print("exists, skip:", out)
        return

    if args.task in TS_TASKS:
        task = TASKS[args.task](args.batch, args.device, seed=args.seed,
                                horizon=args.horizon, causal=bool(args.causal),
                                washout=args.washout)
    else:
        task = TASKS[args.task](args.batch, args.device, seed=args.seed)

    learner = OnlineLearner(task, args.n, f"skrtrl-r{args.r}", lr=args.lr,
                            device=args.device, seed=args.seed, spectral_clip=args.clip,
                            svd_driver="gesvd")
    ref = learner.algo                                     # gesvd, drives the updates
    alt = SKRTRL(learner.cell, task.B, r=args.r, svd_driver="auto")
    shadow = ExactRTRL(learner.cell, task.B)

    keys = ["d_ghat", "d_resid", "d_e", "d_eta", "d_trueE"]
    worst = {k: 0.0 for k in keys}
    worst_at = {k: None for k in keys}
    acc = {k: 0.0 for k in keys}
    cnt = {k: 0 for k in keys}
    t0 = time.time()
    for step in range(args.steps):
        x, y, new_ep = task.step()
        learner._reset_lanes(new_ep)
        if new_ep.any():
            shadow.J[new_ep] = 0.0
            for attr in ("S", "L", "R"):
                getattr(alt, attr)[new_ep] = 0.0
            alt.e[new_ep] = 0.0
        h_prev = learner.h.detach()
        h = learner.cell(x, h_prev)
        A, imm = learner.cell.jac_pieces(x, h_prev, h)
        ref.step_state(A, imm)
        alt.step_state(A, imm)                             # same inputs, other driver
        shadow.step_state(A, imm)
        learner.h = h.detach()

        with torch.no_grad():
            Rr, Ra = ref.residual_dense(), alt.residual_dense()
            nr = Rr.flatten(1).norm(dim=1).mean()
            d = {"d_resid": float(((Ra - Rr).flatten(1).norm(dim=1).mean() /
                                  nr.clamp_min(1e-30))),
                 "d_e": reldev(float(alt.e.mean()), float(ref.e.mean())),
                 "d_eta": reldev(float(alt.last["eta"].mean()),
                                 float(ref.last["eta"].mean()))}
            Er = (shadow.J - Rr).flatten(1).norm(dim=1).mean()
            Ea = (shadow.J - Ra).flatten(1).norm(dim=1).mean()
            d["d_trueE"] = reldev(float(Ea), float(Er))

        if y is not None:
            h_leaf = learner.h.requires_grad_(True)
            outp = learner.readout(h_leaf)
            loss, metric = learner.loss_fn(outp, y)
            learner.opt.zero_grad(set_to_none=True)
            loss.backward()
            delta = h_leaf.grad.detach()
            g_ref = ref.grad_rows(delta)
            with torch.no_grad():
                g_alt = alt.grad_rows(delta)
                d["d_ghat"] = float((g_alt - g_ref).norm() / g_ref.norm().clamp_min(1e-30))
            learner.cell.apply_flat_grad(g_ref)
            learner.opt.step()
            learner.cell.clip_spectral()
            learner.h = learner.h.detach()

        for k, v in d.items():
            if v is None or math.isnan(v):
                continue
            cnt[k] += 1
            acc[k] += v if math.isfinite(v) else 0.0
            if v > worst[k]:
                worst[k], worst_at[k] = v, step

    log = {"args": vars(args),
           "max_reldev": worst, "max_at_step": worst_at,
           "mean_reldev": {k: (acc[k] / cnt[k] if cnt[k] else None) for k in keys},
           "n_compared": cnt, "wall_s": time.time() - t0}
    json.dump(log, open(out, "w"), indent=1)
    print("saved", out)
    for k in keys:
        print(f"  {k:8s} max {worst[k]:.3e} (step {worst_at[k]})  "
              f"mean {log['mean_reldev'][k]:.3e}  n={cnt[k]}")


if __name__ == "__main__":
    main()
