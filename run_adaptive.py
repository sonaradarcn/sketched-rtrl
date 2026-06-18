"""R2b: certificate-guided adaptive-rank SK-RTRL.

A controller reads the normalized certificate c_t = e_t / (||S+LR^T||_F + eps)
every K steps and grows/shrinks the sketch rank r:
  if c_t > tau_high and r < r_max:      r <- min(2r, r_max)
  elif c_t < tau_low for M checks and r > r_min:  r <- max(r//2, r_min)

Because the per-step truncation already keeps the top-r columns and records the
discarded mass in eta_t (hence e_t), shrinking is certificate-safe by construction;
growing adds capacity that the next truncations fill. We log average rank, peak
memory, gradient cosine vs exact, task metric, and certificate violation rate.
Output: results/adaptive/<task>_adaptive_s<seed>.json  (or fixed-r baselines).
"""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F

from skrtrl.tasks import TASKS
from skrtrl.train import OnlineLearner
from skrtrl.algos import ExactRTRL, SKRTRL


def cos(a, b):
    na, nb = a.norm(), b.norm()
    return float("nan") if na < 1e-12 or nb < 1e-12 else ((a * b).sum() / (na * nb)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--shadow", type=int, default=1)
    # controller
    ap.add_argument("--fixed_r", type=int, default=-1)   # -1 -> adaptive; else fixed-r baseline
    ap.add_argument("--r_min", type=int, default=4)
    ap.add_argument("--r_max", type=int, default=32)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--tau_low", type=float, default=0.003)
    ap.add_argument("--tau_high", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=0.5)
    # E5 ablation: which signal drives the controller.
    #   eta    -> per-step relative discarded mass eta_t/||S+LR^T||_F   (our design)
    #   e_t    -> compounded normalized certificate e_t/||S+LR^T||_F     (naive)
    #   oracle -> hindsight TRUE residual ||J_exact - residual||_F norm. (needs shadow)
    ap.add_argument("--ctrl", choices=["eta", "e_t", "oracle"], default="eta")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="results/adaptive")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.ctrl == "oracle":
        args.shadow = 1   # oracle control reads the exact-shadow residual

    os.makedirs(args.outdir, exist_ok=True)
    base = f"fixed{args.fixed_r}" if args.fixed_r > 0 else f"adaptive-{args.ctrl}"
    tag = f"_{args.tag}" if args.tag else ""
    out = f"{args.outdir}/{args.task}_{base}_s{args.seed}{tag}.json"
    if os.path.exists(out):
        print("exists, skip:", out); return

    torch.manual_seed(args.seed)
    task = TASKS[args.task](args.batch, args.device, seed=args.seed)
    # build SK-RTRL allocated for r_max so the controller can grow into it; c fixed from r_max.
    learner = OnlineLearner(task, args.n, "skrtrl-r%d" % args.r_max, lr=args.lr,
                            device=args.device, seed=args.seed, spectral_clip=args.clip)
    algo = learner.algo
    assert isinstance(algo, SKRTRL)
    algo.r = args.r_max if args.fixed_r < 0 else args.fixed_r
    cur_r = args.r_min if args.fixed_r < 0 else args.fixed_r
    algo.r = cur_r
    shadow = ExactRTRL(learner.cell, task.B) if (args.shadow and args.n <= 256) else None

    log = {"args": vars(args), "records": []}
    metrics, coss, ranks = [], [], []
    viol = 0; checks = 0; low_streak = 0
    last_trueE = torch.zeros(task.B, device=args.device)   # for oracle controller (lagged 1 step)
    t0 = time.time()
    for step in range(args.steps):
        x, y, new_ep = task.step()
        learner._reset_lanes(new_ep)
        if shadow is not None and new_ep.any():
            shadow.J[new_ep] = 0.0
        h_prev = learner.h.detach()
        h = learner.cell(x, h_prev)
        A, imm = learner.cell.jac_pieces(x, h_prev, h)
        algo.step_state(A, imm)
        if shadow is not None:
            shadow.step_state(A, imm)
        learner.h = h.detach()
        ranks.append(algo.r)

        # certificate controller every K steps (adaptive only).
        # Control signal = per-step RELATIVE DISCARDED MASS eta_t / ||S+LR^T||_F, which
        # directly measures how much residual information the current rank throws away
        # (large -> rank insufficient -> grow; ~0 -> rank ample -> shrink). This is the
        # right signal: the compounded certificate e_t instead tracks rho_bar conservatism.
        if args.fixed_r < 0 and step > 0 and step % args.K == 0:
            checks += 1
            S_norm2 = (algo.S ** 2).sum(dim=(1, 2))
            L_norm2 = (algo.L ** 2).sum(dim=(1, 2)) if algo.L.numel() else torch.zeros_like(S_norm2)
            denom = torch.sqrt(S_norm2 + L_norm2).clamp_min(1e-8)
            if args.ctrl == "eta":
                signal = algo.last.get("eta", torch.zeros_like(denom))
            elif args.ctrl == "e_t":
                signal = algo.e
            else:  # oracle: hindsight true residual mass
                signal = last_trueE
            c_t = (signal / denom).mean().item()
            if c_t > args.tau_high and algo.r < args.r_max:
                algo.r = min(2 * algo.r, args.r_max); low_streak = 0
            elif c_t < args.tau_low:
                low_streak += 1
                if low_streak >= args.M and algo.r > args.r_min:
                    algo.r = max(algo.r // 2, args.r_min); low_streak = 0
            else:
                low_streak = 0

        if y is not None:
            h_leaf = learner.h.requires_grad_(True)
            outp = learner.readout(h_leaf)
            loss, metric = learner.loss_fn(outp, y)
            learner.opt.zero_grad(set_to_none=True)
            loss.backward()
            delta = h_leaf.grad.detach()
            g_rows = algo.grad_rows(delta)
            if shadow is not None:
                coss.append(cos(g_rows.flatten(), shadow.grad_rows(delta).flatten()))
                E = shadow.J - algo.residual_dense()
                trueE = E.flatten(1).norm(dim=1)
                last_trueE = trueE.detach()
                if (trueE > algo.e + 1e-6 * (1 + shadow.J.flatten(1).norm(dim=1))).any():
                    viol += 1
            learner.cell.apply_flat_grad(g_rows)
            learner.opt.step(); learner.cell.clip_spectral()
            learner.h = learner.h.detach()
            metrics.append(metric)

        if step % args.log_every == 0 and metrics:
            rec = {"step": step, "metric": sum(metrics[-200:]) / len(metrics[-200:]),
                   "rank": algo.r, "avg_rank": sum(ranks[-args.log_every:]) / len(ranks[-args.log_every:]),
                   "grad_cos": (sum(c for c in coss[-200:] if not math.isnan(c)) /
                                max(sum(0 if math.isnan(c) else 1 for c in coss[-200:]), 1)) if coss else None,
                   "e_t": algo.e.mean().item()}
            la = algo.last
            if la and "eta" in la:
                rec["eta"] = la["eta"].mean().item()
                if "rho_bar" in la:
                    rec["rho_bar"] = la["rho_bar"].mean().item()
                if "rho_hat" in la:
                    rec["rho_hat"] = la["rho_hat"].mean().item()
            log["records"].append(rec)
            if step % (args.log_every * 8) == 0:
                print(f"[{args.task}/{base}/s{args.seed}] step {step} metric {rec['metric']:.4f} "
                      f"rank {algo.r} cos {rec.get('grad_cos')}", flush=True)

    log["wall_s"] = time.time() - t0
    log["peak_MB"] = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0
    log["avg_rank"] = sum(ranks) / len(ranks)
    log["cert_violations"] = viol
    log["cert_checks_with_shadow"] = len(coss)
    json.dump(log, open(out, "w"), indent=1)
    print("saved", out, f"(avg_rank {log['avg_rank']:.1f}, viol {viol}, {log['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
