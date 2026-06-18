"""M3 main suite runner: one (task, algo, seed) online-learning run with optional exact shadow.

Logs: task metric curve, gradient cosine vs exact (shadow), certificate trace (SK-RTRL).
Output: results/m3/<task>_<algo>_s<seed>.json
"""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F

from skrtrl.tasks import TASKS
from skrtrl.train import OnlineLearner
from skrtrl.algos import ExactRTRL


def cos(a, b):
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return ((a * b).sum() / (na * nb)).item()


def run_tbptt(args, task, out):
    """Offline reference: truncated BPTT with window k, update at window end."""
    import torch.nn as nn
    torch.manual_seed(args.seed)
    dev = args.device
    cell_lin = nn.RNN(task.n_in, args.n, nonlinearity="tanh", batch_first=False).to(dev)
    readout = nn.Linear(args.n, task.n_out).to(dev)
    opt = torch.optim.Adam(list(cell_lin.parameters()) + list(readout.parameters()), lr=args.lr)
    h = torch.zeros(1, task.B, args.n, device=dev)
    window, metrics = [], []
    log, t0 = {"args": vars(args), "records": []}, time.time()
    for step in range(args.steps):
        x, y, new_ep = task.step()
        if new_ep.any():
            h = h.detach()
            h[0][new_ep] = 0.0
        window.append((x, y))
        if len(window) >= args.tbptt_k or step == args.steps - 1:
            h = h.detach()
            losses = []
            for (xw, yw) in window:
                outp, h = cell_lin(xw.unsqueeze(0), h)
                if yw is not None:
                    o = readout(outp[0])
                    l = F.cross_entropy(o, yw) if task.loss_type == "ce" else F.mse_loss(o, yw)
                    losses.append(l)
                    metrics.append((o.argmax(1) == yw).float().mean().item()
                                   if task.loss_type == "ce" else l.item())
            if losses:
                opt.zero_grad()
                torch.stack(losses).mean().backward()
                opt.step()
            window = []
        if step % args.log_every == 0 and metrics:
            log["records"].append({"step": step, "metric": sum(metrics[-200:]) / len(metrics[-200:])})
    log["wall_s"] = time.time() - t0
    json.dump(log, open(out, "w"), indent=1)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--algo", required=True)   # exact|snap1|skrtrl-rK|uoro|kfrtrl|rflo|tbptt
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--shadow", type=int, default=1)   # exact shadow for grad-cosine (n<=128)
    ap.add_argument("--tbptt_k", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="results/m3")
    ap.add_argument("--tag", default="")
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--horizon", type=int, default=1)     # multi-step prediction horizon
    ap.add_argument("--causal", type=int, default=0)      # 1 = causal normalization
    ap.add_argument("--washout", type=int, default=200)   # reset period (0 = no reset)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = f"{args.outdir}/{args.task}_{args.algo}_s{args.seed}{tag}.json"
    if os.path.exists(out):
        print("exists, skip:", out)
        return
    TS_TASKS = {"henon", "mackeyglass", "lorenz", "sunspot", "laser"}
    if args.task in TS_TASKS:
        task = TASKS[args.task](args.batch, args.device, seed=args.seed,
                                horizon=args.horizon, causal=bool(args.causal), washout=args.washout)
    else:
        task = TASKS[args.task](args.batch, args.device, seed=args.seed)

    if args.algo == "tbptt":
        run_tbptt(args, task, out)
        return

    learner = OnlineLearner(task, args.n, args.algo, lr=args.lr, device=args.device, seed=args.seed,
                            spectral_clip=args.clip)
    shadow = ExactRTRL(learner.cell, task.B) if (args.shadow and args.n <= 256 and args.algo != "exact") else None

    log = {"args": vars(args), "records": []}
    metrics, coss = [], []
    t0 = time.time()
    for step in range(args.steps):
        # --- replicate learner.step but with shadow hooks ---
        x, y, new_ep = task.step()
        learner._reset_lanes(new_ep)
        if shadow is not None and new_ep.any():
            shadow.J[new_ep] = 0.0
        h_prev = learner.h.detach()
        h = learner.cell(x, h_prev)
        A, imm = learner.cell.jac_pieces(x, h_prev, h)
        learner.algo.step_state(A, imm)
        if shadow is not None:
            shadow.step_state(A, imm)
        learner.h = h.detach()
        if y is not None:
            h_leaf = learner.h.requires_grad_(True)
            outp = learner.readout(h_leaf)
            loss, metric = learner.loss_fn(outp, y)
            learner.opt.zero_grad(set_to_none=True)
            loss.backward()
            delta = h_leaf.grad.detach()
            g_rows = learner.algo.grad_rows(delta)
            if shadow is not None:
                g_true = shadow.grad_rows(delta)
                coss.append(cos(g_rows.flatten(), g_true.flatten()))
            learner.cell.apply_flat_grad(g_rows)
            learner.opt.step()
            learner.cell.clip_spectral()
            learner.h = learner.h.detach()
            metrics.append(metric)
        if step % args.log_every == 0 and metrics:
            rec = {"step": step,
                   "metric": sum(metrics[-200:]) / len(metrics[-200:]),
                   "grad_cos": (sum(c for c in coss[-200:] if not math.isnan(c)) /
                                max(sum(0 if math.isnan(c) else 1 for c in coss[-200:]), 1)) if coss else None}
            la = getattr(learner.algo, "last", None)
            if la and "rho_bar" in la:
                rec["rho_bar"] = la["rho_bar"].mean().item()
                if "rho_hat" in la:
                    rec["rho_hat"] = la["rho_hat"].mean().item()
                rec["eta"] = la["eta"].mean().item()
                rec["e_t"] = learner.algo.e.mean().item()
                if shadow is not None and hasattr(learner.algo, "residual_dense"):
                    E = shadow.J - learner.algo.residual_dense()
                    rec["true_E"] = E.flatten(1).norm(dim=1).mean().item()
            log["records"].append(rec)
            if step % (args.log_every * 8) == 0:
                print(f"[{args.task}/{args.algo}/s{args.seed}] step {step} "
                      f"metric {rec['metric']:.4f} cos {rec.get('grad_cos')}", flush=True)
    log["wall_s"] = time.time() - t0
    if args.device == "cuda":
        log["peak_MB"] = torch.cuda.max_memory_allocated() / 2**20
    json.dump(log, open(out, "w"), indent=1)
    print("saved", out, f"({log['wall_s']:.0f}s, {log.get('peak_MB',0):.0f}MB)")


if __name__ == "__main__":
    main()
