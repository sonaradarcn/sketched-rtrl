"""E3: forecasting-architecture baselines on the time-series tasks.

Trains GRU / LSTM (truncated BPTT, same streaming protocol as the vanilla-RNN TBPTT
reference) and an Echo State Network (fixed random reservoir + ridge readout) at a
matched hidden width, and reports the normalized MSE (NMSE) on the held-out tail of
the stream. This gives a forecasting-quality reference point against which SK-RTRL's
online-credit-assignment runs can be read (the paper's claim is about online credit
assignment, not forecasting SOTA, so these are context baselines).

Output: results/round1/arch/<task>_<arch>_s<seed>.json  (same json shape as run_m3).
"""
import argparse, json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrtrl.tasks import TASKS

TS_TASKS = {"henon", "mackeyglass", "lorenz", "sunspot", "laser"}


def make_task(args):
    return TASKS[args.task](args.batch, args.device, seed=args.seed,
                            horizon=args.horizon, causal=bool(args.causal), washout=args.washout)


def run_rnn_family(args, task, kind):
    """GRU/LSTM via truncated BPTT in the streaming loop (window=tbptt_k)."""
    torch.manual_seed(args.seed)
    dev = args.device
    Cell = {"gru": nn.GRU, "lstm": nn.LSTM}[kind]
    net = Cell(task.n_in, args.n, batch_first=False).to(dev)
    readout = nn.Linear(args.n, task.n_out).to(dev)
    opt = torch.optim.Adam(list(net.parameters()) + list(readout.parameters()), lr=args.lr)

    def zero_state(B):
        h = torch.zeros(1, B, args.n, device=dev)
        return (h, torch.zeros_like(h)) if kind == "lstm" else h

    state = zero_state(task.B)
    window, metrics = [], []
    log, t0 = {"args": vars(args), "records": []}, time.time()
    for step in range(args.steps):
        x, y, new_ep = task.step()
        if new_ep.any():
            if kind == "lstm":
                state = (state[0].detach(), state[1].detach())
                state[0][:, new_ep] = 0.0; state[1][:, new_ep] = 0.0
            else:
                state = state.detach(); state[:, new_ep] = 0.0
        window.append((x, y))
        if len(window) >= args.tbptt_k or step == args.steps - 1:
            state = (state[0].detach(), state[1].detach()) if kind == "lstm" else state.detach()
            losses = []
            for (xw, yw) in window:
                outp, state = net(xw.unsqueeze(0), state)
                if yw is not None:
                    o = readout(outp[0])
                    losses.append(F.mse_loss(o, yw))
                    metrics.append(F.mse_loss(o, yw).item())
            if losses:
                opt.zero_grad(set_to_none=True)
                torch.stack(losses).mean().backward()
                opt.step()
            window = []
        if step % args.log_every == 0 and metrics:
            log["records"].append({"step": step, "metric": sum(metrics[-200:]) / len(metrics[-200:])})
    log["wall_s"] = time.time() - t0
    log["final_nmse"] = sum(metrics[-1000:]) / max(len(metrics[-1000:]), 1)
    return log


def run_esn(args, task):
    """Echo State Network: fixed random reservoir + ridge-regression readout.

    Reservoir size = args.esn_res (convention uses a wide reservoir); spectral radius
    rho_res; leaky integration alpha. Readout solved in closed form on the first half
    of the stream, NMSE measured on the held-out tail.
    """
    torch.manual_seed(args.seed)
    dev = args.device
    Nr = args.esn_res
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    Win = (torch.rand(Nr, task.n_in, generator=g) * 2 - 1).to(dev) * args.esn_in_scale
    W = (torch.rand(Nr, Nr, generator=g) * 2 - 1).to(dev)
    sr = torch.linalg.eigvals(W).abs().max().real
    W = W * (args.esn_rho / sr)
    alpha = args.esn_leak

    # collect states
    h = torch.zeros(task.B, Nr, device=dev)
    X, Y = [], []
    t0 = time.time()
    for step in range(args.steps):
        x, y, new_ep = task.step()
        if new_ep.any():
            h[new_ep] = 0.0
        pre = x @ Win.T + h @ W.T
        h = (1 - alpha) * h + alpha * torch.tanh(pre)
        if y is not None and step > args.washout:
            X.append(torch.cat([h, x], dim=1))   # state + input bias term
            Y.append(y)
    Xall = torch.cat(X, dim=0)            # (T*B, Nr+1)
    Yall = torch.cat(Y, dim=0)            # (T*B, 1)
    n = Xall.shape[0]; cut = n // 2
    Xtr, Ytr, Xte, Yte = Xall[:cut], Yall[:cut], Xall[cut:], Yall[cut:]
    # ridge readout
    lam = args.esn_ridge
    A = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], device=dev)
    Wout = torch.linalg.solve(A, Xtr.T @ Ytr)
    pred = Xte @ Wout
    nmse = F.mse_loss(pred, Yte).item()       # series is unit-variance -> MSE == NMSE
    log = {"args": vars(args), "records": [{"step": args.steps, "metric": nmse}],
           "final_nmse": nmse, "wall_s": time.time() - t0, "esn_res": Nr}
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TS_TASKS))
    ap.add_argument("--arch", required=True, choices=["gru", "lstm", "esn"])
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--tbptt_k", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--washout", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="results/round1/arch")
    ap.add_argument("--tag", default="")
    # ESN hyperparameters
    ap.add_argument("--esn_res", type=int, default=200)
    ap.add_argument("--esn_rho", type=float, default=0.95)
    ap.add_argument("--esn_in_scale", type=float, default=1.0)
    ap.add_argument("--esn_leak", type=float, default=0.3)
    ap.add_argument("--esn_ridge", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = f"{args.outdir}/{args.task}_{args.arch}_s{args.seed}{tag}.json"
    if os.path.exists(out):
        print("exists, skip:", out); return

    task = make_task(args)
    log = run_esn(args, task) if args.arch == "esn" else run_rnn_family(args, task, args.arch)
    json.dump(log, open(out, "w"), indent=1)
    print(f"saved {out} (nmse {log['final_nmse']:.4f}, {log['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
