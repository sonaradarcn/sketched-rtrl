"""M5 online-RL suite runner: T-maze POMDP, one (corridor length, algo, seed) run.

Algos: tanh-core online actor-critic (exact | snap1 | skrtrl-rK | uoro | kfrtrl | rflo),
diagonal exact-trace cores (lru | rtu), and truncated-BPTT (window k = 2N) offline
actor-critic reference (tbptt).
Output: results/m5/tmaze<len>_<algo>_s<seed>.json  (saved every --log_every steps:
running mean episode return, episode length, success rate over last 400 episodes).
"""
import argparse, json, os, time
from collections import deque
import torch

from skrtrl.envs import TMaze
from skrtrl.rl import ActorCritic, TanhCore
from skrtrl.diag_cells import OnlineLRU, RTU


class EpStats:
    def __init__(self, batch, device, window=400):
        self.ret = torch.zeros(batch, device=device)
        self.len = torch.zeros(batch, device=device)
        self.rets, self.lens, self.succ = (deque(maxlen=window) for _ in range(3))
        self.episodes = 0

    def update(self, r, done):
        self.ret += r
        self.len += 1
        if done.any():
            fr, fl = self.ret[done].tolist(), self.len[done].tolist()
            self.rets.extend(fr)
            self.lens.extend(fl)
            self.succ.extend((r[done] > 3.0).float().tolist())   # +4 = correct turn
            self.episodes += len(fr)
            self.ret[done] = 0.0
            self.len[done] = 0.0

    def rec(self):
        m = lambda d: sum(d) / max(len(d), 1)
        return {"episodes": self.episodes, "ret": m(self.rets),
                "ep_len": m(self.lens), "success": m(self.succ)}


def log_step(log, out, stats, step, aux, args, t0):
    rec = {"step": step, **stats.rec(), **aux}
    log["records"].append(rec)
    log["wall_s"] = time.time() - t0
    json.dump(log, open(out, "w"), indent=1)
    print(f"[tmaze{args.env_len}/{args.algo}/s{args.seed}] step {step} "
          f"ep {rec['episodes']} ret {rec['ret']:.3f} len {rec['ep_len']:.1f} "
          f"succ {rec['success']:.2f}", flush=True)


def run_online(args, env, out):
    if args.algo in ("lru", "rtu"):
        n = args.n or 64
        core = (OnlineLRU if args.algo == "lru" else RTU)(env.n_obs, n, device=args.device)
    else:
        n = args.n or 128
        core = TanhCore(env.n_obs, n, args.algo, env.B, device=args.device)
    learner = ActorCritic(env, core, lr=args.lr, gamma=args.gamma, beta=args.beta,
                          accumulate_k=args.accumulate_k, device=args.device)
    log = {"args": vars(args), "records": []}
    stats = EpStats(env.B, args.device)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        r, done, aux = learner.step()
        stats.update(r, done)
        if step % args.log_every == 0:
            log_step(log, out, stats, step, aux, args, t0)
    print("saved", out, f"({log['wall_s']:.0f}s)")


def run_tbptt(args, env, out):
    """Offline reference: truncated-window (k = 2N) actor-critic via autograd."""
    import torch.nn as nn
    from torch.distributions import Categorical
    from skrtrl.cells import TanhRNNCell
    n = args.n or 128
    dev = args.device
    cell = TanhRNNCell(env.n_obs, n, device=dev)
    pi = nn.Linear(n, env.n_actions).to(dev)
    vh = nn.Linear(n, 1).to(dev)
    opt = torch.optim.Adam(list(cell.parameters()) + list(pi.parameters())
                           + list(vh.parameters()), lr=args.lr)
    k = 2 * args.env_len
    obs = env.reset()
    h = cell.init_state(env.B)
    prev_done = torch.zeros(env.B, dtype=torch.bool, device=dev)
    buf = []                                            # (logp, ent, V, r, done)
    log = {"args": vars(args), "records": []}
    stats = EpStats(env.B, dev)
    aux, t0 = {}, time.time()
    for step in range(1, args.steps + 1):
        h = cell(obs, h * (~prev_done).float().unsqueeze(1))   # graph across window
        dist = Categorical(logits=pi(h))
        a = dist.sample()
        V = vh(h).squeeze(1)
        obs, r, done = env.step(a)
        buf.append((dist.log_prob(a), dist.entropy(), V, r, done))
        prev_done = done
        stats.update(r, done)
        if len(buf) >= k:
            with torch.no_grad():
                V_boot = vh(cell(obs, h * (~done).float().unsqueeze(1))).squeeze(1)
            V_next = [b[2].detach() for b in buf[1:]] + [V_boot]
            terms = []
            for (logp, ent, V_, r_, d_), Vn in zip(buf, V_next):
                td = r_ + args.gamma * (~d_).float() * Vn - V_
                terms.append(-td.detach() * logp + 0.5 * td.pow(2) - args.beta * ent)
            opt.zero_grad(set_to_none=True)
            torch.stack(terms).mean().backward()
            opt.step()
            h = h.detach()
            buf = []
            aux = {"loss": terms[-1].mean().item(), "entropy": ent.mean().item()}
        if step % args.log_every == 0:
            log_step(log, out, stats, step, aux, args, t0)
    print("saved", out, f"({log['wall_s']:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_len", type=int, default=10)   # {10, 20, 40, 80}
    ap.add_argument("--algo", required=True)  # skrtrl-rK|snap1|exact|uoro|kfrtrl|rflo|lru|rtu|tbptt
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--n", type=int, default=0)          # 0 -> 128 tanh core, 64 diag units
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--accumulate_k", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="results/m5")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out = f"{args.outdir}/tmaze{args.env_len}_{args.algo}_s{args.seed}.json"
    if os.path.exists(out):
        print("exists, skip:", out)
        return
    torch.manual_seed(args.seed)
    env = TMaze(args.batch, args.env_len, args.device, seed=args.seed)
    if args.algo == "tbptt":
        run_tbptt(args, env, out)
    else:
        run_online(args, env, out)


if __name__ == "__main__":
    main()
