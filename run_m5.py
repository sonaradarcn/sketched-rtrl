"""M5 online-RL suite runner: T-maze POMDP, one (corridor length, algo, seed) run.

Algos: tanh-core online actor-critic (exact | snap1 | skrtrl-rK | uoro | kfrtrl | rflo),
diagonal exact-trace cores (lru | rtu), and truncated-BPTT (window k = 2N) offline
actor-critic reference (tbptt).
Output: results/m5/tmaze<len>_<algo>_s<seed>.json  (saved every --log_every steps:
running mean episode return, episode length, success rate over last 400 episodes).

--shadow (R2, plan D1/D2, tanh cores only) adds an exact-RTRL shadow and a SnAp-1
tracker on the same cell (skrtrl.rl.TanhCore), which do not touch the parameters:
  records      grad_cos, delta_norm, ghat_norm, bound_norm, g_norm, gerr_norm,
               J_norm, true_E, e_t, rho_bar, rho_hat, eta, rho_g, tight
  cert_summary D2 online counters over EVERY gradient step (definitions and keys
               as in run_m3.CertCounters, plus n_no_signal / n_rhobar_lt1 and the
               same counters bucketed into --cert_stages training stages)
  spectrum     D1 residual spectrum of R_t = J_t - S_t at --spectrum_ckpts
               (fractions of --steps): sv, mass_top{1,4,8,16,32,64}, stable_rank,
               res_frac_of_J
With --shadow off nothing above is computed and the run is bit-identical to the
pre-R2 driver.
"""
import argparse, json, os, time
from collections import deque
import torch

from skrtrl.envs import TMaze
from skrtrl.rl import ActorCritic, TanhCore
from skrtrl.diag_cells import OnlineLRU, RTU

DIAG_ALGOS = ("lru", "rtu")


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


def ckpt_steps(spec, steps, defer=0, back=None):
    """"0.05,0.25,1.0" -> {nominal step: (fraction, window start, deadline)}.

    All B lanes of TMaze start together and time out together at 2N, so early in
    training (before the agent reaches the junction) the episode resets are
    perfectly synchronised -- and every checkpoint of a 60 000-step run is a
    multiple of 40.  A checkpoint landing on such a step would measure J_t = S_t,
    i.e. an all-zero residual.  Each checkpoint therefore gets a window
    [nominal, nominal + defer] (the last one cannot slip past the end of the run,
    so it looks back over `back` steps instead -- kept short, just wide enough to
    step out of a synchronised reset) in which the spectrum is taken at
    the first step whose MEAN lane age is at least --spectrum_min_age; at the
    deadline it is taken anyway.  The mean is the right statistic: it equals the
    per-lane age exactly in the synchronised regime this guards against, while once
    the resets are asynchronous (some lane is fresh almost every step) it passes at
    the nominal step, where the residual is not degenerate anyway.
    """
    out = {}
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        f = float(tok)
        nom = min(max(int(round(f * steps)), 1), steps)
        start = nom if nom < steps else max(1, steps - (defer if back is None else back))
        out[nom] = (f, start, min(nom + defer, steps))
    return out


def run_online(args, env, out):
    if args.algo in DIAG_ALGOS:
        n = args.n or 64
        core = (OnlineLRU if args.algo == "lru" else RTU)(env.n_obs, n, device=args.device)
    else:
        n = args.n or 128
        core = TanhCore(env.n_obs, n, args.algo, env.B, device=args.device,
                        shadow=bool(args.shadow), cert_warmup=args.cert_warmup,
                        cert_min_age=args.cert_min_age, cert_every=args.cert_every,
                        spectral_clip=args.clip, svd_driver=args.svd_driver,
                        total_steps=args.steps, n_stages=args.cert_stages)
    learner = ActorCritic(env, core, lr=args.lr, gamma=args.gamma, beta=args.beta,
                          accumulate_k=args.accumulate_k, device=args.device)
    shadow_on = getattr(core, "shadow", None) is not None
    log = {"args": vars(args), "records": []}
    stats = EpStats(env.B, args.device)
    ckpts = (ckpt_steps(args.spectrum_ckpts, args.steps, args.spectrum_defer,
                        back=min(args.spectrum_defer,
                                 int(2 * args.spectrum_min_age) + 8))
             if shadow_on else {})
    if shadow_on:
        log["spectrum"] = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        r, done, aux = learner.step()
        stats.update(r, done)
        for nom, (frac, start, dead) in list(ckpts.items()):
            if step < start:
                continue
            if (float(core.age.float().mean()) >= args.spectrum_min_age
                    or step >= dead):
                del ckpts[nom]
                spec = core.spectrum()
                log["spectrum"].append({"step": step, "step_nominal": nom,
                                        "frac": frac, **spec})
                print(f"[tmaze{args.env_len}/{args.algo}/s{args.seed}] spectrum @{step} "
                      f"(nom {nom}, age {spec['age_mean']:.1f}) "
                      f"top16 {spec.get('mass_top16', float('nan')):.3f} "
                      f"sr {spec['stable_rank']:.2f} "
                      f"resfrac {spec['res_frac_of_J']:.3f}", flush=True)
        if step % args.log_every == 0:
            if shadow_on:
                aux = {**aux, **core.cert_record()}
                log["cert_summary"] = core.cert_summary()
            log_step(log, out, stats, step, aux, args, t0)
    if shadow_on:                       # shadow-only summary field
        log["cert_summary"] = core.cert_summary()
    log["peak_MB"] = (torch.cuda.max_memory_allocated() / 2 ** 20
                      if args.device == "cuda" else None)
    log["wall_s"] = time.time() - t0    # final flush (steps may not divide log_every)
    json.dump(log, open(out, "w"), indent=1)
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
    # ---- R2 shadow instrumentation (plan D1/D2); default off = pre-R2 behaviour ----
    ap.add_argument("--shadow", type=int, default=0,
                    help="1 = exact-RTRL shadow + SnAp-1 tracker (tanh cores only)")
    ap.add_argument("--spectrum_ckpts", default="0.05,0.25,0.5,0.75,1.0",
                    help="D1 residual-spectrum checkpoints as fractions of --steps "
                         "(only used with --shadow 1)")
    ap.add_argument("--spectrum_min_age", type=float, default=4.0,
                    help="a D1 checkpoint waits for the mean lane age to reach "
                         "this many steps into the episode (all lanes reset "
                         "synchronously while the agent still times out, which "
                         "would make the residual J_t - S_t exactly zero)")
    ap.add_argument("--spectrum_defer", type=int, default=200,
                    help="how far a checkpoint may slip to satisfy "
                         "--spectrum_min_age before it is taken anyway")
    ap.add_argument("--cert_warmup", type=int, default=200,
                    help="D2 counters start after this many gradient steps")
    ap.add_argument("--cert_min_age", type=int, default=0,
                    help="skip steps whose minimum per-lane age since the episode "
                         "reset is below this (0 = off; lanes reset asynchronously)")
    ap.add_argument("--cert_stages", type=int, default=3,
                    help="number of equal training-stage buckets the D2 counters "
                         "are reported in as well (0 = run average only)")
    ap.add_argument("--cert_every", type=int, default=1,
                    help="stride of the D2 counters (1 = every gradient step)")
    ap.add_argument("--clip", type=float, default=0.0,
                    help="spectral clip of W after each update (0 = off, as in R1); "
                         "the D2 knob that keeps rho_bar_t from making e_t vacuous")
    ap.add_argument("--svd_driver", choices=["gesvd", "auto"], default="gesvd",
                    help="SK-RTRL SVD driver: gesvd = kernel of record (all R1 "
                         "numbers), auto = library default first, same fallbacks")
    args = ap.parse_args()

    if args.shadow and (args.algo in DIAG_ALGOS or args.algo == "tbptt"):
        raise SystemExit(f"--shadow is only supported for tanh cores, not {args.algo!r} "
                         "(the diagonal cores and tbptt keep no influence matrix)")

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
