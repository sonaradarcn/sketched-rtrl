"""M3 main suite runner: one (task, algo, seed) online-learning run with optional exact shadow.

Logs: task metric curve, gradient cosine vs exact (shadow), certificate trace (SK-RTRL),
and (R2 / D2) the certificate-informativeness quantities:
  per record  ||delta_t||, ||ghat_t||, ||g_t||, ||g_t - ghat_t||, ||J_t||_F
  top level   `cert_summary` -- online counters accumulated at EVERY post-washout step
              (independent of --log_every).

Output: results/m3/<task>_<algo>_s<seed>.json          (--cell tanh, unchanged)
        results/m3/<task>_<cell>_<algo>_s<seed>.json  (--cell gru|lstm)
"""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F

from skrtrl.tasks import TASKS
from skrtrl.train import OnlineLearner
from skrtrl.algos import (ExactRTRL, SKRTRL, SVD_STATS, _svd_stats_reset,
                          svd_fallbacks)
from skrtrl.rl import CertCounters as _CertBase   # single source of truth for D2

TS_TASKS = {"henon", "mackeyglass", "lorenz", "sunspot", "laser"}


def cos(a, b):
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return ((a * b).sum() / (na * nb)).item()


def flat_grad_rows(cell):
    """(n_units, p_cell) view of the autograd gradient of the recurrent params.

    Delegated to the cell: the gated cells use the opposite W/U convention
    (U recurrent, W input) and hold all gates in one stacked tensor, so
    hard-coding TanhRNNCell's three tensors here would silently compare a
    transposed gradient in the TBPTT grad-cosine path.
    """
    return cell.flat_grad_rows()


def readout_in(cell, state):
    """The part of the carried state the read-out sees (h_t).

    Identity for the vanilla cell and the GRU; for the LSTM the carried state
    is [h_t; c_t], and the read-out is on h_t only (see
    skrtrl.train.OnlineLearner.readout_in).
    """
    return state[:, :cell.n] if getattr(cell, "n_state", cell.n) != cell.n else state


def shadow_ok(cell, n_max, n_in):
    """Memory guard on the O(n_state * n_units * p_cell) exact shadow.

    --shadow_max_n keeps its meaning as a VANILLA-cell width: a gated cell is
    admitted up to the width whose shadow costs no more than a vanilla cell of
    width n_max.  The GRU triples p and the LSTM quadruples it while doubling
    the state, so the affordable width drops to roughly 0.69 n_max (GRU) and
    0.5 n_max (LSTM).  For the vanilla cell the test is exactly n <= n_max.
    """
    budget = n_max * n_max * (n_max + n_in + 1)
    cost = getattr(cell, "n_state", cell.n) * cell.n * getattr(cell, "p_cell", cell.p)
    return cost <= budget


class CertCounters(_CertBase):
    """D2 counters.

    The definitions live in ONE place, ``skrtrl.rl.CertCounters``, which both this
    runner and ``run_m5.py`` use, so the two cannot drift: n_steps (signal-bearing
    steps only), n_no_signal, n_rhobar_lt1, n_rel_lt1 / n_rel_lt05, tightness
    (n_tight_checked / n_tight_le10), the Theorem-1 checks
    (n_bound_violation, n_bound_violation_fp), n_cert_invalid, and the same counters
    bucketed into training stages.  Batch aggregation: the bound is
    mean_b(||delta_b||_2 * e_b), i.e. the lane-wise product is averaged.

    Added here:
      age buckets     the same counters bucketed by the age since the last episode
                      reset, edges [0, T/4), [T/4, T/2), [T/2, inf), with T the reset
                      period (--washout on the time-series tasks, the episode length
                      otherwise).  Post-reset transients and the deep-history regime
                      are then reported separately instead of being averaged together.
      n_bound_finite  signal-bearing steps whose bound is finite, so a Theorem-1
                      "0 violations" can never be read off an infinite certificate.
    """

    def __init__(self, has_shadow, n_stages=0, span=None, age_span=0):
        super().__init__(has_shadow, n_stages=n_stages, span=span)
        self.n_bound_finite = 0
        self.age_span = int(age_span or 0)
        if self.age_span >= 4:
            self.age_edges = [max(1, self.age_span // 4), max(2, self.age_span // 2)]
            self.age_buckets = [_CertBase(has_shadow) for _ in range(3)]
        else:
            self.age_edges, self.age_buckets = [], []

    def update(self, bound, ghat_norm, gerr_norm, e_mean, trueE_mean,
               rho_bar=None, delta_norm=None, stage=None, age=None):
        if self.age_buckets and age is not None:
            i = 0
            while i < len(self.age_edges) and age >= self.age_edges[i]:
                i += 1
            self.age_buckets[i].update(bound, ghat_norm, gerr_norm, e_mean, trueE_mean,
                                       rho_bar=rho_bar, delta_norm=delta_norm)
        no_signal = ghat_norm <= 0 or (delta_norm is not None and delta_norm <= 0)
        if not no_signal and gerr_norm is not None and math.isfinite(bound):
            self.n_bound_finite += 1
        super().update(bound, ghat_norm, gerr_norm, e_mean, trueE_mean,
                       rho_bar=rho_bar, delta_norm=delta_norm, stage=stage)

    def summary(self):
        out = super().summary()
        out["n_bound_finite"] = self.n_bound_finite
        out["grad_agg"] = ("||ghat_t||_F and ||g_t||_F are Frobenius norms of the "
                           "BATCH-MEAN (n, p) gradient rows actually applied to the "
                           "parameters; delta_norm is mean_b ||delta_b||_2 (per-lane "
                           "2-norms, averaged); e_t is mean_b e_b; the Theorem-1 bound "
                           "is mean_b(||delta_b||_2 * e_b)")
        out["counted_steps"] = ("labelled steps with step >= cert_warmup and "
                                "age >= cert_min_age; delta_t (hence rho^g_t) exists "
                                "only where the task emits a target")
        if self.age_buckets:
            out["age_span"] = self.age_span
            out["age_edges"] = self.age_edges
            out["age_buckets"] = [c.summary() for c in self.age_buckets]
        return out


@torch.no_grad()
def sketch_efficiency(shadow, algo, ranks=(4, 8, 16, 32)):
    """A8: is the sketch as good as the best rank-r approximation of the residual?

    R_t = J_t - S_t is the part of the influence matrix the SnAp-1 trace misses, and
    the sketch spends rank r on it.  The best any rank-r approximation can do is
    best_r = sqrt(sum_{i>r} sigma_i(R_t)^2), so

        sketch_eff = ||E_t||_F / best_r     (>= 1;  1 = optimal rank-r sketch)

    says how much of the achievable accuracy the kernel actually realises -- a
    different question from whether the certificate e_t is tight.
    Singular values are taken lane by lane (P = n*p is wide); the reported ratio uses
    lane means, matching the res_frac_of_J convention of the D1 spectrum code.
    """
    # ns = state rows, nu = parameter blocks (they differ on the LSTM, whose
    # state rows h_i and c_i share unit i's block)
    ns, nu, p, P = algo.n, algo.n_units, algo.p, algo.P
    r_eff = algo.L.shape[2] if algo.L.numel() else 0
    Eres = shadow.J - algo.residual_dense()
    e_norm = float(Eres.flatten(1).norm(dim=1).mean())
    svs, rn = [], []
    for b in range(algo.B):
        R = shadow.J[b].clone().view(ns, nu, p)
        srow = torch.arange(ns, device=R.device)
        R[srow, algo.blk_idx[srow], :] -= algo.S[b]      # R = J - S (SnAp-1 trace)
        R = R.view(ns, P)
        try:
            sv = torch.linalg.svdvals(R)
        except Exception:
            sv = torch.linalg.svdvals(R.cpu()).to(R.device)
        svs.append(sv)
        rn.append(float(R.norm()))
    S = torch.stack(svs)                                # (B, min(n, P))
    tot = (S ** 2).sum(dim=1).clamp_min(1e-30)

    def best(k):
        return float(torch.sqrt((S[:, k:] ** 2).sum(dim=1).clamp_min(0)).mean())             if k < S.shape[1] else 0.0

    b_eff = best(r_eff)
    rec = {"r": int(algo.r), "r_factor_width": int(r_eff),
           "res_norm": sum(rn) / len(rn),
           "E_norm": e_norm,
           "best_rank_err": b_eff,
           "sketch_eff": (e_norm / b_eff) if b_eff > 0 else None,
           "stable_rank": float((tot / (S[:, 0] ** 2).clamp_min(1e-30)).mean()),
           "e_t": float(algo.e.mean())}
    for k in ranks:
        if k < S.shape[1]:
            rec[f"mass_top{k}"] = float(((S[:, :k] ** 2).sum(dim=1) / tot).mean())
            rec[f"best_rank_err_r{k}"] = best(k)
    return rec


def holdout_pass(task, cell, readout, loss_fn, args):
    """A11: stream the held-out tail of a time-series task exactly ONCE, online,
    with no parameter updates and no estimator involved -- one-step-ahead tracking
    on data that was never cycled during training.

    The hidden state starts from zero and the same --washout reset period is applied
    inside the tail, so the evaluation protocol matches training step for step; the
    only difference is that nothing is learned.  Returns a dict or None.
    """
    n_hold = task.enter_holdout() if hasattr(task, "enter_holdout") else 0
    if not n_hold:
        return None
    h = cell.init_state(task.B)
    vals, steps = [], 0
    with torch.no_grad():
        while steps < n_hold and not task.holdout_exhausted():
            x, y, new_ep = task.step()
            if new_ep.any():
                h = h.clone()
                h[new_ep] = 0.0
            h = cell(x, h)
            steps += 1
            if y is not None:
                _, m = loss_fn(readout(readout_in(cell, h)), y)
                vals.append(m)
    if not vals:
        return None
    return {"metric_holdout": sum(vals) / len(vals),
            "metric_holdout_last200": sum(vals[-200:]) / len(vals[-200:]),
            "holdout_steps": steps,
            "holdout_frac": args.holdout_frac,
            "holdout_split_step": int(getattr(task, "t_split", -1)),
            "holdout_note": "single pass, no parameter updates, hidden state from zero"}


def run_tbptt_nnrnn(args, task, out):
    """R1 implementation of record: nn.RNN + autograd, window k, update at window end.

    Kept for the D4 fairness comparison only -- it uses a DIFFERENT parameterisation
    (nn.RNN has separate input and hidden biases) and a different initialisation from
    every other estimator.  `--algo tbptt` is the unified re-implementation below.
    """
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
        if len(window) >= args.tbptt_window or step == args.steps - 1:
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
    log["cert_summary"] = None
    json.dump(log, open(out, "w"), indent=1)
    print("saved", out)


def run_tbptt(args, task, out):
    """Truncated BPTT on the SAME TanhRNNCell / readout / optimiser as every online
    estimator (D4 fairness).

    Identical parameterisation (W, U, b -- one bias), identical initialisation (same
    seeded construction order via OnlineLearner), identical readout, optimiser and lr,
    identical washout / episode-reset handling.  The ONLY difference to the online
    estimators is the gradient estimate: autograd truncated to a window of
    `--tbptt_k` steps, updating at window end.

    With the exact shadow on, the window's autograd parameter gradient is compared
    against the exact full-history online gradient sum_t delta_t^T J_t accumulated over
    the same window, giving a `grad_cos` on the same footing as the other estimators.
    """
    learner = OnlineLearner(task, args.n, "snap1", lr=args.lr, device=args.device,
                            seed=args.seed, spectral_clip=args.clip,
                            svd_driver=args.svd_driver, cell=args.cell)
    cell, readout, opt = learner.cell, learner.readout, learner.opt
    shadow = (ExactRTRL(cell, task.B)
              if (args.shadow and shadow_ok(cell, args.shadow_max_n, task.n_in))
              else None)

    h = cell.init_state(task.B)
    losses, metrics, coss = [], [], []
    win_len = 0          # window length in STEPS (not in labelled steps): tasks such as
                         # `adding` emit one label per episode, and truncation must still
                         # happen every tbptt_k steps -- as in the nn.RNN implementation.
    g_true_win = torch.zeros(args.n, cell.p, device=args.device,
                             dtype=next(cell.parameters()).dtype)
    log, t0 = {"args": vars(args), "records": []}, time.time()
    for step in range(args.steps):
        x, y, new_ep = task.step()
        if new_ep.any():
            h = h.detach().clone()
            h[new_ep] = 0.0
            if shadow is not None:
                shadow.J[new_ep] = 0.0
        h_prev = h
        h = cell(x, h_prev)                            # graph retained inside the window
        if shadow is not None:
            A, imm = cell.jac_pieces(x, h_prev.detach(), h.detach())
            shadow.step_state(A, imm)
        if y is not None:
            loss, metric = learner.loss_fn(readout(readout_in(cell, h)), y)
            losses.append(loss)
            metrics.append(metric)
            if shadow is not None:
                h_leaf = h.detach().requires_grad_(True)
                l_inst, _ = learner.loss_fn(readout(readout_in(cell, h_leaf)), y)
                delta = torch.autograd.grad(l_inst, h_leaf)[0].detach()
                g_true_win += shadow.grad_rows(delta)
        win_len += 1
        if win_len >= args.tbptt_window or step == args.steps - 1:
            if losses:
                opt.zero_grad(set_to_none=True)
                torch.stack(losses).mean().backward()
                if shadow is not None:
                    coss.append(cos(flat_grad_rows(cell).flatten(), g_true_win.flatten()))
                opt.step()
                cell.clip_spectral()
            h = h.detach()
            losses = []
            win_len = 0
            g_true_win = torch.zeros_like(g_true_win)
        if step % args.log_every == 0 and metrics:
            rec = {"step": step,
                   "metric": sum(metrics[-200:]) / len(metrics[-200:]),
                   "grad_cos": (sum(c for c in coss[-50:] if not math.isnan(c)) /
                                max(sum(0 if math.isnan(c) else 1 for c in coss[-50:]), 1))
                   if coss else None}
            log["records"].append(rec)
            if step % (args.log_every * 8) == 0:
                print(f"[{args.task}/tbptt/s{args.seed}] step {step} "
                      f"metric {rec['metric']:.4f} cos {rec.get('grad_cos')}", flush=True)
    log["wall_s"] = time.time() - t0
    if args.device == "cuda":
        log["peak_MB"] = torch.cuda.max_memory_allocated() / 2 ** 20
    ho = holdout_pass(task, cell, readout, learner.loss_fn, args)
    if ho:
        log.update(ho)
    log["cert_summary"] = None
    log["tbptt_impl"] = "tanhrnncell_autograd"
    log["svd_stats"] = dict(SVD_STATS)
    log["svd_fallbacks"] = svd_fallbacks()
    json.dump(log, open(out, "w"), indent=1)
    print("saved", out, f"({log['wall_s']:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--algo", required=True)   # exact|snap1|skrtrl-rK|uoro|kfrtrl|rflo|tbptt|tbptt_nnrnn
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--cell", default="tanh", choices=["tanh", "gru", "lstm"],
                    help="recurrent cell; tanh is the cell of record.  gru/lstm "
                         "run the same estimators through the SK-RTRL cell "
                         "protocol (paper/revise_r2/GRU_EXTENSION.tex sec. 8); "
                         "kfrtrl is not applicable there and is refused")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--shadow", type=int, default=1)   # exact shadow for grad-cosine
    ap.add_argument("--shadow_max_n", type=int, default=256, choices=[64, 128, 256, 512],
                    help="hidden width above which the exact shadow is refused "
                         "(memory).  Expressed as a VANILLA-cell width: on a "
                         "gated cell the admissible width is scaled down so the "
                         "shadow costs no more than a vanilla cell of this width")
    ap.add_argument("--tbptt_k", type=int, default=25,
                    help="deprecated alias of --tbptt_window (kept so the R1 commands "
                         "still run); ignored when --tbptt_window is given")
    ap.add_argument("--tbptt_window", type=int, default=None,
                    help="truncation window of the TBPTT baselines, in STEPS "
                         "(default 25); tasks with a sparse target truncate every "
                         "window steps regardless of how many targets fell inside")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="results/m3")
    ap.add_argument("--tag", default="")
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--horizon", type=int, default=1)     # multi-step prediction horizon
    ap.add_argument("--causal", type=int, default=0)      # 1 = causal normalization
    ap.add_argument("--washout", type=int, default=200)   # reset period (0 = no reset)
    ap.add_argument("--holdout_frac", type=float, default=0.0,
                    help="A11 time-ordered holdout for the time-series tasks: the last "
                         "fraction of the stream is never cycled during training and is "
                         "streamed exactly once afterwards, with NO parameter updates, "
                         "reported as metric_holdout")
    ap.add_argument("--svd_driver", choices=["gesvd", "auto"], default="gesvd",
                    help="gesvd = kernel of record (all R1 numbers); auto = default "
                         "cuSOLVER driver first, same fallback chain behind it")
    # D2 certificate-informativeness counters
    ap.add_argument("--cert_warmup", type=int, default=200,
                    help="global burn-in: counters start at this step")
    ap.add_argument("--cert_min_age", type=int, default=1,
                    help="skip steps whose distance to the last episode reset is below this")
    ap.add_argument("--cert_every", type=int, default=1,
                    help="stride of the online counters (1 = every post-washout step)")
    ap.add_argument("--cert_stages", type=int, default=3,
                    help="number of equal training stages the counters are bucketed "
                         "into (0 = run average only)")
    ap.add_argument("--cert_age_span", type=int, default=0,
                    help="reset period T used for the age buckets [0,T/4) [T/4,T/2) "
                         "[T/2,inf); 0 = infer (--washout on time-series tasks, the "
                         "episode length otherwise)")
    ap.add_argument("--sketch_ckpts", default="0.01,0.05,0.25,0.5,0.75,1.0",
                    help="fractions of --steps at which the sketch-efficiency "
                         "checkpoint ||E_t||/best-rank-r error is measured "
                         "(SK-RTRL with shadow only)")
    ap.add_argument("--sketch_defer", type=int, default=50,
                    help="how far a sketch checkpoint may slip to get past a "
                         "synchronised episode reset (where J_t - S_t is exactly 0)")
    args = ap.parse_args()
    # A10: one effective window, recorded in the json so the protocol is unambiguous
    args.tbptt_window = args.tbptt_k if args.tbptt_window is None else args.tbptt_window
    args.tbptt_k = args.tbptt_window

    if args.cell != "tanh":
        if args.algo in ("kfrtrl", "am-kfrtrl"):
            raise SystemExit(
                "KF-RTRL is NOT APPLICABLE to a gated cell: its Kronecker "
                "factorisation J ~ u (x) B rests on A_t = D_t W and a rank-1 "
                "immediate Jacobian, and a GRU's immediate Jacobian is not even "
                "block diagonal (paper/revise_r2/GRU_EXTENSION.tex sec. 8).  "
                "Report it as not-applicable for --cell gru/lstm rather than "
                "running a variant whose bias is undocumented.")
        if args.algo in ("tbptt_nnrnn", "tbptt-nnrnn"):
            raise SystemExit("--algo tbptt_nnrnn is tanh-only by construction "
                             "(it builds an nn.RNN); use --algo tbptt, which "
                             "differentiates the selected cell itself.")

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    # the cell goes into the name so gated runs cannot overwrite the tanh
    # results; `tanh` keeps the historical name, so every existing job list and
    # every file already in results/m3 stays valid.
    cellpart = "" if args.cell == "tanh" else f"_{args.cell}"
    out = f"{args.outdir}/{args.task}{cellpart}_{args.algo}_s{args.seed}{tag}.json"
    if os.path.exists(out):
        print("exists, skip:", out)
        return
    _svd_stats_reset()
    if args.task in TS_TASKS:
        task = TASKS[args.task](args.batch, args.device, seed=args.seed,
                                horizon=args.horizon, causal=bool(args.causal),
                                washout=args.washout, holdout_frac=args.holdout_frac)
    elif args.holdout_frac > 0:
        raise SystemExit("--holdout_frac applies to the time-series tasks only "
                         f"({sorted(TS_TASKS)}); {args.task} is episodic")
    else:
        task = TASKS[args.task](args.batch, args.device, seed=args.seed)

    if args.algo == "tbptt":
        run_tbptt(args, task, out)
        return
    if args.algo in ("tbptt_nnrnn", "tbptt-nnrnn"):
        run_tbptt_nnrnn(args, task, out)
        return

    learner = OnlineLearner(task, args.n, args.algo, lr=args.lr, device=args.device, seed=args.seed,
                            spectral_clip=args.clip, svd_driver=args.svd_driver,
                            cell=args.cell)
    shadow = ExactRTRL(learner.cell, task.B) if (
        args.shadow and args.algo != "exact"
        and shadow_ok(learner.cell, args.shadow_max_n, task.n_in)) else None
    if args.shadow and shadow is None and args.algo != "exact":
        print(f"[{args.task}/{args.algo}] exact shadow refused: cell={args.cell} "
              f"n={args.n} exceeds the --shadow_max_n={args.shadow_max_n} memory "
              f"budget (p_cell={learner.cell.p}, n_state="
              f"{getattr(learner.cell, 'n_state', args.n)})", flush=True)
    has_cert = hasattr(learner.algo, "e")
    has_dense = hasattr(learner.algo, "residual_dense")
    age_span = args.cert_age_span or (args.washout if args.task in TS_TASKS
                                      else int(getattr(task, "T", 0) or 0))
    args.cert_age_span_used = int(age_span)
    n_stages = max(int(args.cert_stages), 0)
    cert = CertCounters(shadow is not None, n_stages=n_stages,
                        age_span=age_span) if has_cert else None
    if cert is not None and n_stages:
        edges = [round(i * args.steps / n_stages) for i in range(n_stages + 1)]
        for i, c in enumerate(cert.stages):
            c.span = (edges[i], edges[i + 1])

    # A8 sketch-efficiency checkpoints: nominal step -> (fraction, deadline)
    ckpts = {}
    if shadow is not None and has_dense and isinstance(learner.algo, SKRTRL):
        for tok in str(args.sketch_ckpts).split(","):
            tok = tok.strip()
            if not tok:
                continue
            f = float(tok)
            nom = min(max(int(round(f * args.steps)), 1), args.steps - 1)
            ckpts[nom] = (f, min(nom + args.sketch_defer, args.steps - 1))
        log_ckpt = []

    log = {"args": vars(args), "records": []}
    metrics, coss = [], []
    age = 0                     # steps since the last episode reset
    last = {}                   # most recent per-step diagnostics, for the log records
    t0 = time.time()
    for step in range(args.steps):
        # --- replicate learner.step but with shadow hooks ---
        x, y, new_ep = task.step()
        learner._reset_lanes(new_ep)
        if new_ep.any():
            age = 0
            if shadow is not None:
                shadow.J[new_ep] = 0.0
        else:
            age += 1
        h_prev = learner.h.detach()
        h = learner.cell(x, h_prev)
        A, imm = learner.cell.jac_pieces(x, h_prev, h)
        learner.algo.step_state(A, imm)
        if shadow is not None:
            shadow.step_state(A, imm)
        learner.h = h.detach()
        if ckpts:
            for nom, (frac, dead) in list(ckpts.items()):
                if step < nom:
                    continue
                if age >= max(args.cert_min_age, 1) or step >= dead:
                    del ckpts[nom]
                    rec_ck = sketch_efficiency(shadow, learner.algo)
                    rec_ck.update({"step": step, "step_nominal": nom,
                                   "frac": frac, "age": age})
                    log_ckpt.append(rec_ck)
                    print(f"[{args.task}/{args.algo}/s{args.seed}] sketch@{step} "
                          f"r{rec_ck['r_factor_width']} "
                          f"eff {rec_ck['sketch_eff']} "
                          f"resfrac {rec_ck['res_norm']:.3g}", flush=True)
        if y is not None:
            h_leaf = learner.h.requires_grad_(True)
            outp = learner.readout(learner.readout_in(h_leaf))
            loss, metric = learner.loss_fn(outp, y)
            learner.opt.zero_grad(set_to_none=True)
            loss.backward()
            delta = h_leaf.grad.detach()
            g_rows = learner.algo.grad_rows(delta)
            g_true = shadow.grad_rows(delta) if shadow is not None else None
            if g_true is not None:
                coss.append(cos(g_rows.flatten(), g_true.flatten()))

            # ---- D2 instrumentation (every step) ----
            do_cert = (cert is not None and step >= args.cert_warmup
                       and age >= args.cert_min_age and step % args.cert_every == 0)
            need = do_cert or (step % args.log_every == 0)
            if need:
                with torch.no_grad():
                    dn_lane = delta.norm(dim=1)                        # (B,)
                    last["delta_norm"] = float(dn_lane.mean())
                    last["ghat_norm"] = float(g_rows.norm())
                    e_lane = learner.algo.e if has_cert else None
                    # Theorem-1 bound, batch-consistent: mean_b ||delta_b|| * e_b
                    bound = float((dn_lane * e_lane).mean()) if has_cert else None
                    if bound is not None:
                        last["bound_norm"] = bound
                    last["step"] = step
                    gerr = None
                    trueE = None
                    if g_true is not None:
                        last["g_norm"] = float(g_true.norm())
                        gerr = float((g_true - g_rows).norm())
                        last["gerr_norm"] = gerr
                        if has_dense:
                            E = shadow.J - learner.algo.residual_dense()
                            trueE = float(E.flatten(1).norm(dim=1).mean())
                    if do_cert:
                        la0 = getattr(learner.algo, "last", None) or {}
                        rb = float(la0["rho_bar"].mean()) if "rho_bar" in la0 else None
                        stage = (min(n_stages - 1, int(n_stages * step / max(args.steps, 1)))
                                 if n_stages else None)
                        cert.update(bound, last["ghat_norm"], gerr,
                                    float(e_lane.mean()), trueE,
                                    rho_bar=rb, delta_norm=last["delta_norm"],
                                    stage=stage, age=age)
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
                if shadow is not None and has_dense:
                    E = shadow.J - learner.algo.residual_dense()
                    rec["true_E"] = E.flatten(1).norm(dim=1).mean().item()
            if shadow is not None:
                rec["J_norm"] = shadow.J.flatten(1).norm(dim=1).mean().item()
            # delta-dependent quantities exist only at LABELLED steps; on tasks with a
            # sparse target (adding, copy) the log step need not be one, so they are
            # carried from the most recent labelled step and `diag_step` says which.
            for k in ("delta_norm", "ghat_norm", "bound_norm", "g_norm", "gerr_norm"):
                if k in last:
                    rec[k] = last[k]
            if "step" in last:
                rec["diag_step"] = last["step"]
            log["records"].append(rec)
            if step % (args.log_every * 8) == 0:
                print(f"[{args.task}/{args.algo}/s{args.seed}] step {step} "
                      f"metric {rec['metric']:.4f} cos {rec.get('grad_cos')}", flush=True)
    log["wall_s"] = time.time() - t0
    if args.device == "cuda":
        log["peak_MB"] = torch.cuda.max_memory_allocated() / 2**20
    ho = holdout_pass(task, learner.cell, learner.readout, learner.loss_fn, args)
    if ho:
        log.update(ho)
        print(f"[{args.task}/{args.algo}/s{args.seed}] holdout "
              f"{ho['holdout_steps']} steps  metric {ho['metric_holdout']:.4f}", flush=True)
    log["cert_summary"] = cert.summary() if cert is not None else None
    if shadow is not None and has_dense and isinstance(learner.algo, SKRTRL):
        log["checkpoints"] = log_ckpt
    log["svd_stats"] = dict(SVD_STATS)
    log["svd_fallbacks"] = svd_fallbacks()
    json.dump(log, open(out, "w"), indent=1)
    print("saved", out, f"({log['wall_s']:.0f}s, {log.get('peak_MB',0):.0f}MB)")


if __name__ == "__main__":
    main()
