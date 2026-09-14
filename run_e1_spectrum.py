"""E1 / D1 (R2, Reviewer #3 point 1): spectrum of the TRUE SnAp-1 residual R_t = J_t - S_t.

Definition of record (R2_EXPERIMENT_PLAN.md, D1):
    R_t := J_t - S_t
where J_t = dh_t/dtheta is the exact influence matrix (ExactRTRL, "shadow") and S_t is the
SnAp-1 trace the paper actually maintains, S_{t,i} = A_{ii} S_{t-1,i} + imm_i (SnAp1.S).
S_t lives in the block-diagonal support, so R_t = J_t - blockdiag_embed(S_t) differs from the
R1 quantity J_t - blockdiag(J_t): the latter zeroes the diagonal blocks, the former subtracts
the *recursively propagated* trace from them. The R1 quantity is still logged, under the
`bd_` prefix, so the new figure remains comparable with R1's Fig. 2 (fig_pilot_residual).

Training protocol (identical to the R1 pilot, run_m1_spectrum.py): the network is trained
online with **exact RTRL**, so the residual measured is a property of the network/task pair
and not of the approximation being tested.  A SnAp-1 instance is attached as a passive
tracker: it receives exactly the same (A_t, imm_t) and the same lane resets, and never
contributes to a parameter update.

Metrics at each checkpoint, per batch lane, then mean/std over lanes:
    sv                 first 64 singular values of R_t
    mass_top{1,4,8,16,32,64}, mass_topn4   sum_{i<=r} s_i^2 / sum_i s_i^2  (r = n//4 for topn4)
    r_eps{10,05}       smallest r with ||R - R_r||_F / ||R||_F <= eps, eps = 0.10 / 0.05
    r_eps{10,05}_over_n   the same as a fraction of n (does the required rank scale with width?)
    stable_rank        ||R||_F^2 / ||R||_2^2
    res_frac_of_J      ||R_t||_F / ||J_t||_F
    diag_gap_*         ||blockdiag(J) - S||_F, the entire difference between the two conventions
    bd_*               the same quantities for J_t - blockdiag(J_t)   (R1 comparability)

Checkpoints are stratified two ways (--ckpt_align age, the default): training stage
(--ckpt_fracs) x age of the influence matrix within its reset interval
({T/8, T/4, T/2, 3T/4, T-1} of T_reset), so that accumulation and learning are not
confounded.  See make_checkpoints().

Output: <outdir>/<task>_n<n>_s<seed>[_<tag>].json   (exists-skip)
"""
import argparse, json, os, time
import torch

from skrtrl.tasks import TASKS
from skrtrl.train import OnlineLearner
from skrtrl.algos import SnAp1

TS_TASKS = {"henon", "mackeyglass", "lorenz", "sunspot", "laser"}
MASS_RANKS = (1, 4, 8, 16, 32, 64)
EPS_LEVELS = ((0.10, "10"), (0.05, "05"))     # r_eps: relative Frobenius truncation error
AGE_FRACS = (0.125, 0.25, 0.5, 0.75, None)    # None -> T_reset - 1 (oldest usable age)


# --------------------------------------------------------------------------------------
# spectra
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _svals(M, method):
    """Singular values of a single (n, P) matrix, descending, returned in float64.

    method="svd"  : torch.linalg.svdvals  (exact, fine up to n~128 where P = n*(n+m+1))
    method="gram" : eigenvalues of M M^T in fp64 (cheap at n=512, where P ~ 2.7e5;
                    squares the condition number, so only the leading sigma are trusted --
                    which is all the top-r mass / stable-rank metrics use).
    """
    if method == "svd":
        return torch.linalg.svdvals(M).double()
    G = (M @ M.T).double()
    G = 0.5 * (G + G.T)
    ev = torch.linalg.eigvalsh(G)
    return ev.clamp_min(0).flip(0).sqrt()


@torch.no_grad()
def _metrics_from_sv(sv, n, keep=64):
    """sv: descending fp64 singular values of one lane. Returns dict of scalars + sv list."""
    sq = sv * sv
    tot = sq.sum().clamp_min(1e-300)
    out = {}
    for r in MASS_RANKS:
        if r <= sv.numel():
            out[f"mass_top{r}"] = (sq[:r].sum() / tot).item()
    rn4 = max(1, n // 4)
    out["mass_topn4"] = (sq[:rn4].sum() / tot).item()
    s1sq = sq[0].clamp_min(1e-300)
    out["stable_rank"] = (tot / s1sq).item()
    out["fro"] = tot.sqrt().item()
    out["sigma_max"] = sv[0].item()
    # r_eps: smallest r with ||R - R_r||_F / ||R||_F <= eps, i.e. cumulative mass >= 1-eps^2
    cum = torch.cumsum(sq, 0) / tot
    for eps, key in EPS_LEVELS:
        thr = 1.0 - eps * eps
        hit = (cum >= thr).nonzero()
        r_eps = int(hit[0].item()) + 1 if hit.numel() else int(sv.numel())
        out[f"r_eps{key}"] = float(r_eps)
        out[f"r_eps{key}_over_n"] = r_eps / float(n)
    return out, sv[:keep].cpu().tolist()


@torch.no_grad()
def checkpoint_spectrum(J, S, n, p, method, fp64_check=False):
    """J (B, n, P) exact influence matrix, S (B, n, p) SnAp-1 trace.

    Returns (record_dict, sanity_dict).  Works one lane at a time so that only a single
    (n, P) residual copy is materialised (2.7e5 * 512 * 4 B = 0.55 GB at n=512).
    """
    B = J.shape[0]
    P = J.shape[2]
    idx = torch.arange(n, device=J.device)
    per_lane, per_lane_bd, svs, svs_bd = [], [], [], []
    j_fro = []
    sanity = {}
    buf = torch.empty(n, P, device=J.device, dtype=J.dtype)
    for b in range(B):
        Jb = J[b]
        j_fro.append(Jb.norm().double().item())
        # ---- R = J - blockdiag_embed(S) ----
        buf.copy_(Jb)
        bv = buf.view(n, n, p)
        bv[idx, idx, :] -= S[b]
        sv = _svals(buf, method)
        m, svlist = _metrics_from_sv(sv, n)
        # ||blockdiag(J) - S||_F: the whole difference between the reviewer's object
        # (J - S) and the R1 object (J - blockdiag(J)) sits in the diagonal blocks
        m["diag_gap"] = bv[idx, idx, :].norm().double().item()
        per_lane.append(m)
        svs.append(svlist)
        if b == 0:
            # Sum sigma^2 == ||R||_F^2 (the spectral decomposition must conserve mass).
            fro_direct = buf.norm().double().item()
            sanity["fro_reldiff"] = abs(m["fro"] - fro_direct) / max(fro_direct, 1e-30)
            sanity["fro_direct"] = fro_direct
            if fp64_check:
                sv_svd32 = _svals(buf, "svd")
                sv_gram32 = _svals(buf, "gram")
                sv_svd64 = torch.linalg.svdvals(buf.double())
                k = min(64, sv_svd64.numel())

                def _rel(a, b_):
                    a, b_ = a[:k], b_[:k]
                    return (a - b_).abs().div(b_.abs().clamp_min(1e-30)).max().item()

                m64, _ = _metrics_from_sv(sv_svd64, n)
                sanity["fp64"] = {
                    "max_rel_sv_fp32svd_vs_fp64svd": _rel(sv_svd32, sv_svd64),
                    "max_rel_sv_fp32gram_vs_fp64svd": _rel(sv_gram32, sv_svd64),
                    "mass_top16_fp32": per_lane[0].get("mass_top16"),
                    "mass_top16_fp64": m64.get("mass_top16"),
                    "stable_rank_fp32": per_lane[0]["stable_rank"],
                    "stable_rank_fp64": m64["stable_rank"],
                    "n_sv_compared": k,
                }
        # ---- bd residual = J - blockdiag(J)  (R1 quantity; reuse the same buffer) ----
        bv[idx, idx, :] = 0.0
        sv_bd = _svals(buf, method)
        m_bd, svlist_bd = _metrics_from_sv(sv_bd, n)
        per_lane_bd.append(m_bd)
        svs_bd.append(svlist_bd)
    del buf

    def agg(dicts, prefix=""):
        rec = {}
        keys = dicts[0].keys()
        for k in keys:
            v = torch.tensor([d[k] for d in dicts], dtype=torch.float64)
            rec[prefix + k] = v.mean().item()
            rec[prefix + k + "_std"] = v.std(unbiased=False).item()
        return rec

    jf = torch.tensor(j_fro, dtype=torch.float64)
    rec = {"j_fro": jf.mean().item(), "j_fro_std": jf.std(unbiased=False).item()}
    rec.update(agg(per_lane))
    rec.update(agg(per_lane_bd, prefix="bd_"))
    for pre in ("", "bd_"):
        src = per_lane if pre == "" else per_lane_bd
        fr = torch.tensor([d["fro"] / max(f, 1e-30) for d, f in zip(src, j_fro)], dtype=torch.float64)
        rec[pre + "res_frac_of_J"] = fr.mean().item()
        rec[pre + "res_frac_of_J_std"] = fr.std(unbiased=False).item()
    dg = torch.tensor([d["diag_gap"] / max(f, 1e-30) for d, f in zip(per_lane, j_fro)],
                      dtype=torch.float64)
    rec["diag_gap_frac_of_J"] = dg.mean().item()
    rec["diag_gap_frac_of_J_std"] = dg.std(unbiased=False).item()
    dgr = torch.tensor([d["diag_gap"] / max(d["fro"], 1e-30) for d in per_lane], dtype=torch.float64)
    rec["diag_gap_over_res"] = dgr.mean().item()
    svt = torch.tensor(svs, dtype=torch.float64)
    rec["sv"] = svt.mean(0).tolist()
    rec["sv_std"] = svt.std(0, unbiased=False).tolist()
    svbt = torch.tensor(svs_bd, dtype=torch.float64)
    rec["bd_sv"] = svbt.mean(0).tolist()
    rec["sv_method"] = method
    return rec, sanity


# --------------------------------------------------------------------------------------
# checkpoint schedule
# --------------------------------------------------------------------------------------
def age_targets(period):
    """Ages within one reset interval at which the residual is sampled."""
    if not period or period < 2:
        return []
    raw = [period - 1 if a is None else a * period for a in AGE_FRACS]
    return sorted({min(period, max(1, int(round(a)))) for a in raw})


def make_checkpoints(steps, fracs, period, align, log_every=500, stride=0):
    """Return sorted (step, frac, target_age) triples; step is a 1-based step count.

    `period` = T_reset, the reset period of the stream (washout for the time-series tasks,
    the episode length T for the episodic ones, 0 when lanes reset at lane-specific times).
    Because lanes reset at step indices that are multiples of T_reset, the age of J_t after
    completing step count `d` is ((d - 1) mod T_reset) + 1, i.e. d == age (mod T_reset).

    align="age" (default): two-way stratification, training stage x age-within-interval.
      For every stage in --ckpt_fracs and every age in {T/8, T/4, T/2, 3T/4, T-1} the
      nearest step with exactly that age is used.  This separates "the residual grows as
      the influence matrix accumulates" from "the residual grows as the network trains",
      which a single-step-per-stage grid confounds.  For a stream with no fixed period
      (anbn, or washout=0) the ages cannot be dialled, so five consecutive log points
      around the stage are taken instead and the realised (lane-dependent) age logged.
    align="mid" / "none": the one-checkpoint-per-stage controls.  "none" is the naive
      steps*frac grid, which for every time-series task lands exactly on reset boundaries
      (washout=200, steps=20000 -> 200/1000/5000/...), where J_t has just been zeroed;
      "mid" moves it to the middle of the interval.
    """
    used, ck = set(), []

    def add(d, f, a):
        d = int(d)
        if d < 1 or d > steps or d in used:
            return
        used.add(d)
        ck.append((d, f, a))

    if align == "age":
        ages = age_targets(period)
        if ages:
            for f in sorted(fracs):
                s = max(1, min(steps, int(round(steps * f))))
                for a in ages:
                    d = round((s - a) / period) * period + a
                    while d < 1:
                        d += period
                    while d > steps:
                        d -= period
                    add(d, f, a)
        else:
            st0 = max(1, stride or log_every or 1)
            for f in sorted(fracs):
                s = max(1, min(steps, int(round(steps * f))))
                st = max(1, min(st0, max(1, s // 4)))       # early stages get a tighter window
                start = min(max(1, s - 2 * st), max(1, steps - 4 * st))
                for i in range(5):
                    add(min(steps, start + i * st), f, 0)
        return sorted(ck)

    for f in sorted(fracs):
        s = max(1, min(steps, int(round(steps * f))))
        if align == "mid" and period and period > 1:
            s2 = (s // period) * period + period // 2
            while s2 > steps:
                s2 -= period
            if s2 < 1:
                s2 = min(steps, max(1, period // 2))
            # two fracs can land in the same reset interval on very short runs: push the
            # later one forward by whole periods so no checkpoint is silently dropped
            while s2 in used and s2 + period <= steps:
                s2 += period
            s = s2
        add(s, f, None)
    return sorted(ck)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt_fracs", default="0.01,0.05,0.25,0.5,0.75,1.0")
    ap.add_argument("--ckpt_align", default="age", choices=["age", "mid", "none"],
                    help="age = stage x age-within-reset-interval stratification (default); "
                         "mid/none = one checkpoint per stage (controls)")
    ap.add_argument("--age_stride", type=int, default=0,
                    help="spacing of the 5 probes per stage when the stream has no fixed "
                         "reset period (0 = use --log_every)")
    ap.add_argument("--outdir", default="results/r2/d1_spectrum")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--sv_method", default="auto", choices=["auto", "svd", "gram"])
    ap.add_argument("--fp64_check", type=int, default=-1,
                    help="-1 = auto (on when n<=128), 0/1 to force")
    # time-series protocol, passed through exactly like run_m3.py
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--washout", type=int, default=200)
    # accepted for queue uniformity; this script never runs the SK-RTRL kernel
    ap.add_argument("--svd_driver", default="")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = f"{args.outdir}/{args.task}_n{args.n}_s{args.seed}{tag}.json"
    if os.path.exists(out):
        print("exists, skip:", out)
        return

    # --svd_driver exists only for queue uniformity: this script runs exact RTRL plus a
    # SnAp-1 tracker (r = 0), and neither ever calls the SK-RTRL SVD.  Pass it through if
    # the installed skrtrl supports it, otherwise say so and carry on.
    drv_kw = {}
    if args.svd_driver:
        import inspect
        from skrtrl.train import OnlineLearner as _OL
        supported = ("svd_driver" in inspect.signature(_OL.__init__).parameters
                     and "svd_driver" in inspect.signature(SnAp1.__init__).parameters)
        if supported:
            drv_kw = {"svd_driver": args.svd_driver}
            print(f"svd_driver -> {args.svd_driver} (unused: no SK-RTRL kernel in this script)")
        else:
            print(f"note: installed skrtrl has no svd_driver argument; --svd_driver="
                  f"{args.svd_driver} ignored (exact RTRL + SnAp-1 only)")

    if args.task in TS_TASKS:
        task = TASKS[args.task](args.batch, args.device, seed=args.seed,
                                horizon=args.horizon, causal=bool(args.causal),
                                washout=args.washout)
        period = args.washout
    else:
        task = TASKS[args.task](args.batch, args.device, seed=args.seed)
        period = int(getattr(task, "T", 0) or 0)

    fracs = [float(x) for x in args.ckpt_fracs.split(",") if x.strip()]
    ckpts = make_checkpoints(args.steps, fracs, period, args.ckpt_align,
                             log_every=args.log_every, stride=args.age_stride)
    ck_map = {s: (f, a) for s, f, a in ckpts}
    method = args.sv_method
    if method == "auto":
        method = "svd" if args.n <= 128 else "gram"
    fp64_check = (args.n <= 128) if args.fp64_check < 0 else bool(args.fp64_check)

    learner = OnlineLearner(task, args.n, "exact", lr=args.lr, device=args.device,
                            seed=args.seed, **drv_kw)
    # passive tracker: identical (A_t, imm_t) and identical lane resets, never updates params
    tracker = SnAp1(learner.cell, task.B, **drv_kw)
    n, p = learner.cell.n, learner.cell.p

    log = {"args": vars(args), "meta": {"P": n * p, "p": p, "reset_period": period,
                                        "sv_method": method,
                                        "checkpoints": [s for s, _, _ in ckpts],
                                        "ckpt_fracs_realised": [f for _, f, _ in ckpts],
                                        "ckpt_target_ages": [a for _, _, a in ckpts],
                                        "age_targets": age_targets(period),
                                        "eps_levels": [e for e, _ in EPS_LEVELS],
                                        "train_algo": "exact", "residual": "J_t - S_t(snap1)"},
           "records": [], "curve": []}
    age = torch.zeros(task.B, dtype=torch.long, device=args.device)
    metrics = []
    sanity_done = False
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for step in range(args.steps):
        x, y, new_ep = task.step()
        learner._reset_lanes(new_ep)               # resets learner.h and the exact J lanes
        if new_ep.any():
            tracker.S[new_ep] = 0.0
            tracker.e[new_ep] = 0.0
            age[new_ep] = 0
        h_prev = learner.h.detach()
        h = learner.cell(x, h_prev)
        A, imm = learner.cell.jac_pieces(x, h_prev, h)
        learner.algo.step_state(A, imm)
        tracker.step_state(A, imm)                 # identical (A, imm) -> identical S_t
        age += 1
        learner.h = h.detach()
        if y is not None:
            h_leaf = learner.h.requires_grad_(True)
            outp = learner.readout(h_leaf)
            loss, metric = learner.loss_fn(outp, y)
            learner.opt.zero_grad(set_to_none=True)
            loss.backward()
            delta = h_leaf.grad.detach()
            learner.cell.apply_flat_grad(learner.algo.grad_rows(delta))
            learner.opt.step()
            learner.cell.clip_spectral()
            learner.h = learner.h.detach()
            metrics.append(metric)
        done = step + 1
        if done in ck_map:
            rec, sanity = checkpoint_spectrum(learner.algo.J, tracker.S, n, p, method,
                                              fp64_check=(fp64_check and not sanity_done))
            rec = {"step": done, "frac": ck_map[done][0], "target_age": ck_map[done][1],
                   "age_mean": age.float().mean().item(),
                   "age_min": int(age.min().item()), "age_max": int(age.max().item()),
                   "metric": (sum(metrics[-200:]) / len(metrics[-200:])) if metrics else None,
                   **rec}
            if sanity:
                if "first_ckpt" not in log.get("sanity", {}):
                    log.setdefault("sanity", {})["first_ckpt"] = {"step": done, **sanity}
                if "fp64" in sanity:
                    sanity_done = True
                rec["fro_reldiff"] = sanity["fro_reldiff"]
            log["records"].append(rec)
            print(f"[{args.task}/n{args.n}/s{args.seed}] ck step {done} "
                  f"(frac {rec['frac']}, age {rec['age_mean']:.0f}"
                  f"{'' if rec['target_age'] in (None, 0) else '/tgt %d' % rec['target_age']}) "
                  f"resfrac {rec['res_frac_of_J']:.4f} mass16 {rec.get('mass_top16'):.4f} "
                  f"sr {rec['stable_rank']:.2f} r10 {rec['r_eps10']:.1f} "
                  f"r05 {rec['r_eps05']:.1f} | bd mass16 {rec.get('bd_mass_top16'):.4f} "
                  f"bd r10 {rec['bd_r_eps10']:.1f}", flush=True)
        if step % args.log_every == 0 and metrics:
            log["curve"].append({"step": done, "metric": sum(metrics[-200:]) / len(metrics[-200:])})
    log["wall_s"] = time.time() - t0
    if args.device == "cuda":
        log["peak_MB"] = torch.cuda.max_memory_allocated() / 2 ** 20
    with open(out, "w") as f:
        json.dump(log, f, indent=1)
    print("saved", out, f"({log['wall_s']:.0f}s, {log.get('peak_MB', 0):.0f}MB, "
          f"{len(log['records'])} checkpoints)")


if __name__ == "__main__":
    main()
