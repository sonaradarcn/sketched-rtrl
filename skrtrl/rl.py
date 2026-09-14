"""Online actor-critic for POMDPs: per-step TD(0) updates, NO replay buffer.

Shared recurrent core h_t -> softmax policy head + linear value head.
Per env step:
  td      = r + gamma * (1 - done) * V(h_{t+1}).detach() - V(h_t)
  loss    = -td.detach() * logpi(a) + 0.5 * td^2 - beta * entropy
  delta_h = dLoss/dh_t via autograd on the head graph (h_t is a leaf);
  recurrent params get grads from the core's online estimator (grad_rows-style),
  heads get direct autograd grads; one Adam step per env step (or every
  accumulate_k steps, averaging the accumulated grads).

Cores implement the protocol documented in diag_cells.py (begin / features /
advance / backward_grads / commit). TanhCore adapts TanhRNNCell + OnlineGrad.

R2 shadow instrumentation -- TanhCore(shadow=True), plan D1/D2
--------------------------------------------------------------
Two extra estimators ride along on the SAME cell, are fed the SAME (A_t, imm_t)
pair produced by advance(), and are lane-reset exactly like the main estimator,
so they stay time-aligned with it step for step (same skeleton as run_m3.py):
  * ExactRTRL  -> the true influence matrix J_t
  * SnAp1      -> the SnAp-1 trace S_t the paper actually maintains, used as the
                  reference for the D1 residual  R_t = J_t - S_t  (NOT blkdiag(J_t))
Neither ever writes to .grad, so the learning trajectory is unchanged; the
SnAp-1 tracker's power-iteration probe is seeded deterministically so that
turning the shadow on does not consume any global RNG either -- a --shadow run
and a no-shadow run of the same seed follow the same trajectory.

Batch aggregation of the D2 quantities (RL-specific, stated once here)
----------------------------------------------------------------------
delta_t = dLoss/dh_t is (B, n) and the gradient actually applied to the recurrent
parameters is the batch mean returned by grad_rows(), so with
  ghat_t = algo.grad_rows(delta_t),  g_t = shadow.grad_rows(delta_t)   (both (n, p))
the batch-consistent form of the Theorem-1 bound is
  ||g_t - ghat_t||_F <= (1/B) sum_b ||delta_b|| ||E_b||_F
                     <= (1/B) sum_b ||delta_b|| e_b  =:  bound_t,
i.e. the lane-wise product is averaged (NOT mean||delta|| times mean e).  This is
the same convention as run_m3.CertCounters; `delta_norm` is still reported as
mean_b ||delta_b|| for readability, e_t as mean_b e_b and ||E_t||_F as
mean_b ||J_b - residual_dense_b||_F, and the criteria are
  (i)  rho^g_t = bound_t / ||ghat_t||_F  < 1   (also < 0.5)   -- no shadow needed
  (ii) T_t     = e_t / ||E_t||_F        <= 10                 -- shadow needed
plus the Theorem-1 check ||g-ghat|| <= bound_t (1+1e-4) and the Lemma-1 validity
check ||E_t||_F <= e_t (1+1e-4).

Unlike the supervised tasks there is no washout in RL: lanes auto-reset
asynchronously, so a global "steps since reset" gate would discard most steps at
B=16.  Counters therefore run at every gradient step after --cert_warmup, with an
optional per-lane age gate (--cert_min_age, default 0 = off); a freshly reset lane
has e_b = 0 and ||E_b||_F = 0 and so contributes 0 to both sides of the bound.
"""
import math
from collections import deque

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .algos import ExactRTRL, SnAp1
from .cells import TanhRNNCell
from .train import make_algo

# Instantaneous per-step diagnostics written into every log record (same key
# names as run_m3.py so one analysis script reads both suites).
CERT_KEYS = ("delta_norm", "ghat_norm", "bound_norm", "g_norm", "gerr_norm",
             "J_norm", "true_E", "e_t", "rho_bar", "rho_hat", "eta",
             "rho_g", "tight")


def cos(a, b):
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return ((a * b).sum() / (na * nb)).item()


@torch.no_grad()
def _reset_estimator_lanes(algo, mask):
    """Zero per-lane estimator state (mirrors train.OnlineLearner._reset_lanes)."""
    if hasattr(algo, "J"):
        algo.J[mask] = 0.0
    for attr in ("S", "L", "R"):
        t = getattr(algo, attr, None)
        if t is not None and torch.is_tensor(t) and t.numel():
            t[mask] = 0.0
    if hasattr(algo, "e"):
        algo.e[mask] = 0.0
    if hasattr(algo, "reset_lanes"):
        algo.reset_lanes(mask)


class CertCounters:
    """Online (every-step) certificate-informativeness counters -- plan D2.

    Definitions and output keys follow run_m3.CertCounters (package A); see the
    module docstring for the batch aggregation.  Additions:

    n_no_signal      steps with no usable gradient signal (||delta_t|| = 0 or
                     ||ghat_t|| = 0 -- the m3 analogue is a y = None step).  They
                     are counted separately and are NOT in the denominator of any
                     fraction, so `n_steps` counts signal-bearing steps only and
                     e.g. frac_rel_lt1 = n_rel_lt1 / n_steps stays unpolluted.
                     With T-maze's dense reward this is expected to stay 0.
    n_rhobar_lt1     steps whose certificate contraction factor rho_bar_t < 1,
                     i.e. the steps on which e_t is not being inflated -- the
                     direct explanation of an uninformative certificate.
    stages           the same counters bucketed by training stage (thirds of the
                     run by default), so early / intermediate / late are reported
                     separately and not only as a run average.
    n_bound_violation_fp
                     the Theorem-1 test with an absolute fp32 floor, which
                     separates a real violation from the rounding noise of the
                     first step of an episode, where e_t = 0 exactly while
                     ||g - ghat|| is zero only up to fp32.
    """

    ATOL = 1e-6            # absolute fp32 floor of the *_fp violation test

    def __init__(self, has_shadow, n_stages=0, span=None):
        self.has_shadow = has_shadow
        self.span = span
        self.n_steps = 0
        self.n_no_signal = 0
        self.n_rhobar_lt1 = 0
        self.n_rel_lt1 = 0
        self.n_rel_lt05 = 0
        self.n_tight_checked = 0
        self.n_tight_le10 = 0
        self.n_bound_violation = 0
        self.n_bound_violation_fp = 0
        self.n_cert_invalid = 0
        self.sum_rel = 0.0
        self.sum_tight = 0.0
        self.max_rel = 0.0
        self.max_tight = 0.0
        self.stages = [CertCounters(has_shadow) for _ in range(n_stages)]

    def update(self, bound, ghat_norm, gerr_norm, e_mean, trueE_mean,
               rho_bar=None, delta_norm=None, stage=None):
        if stage is not None and self.stages:
            self.stages[min(stage, len(self.stages) - 1)].update(
                bound, ghat_norm, gerr_norm, e_mean, trueE_mean,
                rho_bar=rho_bar, delta_norm=delta_norm)
        if ghat_norm <= 0 or (delta_norm is not None and delta_norm <= 0):
            self.n_no_signal += 1          # deliberately not in any denominator
            return
        self.n_steps += 1
        if rho_bar is not None and rho_bar < 1.0:
            self.n_rhobar_lt1 += 1
        rel = bound / ghat_norm
        self.sum_rel += rel
        self.max_rel = max(self.max_rel, rel)
        if rel < 1.0:
            self.n_rel_lt1 += 1
        if rel < 0.5:
            self.n_rel_lt05 += 1
        if gerr_norm is not None:
            slack = bound * (1.0 + 1e-4)
            if gerr_norm > slack:
                self.n_bound_violation += 1
            if gerr_norm > slack + self.ATOL * max(ghat_norm, 1e-30):
                self.n_bound_violation_fp += 1
        if trueE_mean is not None:
            if trueE_mean > e_mean * (1.0 + 1e-4):
                self.n_cert_invalid += 1
            if trueE_mean > 1e-12:
                tight = e_mean / trueE_mean
                self.n_tight_checked += 1
                self.sum_tight += tight
                self.max_tight = max(self.max_tight, tight)
                if tight <= 10.0:
                    self.n_tight_le10 += 1

    def summary(self):
        ns = max(self.n_steps, 1)
        nt = max(self.n_tight_checked, 1)
        out = {
            "n_steps": self.n_steps,
            "n_no_signal": self.n_no_signal,
            "n_rhobar_lt1": self.n_rhobar_lt1,
            "frac_rhobar_lt1": self.n_rhobar_lt1 / ns if self.n_steps else None,
            "n_rel_lt1": self.n_rel_lt1,
            "n_rel_lt05": self.n_rel_lt05,
            "n_tight_checked": self.n_tight_checked,
            "n_tight_le10": self.n_tight_le10,
            "n_bound_violation": self.n_bound_violation,
            "n_bound_violation_fp": self.n_bound_violation_fp,
            "n_cert_invalid": self.n_cert_invalid,
            "frac_rel_lt1": self.n_rel_lt1 / ns,
            "frac_rel_lt05": self.n_rel_lt05 / ns,
            "frac_tight_le10": (self.n_tight_le10 / nt) if self.n_tight_checked else None,
            "mean_rel_g": self.sum_rel / ns,
            "max_rel_g": self.max_rel,
            "mean_tightness": (self.sum_tight / nt) if self.n_tight_checked else None,
            "max_tightness": self.max_tight if self.n_tight_checked else None,
            "has_shadow": self.has_shadow,
            "delta_agg": "mean_b(||delta_b||_2 * e_b)",
            "denominator": "n_steps excludes the n_no_signal steps",
        }
        if self.span is not None:
            out["step_range"] = list(self.span)
        if self.stages:
            out["stages"] = [c.summary() for c in self.stages]
        return out


class TanhCore:
    """TanhRNNCell + OnlineGrad estimator behind the diag-cell core protocol.

    advance() stages (h_{t+1}, A_{t+1}, imm_{t+1}) without touching the estimator,
    so backward_grads() contracts delta_h with J_t; commit() then resets done lanes
    and steps the estimator to J_{t+1}.

    shadow=True additionally maintains an exact-RTRL shadow and a SnAp-1 tracker
    for the D1 residual spectrum and the D2 certificate counters (module docstring).
    """

    def __init__(self, n_in, n_hid, algo_name, batch, device="cuda", dtype=torch.float32,
                 shadow=False, cert_warmup=0, cert_min_age=0, cert_every=1,
                 cos_window=200, spectral_clip=0.0, total_steps=0, n_stages=3,
                 **algo_kw):
        self.cell = TanhRNNCell(n_in, n_hid, device=device, dtype=dtype,
                                spectral_clip=spectral_clip)
        self.algo = make_algo(algo_name, self.cell, batch, **algo_kw)
        self.n_feat = n_hid
        self.B = batch
        self.shadow = self.snap = self.cert = None
        self._trackers = ()
        self.last_cert = {}
        self._t = 0                                # gradient steps taken
        self._cos = deque(maxlen=cos_window)
        self._n_skipped = 0
        self.cert_warmup, self.cert_min_age = cert_warmup, max(0, cert_min_age)
        self.cert_every = max(1, cert_every)
        # stage buckets: thirds of the run (needs the total step budget; 0 = off)
        self.total_steps = int(total_steps or 0)
        self.n_stages = int(n_stages) if self.total_steps > 0 else 0
        self.age = torch.zeros(batch, dtype=torch.long, device=self.cell.W.device)
        if shadow:
            self.shadow = ExactRTRL(self.cell, batch)          # true J_t
            self.snap = SnAp1(self.cell, batch)                # SnAp-1 trace S_t
            # deterministic power-iteration probe: keeps the tracker free of any
            # global-RNG side effect, so --shadow does not perturb the trajectory
            self.snap._pv = torch.full((batch, n_hid, 1), n_hid ** -0.5,
                                       device=self.cell.W.device, dtype=dtype)
            self._trackers = (self.shadow, self.snap)
            self.cert = (CertCounters(True, n_stages=self.n_stages)
                         if hasattr(self.algo, "e") else None)
            if self.cert is not None and self.n_stages:
                edges = [round(i * self.total_steps / self.n_stages)
                         for i in range(self.n_stages + 1)]
                for i, c in enumerate(self.cert.stages):
                    c.span = (edges[i] + 1, edges[i + 1])

    def parameters(self):
        return list(self.cell.parameters())

    @torch.no_grad()
    def clip(self):
        """Post-update spectral clip of W (no-op unless spectral_clip > 0)."""
        self.cell.clip_spectral()

    def begin(self, batch):
        self.h = self.cell.init_state(batch)
        self._stage, self._leaf = None, None

    def features(self):
        self._leaf = self.h.detach().requires_grad_(True)
        return self._leaf

    @torch.no_grad()
    def advance(self, x, done):
        h_prev = self.h * (~done).to(self.h.dtype).unsqueeze(1)
        h = self.cell(x, h_prev)
        A, imm = self.cell.jac_pieces(x, h_prev, h)
        self._stage = (h, A, imm, done)
        return h

    @torch.no_grad()
    def backward_grads(self, scale=1.0):
        delta = self._leaf.grad
        if delta is None:
            return
        g_rows = self.algo.grad_rows(delta)
        if self.shadow is not None:
            self._measure(delta.detach(), g_rows)
        self.cell.apply_flat_grad(g_rows * scale)

    @torch.no_grad()
    def commit(self):
        h, A, imm, done = self._stage
        if done.any():
            _reset_estimator_lanes(self.algo, done)
            for tr in self._trackers:
                _reset_estimator_lanes(tr, done)
        self.algo.step_state(A, imm)
        for tr in self._trackers:
            tr.step_state(A, imm)
        self.h = h
        self._stage = None
        self.age += 1                      # age of the state h_{t+1} just adopted
        if done.any():
            self.age[done] = 1

    # ---------------- shadow diagnostics: D2 certificate, D1 spectrum ----------------

    @torch.no_grad()
    def _measure(self, delta, ghat):
        """One gradient step of shadow diagnostics (aggregation: module docstring)."""
        self._t += 1
        g = self.shadow.grad_rows(delta)                  # true batch-mean grad rows
        self._cos.append(cos(ghat.flatten(), g.flatten()))
        do_cert = (self.cert is not None and self._t > self.cert_warmup
                   and self._t % self.cert_every == 0
                   and (self.cert_min_age <= 0                    # short-circuit: the
                        or int(self.age.min()) >= self.cert_min_age))   # min() syncs
        last = self.last_cert
        dn_lane = delta.norm(dim=1)                       # (B,)
        last["delta_norm"] = float(dn_lane.mean())
        last["ghat_norm"] = float(ghat.norm())
        last["g_norm"] = float(g.norm())
        gerr = float((g - ghat).norm())
        last["gerr_norm"] = gerr
        last["J_norm"] = float(self.shadow.J.flatten(1).norm(dim=1).mean())
        la = getattr(self.algo, "last", None) or {}
        for k in ("rho_bar", "rho_hat", "eta"):
            if k in la:
                last[k] = float(la[k].mean())
        e_lane = getattr(self.algo, "e", None)
        if e_lane is None or not torch.is_tensor(e_lane):
            return                                        # estimator without certificate
        e_mean = float(e_lane.mean())
        bound = float((dn_lane * e_lane).mean())          # Theorem-1 bound, batch form
        last["e_t"] = e_mean
        last["bound_norm"] = bound
        last["rho_g"] = bound / last["ghat_norm"] if last["ghat_norm"] > 0 else None
        trueE = None
        if hasattr(self.algo, "residual_dense"):
            E = self.shadow.J - self.algo.residual_dense()
            trueE = float(E.flatten(1).norm(dim=1).mean())
            last["true_E"] = trueE
            last["tight"] = e_mean / trueE if trueE > 1e-12 else None
        if do_cert:
            stage = None
            if self.n_stages:
                stage = min(self.n_stages - 1,
                            int(self.n_stages * (self._t - 1) / max(self.total_steps, 1)))
            self.cert.update(bound, last["ghat_norm"], gerr, e_mean, trueE,
                             rho_bar=last.get("rho_bar"),
                             delta_norm=last["delta_norm"], stage=stage)
        elif self.cert is not None:
            self._n_skipped += 1

    def cert_record(self):
        """Log-record fields: windowed grad_cos + the latest per-step diagnostics."""
        fin = [c for c in self._cos if not math.isnan(c)]
        rec = {"grad_cos": (sum(fin) / len(fin)) if fin else None}
        for k in CERT_KEYS:
            rec[k] = self.last_cert.get(k)
        return rec

    def cert_summary(self):
        if self.shadow is None:
            return None
        s = {"n_grad_steps": self._t, "n_skipped": self._n_skipped,
             "cert_warmup": self.cert_warmup, "cert_min_age": self.cert_min_age,
             "cert_every": self.cert_every}
        s.update(self.cert.summary() if self.cert is not None
                 else {"has_shadow": True, "n_steps": 0,
                       "note": "estimator exposes no certificate (no e_t)"})
        return s

    @torch.no_grad()
    def spectrum(self, max_sv=64, ranks=(1, 4, 8, 16, 32, 64)):
        """D1 residual spectrum of R_t = J_t - S_t, per lane, aggregated over lanes.

        Singular values are taken lane by lane (P = n*p is wide, so the dense
        batched form is the only memory risk); mass_top{r} and stable_rank are lane
        means of the per-lane quantities and res_frac_of_J is the ratio of the lane
        means, the same convention as run_m1_spectrum.py (the D1 implementation).
        """
        if self.shadow is None:
            return None
        n, p, P = self.cell.n, self.cell.p, self.shadow.P
        idx = torch.arange(n, device=self.shadow.J.device)
        svs, rn, jn = [], [], []
        for b in range(self.B):
            R = self.shadow.J[b].clone().view(n, n, p)
            R[idx, idx, :] -= self.snap.S[b]               # R = J - S (SnAp-1 trace)
            R = R.view(n, P)
            try:
                s = torch.linalg.svdvals(R)
            except Exception:                              # cuSOLVER fallback
                s = torch.linalg.svdvals(R.cpu()).to(R.device)
            svs.append(s)
            rn.append(float(R.norm()))
            jn.append(float(self.shadow.J[b].norm()))
        S = torch.stack(svs)                               # (B, min(n, P))
        tot = (S ** 2).sum(dim=1).clamp_min(1e-30)
        res_norm, j_norm = sum(rn) / len(rn), sum(jn) / len(jn)
        rec = {"n_lanes": self.B, "n": n, "P": P,
               "sv": S.mean(0)[:max_sv].tolist(),
               "stable_rank": float((tot / (S[:, 0] ** 2).clamp_min(1e-30)).mean()),
               "res_norm": res_norm, "J_norm": j_norm,
               "res_frac_of_J": res_norm / max(j_norm, 1e-30),
               "age_min": int(self.age.min()), "age_mean": float(self.age.float().mean())}
        for k in ranks:
            if k <= S.shape[1]:
                rec[f"mass_top{k}"] = float(((S[:, :k] ** 2).sum(dim=1) / tot).mean())
        return rec


class ActorCritic:
    """Online A2C(0) over a batched auto-reset env and a recurrent core."""

    def __init__(self, env, core, lr=3e-4, gamma=0.99, beta=0.01, accumulate_k=1,
                 device="cuda", dtype=torch.float32):
        self.env, self.core = env, core
        self.gamma, self.beta, self.k = gamma, beta, max(1, accumulate_k)
        self.pi = nn.Linear(core.n_feat, env.n_actions).to(device=device, dtype=dtype)
        self.v = nn.Linear(core.n_feat, 1).to(device=device, dtype=dtype)
        self.opt = torch.optim.Adam(list(core.parameters())
                                    + list(self.pi.parameters()) + list(self.v.parameters()), lr=lr)
        core.begin(env.B)
        obs = env.reset()                       # feed first obs: h_1 = cell(o_0, 0)
        core.advance(obs, torch.zeros(env.B, dtype=torch.bool, device=device))
        core.commit()
        self._acc = 0
        self.opt.zero_grad(set_to_none=True)

    def step(self):
        f = self.core.features()                       # graph at h_t (leaf)
        dist = Categorical(logits=self.pi(f))
        a = dist.sample()
        V = self.v(f).squeeze(1)
        obs, r, done = self.env.step(a)
        f_next = self.core.advance(obs, done)          # detached h_{t+1} features
        with torch.no_grad():
            V_next = self.v(f_next).squeeze(1)
        td = r + self.gamma * (~done).to(V.dtype) * V_next - V
        ent = dist.entropy()
        loss = (-td.detach() * dist.log_prob(a) + 0.5 * td.pow(2) - self.beta * ent).mean()
        (loss / self.k).backward()                     # head grads + leaf delta_h
        self.core.backward_grads(1.0 / self.k)         # recurrent grads via estimator
        self._acc += 1
        if self._acc >= self.k:
            self.opt.step()
            clip = getattr(self.core, "clip", None)
            if clip is not None:
                clip()                                 # no-op unless --clip > 0
            self.opt.zero_grad(set_to_none=True)
            self._acc = 0
        self.core.commit()                             # resets done lanes, J_t -> J_{t+1}
        return r, done, {"loss": loss.item(), "entropy": ent.mean().item()}
