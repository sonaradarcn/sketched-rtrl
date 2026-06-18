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
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .cells import TanhRNNCell
from .train import make_algo


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


class TanhCore:
    """TanhRNNCell + OnlineGrad estimator behind the diag-cell core protocol.

    advance() stages (h_{t+1}, A_{t+1}, imm_{t+1}) without touching the estimator,
    so backward_grads() contracts delta_h with J_t; commit() then resets done lanes
    and steps the estimator to J_{t+1}.
    """

    def __init__(self, n_in, n_hid, algo_name, batch, device="cuda", dtype=torch.float32):
        self.cell = TanhRNNCell(n_in, n_hid, device=device, dtype=dtype)
        self.algo = make_algo(algo_name, self.cell, batch)
        self.n_feat = n_hid

    def parameters(self):
        return list(self.cell.parameters())

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
        self.cell.apply_flat_grad(self.algo.grad_rows(delta) * scale)

    @torch.no_grad()
    def commit(self):
        h, A, imm, done = self._stage
        if done.any():
            _reset_estimator_lanes(self.algo, done)
        self.algo.step_state(A, imm)
        self.h = h
        self._stage = None


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
            self.opt.zero_grad(set_to_none=True)
            self._acc = 0
        self.core.commit()                             # resets done lanes, J_t -> J_{t+1}
        return r, done, {"loss": loss.item(), "entropy": ent.mean().item()}
