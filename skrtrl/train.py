"""Online training loop: forward 1 step -> instantaneous loss -> online grad -> update."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cells import TanhRNNCell
from .algos import ExactRTRL, SKRTRL, SnAp1


def make_algo(name, cell, batch, **kw):
    if name == "exact":
        return ExactRTRL(cell, batch)
    if name == "snap1":
        return SnAp1(cell, batch)
    if name.startswith("skrtrl"):
        mode = "randproj" if name.startswith("skrtrl-rp") else "svd"
        rpart = name.split("-rp")[1] if mode == "randproj" else (name.split("-r")[1] if "-r" in name else "")
        r = int(rpart) if rpart else kw.get("r", 16)
        return SKRTRL(cell, batch, r=r, mode=mode)
    if name in ("uoro", "kfrtrl", "eprop", "rflo"):
        from .baselines import make_baseline
        return make_baseline(name, cell, batch)
    raise ValueError(name)


class OnlineLearner:
    def __init__(self, task, n_hid, algo_name, lr=1e-3, device="cuda", seed=0,
                 dtype=torch.float32, spectral_clip=0.0, algo_kw=None):
        torch.manual_seed(seed)
        self.task = task
        self.cell = TanhRNNCell(task.n_in, n_hid, device=device, dtype=dtype,
                                spectral_clip=spectral_clip)
        self.readout = nn.Linear(n_hid, task.n_out).to(device=device, dtype=dtype)
        self.algo = make_algo(algo_name, self.cell, task.B, **(algo_kw or {}))
        self.opt = torch.optim.Adam(list(self.cell.parameters()) + list(self.readout.parameters()), lr=lr)
        self.h = self.cell.init_state(task.B)
        self.loss_type = task.loss_type
        self.device = device

    def loss_fn(self, out, y):
        if self.loss_type == "ce":
            return F.cross_entropy(out, y), (out.argmax(1) == y).float().mean().item()
        return F.mse_loss(out, y), F.mse_loss(out, y).item()

    @torch.no_grad()
    def _reset_lanes(self, new_ep):
        if new_ep.any():
            self.h[new_ep] = 0.0
            # reset estimator state of those lanes
            a = self.algo
            if hasattr(a, "J"):
                a.J[new_ep] = 0.0
            for attr in ("S", "L", "R"):
                if hasattr(a, attr) and getattr(a, attr) is not None and getattr(a, attr).numel():
                    getattr(a, attr)[new_ep] = 0.0
            if hasattr(a, "e"):
                a.e[new_ep] = 0.0
            if hasattr(a, "reset_lanes"):
                a.reset_lanes(new_ep)

    def step(self, update=True):
        x, y, new_ep = self.task.step()
        self._reset_lanes(new_ep)
        h_prev = self.h.detach()
        h = self.cell(x, h_prev)
        A, imm = self.cell.jac_pieces(x, h_prev, h)
        self.algo.step_state(A, imm)
        self.h = h.detach()

        metrics = {}
        if y is not None:
            h_leaf = self.h.requires_grad_(True)
            out = self.readout(h_leaf)
            loss, metric = self.loss_fn(out, y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()                      # fills readout grads + h_leaf.grad
            delta = h_leaf.grad.detach()         # (B, n)
            g_rows = self.algo.grad_rows(delta)
            self.cell.apply_flat_grad(g_rows)
            if update:
                self.opt.step()
                self.cell.clip_spectral()
            self.h = self.h.detach()
            metrics = {"loss": loss.item(), "metric": metric}
        return metrics
