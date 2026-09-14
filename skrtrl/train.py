"""Online training loop: forward 1 step -> instantaneous loss -> online grad -> update."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cells import TanhRNNCell
from .cells_gated import make_gated_cell
from .algos import ExactRTRL, SKRTRL, SnAp1


def make_cell(kind, n_in, n_hid, device=None, dtype=torch.float32,
              spectral_clip=0.0):
    """Build the recurrent cell.  ``kind="tanh"`` is the cell of record.

    The gated cells (``"gru"``, ``"lstm"``) implement the same SK-RTRL cell
    protocol, so nothing downstream branches on the cell type; see
    skrtrl/cells_gated.py and paper/revise_r2/GRU_EXTENSION.tex.
    """
    if kind in (None, "tanh"):
        return TanhRNNCell(n_in, n_hid, device=device, dtype=dtype,
                           spectral_clip=spectral_clip)
    return make_gated_cell(kind, n_in, n_hid, device=device, dtype=dtype,
                           spectral_clip=spectral_clip)


def make_algo(name, cell, batch, **kw):
    """Build an online-gradient estimator.

    Recognised keyword arguments (all optional):
      r                 fallback rank when the name carries none
      svd_driver        "gesvd" (kernel of record) | "auto"   -- SK-RTRL only
      c                 explicit append budget, overrides max(4, ceil(r/4))
      force_preproject  keep pre-projection + c even when r >= n
      disable_preproject  switch the top-c pre-projection off entirely (c = n)
      collapse_every    amortised kernel collapse period ("am-" prefix)
    """
    svd_driver = kw.get("svd_driver", "gesvd")
    if name == "exact":
        return ExactRTRL(cell, batch)
    if name == "snap1":
        return SnAp1(cell, batch, svd_driver=svd_driver)
    # "am-" prefix selects the amortised-rotation kernel (skrtrl.algos_amortised);
    # it is mathematically equivalent to the kernel of record, see
    # tests/test_amortised_kernel.py.  E.g. "am-skrtrl-r16", "am-skrtrl-rp16".
    amortised = name.startswith("am-")
    if amortised:
        name = name[len("am-"):]
    if name.startswith("skrtrl"):
        mode = "randproj" if name.startswith("skrtrl-rp") else "svd"
        rpart = name.split("-rp")[1] if mode == "randproj" else (name.split("-r")[1] if "-r" in name else "")
        r = int(rpart) if rpart else kw.get("r", 16)
        if amortised:
            from .algos_amortised import SKRTRLAmortised
            return SKRTRLAmortised(cell, batch, r=r, mode=mode,
                                   collapse_every=kw.get("collapse_every"),
                                   c=kw.get("c"), svd_driver=svd_driver,
                                   force_preproject=kw.get("force_preproject", False),
                                   disable_preproject=kw.get("disable_preproject", False))
        return SKRTRL(cell, batch, r=r, mode=mode, c=kw.get("c"), svd_driver=svd_driver,
                      force_preproject=kw.get("force_preproject", False),
                      disable_preproject=kw.get("disable_preproject", False))
    if name in ("uoro", "kfrtrl", "eprop", "rflo"):
        from .baselines import make_baseline
        return make_baseline(name, cell, batch)
    raise ValueError(name)


class OnlineLearner:
    def __init__(self, task, n_hid, algo_name, lr=1e-3, device="cuda", seed=0,
                 dtype=torch.float32, spectral_clip=0.0, algo_kw=None,
                 svd_driver="gesvd", cell="tanh"):
        torch.manual_seed(seed)
        self.task = task
        self.cell_kind = cell or "tanh"
        self.cell = make_cell(self.cell_kind, task.n_in, n_hid, device=device,
                              dtype=dtype, spectral_clip=spectral_clip)
        self.n_hid = n_hid
        # The read-out sees h_t only.  For the LSTM the carried state is
        # [h_t; c_t] in R^{2n}, so h_leaf.grad is then the 2n-vector
        # [dL/dh_t; 0] -- exactly the delta_t of Theorem 1 step 3, with no
        # padding needed.  (Letting the read-out also read c_t is a different
        # model; this keeps a standard LSTM.)
        self.state_is_stacked = getattr(self.cell, "n_state", n_hid) != n_hid
        self.readout = nn.Linear(n_hid, task.n_out).to(device=device, dtype=dtype)
        akw = dict(algo_kw or {})
        akw.setdefault("svd_driver", svd_driver)
        self.algo = make_algo(algo_name, self.cell, task.B, **akw)
        self.opt = torch.optim.Adam(list(self.cell.parameters()) + list(self.readout.parameters()), lr=lr)
        self.h = self.cell.init_state(task.B)
        self.loss_type = task.loss_type
        self.device = device

    def readout_in(self, state):
        """The part of the carried state the read-out sees (h_t)."""
        return state[:, :self.n_hid] if self.state_is_stacked else state

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
            out = self.readout(self.readout_in(h_leaf))
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
