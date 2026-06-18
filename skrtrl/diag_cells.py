"""Exact-online-gradient diagonal-recurrence cells: OnlineLRU, RTU.

Element-wise (block-diagonal) recurrence makes the RTRL influence block-diagonal:
each unit's state depends only on its own (nu_k, theta_k) and its own input-projection
row, so EXACT per-parameter eligibility traces are O(n) / O(n*m) scalar recursions —
the diagonal analogue of OnlineGrad.step_state + grad_rows.

Core protocol (consumed by rl.ActorCritic; TanhCore in rl.py implements the same):
  begin(batch)            allocate hidden state + eligibility traces
  features()              (B, n_feat) graph rooted at fresh state leaves (for heads)
  advance(x, done)        compute next state + next traces (STAGED, not committed),
                          zeroing state/traces of done lanes; return detached next
                          features (for the V(h_{t+1}) bootstrap)
  backward_grads(scale)   after loss.backward(): leaf grads (= dLoss/dh_t) contracted
                          with the CURRENT traces -> accumulate recurrent-param .grad
  commit()                adopt staged state/traces (call after optimizer step)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _acc(p, g):
    if p.grad is None:
        p.grad = g.detach().clone()
    else:
        p.grad += g


def _nu_init(n, r_min, r_max, device, dtype):
    """nu with r = exp(-exp(nu)) ~ U[r_min, r_max]."""
    r = r_min + (r_max - r_min) * torch.rand(n, device=device, dtype=dtype)
    return torch.log(-torch.log(r))


class OnlineLRU(nn.Module):
    """Orvieto et al. 2023 LRU with exact online (RTRL) gradients.

    u_t = lam (.) u_{t-1} + gam (.) (B_in x_t),  lam = exp(-exp(nu) + i theta),
    gam = sqrt(1 - |lam|^2)  (gamma normalization);  B_in real input projection.
    features = ReLU(C [Re u; Im u])  -- C (self.out) trains by autograd on the head
    graph; nu / theta / B_in get exact eligibility-trace grads:
      e^nu_t  = lam e^nu_{t-1}  - exp(nu) lam u_{t-1} + (dgam/dnu) B_in x_t
      e^th_t  = lam e^th_{t-1}  + i lam u_{t-1}
      T^B_t   = lam T^B_{t-1}   + gam x_t^T          (per-row scalar decay)
    All complex arithmetic kept as (re, im) real pairs.
    """
    name = "lru"

    def __init__(self, n_in, n_units, n_feat=None, device=None, dtype=torch.float32,
                 r_min=0.9, r_max=0.999, max_phase=math.pi / 10):
        super().__init__()
        self.n, self.m = n_units, n_in
        self.n_feat = n_feat or 2 * n_units
        self.nu = nn.Parameter(_nu_init(n_units, r_min, r_max, device, dtype))
        self.theta = nn.Parameter(max_phase * torch.rand(n_units, device=device, dtype=dtype))
        k = 1.0 / n_in ** 0.5
        self.Bin = nn.Parameter(torch.empty(n_units, n_in, device=device, dtype=dtype).uniform_(-k, k))
        self.out = nn.Linear(2 * n_units, self.n_feat).to(device=device, dtype=dtype)

    def _coeffs(self):
        en = torch.exp(self.nu)
        rr = torch.exp(-en)
        lr_, li = rr * torch.cos(self.theta), rr * torch.sin(self.theta)
        gam = torch.sqrt((1.0 - rr * rr).clamp_min(1e-12))
        return en, rr, lr_, li, gam

    def begin(self, batch):
        z = lambda *s: torch.zeros(batch, *s, device=self.nu.device, dtype=self.nu.dtype)
        self.ur, self.ui = z(self.n), z(self.n)
        self.enu_r, self.enu_i = z(self.n), z(self.n)
        self.eth_r, self.eth_i = z(self.n), z(self.n)
        self.Tb_r, self.Tb_i = z(self.n, self.m), z(self.n, self.m)
        self._stage, self._leaves = None, None

    def features(self):
        ur = self.ur.detach().requires_grad_(True)
        ui = self.ui.detach().requires_grad_(True)
        self._leaves = (ur, ui)
        return F.relu(self.out(torch.cat([ur, ui], dim=1)))

    @torch.no_grad()
    def advance(self, x, done):
        en, rr, lr_, li, gam = self._coeffs()
        keep = (~done).to(x.dtype).unsqueeze(1)
        ur, ui = self.ur * keep, self.ui * keep
        bx = x @ self.Bin.T
        lu_r, lu_i = lr_ * ur - li * ui, li * ur + lr_ * ui          # lam u_{t-1}
        st = {"ur": lu_r + gam * bx, "ui": lu_i}
        dgam = en * rr * rr / gam
        enr, eni = self.enu_r * keep, self.enu_i * keep
        etr, eti = self.eth_r * keep, self.eth_i * keep
        st["enu_r"] = lr_ * enr - li * eni - en * lu_r + dgam * bx
        st["enu_i"] = li * enr + lr_ * eni - en * lu_i
        st["eth_r"] = lr_ * etr - li * eti - lu_i                    # i lam u = (-Im, +Re)
        st["eth_i"] = li * etr + lr_ * eti + lu_r
        lrc, lic, gmc = lr_.view(1, -1, 1), li.view(1, -1, 1), gam.view(1, -1, 1)
        Tr, Ti = self.Tb_r * keep.unsqueeze(2), self.Tb_i * keep.unsqueeze(2)
        st["Tb_r"] = lrc * Tr - lic * Ti + gmc * x.unsqueeze(1)
        st["Tb_i"] = lic * Tr + lrc * Ti
        self._stage = st
        return F.relu(self.out(torch.cat([st["ur"], st["ui"]], dim=1)))

    @torch.no_grad()
    def backward_grads(self, scale=1.0):
        ur, ui = self._leaves
        if ur.grad is None and ui.grad is None:
            return
        dr = ur.grad if ur.grad is not None else torch.zeros_like(ur)
        di = ui.grad if ui.grad is not None else torch.zeros_like(ui)
        _acc(self.nu, scale * (dr * self.enu_r + di * self.enu_i).mean(0))
        _acc(self.theta, scale * (dr * self.eth_r + di * self.eth_i).mean(0))
        _acc(self.Bin, scale * (dr.unsqueeze(2) * self.Tb_r + di.unsqueeze(2) * self.Tb_i).mean(0))

    @torch.no_grad()
    def commit(self):
        for k_, v in self._stage.items():
            setattr(self, k_, v)
        self._stage = None


class RTU(nn.Module):
    """Recurrent Trace Units (Elelimy et al. 2024, arXiv:2409.01449), NONLINEAR variant.

    Paper Eq.(4), cosine parametrization with real-valued state pairs:
      h1_t = f(g (.) h1_{t-1} - ph (.) h2_{t-1} + gam (.) W1 x_t),
      h2_t = f(g (.) h2_{t-1} + ph (.) h1_{t-1} + gam (.) W2 x_t),  h_t = [h1; h2],
      g = r cos(th), ph = r sin(th), r = exp(-exp(nu)), th = exp(tl),
      gam = sqrt(1 - r^2),  f = ReLU.
    Exact RTRL traces (paper Sec 3.3): per unit, 2x2-coupled scalar recursions.
    Deviations from the paper's write-up (both still exact for this recurrence):
      - traces are taken w.r.t. (nu, tl) = (nu^log, theta^log) directly, instead of
        (r, theta) followed by an outer chain rule;
      - the nonlinearity factor f'(z_t) is folded into the trace recursion (the paper
        states traces for the linear RTU only; Eq.(4) defines the nonlinear cell).
    features = [h1; h2] fed straight to the heads (paper feeds h_t to a linear layer).
    """
    name = "rtu"

    def __init__(self, n_in, n_units, device=None, dtype=torch.float32,
                 r_min=0.9, r_max=0.999, th_min=0.01, th_max=math.pi / 10):
        super().__init__()
        self.n, self.m = n_units, n_in
        self.n_feat = 2 * n_units
        self.nu = nn.Parameter(_nu_init(n_units, r_min, r_max, device, dtype))
        lo, hi = math.log(th_min), math.log(th_max)
        self.tl = nn.Parameter(lo + (hi - lo) * torch.rand(n_units, device=device, dtype=dtype))
        k = 1.0 / n_in ** 0.5
        self.W1 = nn.Parameter(torch.empty(n_units, n_in, device=device, dtype=dtype).uniform_(-k, k))
        self.W2 = nn.Parameter(torch.empty(n_units, n_in, device=device, dtype=dtype).uniform_(-k, k))

    def begin(self, batch):
        z = lambda *s: torch.zeros(batch, *s, device=self.nu.device, dtype=self.nu.dtype)
        self.h1, self.h2 = z(self.n), z(self.n)
        self.enu1, self.enu2 = z(self.n), z(self.n)
        self.eth1, self.eth2 = z(self.n), z(self.n)
        self.T1a, self.T1b = z(self.n, self.m), z(self.n, self.m)   # d(h1,h2)/dW1
        self.T2a, self.T2b = z(self.n, self.m), z(self.n, self.m)   # d(h1,h2)/dW2
        self._stage, self._leaves = None, None

    def features(self):
        h1 = self.h1.detach().requires_grad_(True)
        h2 = self.h2.detach().requires_grad_(True)
        self._leaves = (h1, h2)
        return torch.cat([h1, h2], dim=1)

    @torch.no_grad()
    def advance(self, x, done):
        en = torch.exp(self.nu)
        rr = torch.exp(-en)
        th = torch.exp(self.tl)
        g_, ph = rr * torch.cos(th), rr * torch.sin(th)
        gam = torch.sqrt((1.0 - rr * rr).clamp_min(1e-12))
        dg_nu, dph_nu, dgam_nu = -en * g_, -en * ph, en * rr * rr / gam
        dg_tl, dph_tl = -ph * th, g_ * th
        keep = (~done).to(x.dtype).unsqueeze(1)
        h1, h2 = self.h1 * keep, self.h2 * keep
        w1x, w2x = x @ self.W1.T, x @ self.W2.T
        z1 = g_ * h1 - ph * h2 + gam * w1x
        z2 = g_ * h2 + ph * h1 + gam * w2x
        d1, d2 = (z1 > 0).to(z1.dtype), (z2 > 0).to(z2.dtype)        # f' for ReLU
        st = {"h1": F.relu(z1), "h2": F.relu(z2)}
        e1, e2 = self.enu1 * keep, self.enu2 * keep
        st["enu1"] = d1 * (dg_nu * h1 + g_ * e1 - dph_nu * h2 - ph * e2 + dgam_nu * w1x)
        st["enu2"] = d2 * (dg_nu * h2 + g_ * e2 + dph_nu * h1 + ph * e1 + dgam_nu * w2x)
        e1, e2 = self.eth1 * keep, self.eth2 * keep
        st["eth1"] = d1 * (dg_tl * h1 + g_ * e1 - dph_tl * h2 - ph * e2)
        st["eth2"] = d2 * (dg_tl * h2 + g_ * e2 + dph_tl * h1 + ph * e1)
        k2 = keep.unsqueeze(2)
        gc, pc, gmc = g_.view(1, -1, 1), ph.view(1, -1, 1), gam.view(1, -1, 1)
        d1c, d2c, xc = d1.unsqueeze(2), d2.unsqueeze(2), x.unsqueeze(1)
        Ta, Tb = self.T1a * k2, self.T1b * k2
        st["T1a"] = d1c * (gc * Ta - pc * Tb + gmc * xc)
        st["T1b"] = d2c * (gc * Tb + pc * Ta)
        Ta, Tb = self.T2a * k2, self.T2b * k2
        st["T2a"] = d1c * (gc * Ta - pc * Tb)
        st["T2b"] = d2c * (gc * Tb + pc * Ta + gmc * xc)
        self._stage = st
        return torch.cat([st["h1"], st["h2"]], dim=1)

    @torch.no_grad()
    def backward_grads(self, scale=1.0):
        h1, h2 = self._leaves
        if h1.grad is None and h2.grad is None:
            return
        d1 = h1.grad if h1.grad is not None else torch.zeros_like(h1)
        d2 = h2.grad if h2.grad is not None else torch.zeros_like(h2)
        _acc(self.nu, scale * (d1 * self.enu1 + d2 * self.enu2).mean(0))
        _acc(self.tl, scale * (d1 * self.eth1 + d2 * self.eth2).mean(0))
        _acc(self.W1, scale * (d1.unsqueeze(2) * self.T1a + d2.unsqueeze(2) * self.T1b).mean(0))
        _acc(self.W2, scale * (d1.unsqueeze(2) * self.T2a + d2.unsqueeze(2) * self.T2b).mean(0))

    @torch.no_grad()
    def commit(self):
        for k_, v in self._stage.items():
            setattr(self, k_, v)
        self._stage = None
