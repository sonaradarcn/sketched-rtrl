"""Recurrent cells exposing the quantities online-gradient algorithms need.

Parameter layout (per-row view, the SnAp-1 pattern):
  recurrent params of unit i = [W[i, :n], U[i, :m], b[i]]  -> p = n + m + 1 per row
  flat param index (i, j) -> i * p + j, total P = n * p.
"""
import torch
import torch.nn as nn


class TanhRNNCell(nn.Module):
    """h_t = tanh(W h_{t-1} + U x_t + b).

    Exposes per-step:
      A_t = D_t W            (n x n), D_t = diag(1 - h_t^2)
      imm_t[i, :] = (1 - h_{t,i}^2) * [h_{t-1}, x_t, 1]   (n x p)  immediate Jacobian rows
    """

    def __init__(self, n_in: int, n_hid: int, device=None, dtype=torch.float32,
                 w_init: str = "default", spectral_clip: float = 0.0):
        super().__init__()
        self.n, self.m = n_hid, n_in
        self.p = n_hid + n_in + 1
        k = 1.0 / n_hid ** 0.5
        W = torch.empty(n_hid, n_hid, device=device, dtype=dtype).uniform_(-k, k)
        if w_init == "orthogonal":
            torch.nn.init.orthogonal_(W)
        self.W = nn.Parameter(W)
        self.U = nn.Parameter(torch.empty(n_hid, n_in, device=device, dtype=dtype).uniform_(-k, k))
        self.b = nn.Parameter(torch.zeros(n_hid, device=device, dtype=dtype))
        self.spectral_clip = spectral_clip

    def init_state(self, batch: int):
        return torch.zeros(batch, self.n, device=self.W.device, dtype=self.W.dtype)

    def forward(self, x, h_prev):
        z = h_prev @ self.W.T + x @ self.U.T + self.b
        h = torch.tanh(z)
        return h

    @torch.no_grad()
    def jac_pieces(self, x, h_prev, h):
        """Return (A, imm): A (B,n,n), imm (B,n,p)."""
        D = 1.0 - h * h                                   # (B, n)
        A = D.unsqueeze(2) * self.W.unsqueeze(0)          # (B, n, n)
        ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        feats = torch.cat([h_prev, x, ones], dim=1)       # (B, p)
        imm = D.unsqueeze(2) * feats.unsqueeze(1)         # (B, n, p)
        return A, imm

    @torch.no_grad()
    def apply_flat_grad(self, g_rows):
        """g_rows (n, p) -> write into .grad of W, U, b (adds)."""
        n, m = self.n, self.m
        for par, sl in ((self.W, slice(0, n)), (self.U, slice(n, n + m))):
            if par.grad is None:
                par.grad = torch.zeros_like(par)
            par.grad += g_rows[:, sl]
        if self.b.grad is None:
            self.b.grad = torch.zeros_like(self.b)
        self.b.grad += g_rows[:, n + m]

    @torch.no_grad()
    def clip_spectral(self):
        if self.spectral_clip > 0:
            s = torch.linalg.matrix_norm(self.W, ord=2)
            if s > self.spectral_clip:
                self.W.mul_(self.spectral_clip / s)
