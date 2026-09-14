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
        # --- SK-RTRL cell protocol (mirrors skrtrl/cells_gated.py) -------------
        # One gate, one state row per parameter block: every structural quantity
        # the estimator asks a gated cell for is the identity here.
        self.G = 1
        self.p_gate = self.p
        self.p_cell = self.p
        self.n_state = self.n
        self.register_buffer("blk_idx", torch.arange(n_hid, device=device),
                             persistent=False)

    @property
    def append_width(self):
        """omega: the append N_t = Ahat_t S_{t-1} has exactly n columns."""
        return self.n

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

    # ------------------------------------------------------------------
    # SK-RTRL cell protocol.  Every method below reproduces, bit for bit,
    # the algebra that used to be inlined in skrtrl/algos.py -- including
    # the zero-row convention of the append right factor (a zero row, NOT
    # the canonical e_0 of cells_gated._row_normalise: the matching
    # C-column vanishes either way, and keeping the zero row is what makes
    # the vanilla path numerically unchanged).
    # ------------------------------------------------------------------
    def blkdiag_of_A(self, A):
        return torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))

    def offdiag_of_A(self, A):
        return A - self.blkdiag_of_A(A)

    @torch.no_grad()
    def trace_update(self, A, S_prev, imm):
        """S_t = diag(A_t) (.) S_{t-1} + I_t."""
        return torch.diagonal(A, dim1=1, dim2=2).unsqueeze(2) * S_prev + imm

    @torch.no_grad()
    def append_factors(self, A, S_prev):
        """N_t = Ahat_t S_{t-1} = (Ahat_t D_s) Q_t^T, omega = n columns."""
        from .cells_gated import StructuredQ
        s_norm = S_prev.norm(dim=2)                                # (B, n)
        C = self.offdiag_of_A(A) * s_norm.unsqueeze(1)             # (B, n, n)
        nz = s_norm > 0
        q = torch.where(nz.unsqueeze(2),
                        S_prev / s_norm.clamp_min(1e-30).unsqueeze(2),
                        torch.zeros_like(S_prev))                  # (B, n, p)
        return C, StructuredQ(q, self.blk_idx, self.p, self.n)

    @torch.no_grad()
    def append_mass(self, A, S_prev):
        """||N_t||_F exactly (Q_t's nonzero rows are unit norm)."""
        s_norm = S_prev.norm(dim=2)
        return torch.linalg.matrix_norm(self.offdiag_of_A(A) * s_norm.unsqueeze(1),
                                        ord="fro", dim=(1, 2))

    @torch.no_grad()
    def imm_offblock_norm(self):
        """0: the vanilla immediate Jacobian is exactly block diagonal."""
        return torch.zeros(1, device=self.W.device, dtype=self.W.dtype)

    def imm_offblock_factors(self):
        return None

    @torch.no_grad()
    def imm_dense(self, imm):
        """(B, n, P) full immediate Jacobian (tests / exact shadow only)."""
        B = imm.shape[0]
        out = imm.new_zeros(B, self.n, self.n, self.p)
        i = torch.arange(self.n, device=imm.device)
        out[:, i, i, :] = imm
        return out.reshape(B, self.n, self.n * self.p)

    @torch.no_grad()
    def flat_grad_rows(self):
        """(n, p) view of the autograd gradient; inverse of apply_flat_grad."""
        return torch.cat([self.W.grad, self.U.grad, self.b.grad.unsqueeze(1)], dim=1)

    @torch.no_grad()
    def spectral_clip_(self, clip=None):
        c = self.spectral_clip if clip is None else clip
        if c and c > 0:
            s = torch.linalg.matrix_norm(self.W, ord=2)
            if s > c:
                self.W.mul_(c / s)
