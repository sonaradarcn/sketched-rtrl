"""Gated recurrent cells (GRU, LSTM) exposing everything SK-RTRL needs.

New file: nothing here modifies `cells.py`, `algos.py` or `train.py`.
The derivation these implement is `paper/revise_r2/GRU_EXTENSION.tex`; the
numbers in the comments below refer to its propositions.

Why a gated cell needs more than `(A_t, imm)`
---------------------------------------------
For the vanilla tanh cell the immediate Jacobian lies entirely inside the
block-diagonal support that the SnAp-1 trace maintains, so `jac_pieces`
returning `(A_t, imm)` is enough and the append is always
`N_t = Ahat_t S_{t-1} = (Ahat_t D_s) Q_t^T`.

Neither holds verbatim for a GRU. The reset gate reaches every unit in one
step through `U_h (r_t (.) h_{t-1})`, so (Prop. 1)

    d h_{t,i} / d theta^r_{j.} = (K_t)_{ij} phi_t^T,
    K_t = diag(z_t (.) (1 - h~_t^2)) U_h diag(r'_t (.) h_{t-1}),

which is dense in (i, j). The immediate Jacobian therefore splits as
I_t = I_t^o + I_t^perp: a block-diagonal part the trace keeps, plus an
off-block part that is exactly rank <= n. Proposition 3 factors the whole
appended object,

    N_t = Ahat_t S_{t-1} + I_t^perp = C_t Qt_t^T,
    C_t = [ Ahat_t | Khat_t ] T_t  in R^{n x 2n},

with Qt_t having EXACTLY orthonormal columns, each supported on a single
parameter block -- the two properties Lemma 1's Pythagoras step needs.
`append_factors` returns exactly that pair.

The LSTM is the opposite case. Its immediate Jacobian IS block diagonal on
the stacked state x_t = [h_t; c_t] (Prop. 5), but (i) the trace update needs
the per-unit 2x2 blocks of A_t rather than its scalar diagonal, because
(A_t)_{i,n+i} = o (.) (1-tanh^2 c) (.) f is the main memory path, and
(ii) in exchange the c-columns of A_t are exactly diagonal, so
Ahat_t = A_t - blkdiag_2(A_t) annihilates them and the append collapses back
to the vanilla row-normalised factorisation of width omega = n (Prop. 7).

Interface (a superset of TanhRNNCell's)
---------------------------------------
    n, m                units, input width
    p_gate = n + m + 1  parameters of one unit in ONE gate
    p_cell = G * p_gate parameters of one unit across all G gates
    p                   alias of p_cell, so that code computing
                        `P = cell.n * cell.p` keeps working
    n_state             width of the carried state (n for GRU, 2n for LSTM)
    P = n * p_cell      total recurrent parameter count
    blk_idx (n_state,)  parameter block that each state row belongs to
                        (arange(n) for GRU; [0..n-1, 0..n-1] for LSTM)

    init_state(batch)                     -> (B, n_state) zeros
    forward(x, state_prev)                -> (B, n_state)
    jac_pieces(x, state_prev, state)      -> (A, imm_diag)
        A        (B, n_state, n_state)
        imm_diag (B, n_state, p_cell)  row s holds the nonzero block of
                 state row s, which sits in parameter block blk_idx[s]
    blkdiag_of_A(A)                       -> (B, n_state, n_state)
    offdiag_of_A(A)                       -> A - blkdiag_of_A(A)
    trace_update(A, S_prev, imm_diag)     -> (B, n_state, p_cell)
    append_factors(A, S_prev)             -> (C, Q) with Q a StructuredQ
    append_mass(A, S_prev)                -> ||N_t||_F, exact (= ||C||_F)
    imm_offblock_norm()                   -> ||I_t^perp||_F, exact
    apply_flat_grad(g_rows)               g_rows (n, p_cell)
    spectral_clip(clip) / clip_spectral() clip each U_a separately

Call order.  `jac_pieces` caches the per-step quantities that
`append_factors` and `imm_offblock_norm` need (phi_t and Khat_t for the GRU).
Call `forward` -> `jac_pieces` -> {`trace_update`, `append_factors`} within one
time step; a stale cache raises rather than returning a wrong factorisation.

Flat parameter layout (per-unit blocks -- the SnAp-1 view)
---------------------------------------------------------
    unit i, gate a in [0, G), offset j in [0, p_gate):
        flat index = i * p_cell + a * p_gate + j
        j in [0, n)      -> U_a[i, j]      recurrent, acts on h_{t-1}
        j in [n, n+m)    -> W_a[i, j-n]    input,     acts on x_t
        j = n+m          -> b_a[i]
    Gate order: GRU ("z", "r", "h"); LSTM ("i", "f", "g", "o").
NOTE the gated-RNN convention is used: U_. is recurrent and W_. is the input
matrix, which is the OPPOSITE of TanhRNNCell's W/U. The layout above is
authoritative.
"""
import math

import torch
import torch.nn as nn

_EPS = 1e-12


# ---------------------------------------------------------------------------
# structured right factor
# ---------------------------------------------------------------------------
class StructuredQ:
    """Append right factor: one nonzero `p_cell`-block per column.

    rows   (B, w, p_cell)  the nonzero block of each column (unit norm, or
                           zero only for columns whose C-column vanishes)
    blocks (w,) long       which parameter block each column occupies

    The dense form is Q (B, P, w) with
        Q[b, blocks[k]*p_cell : (blocks[k]+1)*p_cell, k] = rows[b, k, :].
    Keeping it factored is the point: materialising Q costs O(P w) whereas
    every consumer in Algorithm 1 (the Gram against R, the thin QR, the
    rotation) can work from `rows`/`blocks` at O(n p_cell w).
    """

    __slots__ = ("rows", "blocks", "p_cell", "n_units")

    def __init__(self, rows, blocks, p_cell, n_units):
        self.rows = rows
        self.blocks = blocks
        self.p_cell = p_cell
        self.n_units = n_units

    @property
    def width(self):
        return self.rows.shape[1]

    @property
    def P(self):
        return self.n_units * self.p_cell

    def dense(self):
        """(B, P, w) dense materialisation. For tests and the r=n endpoint."""
        B, w, pc = self.rows.shape
        out = self.rows.new_zeros(B, w, self.n_units, pc)
        idx = torch.arange(w, device=self.rows.device)
        out[:, idx, self.blocks, :] = self.rows
        return out.permute(0, 2, 3, 1).reshape(B, self.P, w)

    def gram(self):
        """(B, w, w) = Q^T Q, computed in O(n p_cell w^2) without densifying.

        Columns in different parameter blocks are orthogonal by construction,
        so only same-block pairs contribute.
        """
        B, w, _ = self.rows.shape
        same = (self.blocks.unsqueeze(0) == self.blocks.unsqueeze(1))
        g = torch.einsum("bkp,blp->bkl", self.rows, self.rows)
        return g * same.to(g.dtype)


# ---------------------------------------------------------------------------
# per-block Gram-Schmidt helpers (Prop. 3)
# ---------------------------------------------------------------------------
def _canonical_orth(q1, seg_offsets):
    """Unit vectors inside each block, orthogonal to q1.

    Picks, per (batch, unit), whichever segment-start canonical direction is
    least aligned with q1: with G >= 3 candidates the smallest |q1[o]| is at
    most 1/sqrt(3), so the orthogonalised vector has norm >= sqrt(2/3).
    """
    comp = torch.stack([q1[:, :, o] for o in seg_offsets], dim=-1).abs()
    k = comp.argmin(dim=-1)
    off = torch.as_tensor(seg_offsets, device=q1.device)[k]           # (B,n)
    e = torch.zeros_like(q1)
    e.scatter_(2, off.unsqueeze(2), 1.0)
    v = e - (e * q1).sum(dim=2, keepdim=True) * q1
    return v / v.norm(dim=2, keepdim=True).clamp_min(_EPS)


def _block_gs2(sa, sb, seg_offsets):
    """Per-(batch, unit) Gram-Schmidt of two block-local vectors.

    Returns (q1, q2, nu1, delta, nu2) with, for every (b, i),
        sa = nu1 * q1,   sb = delta * q1 + nu2 * q2,
        ||q1|| = ||q2|| = 1,  q1 . q2 = 0   (all exact).
    Degenerate rows (sa = 0, or sb parallel to q1) take a canonical unit
    vector and a zero scale, so the reconstructed product is unaffected --
    the convention the manuscript already uses for ||s_{i.}|| = 0.
    """
    nu1 = sa.norm(dim=2)                                              # (B,n)
    q1 = sa / nu1.clamp_min(_EPS).unsqueeze(2)
    bad1 = nu1 <= _EPS
    if bool(bad1.any()):
        e0 = torch.zeros_like(sa)
        e0[:, :, 0] = 1.0
        q1 = torch.where(bad1.unsqueeze(2), e0, q1)
        nu1 = torch.where(bad1, torch.zeros_like(nu1), nu1)

    delta = (sb * q1).sum(dim=2)                                      # (B,n)
    w = sb - delta.unsqueeze(2) * q1
    nu2 = w.norm(dim=2)
    q2 = w / nu2.clamp_min(_EPS).unsqueeze(2)
    bad2 = nu2 <= _EPS * sb.norm(dim=2).clamp_min(1.0)
    if bool(bad2.any()):
        q2 = torch.where(bad2.unsqueeze(2), _canonical_orth(q1, seg_offsets), q2)
        nu2 = torch.where(bad2, torch.zeros_like(nu2), nu2)
    return q1, q2, nu1, delta, nu2


def _row_normalise(s, eps=_EPS):
    """Vanilla factorisation: s = D_s Q with Q's rows of unit norm."""
    ds = s.norm(dim=2)
    q = s / ds.clamp_min(eps).unsqueeze(2)
    bad = ds <= eps
    if bool(bad.any()):
        e0 = torch.zeros_like(s)
        e0[:, :, 0] = 1.0
        q = torch.where(bad.unsqueeze(2), e0, q)
        ds = torch.where(bad, torch.zeros_like(ds), ds)
    return ds, q


# ---------------------------------------------------------------------------
# base class
# ---------------------------------------------------------------------------
class GatedCellBase(nn.Module):
    """Shared parameter storage, layout and gradient plumbing."""

    gate_names = ()
    state_mult = 1

    def __init__(self, n_in, n_hid, device=None, dtype=torch.float32,
                 w_init="default", spectral_clip=0.0):
        super().__init__()
        G = len(self.gate_names)
        self.G = G
        self.n, self.m = n_hid, n_in
        self.p_gate = n_hid + n_in + 1
        self.p_cell = G * self.p_gate
        self.p = self.p_cell          # so that `P = cell.n * cell.p` is right
        self.n_state = self.state_mult * n_hid
        self.P = n_hid * self.p_cell
        self.spectral_clip = spectral_clip
        self.seg_offsets = [a * self.p_gate for a in range(G)]

        k = 1.0 / math.sqrt(n_hid)
        U = torch.empty(G, n_hid, n_hid, device=device, dtype=dtype).uniform_(-k, k)
        if w_init == "orthogonal":
            for a in range(G):
                torch.nn.init.orthogonal_(U[a])
        self.U = nn.Parameter(U)
        self.W = nn.Parameter(
            torch.empty(G, n_hid, n_in, device=device, dtype=dtype).uniform_(-k, k))
        self.b = nn.Parameter(torch.zeros(G, n_hid, device=device, dtype=dtype))

        # state row -> parameter block
        idx = torch.arange(n_hid, device=device)
        self.register_buffer("blk_idx", idx.repeat(self.state_mult), persistent=False)
        self._stamp = 0
        self._cache = None

    # -- gate index by name -------------------------------------------------
    def gate(self, name):
        return self.gate_names.index(name)

    @property
    def append_width(self):
        """omega: number of columns `append_factors` returns.

        The append budget c must be capped at this, and the Corollary-3 exact
        endpoint is (r, c) = (n_state, omega): (n, n) for the vanilla cell,
        (n, 2n) for the GRU, (2n, n) for the LSTM.
        """
        raise NotImplementedError

    def init_state(self, batch):
        return torch.zeros(batch, self.n_state,
                           device=self.U.device, dtype=self.U.dtype)

    def _features(self, x, h_prev):
        """phi_t = [h_{t-1}; x_t; 1], the row driving every gate."""
        ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        return torch.cat([h_prev, x, ones], dim=1)

    def _preacts(self, x, h_prev):
        """(B, G, n) pre-activations of all gates from the same h_{t-1}."""
        return (torch.einsum("bj,gij->bgi", h_prev, self.U)
                + torch.einsum("bj,gij->bgi", x, self.W)
                + self.b.unsqueeze(0))

    # -- structure ----------------------------------------------------------
    def blkdiag_of_A(self, A):
        raise NotImplementedError

    def offdiag_of_A(self, A):
        return A - self.blkdiag_of_A(A)

    def trace_update(self, A, S_prev, imm_diag):
        raise NotImplementedError

    def append_factors(self, A, S_prev):
        raise NotImplementedError

    @torch.no_grad()
    def append_mass(self, A, S_prev):
        """||N_t||_F exactly, without densifying anything.

        Because the append right factor has exactly orthonormal columns
        (Prop. 3 / Prop. 7), ||C_t Qt_t^T||_F = ||C_t||_F. This is the
        quantity the r=0 endpoint must record as eta_t: it already includes
        the dropped immediate mass ||I_t^perp||_F (Remark 3), so the
        certificate covers the extra bias a block-diagonal SnAp-1 incurs on a
        gated cell.
        """
        C, _ = self.append_factors(A, S_prev)
        return torch.linalg.matrix_norm(C, ord="fro", dim=(1, 2))

    def imm_offblock_norm(self):
        """||I_t^perp||_F for the step whose jac_pieces was called last."""
        raise NotImplementedError

    def imm_offblock_factors(self):
        """Exact factorisation of I_t^perp, or None when it vanishes.

        Returns (M, F_rows, blocks) meaning
            I_t^perp[b, i, block(blocks[j])] += M[b, i, j] * F_rows[b, j, :],
        i.e. I_t^perp = M @ F with F block-supported -- rank <= n, the same
        shape as the trace factor. `append_factors` consumes this internally;
        it is exposed because the exact shadow (ExactRTRL) needs the FULL
        immediate Jacobian, not just the block-diagonal part.
        """
        raise NotImplementedError

    @torch.no_grad()
    def imm_dense(self, imm_diag):
        """(B, n_state, P) full immediate Jacobian = I_t^o + I_t^perp.

        O(n^2 p_cell) in memory, so this is for the exact shadow and for
        tests -- never for the streaming path, which only ever needs the
        compact `imm_diag` plus `append_factors`.
        """
        B = imm_diag.shape[0]
        out = imm_diag.new_zeros(B, self.n_state, self.n, self.p_cell)
        s = torch.arange(self.n_state, device=imm_diag.device)
        out[:, s, self.blk_idx[s], :] = imm_diag
        fac = self.imm_offblock_factors()
        if fac is not None:
            M, F_rows, blocks = fac
            out.index_add_(2, blocks, torch.einsum("bij,bjq->bijq", M, F_rows))
        return out.reshape(B, self.n_state, self.P)

    # -- gradient plumbing --------------------------------------------------
    @torch.no_grad()
    def apply_flat_grad(self, g_rows):
        """g_rows (n, p_cell) -> accumulate into .grad of U, W, b."""
        n, m, pg = self.n, self.m, self.p_gate
        v = g_rows.view(n, self.G, pg)
        for par, val in ((self.U, v[:, :, :n].permute(1, 0, 2)),
                         (self.W, v[:, :, n:n + m].permute(1, 0, 2)),
                         (self.b, v[:, :, n + m].permute(1, 0))):
            if par.grad is None:
                par.grad = torch.zeros_like(par)
            par.grad += val

    @torch.no_grad()
    def flat_grad_rows(self):
        """(n, p_cell) view of the autograd gradient of the recurrent params.

        Exact inverse of `apply_flat_grad`; the gated analogue of
        `run_m3.flat_grad_rows`, which hard-codes TanhRNNCell's three tensors.
        Recurrent block first, then input, then bias -- the row order of
        `cells.py`.
        """
        n, m, pg = self.n, self.m, self.p_gate
        out = self.U.new_zeros(n, self.p_cell)
        v = out.view(n, self.G, pg)
        v[:, :, :n] = self.U.grad.permute(1, 0, 2)
        v[:, :, n:n + m] = self.W.grad.permute(1, 0, 2)
        v[:, :, n + m] = self.b.grad.permute(1, 0)
        return out

    @torch.no_grad()
    def spectral_clip_(self, clip=None):
        """Clip each recurrent matrix U_a separately to spectral norm `clip`.

        All of U_z/U_r/U_h (GRU) and U_i/U_f/U_g/U_o (LSTM) enter A_t, so
        clipping only the candidate-gate matrix would not control ||A_t||_2.
        """
        c = self.spectral_clip if clip is None else clip
        if not c or c <= 0:
            return
        for a in range(self.G):
            s = torch.linalg.matrix_norm(self.U[a], ord=2)
            if s > c:
                self.U[a].mul_(c / s)

    # requested spelling, plus the spelling train.py already calls
    def spectral_clip_apply(self, clip=None):
        self.spectral_clip_(clip)

    def clip_spectral(self):
        self.spectral_clip_(None)

    # -- cache discipline ---------------------------------------------------
    def _put_cache(self, **kw):
        self._stamp += 1
        kw["_stamp"] = self._stamp
        self._cache = kw

    def _get_cache(self):
        if self._cache is None:
            raise RuntimeError(
                f"{type(self).__name__}: call jac_pieces() for this time step "
                "before append_factors()/imm_offblock_norm().")
        return self._cache


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------
class GRUCell(GatedCellBase):
    """h_t = (1-z_t) (.) h_{t-1} + z_t (.) h~_t   (the D5 / R2 plan convention)

        z_t  = sigmoid(U_z h_{t-1} + W_z x_t + b_z)
        r_t  = sigmoid(U_r h_{t-1} + W_r x_t + b_r)
        h~_t = tanh(U_h (r_t (.) h_{t-1}) + W_h x_t + b_h)

    The other common convention h_t = z (.) h_{t-1} + (1-z) (.) h~ is this one
    with z -> 1-z; it flips the sign of d^z_t and swaps diag(1-z) for diag(z)
    in A_t, and changes nothing structural.
    """

    gate_names = ("z", "r", "h")
    state_mult = 1

    @property
    def append_width(self):
        return 2 * self.n          # Ahat_t S_{t-1} plus the rank-<=n I_t^perp

    # -- forward ------------------------------------------------------------
    def _gates(self, x, h_prev):
        a = self._preacts(x, h_prev)                           # (B,3,n)
        z = torch.sigmoid(a[:, 0])
        r = torch.sigmoid(a[:, 1])
        # the candidate reads the GATED previous state
        ah = (torch.einsum("bj,ij->bi", r * h_prev, self.U[2])
              + x @ self.W[2].T + self.b[2])
        hh = torch.tanh(ah)
        return z, r, hh

    def forward(self, x, h_prev):
        z, r, hh = self._gates(x, h_prev)
        return (1.0 - z) * h_prev + z * hh

    # -- Jacobians (Prop. 1, Prop. 2) ---------------------------------------
    @torch.no_grad()
    def jac_pieces(self, x, h_prev, h=None):
        n, m, pg = self.n, self.m, self.p_gate
        Uz, Ur, Uh = self.U[0], self.U[1], self.U[2]
        z, r, hh = self._gates(x, h_prev)
        zp = z * (1.0 - z)
        rp = r * (1.0 - r)
        hp = 1.0 - hh * hh

        # A_t = dh_t/dh_{t-1}, Prop. 2 -- three paths, dense
        dz_dh = zp.unsqueeze(2) * Uz                                  # (B,n,n)
        dr_dh = rp.unsqueeze(2) * Ur
        dRH = torch.diag_embed(r) + h_prev.unsqueeze(2) * dr_dh       # d(r*h)/dh
        dhh_dh = hp.unsqueeze(2) * (Uh @ dRH)
        A = (torch.diag_embed(1.0 - z)
             + (hh - h_prev).unsqueeze(2) * dz_dh
             + z.unsqueeze(2) * dhh_dh)

        # immediate Jacobian, Prop. 1
        phi = self._features(x, h_prev)                               # (B,p_gate)
        psi = torch.cat([r * h_prev, x,
                         torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)],
                        dim=1)
        dz_c = (hh - h_prev) * zp                                     # d^z_t
        dh_c = z * hp                                                 # d^h_t
        K = dh_c.unsqueeze(2) * Uh * (rp * h_prev).unsqueeze(1)       # (B,n,n)

        imm = torch.zeros(x.shape[0], n, self.p_cell,
                          device=x.device, dtype=x.dtype)
        v = imm.view(x.shape[0], n, self.G, pg)
        v[:, :, 0, :] = dz_c.unsqueeze(2) * phi.unsqueeze(1)
        v[:, :, 1, :] = torch.diagonal(K, dim1=1, dim2=2).unsqueeze(2) * phi.unsqueeze(1)
        v[:, :, 2, :] = dh_c.unsqueeze(2) * psi.unsqueeze(1)

        Khat = K - torch.diag_embed(torch.diagonal(K, dim1=1, dim2=2))
        self._put_cache(phi=phi, Khat=Khat)
        return A, imm

    # -- structure ----------------------------------------------------------
    def blkdiag_of_A(self, A):
        return torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))

    @torch.no_grad()
    def trace_update(self, A, S_prev, imm_diag):
        """S_t = diag(A_t) (.) S_{t-1} + I_t^o -- unchanged in form."""
        return torch.diagonal(A, dim1=1, dim2=2).unsqueeze(2) * S_prev + imm_diag

    @torch.no_grad()
    def append_factors(self, A, S_prev):
        """N_t = Ahat_t S_{t-1} + I_t^perp = C_t Qt_t^T exactly (Prop. 3).

        C_t = [ Ahat_t | Khat_t ] T_t  (B, n, 2n), Qt_t of width 2n with
        exactly orthonormal, single-block columns.
        """
        Khat, f_row, _ = self.imm_offblock_factors()
        n = self.n
        Ahat = self.offdiag_of_A(A)
        q1, q2, nu1, delta, nu2 = _block_gs2(S_prev, f_row, self.seg_offsets)

        # C = [Ahat | Khat] T  with T supported on (i,i), (n+i,i), (n+i,n+i)
        C = torch.cat([Ahat * nu1.unsqueeze(1) + Khat * delta.unsqueeze(1),
                       Khat * nu2.unsqueeze(1)], dim=2)               # (B,n,2n)
        rows = torch.cat([q1, q2], dim=1)                             # (B,2n,p_cell)
        blocks = torch.cat([self.blk_idx[:n], self.blk_idx[:n]])
        return C, StructuredQ(rows, blocks, self.p_cell, n)

    @torch.no_grad()
    def imm_offblock_factors(self):
        """I_t^perp = Khat_t @ F_t, with F_t's row j carrying phi_t in the
        r-segment of block j (the same block-local vector for every unit)."""
        cache = self._get_cache()
        phi, Khat = cache["phi"], cache["Khat"]
        B, n, pg = phi.shape[0], self.n, self.p_gate
        F = torch.zeros(B, n, self.p_cell, device=phi.device, dtype=phi.dtype)
        F.view(B, n, self.G, pg)[:, :, 1, :] = phi.unsqueeze(1)
        return Khat, F, self.blk_idx[:n].clone()

    @torch.no_grad()
    def imm_offblock_norm(self):
        """||I_t^perp||_F = ||Khat_t||_F ||phi_t||_2, exact and O(n^2).

        Rows of I_t^perp sit in disjoint parameter blocks and every block
        carries the same phi_t, so the Frobenius norm factorises.
        """
        cache = self._get_cache()
        return (torch.linalg.matrix_norm(cache["Khat"], ord="fro", dim=(1, 2))
                * cache["phi"].norm(dim=1))


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
class LSTMCell(GatedCellBase):
    """State x_t = [h_t; c_t] in R^{2n}, gates (i, f, g~, o), no peepholes.

        c_t = f_t (.) c_{t-1} + i_t (.) g~_t,     h_t = o_t (.) tanh(c_t)

    Immediate Jacobian is block diagonal (Prop. 5): the two state rows h_i and
    c_i share unit i's 4p parameter columns. The trace needs the per-unit 2x2
    blocks of A_t (Prop. 6 / Remark 4); the append collapses to width n
    because Ahat_t has zero c-columns (Prop. 7).
    """

    gate_names = ("i", "f", "g", "o")
    state_mult = 2

    @property
    def append_width(self):
        return self.n              # Ahat_t kills the c-columns (Prop. 7)

    def split(self, state):
        n = self.n
        return state[:, :n], state[:, n:]

    def _gates(self, x, state_prev):
        h_prev, c_prev = self.split(state_prev)
        a = self._preacts(x, h_prev)                                  # (B,4,n)
        ii = torch.sigmoid(a[:, 0])
        ff = torch.sigmoid(a[:, 1])
        gg = torch.tanh(a[:, 2])
        oo = torch.sigmoid(a[:, 3])
        c = ff * c_prev + ii * gg
        return ii, ff, gg, oo, c

    def forward(self, x, state_prev):
        ii, ff, gg, oo, c = self._gates(x, state_prev)
        return torch.cat([oo * torch.tanh(c), c], dim=1)

    # -- Jacobians (Prop. 5, Prop. 6) ---------------------------------------
    @torch.no_grad()
    def jac_pieces(self, x, state_prev, state=None):
        n, m, pg = self.n, self.m, self.p_gate
        Ui, Uf, Ug, Uo = self.U[0], self.U[1], self.U[2], self.U[3]
        h_prev, c_prev = self.split(state_prev)
        ii, ff, gg, oo, c = self._gates(x, state_prev)
        ip, fp, op = ii * (1 - ii), ff * (1 - ff), oo * (1 - oo)
        gp = 1.0 - gg * gg
        th = torch.tanh(c)
        thp = 1.0 - th * th

        # A_t on [h; c] -- h-columns dense, c-columns exactly diagonal
        G_t = ((c_prev * fp).unsqueeze(2) * Uf
               + (gg * ip).unsqueeze(2) * Ui
               + (ii * gp).unsqueeze(2) * Ug)                         # dc/dh
        dh_dh = (oo * thp).unsqueeze(2) * G_t + (th * op).unsqueeze(2) * Uo
        dh_dc = torch.diag_embed(oo * thp * ff)
        dc_dc = torch.diag_embed(ff)
        A = torch.cat([torch.cat([dh_dh, dh_dc], dim=2),
                       torch.cat([G_t, dc_dc], dim=2)], dim=1)        # (B,2n,2n)

        # immediate Jacobian: per-gate coefficients on c_t, then on h_t
        phi = self._features(x, h_prev)
        gam_c = torch.stack([gg * ip, c_prev * fp, ii * gp,
                             torch.zeros_like(ii)], dim=1)            # (B,4,n)
        gam_h = torch.stack([(oo * thp) * gam_c[:, 0],
                             (oo * thp) * gam_c[:, 1],
                             (oo * thp) * gam_c[:, 2],
                             th * op], dim=1)                         # (B,4,n)
        imm = torch.zeros(x.shape[0], self.n_state, self.p_cell,
                          device=x.device, dtype=x.dtype)
        vh = imm[:, :n, :].view(x.shape[0], n, self.G, pg)
        vc = imm[:, n:, :].view(x.shape[0], n, self.G, pg)
        vh.copy_(gam_h.permute(0, 2, 1).unsqueeze(3) * phi[:, None, None, :])
        vc.copy_(gam_c.permute(0, 2, 1).unsqueeze(3) * phi[:, None, None, :])

        self._put_cache(phi=phi)
        return A, imm

    # -- structure ----------------------------------------------------------
    def blkdiag_of_A(self, A):
        """The per-unit 2x2 blocks on rows/cols {i, n+i} (Prop. 6)."""
        n = self.n
        out = torch.zeros_like(A)
        i = torch.arange(n, device=A.device)
        out[:, i, i] = A[:, i, i]
        out[:, i, n + i] = A[:, i, n + i]
        out[:, n + i, i] = A[:, n + i, i]
        out[:, n + i, n + i] = A[:, n + i, n + i]
        return out

    @torch.no_grad()
    def trace_update(self, A, S_prev, imm_diag):
        """Per-unit 2x2 product, NOT diag(A_t) (.) S_{t-1}.

        The scalar-diagonal update would drop (A_t)_{i,n+i} = o (.) (1-th^2)
        (.) f, the intra-unit c->h memory path, which is O(1) with the forget
        gate open.
        """
        n = self.n
        i = torch.arange(n, device=A.device)
        Sh, Sc = S_prev[:, :n, :], S_prev[:, n:, :]
        out_h = A[:, i, i].unsqueeze(2) * Sh + A[:, i, n + i].unsqueeze(2) * Sc
        out_c = A[:, n + i, i].unsqueeze(2) * Sh + A[:, n + i, n + i].unsqueeze(2) * Sc
        return torch.cat([out_h, out_c], dim=1) + imm_diag

    @torch.no_grad()
    def append_factors(self, A, S_prev, mode="omega_n"):
        """N_t = Ahat_t S_{t-1} = (Ahat_t^h D_s) Q_t^T (Prop. 7), width n.

        `mode="omega_2n"` is the peephole-ready route: a per-block QR of the
        two rows (S^h_{t-1,i}, S^c_{t-1,i}), width 2n. Both reproduce N_t
        exactly; with no peepholes the c-columns of Ahat_t vanish, so the
        omega_n route is the one to run.
        """
        n = self.n
        Ahat = self.offdiag_of_A(A)
        Sh, Sc = S_prev[:, :n, :], S_prev[:, n:, :]

        if mode == "omega_n":
            ds, q = _row_normalise(Sh)
            C = Ahat[:, :, :n] * ds.unsqueeze(1)                      # (B,2n,n)
            return C, StructuredQ(q, self.blk_idx[:n].clone(), self.p_cell, n)

        if mode == "omega_2n":
            q1, q2, nu1, delta, nu2 = _block_gs2(Sh, Sc, self.seg_offsets)
            Ah, Ac = Ahat[:, :, :n], Ahat[:, :, n:]
            C = torch.cat([Ah * nu1.unsqueeze(1) + Ac * delta.unsqueeze(1),
                           Ac * nu2.unsqueeze(1)], dim=2)             # (B,2n,2n)
            rows = torch.cat([q1, q2], dim=1)
            blocks = torch.cat([self.blk_idx[:n], self.blk_idx[:n]])
            return C, StructuredQ(rows, blocks, self.p_cell, n)

        raise ValueError(mode)

    @torch.no_grad()
    def append_c_column_mass(self, A):
        """max |Ahat_t[:, c-columns]|; must be 0 without peepholes (Prop. 7).

        Diagnostic: if a peephole variant is added this stops being zero and
        `append_factors(mode="omega_n")` would silently drop mass, so the test
        suite checks it rather than trusting the comment.
        """
        return self.offdiag_of_A(A)[:, :, self.n:].abs().amax(dim=(1, 2))

    def imm_offblock_factors(self):
        """None: the LSTM immediate Jacobian is exactly block diagonal (Prop. 5)."""
        self._get_cache()
        return None

    @torch.no_grad()
    def imm_offblock_norm(self):
        """Zero: the LSTM immediate Jacobian is exactly block diagonal."""
        cache = self._get_cache()
        return cache["phi"].new_zeros(cache["phi"].shape[0])


# ---------------------------------------------------------------------------
def make_gated_cell(name, n_in, n_hid, **kw):
    cells = {"gru": GRUCell, "lstm": LSTMCell}
    key = name.lower()
    if key not in cells:
        raise ValueError(f"unknown gated cell {name!r}; have {sorted(cells)}")
    return cells[key](n_in, n_hid, **kw)
