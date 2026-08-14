"""Amortised-rotation kernel for SK-RTRL (paper Sec. 4.3, "Amortised rotation").

The kernel of record in :mod:`skrtrl.algos` applies the rank-r rotation of
Eq. (7) eagerly::

    R_t <- [ R_{t-1} | Qperp_t ] W_t ,

which touches the P x r factor (P ~ n^2) once per step and therefore costs
O(P (r+c) r) = O(n^2 r^2).  On top of that it materialises the P x c dense
append Qc and runs a P x c thin QR, two more O(n^2 r^2) / O(n^2 r c) terms.

This module implements the *deferred* variant.  Instead of rotating R every
step we keep

    R_t = Z_t M_t ,   Z_t = [ Z0 | Q^(1) | ... | Q^(k) ] ,   Gam_t = Z_t^T Z_t ,

where Z0 is a dense P x r base, each Q^(s) is a *buffered append block* stored
in its exact structured form (one nonzero p-block per row of the SnAp-1 part,
see below), and M_t is a small w x r coefficient matrix, w = r + sum_s c_s.
All per-step work then happens either on n x small matrices or on P x r data
that the naive kernel already touches, and the P-sized rotation is collapsed
onto Z0 only once every K = Theta(r) steps.

Why the buffer is cheap.  The append factor is
``Qc[(i,j), b] = Snorm[i,j] * Vc[i,b]`` (row block i of Qc is the outer product
of row i of the normalised SnAp-1 state with row i of the pre-projector), so a
block is stored as the pair ``(Snorm (B,n,p), Vc (B,n,c))`` -- O(n p) = O(n^2)
instead of the O(n^2 c) dense form.  Consequently

  * ``Q^(s)^T Q^(t) = Vc_s^T diag(d_st) Vc_t`` with
    ``d_st[i] = <Snorm_s[i,:], Snorm_t[i,:]>``, costing O(n p + n c^2)
    ("diagonal by disjoint supports" in the paper: for the Cor.-3 endpoint
    Vc = I and the cross-Gram is literally diagonal);
  * applying a buffered block to a small matrix costs O(n p q), independent of c;
  * the collapse costs O(P r^2 + K n p r) = O(n^2 r^2) once per K = Theta(r)
    steps, i.e. O(n^2 r) amortised, with an O(K n p) = O(n^2 r) transient buffer.

Exactness.  In exact arithmetic this kernel returns the same L_t and R_t as the
naive one: it re-associates the same products.  The orthogonalisation of the
append against R is done in the Gram metric (Cholesky-free, via ``eigh`` of the
c x c matrix ``(Qc - R G)^T (Qc - R G) = E^T Gam' E``) rather than by a thin QR
of the P x c residual.  The resulting Theta differs from the QR one by a left
orthogonal factor O (Theta_qr = O Theta_eig); since the core is
``[L' + B G^T | B Theta^T]`` and the basis is ``[R | Qperp]``, the same O cancels
between the two, leaving core singular values, L_t, R_t, eta_t and ghat_t
identical.  See ``tests/test_amortised_kernel.py``.

Numerics.  The coefficient-space algebra (Gam, M, z, G, Theta) is tiny but
involves the cancellation ``Qc - R (R^T Qc)``, so it is carried in float64 by
default even when the estimator itself runs in float32 (``hp_bookkeeping=True``).
"""
import math

import torch

from .algos import SKRTRL, _diag_of, _robust_svd


# --------------------------------------------------------------------------
# structured-block primitives
# --------------------------------------------------------------------------
def _cross_gram(Sn_s, Vc_s, Sn_t, Vc_t):
    """Q^(s)^T Q^(t) for two structured append blocks -> (B, c_s, c_t).

    Q^(s)[(i,j), a] = Sn_s[i,j] Vc_s[i,a]  =>  the (a,b) entry is
    sum_i <Sn_s[i,:], Sn_t[i,:]> Vc_s[i,a] Vc_t[i,b].  Cost O(n p + n c_s c_t).
    """
    d = (Sn_s * Sn_t).sum(dim=2)                              # (B, n)
    return torch.einsum("zia,zi,zib->zab", Vc_s, d, Vc_t)


def _dense_T_block(Z0, Sn, Vc, n, p):
    """Z0^T Qc for a dense Z0 (B, P, q) -> (B, q, c).  Cost O(n p q + n q c)."""
    B, P, q = Z0.shape
    if q == 0:
        return Z0.new_zeros(B, 0, Vc.shape[2])
    G0 = torch.einsum("bipq,bip->bqi", Z0.view(B, n, p, q), Sn)   # (B, q, n)
    return torch.bmm(G0, Vc)


# The buffer is kept as two stacked tensors so that the k deferred blocks are
# consumed by *one* batched kernel instead of k small ones; with a Python loop
# the deferral is launch-bound and loses to collapsing every step.
def _buf_cross_gram(bufS, bufV, Sn, Vc):
    """[Q^(1..k)]^T Qc -> (B, k*c_s, c).  Cost O(k n p + k n c_s c)."""
    d = torch.einsum("bkip,bip->bki", bufS, Sn)               # (B, k, n)
    out = torch.einsum("zkia,zki,zib->zkab", bufV, d, Vc)     # (B, k, c_s, c)
    B, k, cs, c = out.shape
    return out.reshape(B, k * cs, c)


def _buf_apply(bufS, bufV, X):
    """[Q^(1..k)] @ X with X (B, k*c_s, q) -> (B, P, q).  Cost O(k n p q)."""
    B, k, n, cs = bufV.shape
    p = bufS.shape[3]
    q = X.shape[2]
    T = torch.einsum("bkia,bkaq->bkiq", bufV, X.reshape(B, k, cs, q))
    return torch.einsum("bkip,bkiq->bipq", bufS, T).reshape(B, n * p, q)


class SKRTRLAmortised(SKRTRL):
    """SK-RTRL with the deferred / amortised rotation kernel.

    Drop-in replacement for :class:`skrtrl.algos.SKRTRL`: same constructor
    signature plus ``collapse_every`` (K, defaults to Theta(r)) and
    ``max_buffer``.  ``r``/``c`` may be changed between steps exactly as for the
    naive kernel (adaptive-rank controller).

    The dense R is never stored; use :meth:`R_dense` if it is needed, and
    :meth:`reset_lanes` (not ``algo.R[mask] = 0``) to clear a batch lane.
    """

    name = "skrtrl-amort"

    def __init__(self, cell, batch, r: int, c: int | None = None, mode: str = "svd",
                 collapse_every: int | None = None, max_buffer: int | None = None,
                 width_mult: int = 4, hp_bookkeeping: bool = True):
        self._collapse_every = collapse_every
        self._max_buffer = max_buffer
        self.width_mult = width_mult
        self.hp_bookkeeping = hp_bookkeeping
        super().__init__(cell, batch, r, c=c, mode=mode)      # calls reset()

    # ---------------- bookkeeping dtype ----------------
    @property
    def _bdt(self):
        dt = self.cell.W.dtype
        if self.hp_bookkeeping and dt in (torch.float32, torch.float16, torch.bfloat16):
            return torch.float64
        return dt

    @property
    def K(self):
        """Collapse period.

        The paper's construction is K = Theta(r), which makes the collapse
        (O(P r^2 + K n p r)) cost O(n^2 r) per step.  With the paper's append
        budget c = Theta(r) that also makes the *deferred basis width*
        w = r + K c = Theta(r^2), and the coefficient-space algebra (the w x w
        Gram) then costs O(w^2 c) = O(r^5) per step -- n-independent, but the
        dominant term as soon as r^4 > n^2.  We therefore additionally cap K so
        that w <= (1 + width_mult) r, which is what the benchmark uses; pass
        ``width_mult=0`` (or an explicit ``collapse_every``) to get the
        uncapped Theta(r) schedule.
        """
        if self._collapse_every is not None:
            return max(1, int(self._collapse_every))
        r = max(1, self.r)
        c = max(1, min(self.c, self.n))
        k = max(1, r // 2)                       # Theta(r)
        if self.width_mult:
            k = min(k, max(1, (self.width_mult * r) // c))
        if self._max_buffer:
            k = min(k, int(self._max_buffer))
        return k

    # ---------------- state ----------------
    def reset(self):
        W = self.cell.W
        dev, dt, bdt = W.device, W.dtype, self._bdt
        B, n, p, P, r = self.B, self.n, self.p, self.P, self.r
        self.S = torch.zeros(B, n, p, device=dev, dtype=dt)
        self.L = torch.zeros(B, n, r, device=dev, dtype=dt)
        # R = Z M with Z = [Z0 | buffered blocks];  start from R = 0.
        self.Z0 = torch.zeros(B, P, r, device=dev, dtype=dt)
        self.M = torch.zeros(B, r, r, device=dev, dtype=bdt)
        self.Gam = torch.zeros(B, r, r, device=dev, dtype=bdt)
        self._bufS = None               # (B, Kmax, n, p)   buffered Snorm
        self._bufV = None               # (B, Kmax, n, c)   buffered Vc
        self._nbuf = 0
        self.e = torch.zeros(B, device=dev, dtype=dt)
        self.last = {}
        self.n_collapse = 0

    # ---------------- buffer management ----------------
    def _alloc_buf(self, cc, K):
        W = self.cell.W
        self._bufS = torch.empty(self.B, K, self.n, self.p, device=W.device, dtype=W.dtype)
        self._bufV = torch.empty(self.B, K, self.n, cc, device=W.device, dtype=W.dtype)
        self._nbuf = 0

    def _ensure_buf(self, cc):
        """(Re)allocate the buffer if the block width or the period changed."""
        K = self.K
        stale = (self._bufS is None or self._bufS.shape[1] != K
                 or self._bufV.shape[3] != cc)
        if stale:
            if self._nbuf > 0:
                self._collapse()        # flush before changing the layout
            self._alloc_buf(cc, K)

    # ---------------- Z-side helpers ----------------
    def _Z_apply(self, X):
        """Z @ X for a coefficient matrix X (B, w, q) -> dense (B, P, q)."""
        B = self.B
        X = X.to(self.Z0.dtype)
        r0 = self.Z0.shape[2]
        out = torch.bmm(self.Z0, X[:, :r0, :]) if r0 else \
            self.Z0.new_zeros(B, self.P, X.shape[2])
        if self._nbuf:
            out = out + _buf_apply(self._bufS[:, :self._nbuf], self._bufV[:, :self._nbuf],
                                   X[:, r0:, :].contiguous())
        return out

    def _ZT_Qc(self, Sn, Vc):
        """Z^T Qc -> (B, w, c)."""
        parts = [_dense_T_block(self.Z0, Sn, Vc, self.n, self.p)]
        if self._nbuf:
            parts.append(_buf_cross_gram(self._bufS[:, :self._nbuf],
                                         self._bufV[:, :self._nbuf], Sn, Vc))
        return torch.cat(parts, dim=1).to(self._bdt)

    def R_dense(self):
        """Materialise the P x r right factor (testing / diagnostics only).

        Deliberately *not* exposed as an ``R`` attribute: the dense factor does
        not exist in this kernel, and an in-place write such as
        ``algo.R[mask] = 0`` (see ``train.OnlineLearner._reset_lanes``) would
        silently touch a temporary.  Lane resets go through :meth:`reset_lanes`,
        which ``train.py`` already calls when the estimator provides it.
        """
        return self._Z_apply(self.M)

    @torch.no_grad()
    def reset_lanes(self, mask):
        """Zero the sketch of the batch lanes in ``mask`` (episode boundaries).

        Zeroing Z0, M and the buffered blocks of a lane makes R = Z M = 0 and
        Gam = Z^T Z = 0 for that lane, i.e. exactly the naive kernel's R = 0.
        """
        self.Z0[mask] = 0.0
        self.M[mask] = 0.0
        self.Gam[mask] = 0.0
        if self._bufS is not None and self._nbuf:
            self._bufS[mask, :self._nbuf] = 0.0
            self._bufV[mask, :self._nbuf] = 0.0

    def _collapse(self):
        """Fold the buffered rotations onto Z0.  O(P r^2 + K n p r)."""
        M = self.M
        Gam_new = torch.bmm(torch.bmm(M.transpose(1, 2), self.Gam), M)
        Z0_new = self._Z_apply(M)
        q = M.shape[2]
        self.Z0 = Z0_new
        self.M = torch.eye(q, device=M.device, dtype=M.dtype).expand(self.B, q, q).contiguous()
        self.Gam = Gam_new
        self._nbuf = 0
        self.n_collapse += 1

    # ---------------- the step ----------------
    @torch.no_grad()
    def step_state(self, A, imm):
        B, n, p = self.B, self.n, self.p
        r = self.r
        bdt = self._bdt
        r_in = self.L.shape[2] if self.L.numel() else 0
        S_prev = self.S
        diagA = _diag_of(A)
        Ahat = A - torch.diag_embed(diagA)

        # --- certified norm bound (identical to the naive kernel) ---
        nF = torch.linalg.matrix_norm(A, ord="fro", dim=(1, 2))
        n1 = A.abs().sum(dim=1).max(dim=1).values
        ninf = A.abs().sum(dim=2).max(dim=1).values
        rho_bar = torch.minimum(nF, torch.sqrt(n1 * ninf))
        if not hasattr(self, "_pv") or self._pv.shape[0] != B:
            self._pv = torch.randn(B, n, 1, device=A.device, dtype=A.dtype)
        Av = torch.bmm(A, self._pv)
        rho_hat = Av.norm(dim=(1, 2)) / self._pv.norm(dim=(1, 2)).clamp_min(1e-30)
        self._pv = Av / Av.norm(dim=(1, 2), keepdim=True).clamp_min(1e-30)

        # --- exact SnAp-1 part ---
        self.S = diagA.unsqueeze(2) * S_prev + imm

        if r == 0:
            eta = self._offdiag_mass(Ahat, S_prev)
            self.e = rho_bar * self.e + eta
            self.last = {"rho_bar": rho_bar, "rho_hat": rho_hat, "eta": eta}
            return

        # --- propagate residual left factor ---
        Lp = torch.bmm(A, self.L)                                  # (B, n, r_in)

        # --- new off-diagonal mass in exact factored form ---
        s_norm = S_prev.norm(dim=2)
        B0 = Ahat * s_norm.unsqueeze(1)
        nz = s_norm > 0
        Snorm = torch.where(nz.unsqueeze(2), S_prev / s_norm.clamp_min(1e-30).unsqueeze(2),
                            torch.zeros_like(S_prev))              # (B, n, p)

        tau_c = torch.zeros(B, device=A.device, dtype=A.dtype)
        if self.preproject:
            c = min(self.c, n)
            if self.mode == "randproj":
                Vc, _ = torch.linalg.qr(torch.randn(B, n, c, device=A.device, dtype=A.dtype))
                Bc = torch.bmm(B0, Vc)
                tau_c = torch.sqrt((torch.linalg.matrix_norm(B0, ord="fro", dim=(1, 2)) ** 2
                                    - torch.linalg.matrix_norm(Bc, ord="fro", dim=(1, 2)) ** 2).clamp_min(0))
            else:
                Ub, sb, Vbh = _robust_svd(B0)
                Bc = Ub[:, :, :c] * sb[:, :c].unsqueeze(1)
                Vc = Vbh[:, :c, :].transpose(1, 2)
                tau_c = torch.sqrt((sb[:, c:] ** 2).sum(dim=1).clamp_min(0))
        else:
            Bc = B0
            Vc = torch.eye(n, device=A.device, dtype=A.dtype).expand(B, n, n)
        cc = Vc.shape[2]

        # ---------------------------------------------------------------
        # deferred rotation: everything below is O(n p r) + O(small) --
        # no P x c dense append, no P x c QR, no P x (r+c) rotation.
        # ---------------------------------------------------------------
        self._ensure_buf(cc)
        z = self._ZT_Qc(Snorm, Vc)                                 # (B, w, cc)
        Om = _cross_gram(Snorm, Vc, Snorm, Vc).to(bdt)             # (B, cc, cc) = Qc^T Qc
        Mb = self.M                                                # (B, w, r_in)
        G = torch.bmm(Mb.transpose(1, 2), z)                       # (B, r_in, cc) = R^T Qc
        w = z.shape[1]

        # extended Gram of Z' = [Z | Qc]
        Gam2 = torch.cat([torch.cat([self.Gam, z], dim=2),
                          torch.cat([z.transpose(1, 2), Om], dim=2)], dim=1)   # (B, w+cc, w+cc)
        # coefficients of (Qc - R G) in the basis Z'
        E = torch.cat([-torch.bmm(Mb, G),
                       torch.eye(cc, device=A.device, dtype=bdt).expand(B, cc, cc)], dim=1)

        # Qperp = (Qc - R G) Theta^{-1} in the Gram metric
        Kmat = torch.bmm(torch.bmm(E.transpose(1, 2), Gam2), E)    # (B, cc, cc), PSD
        Kmat = 0.5 * (Kmat + Kmat.transpose(1, 2))
        lam, V = torch.linalg.eigh(Kmat)
        lam = lam.clamp_min(0)
        sq = lam.sqrt()
        tol = sq.amax(dim=1, keepdim=True) * (1e-12 if bdt == torch.float64 else 1e-6)
        inv = torch.where(sq > tol, 1.0 / sq.clamp_min(1e-300 if bdt == torch.float64 else 1e-30),
                          torch.zeros_like(sq))
        Theta = sq.unsqueeze(2) * V.transpose(1, 2)                # Theta^T Theta = Kmat
        Cperp = torch.bmm(E, V * inv.unsqueeze(1))                 # (B, w+cc, cc)

        Gt = G.to(A.dtype)
        Th = Theta.to(A.dtype)
        core = torch.cat([Lp + torch.bmm(Bc, Gt.transpose(1, 2)),
                          torch.bmm(Bc, Th.transpose(1, 2))], dim=2)   # (B, n, r_in+cc)

        Mpad = torch.cat([Mb, Mb.new_zeros(B, cc, r_in)], dim=1)       # (B, w+cc, r_in)
        Mcat = torch.cat([Mpad, Cperp], dim=2)                         # (B, w+cc, r_in+cc)

        if self.mode == "randproj" and self.preproject:
            rc = core.shape[2]
            Omr, _ = torch.linalg.qr(torch.randn(B, rc, r, device=A.device, dtype=A.dtype))
            self.L = torch.bmm(core, Omr)
            self.M = torch.bmm(Mcat, Omr.to(bdt))
            tau_r = torch.sqrt((torch.linalg.matrix_norm(core, ord="fro", dim=(1, 2)) ** 2
                                - torch.linalg.matrix_norm(self.L, ord="fro", dim=(1, 2)) ** 2).clamp_min(0))
        else:
            Uc_, sc_, Wch = _robust_svd(core)
            k = min(r, sc_.shape[1])
            self.L = Uc_[:, :, :k] * sc_[:, :k].unsqueeze(1)
            Wfac = Wch.transpose(1, 2)[:, :, :k]                       # (B, r_in+cc, k)
            self.M = torch.bmm(Mcat, Wfac.to(bdt))
            if k < r:                                                  # pad (early steps)
                self.L = torch.cat([self.L, self.L.new_zeros(B, n, r - k)], dim=2)
                self.M = torch.cat([self.M, self.M.new_zeros(B, w + cc, r - k)], dim=2)
            tau_r = torch.sqrt((sc_[:, k:] ** 2).sum(dim=1).clamp_min(0))

        self.Gam = Gam2
        self._bufS[:, self._nbuf] = Snorm
        self._bufV[:, self._nbuf] = Vc
        self._nbuf += 1
        if self._nbuf >= self._bufS.shape[1]:
            self._collapse()

        eta = tau_c + tau_r
        self.e = rho_bar * self.e + eta
        self.last = {"rho_bar": rho_bar, "rho_hat": rho_hat, "eta": eta,
                     "tau_c": tau_c, "tau_r": tau_r}

    # ---------------- read-out ----------------
    @torch.no_grad()
    def grad_rows(self, delta):
        g1 = delta.unsqueeze(2) * self.S
        r_in = self.L.shape[2] if self.L.numel() else 0
        if r_in > 0:
            u = torch.einsum("bn,bnr->br", delta, self.L)              # (B, r_in)
            v = torch.bmm(self.M, u.unsqueeze(2).to(self.M.dtype))     # (B, w, 1)
            g1 = g1 + self._Z_apply(v).view(self.B, self.n, self.p)
        return g1.mean(0)

    @torch.no_grad()
    def residual_dense(self):
        """(B, n, P) dense S + L R^T  (for testing only)."""
        idx = torch.arange(self.n, device=self.S.device)
        out = torch.bmm(self.L, self.R_dense().transpose(1, 2))
        outv = out.view(self.B, self.n, self.n, self.p)
        outv[:, idx, idx, :] += self.S
        return outv.view(self.B, self.n, self.P)


def make_skrtrl(cell, batch, r, c=None, mode="svd", kernel="naive", **kw):
    """Factory: ``kernel='naive'`` (implementation of record) or ``'amortised'``."""
    if kernel in ("naive", "eager", None):
        return SKRTRL(cell, batch, r=r, c=c, mode=mode)
    if kernel in ("amortised", "amortized", "amort"):
        return SKRTRLAmortised(cell, batch, r=r, c=c, mode=mode, **kw)
    raise ValueError(f"unknown kernel {kernel!r}")
