"""Online gradient algorithms over the influence matrix J_t = dh_t/dtheta.

All maintain per-batch-element estimator state; gradients are averaged over batch.
Flat layout: J (B, n, P) with P = n*p; column (i, j) -> i*p + j.

Implemented here: ExactRTRL, SKRTRL (r=0 reduces to SnAp-1).
Stochastic baselines (UORO, KF-RTRL) live in baselines.py.
"""
import math
import torch


def _diag_of(A):
    return torch.diagonal(A, dim1=1, dim2=2)  # (B, n)


# Diagnostic counters for the SVD fallback chain (read by the runners and written
# into the output json so a silent fallback can never be mistaken for a fast path).
#   default             library default driver (gesvdj on CUDA) -- only in "auto" mode
#   gesvd               cuSOLVER gesvd, first choice in "gesvd" mode
#   gesvd_after_default fallback: "auto" tried the default driver and it failed
#   jitter              fallback: SVD of a PERTURBED matrix (see the warning below)
#   cpu                 fallback: exact LAPACK SVD of the original matrix on CPU
SVD_STATS = {"default": 0, "gesvd": 0, "gesvd_after_default": 0, "jitter": 0, "cpu": 0}
FALLBACK_KEYS = ("gesvd_after_default", "jitter", "cpu")

# Test hooks (tests/test_svd_fallback_cert.py), never touched in production:
#   _FORCE_FAIL     names in this set make the corresponding attempt raise, so each
#                   link of the fallback chain can be exercised deterministically
#   _JITTER_SCALE   multiplies the jitter magnitude, so the A7 accounting bug can be
#                   made visible at a perturbation large enough to matter
_FORCE_FAIL = set()
_JITTER_SCALE = 1.0


def _svd_stats_reset():
    for k in SVD_STATS:
        SVD_STATS[k] = 0


def svd_fallbacks():
    """Counts of SVD calls that were NOT served by the first-choice driver."""
    return {k: SVD_STATS[k] for k in FALLBACK_KEYS}


def _finite(*ts):
    return all(torch.isfinite(t).all() for t in ts)


def _svd_call(M, driver=None):
    """``torch.linalg.svd`` with the cuSOLVER driver kwarg applied only where it is
    legal (CUDA tensors).  On CPU the kwarg raises, which used to send every CPU call
    down two dead exception paths before reaching the CPU link; the factors are the
    same LAPACK ones either way, so the chain is simply device-aware now."""
    if driver is not None and M.is_cuda:
        return torch.linalg.svd(M, full_matrices=False, driver=driver)
    return torch.linalg.svd(M, full_matrices=False)


def _robust_svd_ex(M, driver: str = "gesvd"):
    """Batched thin SVD with a fallback chain.  Returns ``(U, S, Vh, path)``.

    ``driver="gesvd"`` (default, kernel of record): gesvd (QR-based, robust) ->
    jitter + gesvd -> CPU.  The default cuSOLVER Jacobi driver can fail to
    converge on ill-conditioned / repeated-singular-value batches at larger n;
    gesvd is slower but stable, and every R1 number was produced with it.

    ``driver="auto"``: try the library default driver first (gesvdj on CUDA,
    LAPACK gesdd on CPU), accept it only if it did not raise and returned finite
    factors, and otherwise fall through the *same* chain as above.  Factors are
    mathematically identical up to sign/rotation of the singular subspaces; the
    call sites only use them through subspace projections and squared singular
    values, so the estimator is unchanged to floating-point accuracy.

    ``path`` is one of the SVD_STATS keys and tells the caller which link served
    the call.  **This matters for the certificate**: on the ``"jitter"`` path the
    returned factors belong to M + Delta, not to M, so the tail singular values
    are NOT the discarded mass of M and must not be used as eta_t -- the caller
    has to measure the discarded mass against the original matrix.  Every other
    path (default / gesvd / cpu) factorises M itself.
    """
    if driver not in ("gesvd", "auto"):
        raise ValueError(f"unknown svd driver {driver!r}")
    if driver == "auto":
        try:
            if "default" in _FORCE_FAIL:
                raise RuntimeError("forced default-driver failure (test hook)")
            U, S, Vh = _svd_call(M)
            if _finite(U, S, Vh):
                SVD_STATS["default"] += 1
                return U, S, Vh, "default"
        except Exception:
            pass
    key = "gesvd_after_default" if driver == "auto" else "gesvd"
    try:
        if "gesvd" in _FORCE_FAIL:
            raise RuntimeError("forced gesvd failure (test hook)")
        U, S, Vh = _svd_call(M, "gesvd")
        if _finite(U, S, Vh):
            SVD_STATS[key] += 1
            return U, S, Vh, key
    except Exception:
        pass
    try:
        if "jitter" in _FORCE_FAIL:
            raise RuntimeError("forced jitter failure (test hook)")
        eps = (1e-6 * _JITTER_SCALE) * M.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        U, S, Vh = _svd_call(M + eps * torch.randn_like(M), "gesvd")
        if _finite(U, S, Vh):
            SVD_STATS["jitter"] += 1
            return U, S, Vh, "jitter"
    except Exception:
        pass
    U, S, Vh = torch.linalg.svd(M.detach().cpu(), full_matrices=False)
    SVD_STATS["cpu"] += 1
    return U.to(M.device), S.to(M.device), Vh.to(M.device), "cpu"


def _robust_svd(M, driver: str = "gesvd"):
    """Back-compatible wrapper: ``(U, S, Vh)`` only.  Prefer ``_robust_svd_ex``
    wherever the discarded mass is accounted for -- see its docstring."""
    U, S, Vh, _ = _robust_svd_ex(M, driver)
    return U, S, Vh


class OnlineGrad:
    name = "base"

    def __init__(self, cell, batch: int):
        self.cell, self.B = cell, batch
        # `n` is the STATE width, because that is how every use of it reads
        # (torch.zeros(B, n, P), Lp = A @ L, delta (B, n)).  The number of
        # PARAMETER BLOCKS gets its own name: the two differ on the LSTM, whose
        # state rows h_i and c_i share unit i's parameter block.
        self.n_units = cell.n                                   # parameter blocks
        self.n = getattr(cell, "n_state", cell.n)               # state rows
        self.p = getattr(cell, "p_cell", cell.p)                # block width
        self.P = self.n_units * self.p
        blk = getattr(cell, "blk_idx", None)
        self.blk_idx = (torch.arange(self.n_units,
                                     device=next(cell.parameters()).device)
                        if blk is None else blk)
        # omega: number of columns cell.append_factors() returns (n for the
        # vanilla cell and the LSTM, 2n for the GRU).  The append budget c is
        # capped at omega and the Corollary-3 exact endpoint is (n_state, omega).
        self.omega = int(getattr(cell, "append_width", self.n_units))
        self._blocks_trivial = bool(self.n == self.n_units)

    def _contract_blocks(self, g):
        """(B, n_state, p) -> (B, n_units, p): sum the state rows of one block.

        Identity for the vanilla cell and the GRU (one state row per block).
        """
        if self._blocks_trivial:
            return g
        return g.new_zeros(g.shape[0], self.n_units,
                           self.p).index_add_(1, self.blk_idx, g)

    @torch.no_grad()
    def _imm_vjp(self, imm, nu):
        """(B, P) = vec(I_t^T nu) in the per-unit block layout.

        Uses the FULL immediate Jacobian: on a GRU that includes I_t^perp,
        whose exact factorisation the cell exposes.  Dropping it would bias the
        stochastic estimators by an O(1) amount.
        """
        t = nu.unsqueeze(2) * imm                               # (B, n_state, p)
        out = t.new_zeros(self.B, self.n_units,
                          self.p).index_add_(1, self.blk_idx, t)
        fac = getattr(self.cell, "imm_offblock_factors", lambda: None)()
        if fac is not None:
            M, F_rows, blocks = fac                             # I^perp = M @ F
            coef = torch.einsum("bi,bij->bj", nu, M)            # (B, n)
            out.index_add_(1, blocks, coef.unsqueeze(2) * F_rows)
        return out.reshape(self.B, self.P)

    def step_state(self, A, imm):
        raise NotImplementedError

    def grad_rows(self, delta):
        """delta (B, n) = dLoss/dh_t. Return (n, p) batch-mean recurrent-param grad rows."""
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class ExactRTRL(OnlineGrad):
    name = "exact"

    def __init__(self, cell, batch):
        super().__init__(cell, batch)
        self.reset()

    def reset(self):
        W = next(self.cell.parameters())
        self.J = torch.zeros(self.B, self.n, self.P, device=W.device, dtype=W.dtype)

    @torch.no_grad()
    def step_state(self, A, imm):
        """J_t = A_t J_{t-1} + I_t, with the FULL immediate Jacobian.

        `imm` carries only the block-diagonal part I_t^o.  On a GRU the
        immediate Jacobian also has an off-block part I_t^perp (the reset gate
        reaches every unit in one step), whose exact rank-<=n factorisation the
        cell exposes; adding only I_t^o would leave this "exact" shadow -- the
        ground truth behind every gradient cosine and residual spectrum --
        wrong by an O(1) amount.  Accumulated in place from the factors rather
        than via cell.imm_dense(), which is the same numbers at O(n^2 p) extra
        memory.
        """
        self.J = torch.bmm(A, self.J)
        Jv = self.J.view(self.B, self.n, self.n_units, self.p)
        s = torch.arange(self.n, device=A.device)
        Jv[:, s, self.blk_idx[s], :] += imm       # I_t^o: row s -> its own block
        fac = getattr(self.cell, "imm_offblock_factors", lambda: None)()
        if fac is not None:
            M, F_rows, blocks = fac
            Jv.index_add_(2, blocks, torch.einsum("bij,bjq->bijq", M, F_rows))
        self.J = Jv.view(self.B, self.n, self.P)

    @torch.no_grad()
    def grad_rows(self, delta):
        g = torch.bmm(delta.unsqueeze(1), self.J).squeeze(1)  # (B, P)
        return g.mean(0).view(self.n_units, self.p)


class SKRTRL(OnlineGrad):
    """SK-RTRL: exact SnAp-1 part S + two-sided rank-r sketch (L, R) of the residual.

    r = 0  -> SnAp-1 exactly.
    r = n with pre-projection skipped -> exact RTRL (Corollary 3).
    Certificate: e_t = rho_bar_t * e_{t-1} + eta_t, valid upper bound on ||J - (S + L R^T)||_F.
    """
    name = "skrtrl"

    def __init__(self, cell, batch, r: int, c: int | None = None, mode: str = "svd",
                 svd_driver: str = "gesvd", force_preproject: bool = False,
                 disable_preproject: bool = False):
        super().__init__(cell, batch)
        self.r = r
        self.mode = mode  # "svd" (deterministic top-r) | "randproj" (matched-cost ablation)
        self.svd_driver = svd_driver
        if disable_preproject:
            # A9 control: no top-c pre-projection at all (c = n, the full off-diagonal
            # mass is appended and only the rank-r truncation discards anything), so the
            # contribution of the pre-projection to eta_t can be isolated.
            self.r = min(r, self.n)
            self.c = self.omega
            self.preproject = False
            self.reset()
            return
        if r >= self.n and not force_preproject:   # Corollary 3 regime
            # exact endpoint (r, c) = (n_state, omega): (n, n) vanilla,
            # (n, 2n) GRU, (2n, n) LSTM
            self.r = self.n
            self.c = self.omega
            self.preproject = False
        else:
            # force_preproject: keep the sketch machinery (explicit append budget c and
            # the top-c pre-projection) even when r >= n, so an adaptive-rank sweep that
            # lifts the ceiling to r_max = n does not silently switch kernels.
            self.r = min(r, self.n)
            self.c = c if c is not None else max(4, math.ceil(self.r / 4))
            self.c = min(self.c, self.omega)
            self.preproject = True
        self.reset()

    def reset(self):
        W = next(self.cell.parameters())
        dev, dt = W.device, W.dtype
        self.S = torch.zeros(self.B, self.n, self.p, device=dev, dtype=dt)
        self.L = torch.zeros(self.B, self.n, self.r, device=dev, dtype=dt)
        self.R = torch.zeros(self.B, self.P, self.r, device=dev, dtype=dt)
        self.e = torch.zeros(self.B, device=dev, dtype=dt)
        self.last = {}

    @torch.no_grad()
    def step_state(self, A, imm):
        # r_out = target rank (self.r, may be changed between steps by an adaptive
        # controller); r_in = actual width of the current L/R factors. When r_out <
        # r_in the truncation drops columns and records their mass in eta_t, so the
        # certificate stays valid under rank shrink.
        B, n, p, c = self.B, self.n, self.p, self.c
        r = self.r
        r_in = self.L.shape[2] if self.L.numel() else 0
        S_prev = self.S

        # --- certified norm bound (computed on full A) ---
        nF = torch.linalg.matrix_norm(A, ord="fro", dim=(1, 2))
        n1 = A.abs().sum(dim=1).max(dim=1).values                  # max col sum
        ninf = A.abs().sum(dim=2).max(dim=1).values                # max row sum
        rho_bar = torch.minimum(nF, torch.sqrt(n1 * ninf))
        # diagnostic lower bound on ||A||_2 (warm-started power iteration)
        if not hasattr(self, "_pv") or self._pv.shape[0] != B:
            self._pv = torch.randn(B, n, 1, device=A.device, dtype=A.dtype)
        Av = torch.bmm(A, self._pv)
        rho_hat = Av.norm(dim=(1, 2)) / self._pv.norm(dim=(1, 2)).clamp_min(1e-30)
        self._pv = Av / Av.norm(dim=(1, 2), keepdim=True).clamp_min(1e-30)

        # --- exact SnAp-1 part ---
        # For the LSTM this is the per-unit 2x2 product of Prop. 6, not a
        # scalar scale: diag(A_t) alone would drop (A_t)_{i,n+i}, the intra-unit
        # c -> h memory path, which is O(1) with the forget gate open.
        self.S = self.cell.trace_update(A, S_prev, imm)

        if r == 0:
            # exact, not a bound: the append right factor has exactly
            # orthonormal columns, so ||C_t Q_t^T||_F = ||C_t||_F.  On a GRU
            # this also includes ||I_t^perp||_F, so the r=0 (SnAp-1)
            # certificate covers the extra bias a block-diagonal SnAp-1 incurs
            # on a gated cell (Remark 3).
            eta = self.cell.append_mass(A, S_prev)
            self.e = rho_bar * self.e + eta
            self.last = {"rho_bar": rho_bar, "rho_hat": rho_hat, "eta": eta}
            return

        # --- propagate residual left factor ---
        Lp = torch.bmm(A, self.L)                                  # (B, n, r)

        # --- new off-diagonal mass, exact factored form ---
        # N_t = Ahat_t S_{t-1} (+ I_t^perp on a GRU) = B0 Q_t^T with Q_t's
        # columns orthonormal, each supported on a single parameter block.
        B0, Qstruct = self.cell.append_factors(A, S_prev)          # (B, n, omega)
        omega = Qstruct.width
        Qrows, Qblocks = Qstruct.rows, Qstruct.blocks
        trivial = self._blocks_trivial and omega == self.n_units

        tau_c = torch.zeros(B, device=A.device, dtype=A.dtype)
        if self.preproject:
            if self.mode == "randproj":
                # matched-cost ablation: random orthonormal projection instead of top-c
                Vc, _ = torch.linalg.qr(torch.randn(B, omega, c, device=A.device, dtype=A.dtype))
                Bc = torch.bmm(B0, Vc)                             # (B, n, c)
                tau_c = torch.sqrt((torch.linalg.matrix_norm(B0, ord="fro", dim=(1, 2)) ** 2
                                    - torch.linalg.matrix_norm(Bc, ord="fro", dim=(1, 2)) ** 2).clamp_min(0))
            else:
                # deterministic top-c of B0 via full SVD of the small n x n matrix
                Ub, sb, Vbh, path = _robust_svd_ex(B0, self.svd_driver)
                Bc = Ub[:, :, :c] * sb[:, :c].unsqueeze(1)         # (B, n, c)
                Vc = Vbh[:, :c, :].transpose(1, 2)                 # (B, n, c)
                if path == "jitter":
                    # The factors belong to B0 + Delta, so sum_{i>=c} sb_i^2 is the tail
                    # of the PERTURBED matrix and would under-count eta_t.  Q_t has
                    # orthonormal rows, so the mass actually discarded from
                    # B0 Q_t is exactly ||B0 - Bc Vc^T||_F -- measured on the original.
                    tau_c = torch.linalg.matrix_norm(
                        B0 - torch.bmm(Bc, Vc.transpose(1, 2)), ord="fro", dim=(1, 2))
                else:
                    tau_c = torch.sqrt((sb[:, c:] ** 2).sum(dim=1).clamp_min(0))
        else:
            Bc = B0
            Vc = torch.eye(omega, device=A.device, dtype=A.dtype).expand(B, omega, omega)

        # right factor of the append in dense form, Q_c = Q_t V_c  (B, P, c).
        # index_add_ rather than an assignment because on a GRU two columns of
        # Q_t share a parameter block, so their contributions must be summed;
        # with one column per block this is exactly the old outer product.
        cc = Bc.shape[2]
        if trivial:
            Qc = Qrows.unsqueeze(3) * Vc.unsqueeze(2)              # (B, n, p, c)
        else:
            Qc = S_prev.new_zeros(B, self.n_units, self.p, cc)
            Qc.index_add_(1, Qblocks, torch.einsum("bwq,bwk->bwqk", Qrows, Vc))
        Qc_flat = Qc.reshape(B, self.P, cc)

        # G = R^T Q_t V_c   (use r_in = actual factor width, not the target rank r)
        Rv = self.R.view(B, self.n_units, self.p, r_in)
        Rvw = Rv if trivial else Rv[:, Qblocks]
        G0 = torch.einsum("bwqr,bwq->brw", Rvw, Qrows)             # (B, r_in, omega)
        G = torch.bmm(G0, Vc)                                      # (B, r_in, c)

        # thin QR of (Qc - R G)
        Qperp_raw = Qc_flat - torch.bmm(self.R, G)                 # (B, P, c)
        Qperp, Theta = torch.linalg.qr(Qperp_raw, mode="reduced")  # (B,P,c), (B,c,c)

        # corrected core and truncation
        core = torch.cat([Lp + torch.bmm(Bc, G.transpose(1, 2)),
                          torch.bmm(Bc, Theta.transpose(1, 2))], dim=2)  # (B, n, r+c)
        if self.mode == "randproj" and self.preproject:
            rc = core.shape[2]
            Om, _ = torch.linalg.qr(torch.randn(B, rc, r, device=A.device, dtype=A.dtype))
            self.L = torch.bmm(core, Om)
            self.R = torch.bmm(torch.cat([self.R, Qperp], dim=2), Om)
            tau_r = torch.sqrt((torch.linalg.matrix_norm(core, ord="fro", dim=(1, 2)) ** 2
                                - torch.linalg.matrix_norm(self.L, ord="fro", dim=(1, 2)) ** 2).clamp_min(0))
            eta = tau_c + tau_r
            self.e = rho_bar * self.e + eta
            self.last = {"rho_bar": rho_bar, "rho_hat": rho_hat, "eta": eta, "tau_c": tau_c, "tau_r": tau_r}
            return
        Uc_, sc_, Wch, path = _robust_svd_ex(core, self.svd_driver)
        k = min(r, sc_.shape[1])
        self.L = Uc_[:, :, :k] * sc_[:, :k].unsqueeze(1)
        Wfac = Wch.transpose(1, 2)[:, :, :k]                       # (B, r+c, k)
        if path == "jitter":
            # factors of core + Delta: measure the truncation residual on `core`
            # itself.  [R, Qperp] has orthonormal columns, so the discarded mass is
            # exactly ||core - L Wfac^T||_F.
            tau_r = torch.linalg.matrix_norm(
                core - torch.bmm(self.L, Wfac.transpose(1, 2)), ord="fro", dim=(1, 2))
        else:
            tau_r = torch.sqrt((sc_[:, k:] ** 2).sum(dim=1).clamp_min(0))
        self.R = torch.bmm(torch.cat([self.R, Qperp], dim=2), Wfac)
        if k < r:  # pad (early steps)
            padL = self.L.new_zeros(B, n, r - k)
            padR = self.R.new_zeros(B, self.P, r - k)
            self.L = torch.cat([self.L, padL], dim=2)
            self.R = torch.cat([self.R, padR], dim=2)

        eta = tau_c + tau_r
        self.e = rho_bar * self.e + eta
        self.last = {"rho_bar": rho_bar, "rho_hat": rho_hat, "eta": eta, "tau_c": tau_c, "tau_r": tau_r}

    @staticmethod
    @torch.no_grad()
    def _offdiag_mass(Ahat, S_prev):
        # || Ahat S_prev ||_F = || (Ahat D_s) ||_F with column scaling (exact, cheap)
        s_norm = S_prev.norm(dim=2)
        return torch.linalg.matrix_norm(Ahat * s_norm.unsqueeze(1), ord="fro", dim=(1, 2))

    @torch.no_grad()
    def residual_dense(self):
        """(B, n_state, P) dense S + L R^T  (for testing only)."""
        out = torch.bmm(self.L, self.R.transpose(1, 2))
        outv = out.view(self.B, self.n, self.n_units, self.p)
        s = torch.arange(self.n, device=self.S.device)
        outv[:, s, self.blk_idx[s], :] += self.S
        return outv.view(self.B, self.n, self.P)

    @torch.no_grad()
    def grad_rows(self, delta):
        # delta is (B, n_state); the trace part is contracted down to the
        # (n_units, p) parameter-block layout, which on the LSTM sums the two
        # state rows h_i, c_i of unit i.
        g1 = self._contract_blocks(delta.unsqueeze(2) * self.S)     # (B, n_units, p)
        r_in = self.L.shape[2] if self.L.numel() else 0            # actual factor width
        if r_in > 0:
            u = torch.einsum("bn,bnr->br", delta, self.L)           # (B, r_in)
            g2 = torch.einsum("br,bipr->bip", u,
                              self.R.view(self.B, self.n_units, self.p, r_in))
            g1 = g1 + g2
        return g1.mean(0)


class SnAp1(SKRTRL):
    name = "snap1"

    def __init__(self, cell, batch, svd_driver: str = "gesvd"):
        super().__init__(cell, batch, r=0, svd_driver=svd_driver)
