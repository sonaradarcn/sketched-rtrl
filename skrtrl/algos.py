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


def _robust_svd(M):
    """Batched thin SVD with fallbacks: gesvd (QR-based, robust) then jitter then CPU.
    The default cuSOLVER Jacobi driver can fail to converge on ill-conditioned /
    repeated-singular-value batches at larger n; gesvd is slower but stable."""
    try:
        return torch.linalg.svd(M, full_matrices=False, driver="gesvd")
    except Exception:
        pass
    try:
        eps = 1e-6 * M.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        return torch.linalg.svd(M + eps * torch.randn_like(M), full_matrices=False, driver="gesvd")
    except Exception:
        U, S, Vh = torch.linalg.svd(M.detach().cpu(), full_matrices=False)
        return U.to(M.device), S.to(M.device), Vh.to(M.device)


class OnlineGrad:
    name = "base"

    def __init__(self, cell, batch: int):
        self.cell, self.B = cell, batch
        self.n, self.p = cell.n, cell.p
        self.P = self.n * self.p

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
        W = self.cell.W
        self.J = torch.zeros(self.B, self.n, self.P, device=W.device, dtype=W.dtype)

    @torch.no_grad()
    def step_state(self, A, imm):
        self.J = torch.bmm(A, self.J)
        idx = torch.arange(self.n, device=A.device)
        Jv = self.J.view(self.B, self.n, self.n, self.p)
        Jv[:, idx, idx, :] += imm  # immediate part hits row i of column block i
        self.J = Jv.view(self.B, self.n, self.P)

    @torch.no_grad()
    def grad_rows(self, delta):
        g = torch.bmm(delta.unsqueeze(1), self.J).squeeze(1)  # (B, P)
        return g.mean(0).view(self.n, self.p)


class SKRTRL(OnlineGrad):
    """SK-RTRL: exact SnAp-1 part S + two-sided rank-r sketch (L, R) of the residual.

    r = 0  -> SnAp-1 exactly.
    r = n with pre-projection skipped -> exact RTRL (Corollary 3).
    Certificate: e_t = rho_bar_t * e_{t-1} + eta_t, valid upper bound on ||J - (S + L R^T)||_F.
    """
    name = "skrtrl"

    def __init__(self, cell, batch, r: int, c: int | None = None, mode: str = "svd"):
        super().__init__(cell, batch)
        self.r = r
        self.mode = mode  # "svd" (deterministic top-r) | "randproj" (matched-cost ablation)
        if r >= self.n:                       # Corollary 3 regime
            self.r = self.n
            self.c = self.n
            self.preproject = False
        else:
            self.c = c if c is not None else max(4, math.ceil(r / 4))
            self.c = min(self.c, self.n)
            self.preproject = True
        self.reset()

    def reset(self):
        W = self.cell.W
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
        diagA = _diag_of(A)                                        # (B, n)
        Ahat = A - torch.diag_embed(diagA)                         # (B, n, n)

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
        self.S = diagA.unsqueeze(2) * S_prev + imm

        if r == 0:
            eta = self._offdiag_mass(Ahat, S_prev)
            self.e = rho_bar * self.e + eta
            self.last = {"rho_bar": rho_bar, "rho_hat": rho_hat, "eta": eta}
            return

        # --- propagate residual left factor ---
        Lp = torch.bmm(A, self.L)                                  # (B, n, r)

        # --- new off-diagonal mass, exact factored form ---
        s_norm = S_prev.norm(dim=2)                                # (B, n)
        B0 = Ahat * s_norm.unsqueeze(1)                            # (B, n, n) columns scaled
        nz = s_norm > 0
        Snorm = torch.where(nz.unsqueeze(2), S_prev / s_norm.clamp_min(1e-30).unsqueeze(2),
                            torch.zeros_like(S_prev))              # (B, n, p) rows of Q_t

        tau_c = torch.zeros(B, device=A.device, dtype=A.dtype)
        if self.preproject:
            if self.mode == "randproj":
                # matched-cost ablation: random orthonormal projection instead of top-c
                Vc, _ = torch.linalg.qr(torch.randn(B, n, c, device=A.device, dtype=A.dtype))
                Bc = torch.bmm(B0, Vc)                             # (B, n, c)
                tau_c = torch.sqrt((torch.linalg.matrix_norm(B0, ord="fro", dim=(1, 2)) ** 2
                                    - torch.linalg.matrix_norm(Bc, ord="fro", dim=(1, 2)) ** 2).clamp_min(0))
            else:
                # deterministic top-c of B0 via full SVD of the small n x n matrix
                Ub, sb, Vbh = _robust_svd(B0)
                Bc = Ub[:, :, :c] * sb[:, :c].unsqueeze(1)         # (B, n, c)
                Vc = Vbh[:, :c, :].transpose(1, 2)                 # (B, n, c)
                tau_c = torch.sqrt((sb[:, c:] ** 2).sum(dim=1).clamp_min(0))
        else:
            Bc = B0
            Vc = torch.eye(n, device=A.device, dtype=A.dtype).expand(B, n, n)

        # right factor of append in dense form: Qc (B, n, p, c) row (i,j) = Snorm[i,j] * Vc[i,:]
        Qc = Snorm.unsqueeze(3) * Vc.unsqueeze(2)                  # (B, n, p, c)
        Qc_flat = Qc.reshape(B, self.P, Bc.shape[2])

        # G = R^T Q_t V_c   (use r_in = actual factor width, not the target rank r)
        Rv = self.R.view(B, n, p, r_in)
        G0 = torch.einsum("bipr,bip->bri", Rv, Snorm)              # (B, r_in, n) = R^T Q_t
        G = torch.bmm(G0, Vc)                                      # (B, r, c)

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
        Uc_, sc_, Wch = _robust_svd(core)
        k = min(r, sc_.shape[1])
        self.L = Uc_[:, :, :k] * sc_[:, :k].unsqueeze(1)
        Wfac = Wch.transpose(1, 2)[:, :, :k]                       # (B, r+c, k)
        self.R = torch.bmm(torch.cat([self.R, Qperp], dim=2), Wfac)
        if k < r:  # pad (early steps)
            padL = self.L.new_zeros(B, n, r - k)
            padR = self.R.new_zeros(B, self.P, r - k)
            self.L = torch.cat([self.L, padL], dim=2)
            self.R = torch.cat([self.R, padR], dim=2)
        tau_r = torch.sqrt((sc_[:, k:] ** 2).sum(dim=1).clamp_min(0))

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
        """(B, n, P) dense S + L R^T  (for testing only)."""
        idx = torch.arange(self.n, device=self.S.device)
        out = torch.bmm(self.L, self.R.transpose(1, 2))
        outv = out.view(self.B, self.n, self.n, self.p)
        outv[:, idx, idx, :] += self.S
        return outv.view(self.B, self.n, self.P)

    @torch.no_grad()
    def grad_rows(self, delta):
        g1 = delta.unsqueeze(2) * self.S                            # (B, n, p)
        r_in = self.L.shape[2] if self.L.numel() else 0            # actual factor width
        if r_in > 0:
            u = torch.einsum("bn,bnr->br", delta, self.L)           # (B, r_in)
            g2 = torch.einsum("br,bipr->bip", u, self.R.view(self.B, self.n, self.p, r_in))
            g1 = g1 + g2
        return g1.mean(0)


class SnAp1(SKRTRL):
    name = "snap1"

    def __init__(self, cell, batch):
        super().__init__(cell, batch, r=0)
