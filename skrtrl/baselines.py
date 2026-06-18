"""Stochastic / local baselines: UORO, KF-RTRL, RFLO. All per-batch-lane state."""
import torch

from .algos import OnlineGrad


class UORO(OnlineGrad):
    """Tallec & Ollivier 2017: rank-1 unbiased J ~ s_tilde theta_tilde^T."""
    name = "uoro"

    def __init__(self, cell, batch, eps=1e-7):
        super().__init__(cell, batch)
        self.eps = eps
        self.reset()

    def reset(self):
        W = self.cell.W
        self.s = torch.zeros(self.B, self.n, device=W.device, dtype=W.dtype)
        self.th = torch.zeros(self.B, self.P, device=W.device, dtype=W.dtype)

    def reset_lanes(self, mask):
        self.s[mask] = 0.0
        self.th[mask] = 0.0

    @torch.no_grad()
    def step_state(self, A, imm):
        B = self.B
        nu = torch.randint(0, 2, (B, self.n), device=A.device, dtype=A.dtype) * 2 - 1
        As = torch.bmm(A, self.s.unsqueeze(2)).squeeze(2)             # (B, n)
        # I_t^T nu: column (i,j) -> imm[i,j] * nu_i
        Itnu = (nu.unsqueeze(2) * imm).reshape(B, self.P)             # (B, P)
        r0 = torch.sqrt(self.th.norm(dim=1) / As.norm(dim=1).clamp_min(self.eps)).clamp(self.eps, 1e7)
        r1 = torch.sqrt(Itnu.norm(dim=1) / nu.norm(dim=1).clamp_min(self.eps)).clamp(self.eps, 1e7)
        self.s = r0.unsqueeze(1) * As + r1.unsqueeze(1) * nu
        self.th = self.th / r0.unsqueeze(1) + Itnu / r1.unsqueeze(1)

    @torch.no_grad()
    def grad_rows(self, delta):
        coef = (delta * self.s).sum(dim=1, keepdim=True)              # (B, 1)
        g = coef * self.th                                            # (B, P)
        return g.mean(0).view(self.n, self.p)


class KFRTRL(OnlineGrad):
    """Mujika et al. 2018: J ~ u (x) Bm (Kronecker), unbiased for vanilla RNN."""
    name = "kfrtrl"

    def __init__(self, cell, batch, eps=1e-7):
        super().__init__(cell, batch)
        self.eps = eps
        self.reset()

    def reset(self):
        W = self.cell.W
        self.u = torch.zeros(self.B, self.p, device=W.device, dtype=W.dtype)
        self.Bm = torch.zeros(self.B, self.n, self.n, device=W.device, dtype=W.dtype)

    def reset_lanes(self, mask):
        self.u[mask] = 0.0
        self.Bm[mask] = 0.0

    @torch.no_grad()
    def step_state(self, A, imm):
        B = self.B
        # imm = D * feats outer: recover feats and D:  imm[b,i,:] = D_i * feats. Use row with max |D|.
        Dvec = imm[:, :, -1]                                          # feats last entry is 1 -> imm[:,:, -1] = D
        feats = imm / Dvec.unsqueeze(2).clamp_min(1e-12)              # (B, n, p) rows ~ feats
        feats = feats.mean(dim=1)                                     # (B, p)
        AB = torch.bmm(A, self.Bm)                                    # (B, n, n)
        diagD = torch.diag_embed(Dvec)                                # (B, n, n)
        c1 = (torch.randint(0, 2, (B, 1), device=A.device, dtype=A.dtype) * 2 - 1)
        c2 = (torch.randint(0, 2, (B, 1), device=A.device, dtype=A.dtype) * 2 - 1)
        nAB = AB.flatten(1).norm(dim=1).clamp_min(self.eps)
        nu_ = self.u.norm(dim=1).clamp_min(self.eps)
        nD = diagD.flatten(1).norm(dim=1).clamp_min(self.eps)
        nf = feats.norm(dim=1).clamp_min(self.eps)
        p1 = torch.sqrt(nAB / nu_).clamp(self.eps, 1e7).unsqueeze(1)
        p2 = torch.sqrt(nD / nf).clamp(self.eps, 1e7).unsqueeze(1)
        self.u = c1 * p1 * self.u + c2 * p2 * feats
        self.Bm = (c1.unsqueeze(2) * AB / p1.unsqueeze(2)) + (c2.unsqueeze(2) * diagD / p2.unsqueeze(2))

    @torch.no_grad()
    def grad_rows(self, delta):
        dB = torch.bmm(delta.unsqueeze(1), self.Bm).squeeze(1)        # (B, n)
        g = dB.unsqueeze(2) * self.u.unsqueeze(1)                     # (B, n, p)
        return g.mean(0)


class RFLO(OnlineGrad):
    """Murray 2019-style local trace: e_t = (1-1/tau) e_{t-1} + imm; block-diag like SnAp-1
    but with a fixed leak instead of the true diagonal propagation."""
    name = "rflo"

    def __init__(self, cell, batch, tau=10.0):
        super().__init__(cell, batch)
        self.alpha = 1.0 - 1.0 / tau
        self.reset()

    def reset(self):
        W = self.cell.W
        self.S = torch.zeros(self.B, self.n, self.p, device=W.device, dtype=W.dtype)

    def reset_lanes(self, mask):
        self.S[mask] = 0.0

    @torch.no_grad()
    def step_state(self, A, imm):
        self.S = self.alpha * self.S + imm

    @torch.no_grad()
    def grad_rows(self, delta):
        return (delta.unsqueeze(2) * self.S).mean(0)


def make_baseline(name, cell, batch):
    return {"uoro": UORO, "kfrtrl": KFRTRL, "eprop": RFLO, "rflo": RFLO}[name](cell, batch)
