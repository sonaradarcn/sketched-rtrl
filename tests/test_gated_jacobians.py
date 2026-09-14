r"""D5 / R3-5 self-verification: analytic Jacobians of gated cells for SK-RTRL.

Standalone (imports nothing from `skrtrl`): it verifies, against torch.autograd,
the derivation written up in paper/revise_r2/GRU_EXTENSION.tex, so that the
derivation and any future repo cell agree on the same conventions.

Run:
  D:\Anaconda\envs\multilingual_lora\python.exe code/tests/test_gated_jacobians.py

What is checked (float64, CPU, tol 1e-8 absolute):

  G1  GRU analytic A_t          == d h_t / d h_{t-1}                (autograd)
  G2  GRU analytic I_t          == d h_t / d theta |_{h_{t-1} fixed}
  G3  GRU block structure: the z- and h-gate segments of I_t vanish off the
      block diagonal, while the r-gate segments do NOT -- the correction to
      the D5 draft.  The off-block part is exactly  Khat_t  x  phi_t.
  G4  GRU append factorisation: Ahat_t S_{t-1} + I^off_t = C_t Qt^T exactly,
      with Qt^T Qt = I and each column of Qt supported on one parameter block.
  G5  GRU T-step exactness: J_t = A_t J_{t-1} + I_t accumulated for T steps
      == d h_T / d theta of the unrolled sequence (autograd).
  G6  GRU factored endpoint (Corollary 3): the SK-RTRL update with r = n,
      c = 2n and pre-projection disabled gives tau_r == 0 and
      S_t + L_t R_t^T == J_t at every step.
  G7  certified spectral bounds: rho_bar_t >= ||A_t||_2, both the paper's
      surrogate and the split-diagonal refinement
      rho_bar = min(surrogate(A_t), ||blkdiag(A_t)||_2 + surrogate(Ahat_t)).

  L1..L7  the same seven checks for the LSTM on the stacked state [h; c],
      where the immediate Jacobian IS block diagonal (2 state rows x 4p
      parameter columns per unit), the trace update needs the 2x2 diagonal
      blocks of A_t rather than its scalar diagonal, and (L4f/L4g) the
      c-columns of A_t are exactly diagonal, so Ahat_t kills them and the
      append reduces to the vanilla row-normalised factorisation of rank <= n.
  P1  certificate tightness probe at n up to 256 (informational).

Flat parameter layout (per-unit blocks, the SnAp-1 view; G = #gates):
  unit i, gate a in [0,G), offset j in [0,p),  p = n + m + 1
      index = i*(G*p) + a*p + j
      j in [0, n)     -> U_a[i, j]     recurrent, acts on h_{t-1}
      j in [n, n+m)   -> W_a[i, j-n]   input,     acts on x_t
      j = n+m         -> b_a[i]
  Gate order: GRU (z, r, h~);  LSTM (i, f, g~, o).
  Total P = n * G * p.  This is a permutation of the usual per-gate storage.
"""
import math
import sys

import torch
from torch.autograd.functional import jacobian

DT = torch.float64
TOL = 1e-8
RESULTS = []


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def unpack(flat, n, m, G):
    p = n + m + 1
    v = flat.view(n, G, p)
    return [(v[:, a, :n], v[:, a, n:n + m], v[:, a, n + m]) for a in range(G)]


def rand_params(n, m, G, gen, scale=1.0):
    p = n + m + 1
    k = scale / math.sqrt(n)
    flat = ((torch.rand(n * G * p, generator=gen, dtype=DT) * 2 - 1) * k)
    return flat


def record(tag, dev, tol=TOL, note=""):
    ok = bool(dev <= tol) and math.isfinite(dev)
    RESULTS.append((tag, dev, ok, note))
    print(f"  [{'ok ' if ok else 'FAIL'}] {tag:<52s} max|dev| = {dev:.3e}   {note}")
    return ok


def maxdev(a, b):
    return (a - b).abs().max().item()


def batch_state_jac(fn, state):
    """d fn(state) / d state, per batch lane. state (B, d) -> out (B, d')."""
    Jfull = jacobian(fn, state, vectorize=True)          # (B, d', B, d)
    B = state.shape[0]
    return torch.stack([Jfull[b, :, b, :] for b in range(B)], 0)


def blockwise_qr(rows, n, G, p, tag_offsets):
    """Exactly orthonormalise, per parameter block, a stack of block-supported rows.

    rows: list of length k of (B, n, P) tensors; rows[a][b, i, :] must be
    supported on parameter block i.  Returns (T, Q) with

        Y = T Q^T,   Y[b, a*n + i, :] = rows[a][b, i, :],
        Q (B, P, k*n) with exactly orthonormal columns, column a*n+i
        supported on parameter block i.

    tag_offsets: fallback canonical directions (one per a) inside a block,
    used when a row is zero / parallel to an earlier one; the corresponding
    entry of T is then zero, so the product is unaffected.
    """
    B, _, P = rows[0].shape
    k = len(rows)
    Q = torch.zeros(B, P, k * n, dtype=DT)
    T = torch.zeros(B, k * n, k * n, dtype=DT)
    blk = G * p
    for b in range(B):
        for i in range(n):
            base = i * blk
            qs = []
            for a in range(k):
                v = rows[a][b, i].clone()
                coef = []
                for aa, q in enumerate(qs):
                    c = float(v @ q)
                    coef.append(c)
                    v = v - c * q
                nv = float(v.norm())
                if nv > 1e-11:
                    q = v / nv
                else:                              # degenerate: zero scale, any unit
                    q = torch.zeros(P, dtype=DT)
                    for off in tag_offsets[a:] + tag_offsets[:a]:
                        cand = torch.zeros(P, dtype=DT)
                        cand[base + off] = 1.0
                        for qq in qs:
                            cand = cand - float(cand @ qq) * qq
                        if float(cand.norm()) > 1e-6:
                            q = cand / cand.norm()
                            break
                    nv = 0.0
                qs.append(q)
                Q[b, :, a * n + i] = q
                for aa, c in enumerate(coef):
                    T[b, a * n + i, aa * n + i] = c
                T[b, a * n + i, a * n + i] = nv
    return T, Q


# ----------------------------------------------------------------------------
# GRU
# ----------------------------------------------------------------------------
GRU_G = 3          # gates (z, r, h~)


def gru_forward(flat, x, h_prev, n, m):
    (Uz, Wz, bz), (Ur, Wr, br), (Uh, Wh, bh) = unpack(flat, n, m, GRU_G)
    z = torch.sigmoid(h_prev @ Uz.T + x @ Wz.T + bz)
    r = torch.sigmoid(h_prev @ Ur.T + x @ Wr.T + br)
    ht = torch.tanh((r * h_prev) @ Uh.T + x @ Wh.T + bh)
    return (1.0 - z) * h_prev + z * ht


def gru_analytic(flat, x, h_prev, n, m):
    """Forward + analytic A_t and I_t, exactly as derived in GRU_EXTENSION.tex."""
    p, G = n + m + 1, GRU_G
    P = n * G * p
    B = x.shape[0]
    (Uz, Wz, bz), (Ur, Wr, br), (Uh, Wh, bh) = unpack(flat, n, m, G)

    z = torch.sigmoid(h_prev @ Uz.T + x @ Wz.T + bz)
    r = torch.sigmoid(h_prev @ Ur.T + x @ Wr.T + br)
    hh = torch.tanh((r * h_prev) @ Uh.T + x @ Wh.T + bh)
    h = (1.0 - z) * h_prev + z * hh

    zp, rp, hp = z * (1 - z), r * (1 - r), 1 - hh * hh

    # --- recurrent Jacobian A_t = dh_t / dh_{t-1} ------------------------------
    dz_dh = zp.unsqueeze(2) * Uz                                    # (B,n,n)
    dr_dh = rp.unsqueeze(2) * Ur                                    # (B,n,n)
    dRH = torch.diag_embed(r) + h_prev.unsqueeze(2) * dr_dh         # d(r*h)/dh
    dhh_dh = hp.unsqueeze(2) * (Uh @ dRH)                           # (B,n,n)
    A = (torch.diag_embed(1.0 - z)
         + (hh - h_prev).unsqueeze(2) * dz_dh
         + z.unsqueeze(2) * dhh_dh)

    # --- immediate Jacobian I_t ----------------------------------------------
    one = torch.ones(B, 1, dtype=DT)
    phi = torch.cat([h_prev, x, one], 1)                            # (B,p)
    psi = torch.cat([r * h_prev, x, one], 1)                        # (B,p)
    dzc = (hh - h_prev) * zp                                        # (B,n)
    dhc = z * hp                                                    # (B,n)
    K = dhc.unsqueeze(2) * Uh * (rp * h_prev).unsqueeze(1)          # (B,n,n)

    I = torch.zeros(B, n, P, dtype=DT)
    v = I.view(B, n, n, G, p)
    idx = torch.arange(n)
    v[:, idx, idx, 0, :] = dzc.unsqueeze(2) * phi.unsqueeze(1)      # z gate, own row
    v[:, idx, idx, 2, :] = dhc.unsqueeze(2) * psi.unsqueeze(1)      # h~ gate, own row
    v[:, :, :, 1, :] = K.unsqueeze(3) * phi[:, None, None, :]       # r gate, ALL rows

    # leaky/mixing split of A_t used by the refined certified bound
    A_leak = 1.0 - z                                                # (B,n) diagonal
    A_mix = A - torch.diag_embed(A_leak)

    return dict(h=h, A=A, I=I, phi=phi, psi=psi, K=K, z=z, r=r, hh=hh,
                A_leak=A_leak, A_mix=A_mix)


def gru_split_immediate(I, n, G, p):
    B = I.shape[0]
    Id = torch.zeros_like(I)
    vi = I.view(B, n, n, G * p)
    vd = Id.view(B, n, n, G * p)
    idx = torch.arange(n)
    vd[:, idx, idx, :] = vi[:, idx, idx, :]
    return Id, I - Id


def gru_offblock_rows(K, phi, n, m):
    """I^off_t = Khat_t @ Frows, with Frows[j] = phi_t in the r-segment of block j."""
    B = K.shape[0]
    p, G = n + m + 1, GRU_G
    P = n * G * p
    Frows = torch.zeros(B, n, P, dtype=DT)
    vf = Frows.view(B, n, n, G, p)
    idx = torch.arange(n)
    vf[:, idx, idx, 1, :] = phi.unsqueeze(1).expand(B, n, p)
    Khat = K - torch.diag_embed(torch.diagonal(K, dim1=1, dim2=2))
    return Khat, Frows


# ----------------------------------------------------------------------------
# LSTM
# ----------------------------------------------------------------------------
LSTM_G = 4         # gates (i, f, g~, o)


def lstm_forward(flat, x, state, n, m):
    """state = [h_{t-1} ; c_{t-1}] as (B, 2n); returns [h_t ; c_t]."""
    h_prev, c_prev = state[:, :n], state[:, n:]
    (Ui, Wi, bi), (Uf, Wf, bf), (Ug, Wg, bg), (Uo, Wo, bo) = unpack(flat, n, m, LSTM_G)
    ii = torch.sigmoid(h_prev @ Ui.T + x @ Wi.T + bi)
    ff = torch.sigmoid(h_prev @ Uf.T + x @ Wf.T + bf)
    gg = torch.tanh(h_prev @ Ug.T + x @ Wg.T + bg)
    oo = torch.sigmoid(h_prev @ Uo.T + x @ Wo.T + bo)
    c = ff * c_prev + ii * gg
    h = oo * torch.tanh(c)
    return torch.cat([h, c], 1)


def lstm_analytic(flat, x, state, n, m):
    p, G = n + m + 1, LSTM_G
    P = n * G * p
    B = x.shape[0]
    h_prev, c_prev = state[:, :n], state[:, n:]
    (Ui, Wi, bi), (Uf, Wf, bf), (Ug, Wg, bg), (Uo, Wo, bo) = unpack(flat, n, m, G)

    ii = torch.sigmoid(h_prev @ Ui.T + x @ Wi.T + bi)
    ff = torch.sigmoid(h_prev @ Uf.T + x @ Wf.T + bf)
    gg = torch.tanh(h_prev @ Ug.T + x @ Wg.T + bg)
    oo = torch.sigmoid(h_prev @ Uo.T + x @ Wo.T + bo)
    c = ff * c_prev + ii * gg
    th = torch.tanh(c)
    h = oo * th

    ip, fp, op = ii * (1 - ii), ff * (1 - ff), oo * (1 - oo)
    gp, thp = 1 - gg * gg, 1 - th * th

    # --- A_t on the stacked state [h; c] -------------------------------------
    dc_dh = ((c_prev * fp).unsqueeze(2) * Uf
             + (gg * ip).unsqueeze(2) * Ui
             + (ii * gp).unsqueeze(2) * Ug)                          # (B,n,n) dense
    dc_dc = torch.diag_embed(ff)                                     # diagonal
    dh_dh = (oo * thp).unsqueeze(2) * dc_dh + (th * op).unsqueeze(2) * Uo
    dh_dc = torch.diag_embed(oo * thp * ff)                          # diagonal
    A = torch.cat([torch.cat([dh_dh, dh_dc], 2),
                   torch.cat([dc_dh, dc_dc], 2)], 1)                 # (B,2n,2n)

    # --- immediate Jacobian on [h; c] ----------------------------------------
    one = torch.ones(B, 1, dtype=DT)
    phi = torch.cat([h_prev, x, one], 1)                             # (B,p)
    # dc_i / dtheta^a_i coefficients (a = i, f, g~, o)
    cc = [gg * ip, c_prev * fp, ii * gp, torch.zeros_like(ii)]
    hh_ = [oo * thp * cc[0], oo * thp * cc[1], oo * thp * cc[2], th * op]

    I = torch.zeros(B, 2 * n, P, dtype=DT)
    vh = I[:, :n, :].view(B, n, n, G, p)
    vc = I[:, n:, :].view(B, n, n, G, p)
    idx = torch.arange(n)
    for a in range(G):
        vh[:, idx, idx, a, :] = hh_[a].unsqueeze(2) * phi.unsqueeze(1)
        vc[:, idx, idx, a, :] = cc[a].unsqueeze(2) * phi.unsqueeze(1)
    I = torch.cat([vh.reshape(B, n, P), vc.reshape(B, n, P)], 1)

    # leaky/mixing split: the c-columns of A are exactly diagonal
    lam_h, lam_c = oo * thp * ff, ff                                 # (B,n)
    A_leak = torch.zeros(B, 2 * n, 2 * n, dtype=DT)
    A_leak[:, :n, n:] = torch.diag_embed(lam_h)
    A_leak[:, n:, n:] = torch.diag_embed(lam_c)
    A_mix = A - A_leak
    leak_2norm = torch.sqrt(lam_h ** 2 + lam_c ** 2).max(dim=1).values

    return dict(state=torch.cat([h, c], 1), A=A, I=I, phi=phi,
                A_leak=A_leak, A_mix=A_mix, leak_2norm=leak_2norm,
                blk2=torch.stack([torch.stack([torch.diagonal(dh_dh, dim1=1, dim2=2),
                                               torch.diagonal(dh_dc, dim1=1, dim2=2)], -1),
                                  torch.stack([torch.diagonal(dc_dh, dim1=1, dim2=2),
                                               torch.diagonal(dc_dc, dim1=1, dim2=2)], -1)],
                                 -2))  # (B, n, 2, 2)


def lstm_split_immediate(I, n, G, p):
    B = I.shape[0]
    Id = torch.zeros_like(I)
    for half, sl in ((0, slice(0, n)), (1, slice(n, 2 * n))):
        vi = I[:, sl, :].reshape(B, n, n, G * p)
        vd = torch.zeros_like(vi)
        idx = torch.arange(n)
        vd[:, idx, idx, :] = vi[:, idx, idx, :]
        Id[:, sl, :] = vd.reshape(B, n, n * G * p)
    return Id, I - Id


def lstm_blockdiag_trace(A, S_prev, n):
    """S_t block update: per unit i, the 2x2 block of A_t on rows/cols (i, n+i)."""
    B, P = S_prev.shape[0], S_prev.shape[2]
    idx = torch.arange(n)
    a_hh, a_hc = A[:, idx, idx], A[:, idx, n + idx]
    a_ch, a_cc = A[:, n + idx, idx], A[:, n + idx, n + idx]
    Sh, Sc = S_prev[:, :n, :], S_prev[:, n:, :]
    out_h = a_hh.unsqueeze(2) * Sh + a_hc.unsqueeze(2) * Sc
    out_c = a_ch.unsqueeze(2) * Sh + a_cc.unsqueeze(2) * Sc
    return torch.cat([out_h, out_c], 1)


def lstm_blockdiag_of_A(A, n):
    B = A.shape[0]
    out = torch.zeros_like(A)
    idx = torch.arange(n)
    out[:, idx, idx] = A[:, idx, idx]
    out[:, idx, n + idx] = A[:, idx, n + idx]
    out[:, n + idx, idx] = A[:, n + idx, idx]
    out[:, n + idx, n + idx] = A[:, n + idx, n + idx]
    return out


# ----------------------------------------------------------------------------
# certified spectral-norm bounds
# ----------------------------------------------------------------------------
def rho_bar_paper(A):
    nF = torch.linalg.matrix_norm(A, ord="fro", dim=(1, 2))
    n1 = A.abs().sum(dim=1).max(dim=1).values
    ninf = A.abs().sum(dim=2).max(dim=1).values
    return torch.minimum(nF, torch.sqrt(n1 * ninf))


def rho_bar_split(A, Ablk, blk_2norm):
    """Refined certified bound, using only objects Alg. 1 already forms.

        rho_bar = min( surrogate(A_t),  ||blkdiag(A_t)||_2 + surrogate(Ahat_t) )

    ||blkdiag(A_t)||_2 is EXACT and O(n): max_i |A_ii| for a scalar-diagonal
    trace, max_i ||A^{(ii)}||_2 over 2x2 blocks for the LSTM [h;c] trace.
    Splitting the leaky diagonal out avoids paying its sqrt(n) Frobenius
    inflation, which is what makes the plain surrogate loose on gated cells,
    where the retained diagonal is O(1) rather than O(1/sqrt(n)).
    """
    return torch.minimum(rho_bar_paper(A), blk_2norm + rho_bar_paper(A - Ablk))


def blk2_2norm(A, n):
    """max_i ||A^{(ii)}||_2 over the per-unit 2x2 blocks on rows/cols (i, n+i)."""
    idx = torch.arange(n)
    M = torch.stack([torch.stack([A[:, idx, idx], A[:, idx, n + idx]], -1),
                     torch.stack([A[:, n + idx, idx], A[:, n + idx, n + idx]], -1)], -2)
    return torch.linalg.svdvals(M)[..., 0].max(dim=1).values


# ----------------------------------------------------------------------------
# factored SK-RTRL endpoint (r = n_state, c = append rank, no pre-projection)
# ----------------------------------------------------------------------------
def sk_step(A, C, Q, L, R, r_target):
    """One SK-RTRL propagate/append/truncate step; returns (L, R, tau_r)."""
    B = A.shape[0]
    r_in = L.shape[2]
    if r_in > 0:
        G = R.transpose(1, 2) @ Q                       # (B, r_in, c)
        Qperp_raw = Q - R @ G
    else:
        G = torch.zeros(B, 0, Q.shape[2], dtype=DT)
        Qperp_raw = Q
    Qperp, Theta = torch.linalg.qr(Qperp_raw, mode="reduced")
    Lp = A @ L if r_in > 0 else torch.zeros(B, A.shape[1], 0, dtype=DT)
    left = Lp + (C @ G.transpose(1, 2) if r_in > 0 else
                 torch.zeros(B, A.shape[1], 0, dtype=DT))
    core = torch.cat([left, C @ Theta.transpose(1, 2)], 2)
    U, s, Vh = torch.linalg.svd(core, full_matrices=False)
    k = min(r_target, s.shape[1])
    L_new = U[:, :, :k] * s[:, :k].unsqueeze(1)
    basis = torch.cat([R, Qperp], 2) if r_in > 0 else Qperp
    R_new = basis @ Vh.transpose(1, 2)[:, :, :k]
    tau_r = torch.sqrt((s[:, k:] ** 2).sum(dim=1).clamp_min(0))
    return L_new, R_new, tau_r


# ----------------------------------------------------------------------------
# the GRU test suite
# ----------------------------------------------------------------------------
def test_gru(n=5, m=3, B=2, T=20, seed=0):
    print("\n=== GRU (n=%d, m=%d, B=%d, T=%d) ===" % (n, m, B, T))
    gen = torch.Generator().manual_seed(seed)
    p, G = n + m + 1, GRU_G
    P = n * G * p
    flat = rand_params(n, m, G, gen, scale=1.2)
    xs = torch.randn(T, B, m, generator=gen, dtype=DT) * 0.8
    ok = True

    # --- single-step checks at a generic (non-zero) state --------------------
    h_prev = torch.randn(B, n, generator=gen, dtype=DT) * 0.7
    x = xs[0]
    an = gru_analytic(flat, x, h_prev, n, m)

    ok &= record("G0  forward matches functional reference",
                 maxdev(an["h"], gru_forward(flat, x, h_prev, n, m)))

    A_ad = batch_state_jac(lambda s: gru_forward(flat, x, s, n, m), h_prev)
    ok &= record("G1  analytic A_t == autograd dh_t/dh_{t-1}", maxdev(an["A"], A_ad))

    I_ad = jacobian(lambda th: gru_forward(th, x, h_prev, n, m), flat, vectorize=True)
    ok &= record("G2  analytic I_t == autograd dh_t/dtheta|imm", maxdev(an["I"], I_ad))

    # --- G3 block structure --------------------------------------------------
    Id, Ioff = gru_split_immediate(an["I"], n, G, p)
    voff = Ioff.view(B, n, n, G, p)
    dev_zh = max(voff[:, :, :, 0, :].abs().max().item(),
                 voff[:, :, :, 2, :].abs().max().item())
    ok &= record("G3a z/h~ segments vanish off the block diagonal", dev_zh)
    r_off_mass = voff[:, :, :, 1, :].abs().max().item()
    record("G3b r segment is NOT block diagonal (expect > 0)",
           0.0 if r_off_mass > 1e-6 else 1.0,
           note=f"max|I^off_r| = {r_off_mass:.3e}")
    ok &= r_off_mass > 1e-6
    Khat, Frows = gru_offblock_rows(an["K"], an["phi"], n, m)
    ok &= record("G3c I^off_t == Khat_t @ Frows (exact rank <= n)",
                 maxdev(Ioff, Khat @ Frows))

    # --- G4 append factorisation, at a generic non-zero trace ---------------
    S_prev = torch.zeros(B, n, P, dtype=DT)
    vS = S_prev.view(B, n, n, G * p)
    idx = torch.arange(n)
    vS[:, idx, idx, :] = torch.randn(B, n, G * p, generator=gen, dtype=DT) * 0.3
    Ahat = an["A"] - torch.diag_embed(torch.diagonal(an["A"], dim1=1, dim2=2))
    Nt = Ahat @ S_prev + Ioff
    Tm, Qm = blockwise_qr([S_prev, Frows], n, G, p, tag_offsets=[0, p])
    Ct = torch.cat([Ahat, Khat], 2) @ Tm                            # (B, n, 2n)
    ok &= record("G4a append N_t == C_t Q_t^T exactly", maxdev(Nt, Ct @ Qm.transpose(1, 2)))
    gram = Qm.transpose(1, 2) @ Qm
    ok &= record("G4b Q_t^T Q_t == I (exactly orthonormal)",
                 maxdev(gram, torch.eye(2 * n, dtype=DT).expand_as(gram)))
    # column support: column a*n+i must live in parameter block i
    supp_bad = 0.0
    for a in range(2):
        for i in range(n):
            col = Qm[:, :, a * n + i].view(B, n, G * p).clone()
            col[:, i, :] = 0.0
            supp_bad = max(supp_bad, col.abs().max().item())
    ok &= record("G4c every Q_t column supported on one block", supp_bad)
    rk = torch.linalg.matrix_rank(Nt, tol=1e-9)
    record("G4d rank(append) <= n", 0.0 if int(rk.max()) <= n else 1.0,
           note=f"rank = {rk.tolist()} (<= n = {n})")

    # --- G5 / G6 T-step recursion -------------------------------------------
    h = torch.zeros(B, n, dtype=DT)
    J = torch.zeros(B, n, P, dtype=DT)
    S = torch.zeros(B, n, P, dtype=DT)
    L = torch.zeros(B, n, 0, dtype=DT)
    R = torch.zeros(B, P, 0, dtype=DT)
    dev_J, dev_end, dev_tau, rho_bad, tight = 0.0, 0.0, 0.0, 0.0, []
    rank_res = 0
    for t in range(T):
        an = gru_analytic(flat, xs[t], h, n, m)
        A, I = an["A"], an["I"]
        Id, Ioff = gru_split_immediate(I, n, G, p)
        Khat, Frows = gru_offblock_rows(an["K"], an["phi"], n, m)
        Ahat = A - torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))

        # SK-RTRL endpoint bookkeeping (needs S_{t-1})
        Tm, Qm = blockwise_qr([S, Frows], n, G, p, tag_offsets=[0, p])
        Ct = torch.cat([Ahat, Khat], 2) @ Tm
        L, R, tau_r = sk_step(A, Ct, Qm, L, R, r_target=n)
        dev_tau = max(dev_tau, tau_r.abs().max().item())

        # exact influence matrix + block-diagonal trace
        J = A @ J + I
        S = torch.diagonal(A, dim1=1, dim2=2).unsqueeze(2) * S + Id
        h = an["h"]

        est = S + L @ R.transpose(1, 2)
        dev_end = max(dev_end, maxdev(est, J))
        rank_res = max(rank_res, int(torch.linalg.matrix_rank(J - S, tol=1e-9).max()))

        # certified bounds
        s2 = torch.linalg.svdvals(A)[:, 0]
        rb = rho_bar_paper(A)
        Ablk = torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))
        rg = rho_bar_split(A, Ablk, Ablk.abs().amax(dim=(1, 2)))
        rho_bad = max(rho_bad, float((s2 - rb).clamp_min(0).max()),
                      float((s2 - rg).clamp_min(0).max()))
        tight.append((float(rb.mean() / s2.mean()), float(rg.mean() / s2.mean())))

    # autograd reference for dh_T/dtheta of the unrolled sequence
    def unroll(th):
        hh = torch.zeros(B, n, dtype=DT)
        for t in range(T):
            hh = gru_forward(th, xs[t], hh, n, m)
        return hh
    J_ad = jacobian(unroll, flat, vectorize=True)
    ok &= record("G5  T-step J_T (recursion) == autograd dh_T/dtheta", maxdev(J, J_ad))
    ok &= record("G6a factored endpoint tau_r == 0 (Cor. 3)", dev_tau)
    ok &= record("G6b S_t + L_t R_t^T == J_t for all t (r=n, c=2n)", dev_end)
    record("G6c rank(J_t - S_t) <= n", 0.0 if rank_res <= n else 1.0,
           note=f"max rank = {rank_res} (<= n = {n})")
    ok &= record("G7  rho_bar_t >= ||A_t||_2 (paper + gated refinement)", rho_bad)
    rb_m = sum(a for a, _ in tight) / len(tight)
    rg_m = sum(b for _, b in tight) / len(tight)
    print(f"      certificate looseness rho_bar/||A||_2 : paper {rb_m:.2f}x, "
          f"gated refinement {rg_m:.2f}x")
    return ok


# ----------------------------------------------------------------------------
# the LSTM test suite
# ----------------------------------------------------------------------------
def test_lstm(n=5, m=3, B=2, T=20, seed=1):
    print("\n=== LSTM (n=%d, m=%d, B=%d, T=%d) ===" % (n, m, B, T))
    gen = torch.Generator().manual_seed(seed)
    p, G = n + m + 1, LSTM_G
    P = n * G * p
    flat = rand_params(n, m, G, gen, scale=1.2)
    xs = torch.randn(T, B, m, generator=gen, dtype=DT) * 0.8
    ok = True

    state = torch.randn(B, 2 * n, generator=gen, dtype=DT) * 0.7
    x = xs[0]
    an = lstm_analytic(flat, x, state, n, m)

    ok &= record("L0  forward matches functional reference",
                 maxdev(an["state"], lstm_forward(flat, x, state, n, m)))
    A_ad = batch_state_jac(lambda s: lstm_forward(flat, x, s, n, m), state)
    ok &= record("L1  analytic A_t == autograd dx_t/dx_{t-1}", maxdev(an["A"], A_ad))
    I_ad = jacobian(lambda th: lstm_forward(th, x, state, n, m), flat, vectorize=True)
    ok &= record("L2  analytic I_t == autograd dx_t/dtheta|imm", maxdev(an["I"], I_ad))

    Id, Ioff = lstm_split_immediate(an["I"], n, G, p)
    ok &= record("L3  I_t IS block diagonal on [h;c] (I^off == 0)",
                 Ioff.abs().max().item())

    # a generic non-zero, block-structured trace
    S_prev = torch.zeros(B, 2 * n, P, dtype=DT)
    idx = torch.arange(n)
    for half, off in ((0, 0), (1, n)):
        v = S_prev[:, off:off + n, :].view(B, n, n, G * p).clone()
        v[:, idx, idx, :] = torch.randn(B, n, G * p, generator=gen, dtype=DT) * 0.3
        S_prev[:, off:off + n, :] = v.reshape(B, n, P)
    Ahat = an["A"] - lstm_blockdiag_of_A(an["A"], n)
    Nt = Ahat @ S_prev
    Tm, Qm = blockwise_qr([S_prev[:, :n, :], S_prev[:, n:, :]], n, G, p,
                          tag_offsets=[0, p])
    Ct = Ahat @ Tm
    ok &= record("L4a append N_t == C_t Q_t^T exactly", maxdev(Nt, Ct @ Qm.transpose(1, 2)))
    gram = Qm.transpose(1, 2) @ Qm
    ok &= record("L4b Q_t^T Q_t == I (per-block QR of the 2 rows)",
                 maxdev(gram, torch.eye(2 * n, dtype=DT).expand_as(gram)))
    rk = torch.linalg.matrix_rank(Nt, tol=1e-9)
    record("L4c rank(append) <= 2n", 0.0 if int(rk.max()) <= 2 * n else 1.0,
           note=f"rank = {rk.tolist()} (<= 2n = {2 * n})")

    # trace update must use the 2x2 blocks, not the scalar diagonal
    dev_blk = maxdev(lstm_blockdiag_trace(an["A"], S_prev, n),
                     lstm_blockdiag_of_A(an["A"], n) @ S_prev)
    ok &= record("L4d 2x2-block trace update == blkdiag2(A) S_{t-1}", dev_blk)

    # the c-columns of A_t are exactly diagonal (no peepholes), hence removed
    # wholesale by blkdiag2 -> Ahat has zero c-columns -> the append only ever
    # touches the h-ROWS of S_{t-1}, one vector per block, so the vanilla
    # row-normalised factorisation applies verbatim and the append rank is <= n.
    ok &= record("L4f Ahat_t has zero c-columns (rank <= n)",
                 Ahat[:, :, n:].abs().max().item())
    s_norm = S_prev[:, :n, :].norm(dim=2)                            # (B,n)
    Qs = torch.where(s_norm.unsqueeze(2) > 0,
                     S_prev[:, :n, :] / s_norm.clamp_min(1e-300).unsqueeze(2),
                     torch.zeros_like(S_prev[:, :n, :]))
    C_simple = Ahat[:, :, :n] * s_norm.unsqueeze(1)                  # (B,2n,n)
    ok &= record("L4g append == (Ahat^h D_s) Q^T, vanilla factorisation",
                 maxdev(Nt, C_simple @ Qs))
    dev_scalar = maxdev(torch.diagonal(an["A"], dim1=1, dim2=2).unsqueeze(2) * S_prev,
                        lstm_blockdiag_of_A(an["A"], n) @ S_prev)
    record("L4e scalar-diagonal update is WRONG (expect > 0)",
           0.0 if dev_scalar > 1e-6 else 1.0, note=f"gap = {dev_scalar:.3e}")

    # --- T-step recursion + factored endpoint --------------------------------
    st = torch.zeros(B, 2 * n, dtype=DT)
    J = torch.zeros(B, 2 * n, P, dtype=DT)
    S = torch.zeros(B, 2 * n, P, dtype=DT)
    L = torch.zeros(B, 2 * n, 0, dtype=DT)
    R = torch.zeros(B, P, 0, dtype=DT)
    dev_end, dev_tau, rho_bad, tight = 0.0, 0.0, 0.0, []
    rank_res = 0
    for t in range(T):
        an = lstm_analytic(flat, xs[t], st, n, m)
        A, I = an["A"], an["I"]
        Id, _ = lstm_split_immediate(I, n, G, p)
        Ahat = A - lstm_blockdiag_of_A(A, n)

        Tm, Qm = blockwise_qr([S[:, :n, :], S[:, n:, :]], n, G, p, tag_offsets=[0, p])
        Ct = Ahat @ Tm
        L, R, tau_r = sk_step(A, Ct, Qm, L, R, r_target=2 * n)
        dev_tau = max(dev_tau, tau_r.abs().max().item())

        J = A @ J + I
        S = lstm_blockdiag_trace(A, S, n) + Id
        st = an["state"]

        dev_end = max(dev_end, maxdev(S + L @ R.transpose(1, 2), J))
        rank_res = max(rank_res, int(torch.linalg.matrix_rank(J - S, tol=1e-9).max()))

        s2 = torch.linalg.svdvals(A)[:, 0]
        rb = rho_bar_paper(A)
        Ablk = lstm_blockdiag_of_A(A, n)
        rg = rho_bar_split(A, Ablk, blk2_2norm(A, n))
        rho_bad = max(rho_bad, float((s2 - rb).clamp_min(0).max()),
                      float((s2 - rg).clamp_min(0).max()))
        tight.append((float(rb.mean() / s2.mean()), float(rg.mean() / s2.mean())))

    def unroll(th):
        s0 = torch.zeros(B, 2 * n, dtype=DT)
        for t in range(T):
            s0 = lstm_forward(th, xs[t], s0, n, m)
        return s0
    J_ad = jacobian(unroll, flat, vectorize=True)
    ok &= record("L5  T-step J_T (recursion) == autograd dx_T/dtheta", maxdev(J, J_ad))
    ok &= record("L6a factored endpoint tau_r == 0 (Cor. 3, r=2n)", dev_tau)
    ok &= record("L6b S_t + L_t R_t^T == J_t for all t", dev_end)
    record("L6c rank(J_t - S_t) <= 2n", 0.0 if rank_res <= 2 * n else 1.0,
           note=f"max rank = {rank_res} (<= 2n = {2 * n})")
    ok &= record("L7  rho_bar_t >= ||A_t||_2 (paper + gated refinement)", rho_bad)
    rb_m = sum(a for a, _ in tight) / len(tight)
    rg_m = sum(b for _, b in tight) / len(tight)
    print(f"      certificate looseness rho_bar/||A||_2 : paper {rb_m:.2f}x, "
          f"gated refinement {rg_m:.2f}x")
    return ok



# ----------------------------------------------------------------------------
# certificate-tightness probe at realistic widths (no autograd, informational)
# ----------------------------------------------------------------------------
def probe_certificate(widths=(32, 64, 128, 256), T=8, m_over_n=0.25, seed=11):
    """How loose is rho_bar_t on gated cells, and does the split-diagonal
    refinement help?  Informational: reports ratios, asserts only validity."""
    print()
    print("=== certificate tightness at realistic widths (informational) ===")
    print(f"  {'cell':<6s}{'n':>6s}{'||A||_2':>10s}{'paper':>10s}{'split':>10s}"
          f"{'paper/2':>9s}{'split/2':>9s}")
    bad = 0.0
    for n in widths:
        m = max(1, int(m_over_n * n))
        for kind, sat in (("gru", 0.0), ("lstm", 0.0), ("gru", 4.0), ("lstm", 4.0)):
            gen = torch.Generator().manual_seed(seed + n)
            G = GRU_G if kind == "gru" else LSTM_G
            flat = rand_params(n, m, G, gen, scale=1.2)
            if sat:
                # saturated-gate regime: GRU update gate closed (z -> 0, A -> I),
                # LSTM forget gate open (f -> 1) -- the long-memory regime a
                # trained gated network actually occupies.
                p_ = n + m + 1
                a_ = 0 if kind == "gru" else 1
                sgn = -1.0 if kind == "gru" else 1.0
                v = flat.view(n, G, p_)
                v[:, a_, n + m] = sgn * sat
            ns = n if kind == "gru" else 2 * n
            st = torch.zeros(1, ns, dtype=DT)
            acc = torch.zeros(3, dtype=DT)
            for t in range(T):
                x = torch.randn(1, m, generator=gen, dtype=DT) * 0.8
                if kind == "gru":
                    an = gru_analytic(flat, x, st, n, m)
                    A = an["A"]
                    Ablk = torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))
                    blkn = Ablk.abs().amax(dim=(1, 2))
                    st = an["h"]
                else:
                    an = lstm_analytic(flat, x, st, n, m)
                    A = an["A"]
                    Ablk = lstm_blockdiag_of_A(A, n)
                    blkn = blk2_2norm(A, n)
                    st = an["state"]
                s2 = torch.linalg.svdvals(A)[:, 0]
                rb = rho_bar_paper(A)
                rg = rho_bar_split(A, Ablk, blkn)
                bad = max(bad, float((s2 - rb).clamp_min(0).max()),
                          float((s2 - rg).clamp_min(0).max()))
                acc += torch.stack([s2.mean(), rb.mean(), rg.mean()])
            a = acc / T
            tag = kind + ("*" if sat else "")
            print(f"  {tag:<6s}{n:>6d}{a[0]:>10.3f}{a[1]:>10.3f}{a[2]:>10.3f}"
                  f"{a[1] / a[0]:>9.2f}{a[2] / a[0]:>9.2f}")
    return record("P1  rho_bar_t >= ||A_t||_2 at all probed widths", bad)


def main():
    torch.manual_seed(0)
    print(f"torch {torch.__version__}  dtype {DT}  tol {TOL:g}")
    ok = True
    ok &= test_gru()
    ok &= test_lstm()
    # a second seed / different width, to rule out a lucky configuration
    print("\n--- repeat at n=7, m=4, seed=5 ---")
    ok &= test_gru(n=7, m=4, B=2, T=20, seed=5)
    ok &= test_lstm(n=7, m=4, B=2, T=20, seed=7)
    ok &= probe_certificate()

    worst = max((d for _, d, _, _ in RESULTS if math.isfinite(d)), default=0.0)
    nfail = sum(1 for _, _, o, _ in RESULTS if not o)
    print("\n" + "=" * 72)
    if ok and nfail == 0:
        print(f"PASS  {len(RESULTS)} checks, max deviation over all checks = {worst:.3e}")
        return 0
    print(f"FAIL  {nfail}/{len(RESULTS)} checks failed")
    for tag, d, o, note in RESULTS:
        if not o:
            print(f"   - {tag}: {d:.3e} {note}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
