r"""Cross-check skrtrl.cells_gated against the reference derivation.

`test_gated_jacobians.py` validates the DERIVATION against torch.autograd with
a standalone re-implementation. This file validates the IMPLEMENTATION in
`skrtrl/cells_gated.py` against that reference, item by item, plus two
end-to-end checks that close the loop back to autograd:

  C1   forward == reference functional forward
  C2   jac_pieces A_t == reference analytic A_t (hence == autograd)
  C3   jac_pieces imm == block-diagonal restriction of the reference I_t
  C4   imm_offblock_norm() == ||I_t^perp||_F from the reference
  C5   blkdiag_of_A / offdiag_of_A == reference (scalar diag for GRU,
       per-unit 2x2 blocks for LSTM)
  C6   trace_update == blkdiag_of_A(A) @ S_dense + I^o, on a random trace
  C7   append_factors: C @ Q.dense()^T == Ahat_t S_{t-1} + I_t^perp exactly
  C8   Q columns exactly orthonormal, each supported on one parameter block;
       StructuredQ.gram() == Q.dense()^T Q.dense()
  C9   append_mass() == ||N_t||_F  (exact, because Q is orthonormal)
  C10  LSTM only: Ahat_t has zero c-columns, and the omega_2n (peephole-ready)
       append route reproduces N_t exactly as well
  C11  degenerate trace (S_{t-1} = 0, i.e. t = 1) still factors exactly and
       still yields an orthonormal Q
  C12  T=20 exact endpoint through the PUBLIC interface only: trace_update +
       append_factors + the SK-RTRL propagate/truncate step at r = n_state
       give tau_r == 0 and S_t + L_t R_t^T == autograd dx_t/dtheta at every t
  C13  apply_flat_grad: g_rows = delta^T J_T scattered into U/W/b grads
       matches autograd's parameter gradients of delta . x_T
  C14  spectral_clip(clip) clips every U_a separately

float64, CPU only (no CUDA needed).
Run:
  D:\Anaconda\envs\multilingual_lora\python.exe code/tests/test_gated_cells.py
"""
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))      # code/   -> `skrtrl`
sys.path.insert(0, _HERE)                       # tests/  -> the reference

from skrtrl.cells_gated import GRUCell, LSTMCell, make_gated_cell  # noqa: E402
import test_gated_jacobians as ref                                 # noqa: E402

DT = torch.float64
TOL = 1e-8
RESULTS = []


# ---------------------------------------------------------------------------
def record(tag, dev, note="", tol=TOL):
    ok = bool(dev <= tol) and math.isfinite(dev)
    RESULTS.append((tag, dev, ok, note))
    print(f"  [{'ok ' if ok else 'FAIL'}] {tag:<54s} max|dev| = {dev:.3e}   {note}")
    return ok


def maxdev(a, b):
    return (a - b).abs().max().item()


def flat_from_cell(cell):
    """Pack the cell's parameters into the reference's flat per-unit layout."""
    n, m, G, pg = cell.n, cell.m, cell.G, cell.p_gate
    flat = torch.zeros(cell.P, dtype=DT)
    v = flat.view(n, G, pg)
    v[:, :, :n] = cell.U.detach().permute(1, 0, 2)
    v[:, :, n:n + m] = cell.W.detach().permute(1, 0, 2)
    v[:, :, n + m] = cell.b.detach().permute(1, 0)
    return flat


def expand(compact, blk_idx, p_cell, n_units):
    """(B, n_s, p_cell) block-compact -> (B, n_s, P) dense."""
    B, ns, pc = compact.shape
    out = compact.new_zeros(B, ns, n_units, pc)
    s = torch.arange(ns, device=compact.device)
    out[:, s, blk_idx[s], :] = compact
    return out.reshape(B, ns, n_units * pc)


def compactify(dense, blk_idx, p_cell, n_units):
    """(B, n_s, P) dense -> (B, n_s, p_cell), keeping only the diagonal blocks."""
    B, ns, _ = dense.shape
    v = dense.view(B, ns, n_units, p_cell)
    s = torch.arange(ns, device=dense.device)
    return v[:, s, blk_idx[s], :]


def offblock_fro(dense, blk_idx, p_cell, n_units):
    """||I^perp||_F per batch lane, from the dense immediate Jacobian."""
    keep = compactify(dense, blk_idx, p_cell, n_units)
    tot = (dense ** 2).sum(dim=(1, 2))
    kept = (keep ** 2).sum(dim=(1, 2))
    return (tot - kept).clamp_min(0).sqrt()


def randomise(cell, gen, scale=1.5):
    """Make the random init non-trivial (the default init is small)."""
    with torch.no_grad():
        cell.U.mul_(scale)
        cell.W.mul_(scale)
        cell.b.copy_((torch.rand(cell.b.shape, generator=gen, dtype=DT) * 2 - 1) * 0.4)


def random_trace(cell, B, gen, scale=0.3):
    return (torch.rand(B, cell.n_state, cell.p_cell, generator=gen, dtype=DT)
            * 2 - 1) * scale


# ---------------------------------------------------------------------------
def run_cell(kind, n=5, m=3, B=2, T=20, seed=0):
    print(f"\n=== {kind.upper()} cell (n={n}, m={m}, B={B}, T={T}, seed={seed}) ===")
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    cell = make_gated_cell(kind, m, n, dtype=DT)
    randomise(cell, gen)
    flat = flat_from_cell(cell)
    blk, pc, nu, ns = cell.blk_idx, cell.p_cell, cell.n, cell.n_state
    ok = True

    # sizes
    exp_G = 3 if kind == "gru" else 4
    ok &= record("C0  sizes (G, p_gate, p_cell, n_state, P, p alias)",
                 0.0 if (cell.G == exp_G and cell.p_gate == n + m + 1
                         and cell.p_cell == exp_G * (n + m + 1)
                         and cell.n_state == (n if kind == "gru" else 2 * n)
                         and cell.P == n * cell.p_cell
                         and cell.p == cell.p_cell) else 1.0,
                 note=f"G={cell.G} p_gate={cell.p_gate} p_cell={cell.p_cell} "
                      f"n_state={cell.n_state} P={cell.P}")

    xs = torch.randn(T, B, m, generator=gen, dtype=DT) * 0.8
    st = torch.randn(B, ns, generator=gen, dtype=DT) * 0.7
    x = xs[0]

    fwd = ref.gru_forward if kind == "gru" else ref.lstm_forward
    ana = ref.gru_analytic if kind == "gru" else ref.lstm_analytic

    # --- C1 / C2 / C3 / C4 -------------------------------------------------
    out = cell(x, st)
    ok &= record("C1  forward == reference functional forward",
                 maxdev(out, fwd(flat, x, st, n, m)))

    A, imm = cell.jac_pieces(x, st, out)
    R = ana(flat, x, st, n, m)
    ok &= record("C2  jac_pieces A_t == reference A_t", maxdev(A, R["A"]))
    ok &= record("C3  jac_pieces imm == blockdiag restriction of I_t",
                 maxdev(imm, compactify(R["I"], blk, pc, nu)))
    ok &= record("C4  imm_offblock_norm == ||I^perp||_F",
                 maxdev(cell.imm_offblock_norm(), offblock_fro(R["I"], blk, pc, nu)))
    if kind == "gru":
        ok &= record("C4b ||I^perp||_F > 0 (reset-gate coupling present)",
                     0.0 if float(cell.imm_offblock_norm().min()) > 1e-6 else 1.0,
                     note=f"||I^perp||_F = {cell.imm_offblock_norm().tolist()}")
    else:
        ok &= record("C4b ||I^perp||_F == 0 (LSTM is block diagonal)",
                     float(cell.imm_offblock_norm().abs().max()))

    # C4c: what the exact shadow (ExactRTRL) will consume -- the FULL I_t,
    # block-diagonal part plus I^perp. Getting this wrong makes the shadow,
    # i.e. the ground truth for every gradient-cosine number, silently wrong.
    ok &= record("C4c imm_dense == reference full I_t",
                 maxdev(cell.imm_dense(imm), R["I"]))

    # --- C5 blkdiag / offdiag ---------------------------------------------
    if kind == "gru":
        ref_blk = torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))
    else:
        ref_blk = ref.lstm_blockdiag_of_A(A, n)
    ok &= record("C5a blkdiag_of_A == reference", maxdev(cell.blkdiag_of_A(A), ref_blk))
    ok &= record("C5b offdiag_of_A == A - blkdiag", maxdev(cell.offdiag_of_A(A), A - ref_blk))

    # --- C6 trace update --------------------------------------------------
    S = random_trace(cell, B, gen)
    S_dense = expand(S, blk, pc, nu)
    Idiag_dense = expand(imm, blk, pc, nu)
    got = expand(cell.trace_update(A, S, imm), blk, pc, nu)
    want = ref_blk @ S_dense + Idiag_dense
    ok &= record("C6  trace_update == blkdiag(A) S_{t-1} + I^o", maxdev(got, want))
    if kind == "lstm":
        scalar = torch.diagonal(A, dim1=1, dim2=2).unsqueeze(2) * S_dense + Idiag_dense
        d = maxdev(got, scalar)
        ok &= record("C6b scalar-diagonal update would be WRONG (expect > 0)",
                     0.0 if d > 1e-6 else 1.0, note=f"gap = {d:.3e}")

    # --- C7 / C8 / C9 append ---------------------------------------------
    Ioff_dense = R["I"] - Idiag_dense_from_ref(R["I"], blk, pc, nu)
    Ahat = A - ref_blk
    N_want = Ahat @ S_dense + Ioff_dense
    C, Q = cell.append_factors(A, S)
    Qd = Q.dense()
    ok &= record("C7  C @ Q^T == Ahat S_{t-1} + I^perp", maxdev(C @ Qd.transpose(1, 2), N_want),
                 note=f"append width omega = {Q.width}")
    gram = Qd.transpose(1, 2) @ Qd
    eye = torch.eye(Q.width, dtype=DT).expand_as(gram)
    ok &= record("C8a Q^T Q == I (exactly orthonormal)", maxdev(gram, eye))
    ok &= record("C8b StructuredQ.gram() == dense gram", maxdev(Q.gram(), gram))
    supp = 0.0
    for k in range(Q.width):
        col = Qd[:, :, k].view(B, nu, pc).clone()
        col[:, int(Q.blocks[k]), :] = 0.0
        supp = max(supp, col.abs().max().item())
    ok &= record("C8c every Q column supported on one parameter block", supp)
    ok &= record("C9  append_mass == ||N_t||_F",
                 maxdev(cell.append_mass(A, S),
                        torch.linalg.matrix_norm(N_want, ord="fro", dim=(1, 2))))

    # --- C10 LSTM specifics ----------------------------------------------
    if kind == "lstm":
        ok &= record("C10a Ahat_t has zero c-columns (Prop. 7)",
                     float(cell.append_c_column_mass(A).abs().max()))
        C2_, Q2_ = cell.append_factors(A, S, mode="omega_2n")
        ok &= record("C10b omega_2n append route also exact",
                     maxdev(C2_ @ Q2_.dense().transpose(1, 2), N_want),
                     note=f"omega = {Q2_.width}")
        g2 = Q2_.dense().transpose(1, 2) @ Q2_.dense()
        ok &= record("C10c omega_2n Q orthonormal",
                     maxdev(g2, torch.eye(Q2_.width, dtype=DT).expand_as(g2)))

    # --- C11 degenerate trace --------------------------------------------
    S0 = torch.zeros_like(S)
    C0, Q0 = cell.append_factors(A, S0)
    N0 = Ahat @ expand(S0, blk, pc, nu) + Ioff_dense
    ok &= record("C11a exact append at S_{t-1} = 0", maxdev(C0 @ Q0.dense().transpose(1, 2), N0))
    g0 = Q0.dense().transpose(1, 2) @ Q0.dense()
    ok &= record("C11b Q still orthonormal at S_{t-1} = 0",
                 maxdev(g0, torch.eye(Q0.width, dtype=DT).expand_as(g0)))

    # --- C12 T-step exact endpoint through the public interface ----------
    S = torch.zeros(B, ns, pc, dtype=DT)
    L = torch.zeros(B, ns, 0, dtype=DT)
    Rf = torch.zeros(B, cell.P, 0, dtype=DT)
    state = torch.zeros(B, ns, dtype=DT)
    dev_end, dev_tau = 0.0, 0.0
    Js = []
    for t in range(T):
        nxt = cell(x=xs[t], **{("h_prev" if kind == "gru" else "state_prev"): state})
        A, imm = cell.jac_pieces(xs[t], state, nxt)
        C, Q = cell.append_factors(A, S)           # must read S_{t-1}
        L, Rf, tau_r = ref.sk_step(A, C, Q.dense(), L, Rf, r_target=ns)
        dev_tau = max(dev_tau, tau_r.abs().max().item())
        S = cell.trace_update(A, S, imm)
        state = nxt.detach()
        Js.append(expand(S, blk, pc, nu) + L @ Rf.transpose(1, 2))

    def unroll(th, upto):
        s0 = torch.zeros(B, ns, dtype=DT)
        for t in range(upto):
            s0 = fwd(th, xs[t], s0, n, m)
        return s0

    for t in (0, T // 2 - 1, T - 1):
        J_ad = torch.autograd.functional.jacobian(
            lambda th, u=t + 1: unroll(th, u), flat, vectorize=True)
        dev_end = max(dev_end, maxdev(Js[t], J_ad))
    ok &= record("C12a exact endpoint tau_r == 0 over T steps", dev_tau)
    ok &= record(f"C12b S_t + L_t R_t^T == autograd dx_t/dtheta (t=1,{T//2},{T})",
                 dev_end, note=f"r = n_state = {ns}")

    # --- C13 apply_flat_grad ---------------------------------------------
    J_T = torch.autograd.functional.jacobian(
        lambda th: unroll(th, T), flat, vectorize=True)          # (B, ns, P)
    delta = torch.randn(B, ns, generator=gen, dtype=DT)
    g_rows = torch.einsum("bs,bsq->q", delta, J_T).view(nu, pc)   # summed over lanes
    cell.zero_grad(set_to_none=True)
    cell.apply_flat_grad(g_rows)
    # autograd reference on the cell's own parameters
    st2 = torch.zeros(B, ns, dtype=DT)
    for t in range(T):
        st2 = cell(x=xs[t], **{("h_prev" if kind == "gru" else "state_prev"): st2})
    loss = (delta * st2).sum()
    gU, gW, gb = torch.autograd.grad(loss, [cell.U, cell.W, cell.b])
    d13 = max(maxdev(cell.U.grad, gU), maxdev(cell.W.grad, gW), maxdev(cell.b.grad, gb))
    ok &= record("C13 apply_flat_grad matches autograd param grads", d13)
    ok &= record("C13b flat_grad_rows is the exact inverse of apply_flat_grad",
                 maxdev(cell.flat_grad_rows(), g_rows))

    # --- C15 the exact algos.py rewrite recommended in the notes ----------
    # These two einsum/index_add_ forms are what GRU_INTEGRATION_NOTES.md sec 1.3
    # tells the next agent to substitute for lines 289-290 and 293-295 of
    # algos.py. Checked here so that diff does not have to be debugged live.
    A, imm = cell.jac_pieces(xs[0], torch.randn(B, ns, generator=gen, dtype=DT) * 0.7,
                             None)
    S = random_trace(cell, B, gen)
    Cf, Qf = cell.append_factors(A, S)
    w, cc, r_in = Qf.width, max(2, Qf.width // 2), 3
    Vc = torch.linalg.qr(torch.randn(B, w, cc, generator=gen, dtype=DT))[0]
    Rmat = torch.randn(B, cell.P, r_in, generator=gen, dtype=DT)
    Qd = Qf.dense()

    Qc = S.new_zeros(B, nu, pc, cc)
    Qc.index_add_(1, Qf.blocks, torch.einsum("bwq,bwk->bwqk", Qf.rows, Vc))
    ok &= record("C15a proposed Qc build == Q.dense() @ Vc",
                 maxdev(Qc.reshape(B, cell.P, cc), Qd @ Vc))

    Rv = Rmat.view(B, nu, pc, r_in)
    G0 = torch.einsum("bwqr,bwq->brw", Rv[:, Qf.blocks], Qf.rows)
    ok &= record("C15b proposed G0 build == R^T Q.dense()",
                 maxdev(G0, Rmat.transpose(1, 2) @ Qd))
    ok &= record("C15c append_width == actual omega",
                 0.0 if cell.append_width == w else 1.0, note=f"omega = {w}")

    # --- C14 spectral clip ------------------------------------------------
    clip = 0.5
    cell.spectral_clip_(clip)
    worst = max(float(torch.linalg.matrix_norm(cell.U[a], ord=2)) for a in range(cell.G))
    ok &= record("C14 spectral_clip clips every U_a", max(0.0, worst - clip - 1e-12),
                 note=f"max ||U_a||_2 = {worst:.6f} <= {clip}")
    return ok


def Idiag_dense_from_ref(I_dense, blk, pc, nu):
    """Dense block-diagonal restriction of a reference immediate Jacobian."""
    return expand(compactify(I_dense, blk, pc, nu), blk, pc, nu)


def main():
    print(f"torch {torch.__version__}  dtype {DT}  tol {TOL:g}  device cpu")
    ok = True
    ok &= run_cell("gru", n=5, m=3, B=2, T=20, seed=0)
    ok &= run_cell("lstm", n=5, m=3, B=2, T=20, seed=1)
    print("\n--- repeat at n=7, m=4 ---")
    ok &= run_cell("gru", n=7, m=4, B=2, T=20, seed=5)
    ok &= run_cell("lstm", n=7, m=4, B=2, T=20, seed=7)

    worst = max((d for _, d, _, _ in RESULTS if math.isfinite(d)), default=0.0)
    nfail = sum(1 for _, _, o, _ in RESULTS if not o)
    print("\n" + "=" * 74)
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
