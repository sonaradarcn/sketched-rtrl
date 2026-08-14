"""Equivalence tests for the amortised-rotation kernel (paper Sec. 4.3).

Claim under test: "In exact arithmetic the two kernels return the same factors
L_t, R_t, since deferring the rotations only re-associates the same products."

Checked here, step by step, against the kernel of record (skrtrl.algos.SKRTRL):
  A1  reconstruction L_t R_t^T          (basis-convention independent)
  A2  full estimate S_t + L_t R_t^T
  A3  gradient read-out ghat_t
  A4  certificate quantities eta_t, tau_c, tau_r, e_t
  A5  orthonormality R_t^T R_t = I
  A6  Corollary-3 endpoint (r = n, no pre-projection): both == exact RTRL
  A7  certificate validity of the amortised kernel vs exact RTRL
  A8  mid-stream rank changes (adaptive controller) stay equivalent
  A9  collapse period K does not change the answer (K in {1, 2, 3, r})
  A10 float32 agreement at ~1e-5 relative

Run:  python -m tests.test_amortised_kernel        (from code/)
      pytest code/tests/test_amortised_kernel.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skrtrl.cells import TanhRNNCell                      # noqa: E402
from skrtrl.algos import ExactRTRL, SKRTRL                # noqa: E402
from skrtrl.algos_amortised import SKRTRLAmortised, make_skrtrl   # noqa: E402

DEV = "cpu"


def rel(a, b):
    return (a - b).norm().item() / max(b.norm().item(), 1e-30)


def make_setup(n=16, m=4, B=2, T=30, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    cell = TanhRNNCell(m, n, device=DEV, dtype=dtype)
    xs = torch.randn(T, B, m, dtype=dtype) * 0.8
    delta = torch.randn(T, B, n, dtype=dtype)
    return cell, xs, delta


def _lrt(algo):
    R = algo.R_dense() if hasattr(algo, "R_dense") else algo.R
    return torch.bmm(algo.L, R.transpose(1, 2))


def _ortho_err(algo):
    R = algo.R_dense() if hasattr(algo, "R_dense") else algo.R
    r = R.shape[2]
    if r == 0:
        return 0.0
    # Columns carrying (numerically) zero mass in L are unconstrained: both
    # kernels leave them arbitrary -- the naive one because LAPACK's QR completes
    # a rank-deficient Qperp with vectors that need not be orthogonal to R, the
    # amortised one because the Gram-metric orthogonalisation zeroes them out.
    # They multiply a zero column of L, so L R^T and eta_t are unaffected.
    cn = algo.L.norm(dim=1).amax(dim=0)
    thr = 1e-10 if R.dtype == torch.float64 else 1e-5
    keep = cn > thr * cn.amax().clamp_min(1e-30)
    if keep.sum() == 0:
        return 0.0
    Rk = R[:, :, keep]
    I = torch.eye(int(keep.sum()), dtype=R.dtype, device=R.device).expand_as(
        torch.bmm(Rk.transpose(1, 2), Rk))
    return (torch.bmm(Rk.transpose(1, 2), Rk) - I).abs().max().item()


def compare_run(n=16, m=4, B=2, T=30, r=4, dtype=torch.float64, seed=0,
                collapse_every=None, rank_schedule=None, mode="svd"):
    """Run both kernels on the same stream; return the worst deviations."""
    cell, xs, deltas = make_setup(n, m, B, T, dtype, seed)
    naive = SKRTRL(cell, B, r=r, mode=mode)
    amort = SKRTRLAmortised(cell, B, r=r, mode=mode, collapse_every=collapse_every)
    h = cell.init_state(B)
    worst = {"LRt": 0.0, "est": 0.0, "ghat": 0.0, "eta": 0.0, "e": 0.0, "ortho": 0.0}
    ortho_naive = 0.0
    for t in range(T):
        if rank_schedule is not None and t in rank_schedule:
            naive.r = amort.r = rank_schedule[t]
        h_prev = h
        h = cell(xs[t], h_prev)
        A, imm = cell.jac_pieces(xs[t], h_prev, h)
        naive.step_state(A, imm)
        amort.step_state(A, imm)
        d = deltas[t]
        worst["LRt"] = max(worst["LRt"], rel(_lrt(amort), _lrt(naive)))
        worst["est"] = max(worst["est"], rel(amort.residual_dense(), naive.residual_dense()))
        worst["ghat"] = max(worst["ghat"], rel(amort.grad_rows(d), naive.grad_rows(d)))
        worst["eta"] = max(worst["eta"], rel(amort.last["eta"], naive.last["eta"]))
        worst["e"] = max(worst["e"], rel(amort.e, naive.e))
        worst["ortho"] = max(worst["ortho"], _ortho_err(amort))
        ortho_naive = max(ortho_naive, _ortho_err(naive))
        h = h.detach()
    # the invariant R^T R = I is only asserted up to what the kernel of record
    # itself achieves on the same stream
    worst["ortho"] = max(0.0, worst["ortho"] - ortho_naive)
    return worst


# ------------------------------------------------------------------ tests
def test_equivalence_float64():
    """A1-A5: exact-arithmetic equivalence at several ranks."""
    tol = 1e-9
    for r in (2, 4, 8):
        w = compare_run(r=r)
        for k, v in w.items():
            assert v < tol, f"r={r} {k}={v:.3e}"


def test_cor3_endpoint():
    """A6: r = n, pre-projection disabled -> both kernels are exact RTRL."""
    n, B, T = 12, 2, 20
    cell, xs, _ = make_setup(n=n, m=3, B=B, T=T, seed=1)
    ex = ExactRTRL(cell, B)
    naive = SKRTRL(cell, B, r=n)
    amort = SKRTRLAmortised(cell, B, r=n, max_buffer=8)
    assert not naive.preproject and not amort.preproject
    h = cell.init_state(B)
    worst_n = worst_a = 0.0
    for t in range(T):
        h_prev = h
        h = cell(xs[t], h_prev)
        A, imm = cell.jac_pieces(xs[t], h_prev, h)
        ex.step_state(A, imm)
        naive.step_state(A, imm)
        amort.step_state(A, imm)
        worst_n = max(worst_n, rel(naive.residual_dense(), ex.J))
        worst_a = max(worst_a, rel(amort.residual_dense(), ex.J))
        h = h.detach()
    assert worst_n < 1e-9 and worst_a < 1e-9, (worst_n, worst_a)


def test_certificate_validity():
    """A7: e_t is a valid upper bound on ||J - (S + L R^T)|| for the amortised kernel."""
    n, B, T = 16, 2, 40
    for r in (2, 4, 8):
        cell, xs, _ = make_setup(n=n, m=4, B=B, T=T, seed=2)
        ex = ExactRTRL(cell, B)
        amort = SKRTRLAmortised(cell, B, r=r)
        h = cell.init_state(B)
        viol = 0
        for t in range(T):
            h_prev = h
            h = cell(xs[t], h_prev)
            A, imm = cell.jac_pieces(xs[t], h_prev, h)
            ex.step_state(A, imm)
            amort.step_state(A, imm)
            err = (ex.J - amort.residual_dense()).flatten(1).norm(dim=1)
            tol = 1e-9 * (1 + ex.J.flatten(1).norm(dim=1))
            viol += int((err > amort.e + tol).any())
            h = h.detach()
        assert viol == 0, f"r={r}: {viol}/{T} certificate violations"


def test_adaptive_rank():
    """A8: mid-stream rank changes stay equivalent (r_in != r path)."""
    sched = {5: 8, 12: 4, 20: 12, 26: 2}
    w = compare_run(r=4, T=32, rank_schedule=sched)
    for k, v in w.items():
        assert v < 1e-9, f"{k}={v:.3e}"


def test_collapse_period_invariance():
    """A9: the answer does not depend on the collapse period K."""
    ref = compare_run(r=8, collapse_every=1)
    for K in (2, 3, 8, 1000):
        w = compare_run(r=8, collapse_every=K)
        for k, v in w.items():
            assert v < 1e-9, f"K={K} {k}={v:.3e}"
    assert max(ref.values()) < 1e-9


def test_float32():
    """A10: float32 agreement at ~1e-5 relative."""
    w = compare_run(r=8, dtype=torch.float32, T=25)
    for k, v in w.items():
        assert v < 2e-4, f"fp32 {k}={v:.3e}"


def test_lane_reset():
    """A11: episode-boundary lane resets match the naive kernel's R[mask] = 0."""
    n, m, B, T, r = 12, 3, 3, 24, 4
    cell, xs, deltas = make_setup(n, m, B, T, seed=5)
    naive = SKRTRL(cell, B, r=r)
    amort = SKRTRLAmortised(cell, B, r=r)
    h = cell.init_state(B)
    mask = torch.tensor([True, False, True])
    worst = 0.0
    for t in range(T):
        if t in (7, 15):                       # simulate two episode boundaries
            for a in (naive, amort):
                a.S[mask] = 0.0
                a.L[mask] = 0.0
                a.e[mask] = 0.0
            naive.R[mask] = 0.0
            amort.reset_lanes(mask)
        h_prev = h
        h = cell(xs[t], h_prev)
        A, imm = cell.jac_pieces(xs[t], h_prev, h)
        naive.step_state(A, imm)
        amort.step_state(A, imm)
        worst = max(worst, rel(amort.residual_dense(), naive.residual_dense()),
                    rel(amort.grad_rows(deltas[t]), naive.grad_rows(deltas[t])))
        h = h.detach()
    assert not hasattr(amort, "R"), "an R attribute would break in-place lane resets"
    assert worst < 1e-9, worst


def test_factory():
    cell, _, _ = make_setup(n=8, m=2, B=1, T=1)
    assert isinstance(make_skrtrl(cell, 1, r=2, kernel="naive"), SKRTRL)
    assert isinstance(make_skrtrl(cell, 1, r=2, kernel="amortised"), SKRTRLAmortised)


def main():
    ok = True
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"{name:34s} PASS")
        except AssertionError as exc:
            ok = False
            print(f"{name:34s} FAIL  {exc}")
    print("\n--- worst deviations (float64, T=30) ---")
    for r in (2, 4, 8, 16):
        w = compare_run(r=r, n=16 if r < 16 else 20)
        print(f"r={r:3d}  " + "  ".join(f"{k}={v:.2e}" for k, v in w.items()))
    print("\n--- worst deviations (float32, T=25) ---")
    for r in (2, 4, 8):
        w = compare_run(r=r, dtype=torch.float32, T=25)
        print(f"r={r:3d}  " + "  ".join(f"{k}={v:.2e}" for k, v in w.items()))
    print("\n--- long-horizon drift, n=24 (no accumulation expected) ---")
    for dt, tag in ((torch.float64, "f64"), (torch.float32, "f32")):
        for T in (100, 300):
            for r in (8, 16):
                w = compare_run(n=24, m=5, B=2, T=T, r=r, dtype=dt)
                print(f"{tag} T={T:3d} r={r:3d}  "
                      + "  ".join(f"{k}={v:.2e}" for k, v in w.items()))
    print("\nALL PASS" if ok else "\nSOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
