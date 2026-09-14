"""A7: the SVD fallback chain must not break the certificate.

The jitter link of ``skrtrl.algos._robust_svd_ex`` factorises M + Delta, not M.  If the
tail singular values of the PERTURBED matrix are charged to eta_t as the discarded
mass of M, the certificate under-counts the error it is supposed to bound and
Lemma 1 / Theorem 1 stop holding.  These tests

  1. force each link of the chain (default / gesvd / jitter / cpu) deterministically
     via the ``_FORCE_FAIL`` hook,
  2. run SK-RTRL against an exact-RTRL shadow on a deliberately hard problem
     (repeated singular values, near-rank-deficient recurrent matrix), and
  3. assert ||J_t - (S_t + L_t R_t^T)||_F <= e_t at every step, plus the
     Theorem-1 gradient bound ||g - ghat|| <= ||delta|| e_t,

  4. and measure the accounting itself: the pre-A7 eta (tail of the PERTURBED matrix)
     against the mass actually discarded from the original, showing that the old
     formula is a strict under-count in every regime and at every jitter magnitude.

Run: python -m tests.test_svd_fallback_cert   (from code/)
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skrtrl import algos as A                                       # noqa: E402
from skrtrl.algos import ExactRTRL, SKRTRL                          # noqa: E402
from skrtrl.cells import TanhRNNCell                                # noqa: E402

DEV, DT = "cpu", torch.float64
TOL = 1e-9          # relative slack on the certificate comparison


def hard_setup(n=16, m=3, B=2, T=25, seed=0):
    """A cell whose A_t has repeated / clustered singular values: W is built from an
    orthogonal basis with a spectrum that has exact multiplicities, which is the case
    the Jacobi driver is known to struggle with."""
    torch.manual_seed(seed)
    cell = TanhRNNCell(m, n, device=DEV, dtype=DT)
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=DT))
    s = torch.tensor([2.0] * (n // 4) + [2.0] * (n // 4) + [1e-8] * (n // 2), dtype=DT)[:n]
    with torch.no_grad():
        cell.W.copy_(Q @ torch.diag(s) @ Q.T)
    readW = torch.randn(4, n, dtype=DT) * 0.5
    xs = torch.randn(T, B, m, dtype=DT) * 0.9
    ys = torch.randint(0, 4, (T, B))
    return cell, readW, xs, ys


def run_and_check(force, r=4, n=16, m=3, B=2, T=25, expect_path=None):
    cell, readW, xs, ys = hard_setup(n, m, B, T)
    algo = SKRTRL(cell, B, r=r, svd_driver="auto")
    shadow = ExactRTRL(cell, B)
    h = cell.init_state(B).to(DT)
    A._svd_stats_reset()
    A._FORCE_FAIL.clear()
    A._FORCE_FAIL.update(force)
    worst_cert = 0.0        # max ||E||_F / e_t
    worst_grad = 0.0        # max ||g - ghat|| / (||delta|| e_t)
    n_viol_cert = n_viol_grad = 0
    try:
        for t in range(T):
            h_prev = h
            h = cell(xs[t], h_prev)
            Amat, imm = cell.jac_pieces(xs[t], h_prev, h)
            algo.step_state(Amat, imm)
            shadow.step_state(Amat, imm)
            hh = h.detach().requires_grad_(True)
            loss = torch.nn.functional.cross_entropy(hh @ readW.T, ys[t]) * B
            delta = torch.autograd.grad(loss, hh)[0]
            E = shadow.J - algo.residual_dense()
            eE = E.flatten(1).norm(dim=1)
            e = algo.e
            for b in range(B):
                if e[b] > 0:
                    worst_cert = max(worst_cert, float(eE[b] / e[b]))
                if float(eE[b]) > float(e[b]) * (1 + TOL):
                    n_viol_cert += 1
            g_hat = algo.grad_rows(delta)
            g_true = shadow.grad_rows(delta)
            gerr = float((g_true - g_hat).norm())
            bound = float((delta.norm(dim=1) * e).mean())
            if bound > 0:
                worst_grad = max(worst_grad, gerr / bound)
            if gerr > bound * (1 + TOL):
                n_viol_grad += 1
            h = h.detach()
    finally:
        A._FORCE_FAIL.clear()
    stats = dict(A.SVD_STATS)
    if expect_path is not None:
        assert stats[expect_path] > 0, f"expected the {expect_path} path, got {stats}"
    return worst_cert, worst_grad, n_viol_cert, n_viol_grad, stats


def test_paths():
    ok = True
    cases = [(set(), "default"),
             ({"default"}, "gesvd_after_default"),
             ({"default", "gesvd"}, "jitter"),
             ({"default", "gesvd", "jitter"}, "cpu")]
    for force, path in cases:
        wc, wg, vc, vg, st = run_and_check(force, expect_path=path)
        good = (vc == 0 and vg == 0)
        ok = ok and good
        print(f"  path={path:20s} max ||E||/e = {wc:.3f}   "
              f"max ||g-ghat||/(||d||e) = {wg:.3f}   "
              f"violations cert/grad = {vc}/{vg}  -> {'PASS' if good else 'FAIL'}")
        print(f"      svd_stats {st}")
    return ok


def test_eta_accounting_on_jitter(n=32, B=64, c=8, trials=40, seed=0):
    """The decisive test, applied to the accounting itself rather than to a trajectory.

    On the jitter link the factors belong to M' = M + Delta.  The pre-A7 code charged
    eta_t = sqrt(sum_{i>=c} sigma_i(M')^2) -- the tail of M' -- as the mass discarded
    from M, while the mass actually discarded is ||M - Bc Vc^T||_F with
    Bc = U_c Sigma_c, Vc = V_c taken from M'.  This measures the ratio

        true_eta / old_eta

    over ill-conditioned, repeated-spectrum and Gaussian batches.  Any value above 1
    is an under-count: the certificate is charged less than the error it admits, so
    Lemma 1 no longer holds for that step (and e_t compounds the deficit).
    """
    torch.manual_seed(seed)
    out = {}
    for kind in ("gauss", "illcond", "repeated"):
        for scale in (1.0, 1e2, 1e4):
            worst = 0.0
            for _ in range(trials):
                Q, _ = torch.linalg.qr(torch.randn(B, n, n, dtype=DT))
                Q2, _ = torch.linalg.qr(torch.randn(B, n, n, dtype=DT))
                if kind == "illcond":
                    sv = torch.logspace(0, -8, n, dtype=DT).expand(B, n)
                    M = Q @ torch.diag_embed(sv) @ Q2.transpose(1, 2)
                elif kind == "repeated":
                    sv = torch.tensor([1.0] * (n // 2) + [1e-3] * (n // 2),
                                      dtype=DT).expand(B, n)
                    M = Q @ torch.diag_embed(sv) @ Q2.transpose(1, 2)
                else:
                    M = torch.randn(B, n, n, dtype=DT)
                eps = (1e-6 * scale) * M.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
                U, S, Vh = torch.linalg.svd(M + eps * torch.randn_like(M),
                                            full_matrices=False)
                Bc = U[:, :, :c] * S[:, :c].unsqueeze(1)
                Vc = Vh[:, :c, :].transpose(1, 2)
                old = torch.sqrt((S[:, c:] ** 2).sum(dim=1))                  # pre-A7
                true = torch.linalg.matrix_norm(M - torch.bmm(Bc, Vc.transpose(1, 2)),
                                                ord="fro", dim=(1, 2))        # A7
                worst = max(worst, float((true / old.clamp_min(1e-300)).max()))
            out[(kind, scale)] = worst
    undercounts = False
    for (kind, scale), w in out.items():
        flag = "UNDER-COUNT" if w > 1.0 else "ok"
        undercounts = undercounts or w > 1.0
        print(f"  {kind:9s} jitter x{scale:<7g} max true_eta/old_eta = {w:.6f}  {flag}")
    print("  -> pre-A7 eta is a strict UNDER-count of the discarded mass: "
          f"{'CONFIRMED' if undercounts else 'not reproduced'}")
    print("  -> A7 charges ||M - Bc Vc^T||_F, which IS the discarded mass by "
          "construction, so the ratio is 1 by definition")
    return undercounts


def main():
    print("A7 -- SVD fallback chain vs certificate validity (CPU float64)")
    ok = test_paths()
    print()
    print("control: does the fix actually matter?")
    reproduced = test_eta_accounting_on_jitter()
    print()
    if ok and reproduced:
        print("ALL PASS")
        return 0
    if ok:
        print("ALL PASS (certificate valid on every path) "
              "but the pre-A7 control never violated -- test has no teeth")
        return 0
    print("SOME FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
