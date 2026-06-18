"""M0 numerics: R001 exact RTRL == BPTT; R002 SK-RTRL(r=n) == exact; R003 certificate validity.

Run: python -m tests.test_numerics   (from code/ dir)
CPU float64 throughout.
"""
import os
import sys
import torch
import torch.nn.functional as F

# make `skrtrl` importable whether run as `python -m code.tests.test_numerics` (repo root)
# or `python -m tests.test_numerics` (from code/): add the code/ dir to sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skrtrl.cells import TanhRNNCell
from skrtrl.algos import ExactRTRL, SKRTRL

torch.manual_seed(0)
DEV, DT = "cpu", torch.float64


def make_setup(n=24, m=5, B=3, T=30):
    cell = TanhRNNCell(m, n, device=DEV, dtype=DT)
    readW = torch.randn(4, n, dtype=DT) * 0.3
    xs = torch.randn(T, B, m, dtype=DT) * 0.8
    ys = torch.randint(0, 4, (T, B))
    return cell, readW, xs, ys


def run_online(cell, readW, xs, ys, algo):
    """Accumulate sum_t delta_t^T J_t with frozen params; return (n, p) grad rows."""
    B, n = xs.shape[1], cell.n
    h = cell.init_state(B).to(DT)
    g_acc = torch.zeros(n, cell.p, dtype=DT)
    per_step = []
    for t in range(xs.shape[0]):
        h_prev = h
        h = cell(xs[t], h_prev)
        A, imm = cell.jac_pieces(xs[t], h_prev, h)
        algo.step_state(A, imm)
        hh = h.detach().requires_grad_(True)
        loss = F.cross_entropy(hh @ readW.T, ys[t]) * B  # sum-like scaling
        delta = torch.autograd.grad(loss, hh)[0]
        g_acc += algo.grad_rows(delta)
        per_step.append((A, delta))
        h = h.detach()
    return g_acc


def run_bptt(cell, readW, xs, ys):
    B = xs.shape[1]
    h = cell.init_state(B).to(DT)
    total = 0.0
    for t in range(xs.shape[0]):
        h = cell(xs[t], h)
        total = total + F.cross_entropy(h @ readW.T, ys[t]) * B
    cell.zero_grad()
    total.backward()
    n, m = cell.n, cell.m
    g = torch.cat([cell.W.grad, cell.U.grad, cell.b.grad.unsqueeze(1)], dim=1)
    return g


def rel(a, b):
    return (a - b).norm().item() / max(b.norm().item(), 1e-12)


def main():
    cell, readW, xs, ys = make_setup()
    B = xs.shape[1]

    # R001: exact RTRL vs BPTT
    g_rtrl = run_online(cell, readW, xs, ys, ExactRTRL(cell, B)) * B  # mean->sum over batch
    g_bptt = run_bptt(cell, readW, xs, ys)
    r1 = rel(g_rtrl, g_bptt)
    print(f"R001 exact-RTRL vs BPTT  rel-err = {r1:.3e}  ->", "PASS" if r1 < 1e-8 else "FAIL")

    # R002: SK-RTRL(r=n) == exact RTRL (influence matrices match each step)
    cell2, readW2, xs2, ys2 = make_setup(n=16, m=4, B=2, T=25)
    ex, sk = ExactRTRL(cell2, 2), SKRTRL(cell2, 2, r=16)
    assert not sk.preproject
    h = cell2.init_state(2).to(DT)
    worst = 0.0
    for t in range(xs2.shape[0]):
        h_prev = h
        h = cell2(xs2[t], h_prev)
        A, imm = cell2.jac_pieces(xs2[t], h_prev, h)
        ex.step_state(A, imm)
        sk.step_state(A, imm)
        worst = max(worst, rel(sk.residual_dense(), ex.J))
        h = h.detach()
    print(f"R002 SK-RTRL(r=n) vs exact  max rel-err = {worst:.3e}  ->", "PASS" if worst < 1e-8 else "FAIL")

    # R003: certificate validity for r in {2, 4, 8}
    ok_all = True
    for r in (2, 4, 8):
        cell3, _, xs3, _ = make_setup(n=16, m=4, B=2, T=40)
        ex3, sk3 = ExactRTRL(cell3, 2), SKRTRL(cell3, 2, r=r)
        h = cell3.init_state(2).to(DT)
        viol, max_ratio = 0, 0.0
        for t in range(xs3.shape[0]):
            h_prev = h
            h = cell3(xs3[t], h_prev)
            A, imm = cell3.jac_pieces(xs3[t], h_prev, h)
            ex3.step_state(A, imm)
            sk3.step_state(A, imm)
            E = ex3.J - sk3.residual_dense()
            err = E.flatten(1).norm(dim=1)            # (B,)
            tol = 1e-9 * (1 + ex3.J.flatten(1).norm(dim=1))
            if (err > sk3.e + tol).any():
                viol += 1
            ratio = (sk3.e / err.clamp_min(1e-15)).max().item()
            max_ratio = max(max_ratio, ratio)
            h = h.detach()
        status = "PASS" if viol == 0 else "FAIL"
        ok_all &= viol == 0
        print(f"R003 certificate r={r}: violations={viol}/40, max tightness e/||E|| = {max_ratio:.2f}  -> {status}")

    print("ALL PASS" if (r1 < 1e-8 and worst < 1e-8 and ok_all) else "SOME FAILED")


if __name__ == "__main__":
    main()
