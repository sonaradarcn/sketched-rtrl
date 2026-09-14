"""Gated-cell numerics through the PRODUCTION estimators (skrtrl/algos.py).

`test_gated_jacobians.py` checks the derivation against autograd and
`test_gated_cells.py` checks `cells_gated.py` against the derivation.  This
file closes the last link: the estimators in `skrtrl/algos.py` and
`skrtrl/baselines.py`, driven exactly as `skrtrl/train.py` drives them, on a
GRU and on an LSTM.  It is the gated twin of `tests/test_numerics.py`.

  R002  SK-RTRL at the Corollary-3 endpoint (r = n_state, c = omega,
        pre-projection off) == exact RTRL == autograd BPTT, every step.
        Also asserts that the endpoint really is (n_state, omega) and that the
        truncation charges tau_r = tau_c = 0 there.
  R003  certificate validity at r in {2, 4} over 200 steps:
          (a) ||E_t||_F <= e_t                     (Lemma 1 / Theorem 1 step 2)
          (b) ||g_t - ghat_t|| <= ||delta_t|| e_t  (Theorem 1)
        both per batch lane, zero violations required.
  R004  the r = 0 endpoint (SnAp-1) on a GRU: eta_t == ||Ahat_t S_{t-1} +
        I_t^perp||_F, i.e. the certificate covers the cross-block immediate
        mass the block-diagonal trace drops.  The control -- the vanilla
        eta_t = ||Ahat_t S_{t-1}||_F, which ignores I_t^perp -- is shown to be
        an INVALID bound on the same run, so the extra term is not decoration.
  R005  UORO and RFLO run on both gated cells and land in the (n, p_cell)
        parameter-block layout; UORO stays unbiased (it uses the full immediate
        Jacobian), RFLO does not and is not claimed to.
  R006  KF-RTRL refuses to construct on a gated cell, with a reason.

float64, CPU only.
Run:
  D:\\Anaconda\\envs\\multilingual_lora\\python.exe code/tests/test_gated_numerics.py
"""
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from skrtrl.algos import ExactRTRL, SKRTRL, SnAp1                  # noqa: E402
from skrtrl.baselines import KFRTRL, RFLO, UORO                    # noqa: E402
from skrtrl.cells_gated import make_gated_cell                    # noqa: E402

DEV, DT = "cpu", torch.float64
RESULTS = []


def record(tag, ok, note=""):
    RESULTS.append((tag, bool(ok), note))
    print(f"  [{'ok ' if ok else 'FAIL'}] {tag:<58s} {note}")
    return bool(ok)


def rel(a, b):
    return (a - b).norm().item() / max(b.norm().item(), 1e-12)


def make_setup(kind, n=8, m=3, B=2, T=20, seed=0, scale=1.4):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    cell = make_gated_cell(kind, m, n, device=DEV, dtype=DT)
    with torch.no_grad():                      # the default init is tiny
        cell.U.mul_(scale)
        cell.W.mul_(scale)
        cell.b.copy_((torch.rand(cell.b.shape, generator=gen, dtype=DT) * 2 - 1) * 0.4)
    readW = torch.randn(4, n, generator=gen, dtype=DT) * 0.3
    xs = torch.randn(T, B, m, generator=gen, dtype=DT) * 0.8
    ys = torch.randint(0, 4, (T, B), generator=gen)
    return cell, readW, xs, ys


def head(cell, state):
    """The part of the state the read-out sees (train.OnlineLearner.readout_in)."""
    return state[:, :cell.n] if cell.n_state != cell.n else state


def loss_of(cell, readW, state, y, B):
    return torch.nn.functional.cross_entropy(head(cell, state) @ readW.T, y) * B


def stream(cell, readW, xs, ys, algos, per_step=None):
    """Drive the estimators exactly as train.OnlineLearner.step does.

    Parameters are frozen, so sum_t delta_t^T J_t is the full-history gradient.
    Returns {name: (n_units, p_cell) accumulated grad rows}.
    """
    T, B = xs.shape[0], xs.shape[1]
    state = cell.init_state(B)
    acc = {k: torch.zeros(cell.n, cell.p_cell, dtype=DT) for k in algos}
    for t in range(T):
        prev = state.detach()
        state = cell(xs[t], prev)
        A, imm = cell.jac_pieces(xs[t], prev, state)
        for a in algos.values():
            a.step_state(A, imm)
        leaf = state.detach().requires_grad_(True)
        delta = torch.autograd.grad(loss_of(cell, readW, leaf, ys[t], B), leaf)[0]
        for k, a in algos.items():
            acc[k] = acc[k] + a.grad_rows(delta)
        if per_step is not None:
            per_step(t, A, imm, prev, delta)
        state = state.detach()
    return acc


def bptt_rows(cell, readW, xs, ys):
    """Autograd reference: d/dtheta sum_t loss_t, in the (n, p_cell) layout."""
    B = xs.shape[1]
    state = cell.init_state(B)
    total = 0.0
    for t in range(xs.shape[0]):
        state = cell(xs[t], state)
        total = total + loss_of(cell, readW, state, ys[t], B)
    cell.zero_grad(set_to_none=True)
    total.backward()
    return cell.flat_grad_rows()


# ---------------------------------------------------------------------------
# R002  exact endpoint
# ---------------------------------------------------------------------------
def r002(kind, n=8, m=3, B=2, T=20):
    cell, readW, xs, ys = make_setup(kind, n=n, m=m, B=B, T=T, seed=0)
    ns, omega = cell.n_state, cell.append_width
    ex = ExactRTRL(cell, B)
    sk = SKRTRL(cell, B, r=ns)
    ok = record(f"R002a[{kind}] endpoint is (r, c) = (n_state, omega) = ({ns}, {omega})",
                sk.r == ns and sk.c == omega and not sk.preproject,
                f"r={sk.r} c={sk.c} preproject={sk.preproject}")
    worst = {"J": 0.0, "tau": 0.0}

    def check(t, A, imm, prev, delta):
        worst["J"] = max(worst["J"], rel(sk.residual_dense(), ex.J))
        la = sk.last
        worst["tau"] = max(worst["tau"],
                           float(la["tau_c"].abs().max()), float(la["tau_r"].abs().max()))

    acc = stream(cell, readW, xs, ys, {"ex": ex, "sk": sk}, per_step=check)
    g_bptt = bptt_rows(cell, readW, xs, ys)
    d_ex = rel(acc["ex"] * B, g_bptt)          # batch mean -> batch sum
    d_sk = rel(acc["sk"] * B, g_bptt)
    ok &= record(f"R002b[{kind}] SK-RTRL(r=n_state) J == exact RTRL J, all t",
                 worst["J"] < 1e-8, f"max rel-err = {worst['J']:.3e}")
    ok &= record(f"R002c[{kind}] tau_c = tau_r = 0 at the endpoint",
                 worst["tau"] < 1e-12, f"max |tau| = {worst['tau']:.3e}")
    ok &= record(f"R002d[{kind}] exact RTRL == autograd BPTT",
                 d_ex < 1e-8, f"rel-err = {d_ex:.3e}")
    ok &= record(f"R002e[{kind}] SK-RTRL(r=n_state) == autograd BPTT",
                 d_sk < 1e-8, f"rel-err = {d_sk:.3e}")
    return ok


# ---------------------------------------------------------------------------
# R003  certificate validity
# ---------------------------------------------------------------------------
def r003(kind, ranks=(2, 4), n=8, m=3, B=2, T=200):
    ok = True
    for r in ranks:
        cell, readW, xs, ys = make_setup(kind, n=n, m=m, B=B, T=T, seed=7 + r)
        ex, sk = ExactRTRL(cell, B), SKRTRL(cell, B, r=r)
        assert sk.preproject, "r < n_state must keep the top-c pre-projection"
        st = {"v_e": 0, "v_g": 0, "tight": 0.0, "nfin": 0}

        def check(t, A, imm, prev, delta):
            E = ex.J - sk.residual_dense()
            err = E.flatten(1).norm(dim=1)                       # (B,)
            tol = 1e-9 * (1 + ex.J.flatten(1).norm(dim=1))
            if bool((err > sk.e + tol).any()):
                st["v_e"] += 1
            # Theorem 1, per lane: ||g_b - ghat_b|| <= ||delta_b|| e_b
            for b in range(B):
                d1 = torch.zeros_like(delta)
                d1[b] = delta[b]
                gap = (ex.grad_rows(d1) - sk.grad_rows(d1)).norm() * B
                bnd = delta[b].norm() * sk.e[b]
                if float(gap) > float(bnd) + 1e-9 * (1 + float(bnd)):
                    st["v_g"] += 1
                if float(gap) > 0:
                    st["tight"] = max(st["tight"], float(bnd / gap))
            st["nfin"] += int(bool(torch.isfinite(sk.e).all()))

        stream(cell, readW, xs, ys, {"ex": ex, "sk": sk}, per_step=check)
        ok &= record(f"R003a[{kind}] r={r}: ||E_t||_F <= e_t",
                     st["v_e"] == 0, f"violations {st['v_e']}/{T}")
        ok &= record(f"R003b[{kind}] r={r}: ||g-ghat|| <= ||delta|| e_t (per lane)",
                     st["v_g"] == 0,
                     f"violations {st['v_g']}/{T * B}, max looseness "
                     f"{st['tight']:.3g}, e_t finite {st['nfin']}/{T}")
    return ok


# ---------------------------------------------------------------------------
# R004  r = 0 endpoint on a GRU covers I_t^perp
# ---------------------------------------------------------------------------
def r004(n=8, m=3, B=2, T=120):
    cell, readW, xs, ys = make_setup("gru", n=n, m=m, B=B, T=T, seed=3)
    ex, sn = ExactRTRL(cell, B), SnAp1(cell, B)
    assert sn.r == 0
    st = {"d_eta": 0.0, "v_cert": 0, "v_naive": 0, "frac": [], "e_naive": torch.zeros(B, dtype=DT)}

    def check(t, A, imm, prev, delta):
        S_prev = st.pop("S_prev", torch.zeros(B, cell.n_state, cell.p_cell, dtype=DT))
        eta = sn.last["eta"]
        rho = sn.last["rho_bar"]
        # dense reference for N_t = Ahat_t S_{t-1} + I_t^perp
        Ahat = cell.offdiag_of_A(A)
        Sd = S_prev.new_zeros(B, cell.n, cell.n, cell.p_cell)
        i = torch.arange(cell.n)
        Sd[:, i, i, :] = S_prev
        Sd = Sd.reshape(B, cell.n, cell.P)
        Khat, F, blocks = cell.imm_offblock_factors()
        Ip = S_prev.new_zeros(B, cell.n, cell.n, cell.p_cell)
        Ip.index_add_(2, blocks, torch.einsum("bij,bjq->bijq", Khat, F))
        Ip = Ip.reshape(B, cell.n, cell.P)
        N_ref = torch.bmm(Ahat, Sd) + Ip
        st["d_eta"] = max(st["d_eta"],
                          float((eta - N_ref.flatten(1).norm(dim=1)).abs().max()))
        n_ip = Ip.flatten(1).norm(dim=1)
        n_naive = torch.bmm(Ahat, Sd).flatten(1).norm(dim=1)
        st["frac"].append(float((n_ip / eta.clamp_min(1e-30)).mean()))
        # the shipped certificate
        E = ex.J - sn.residual_dense()
        err = E.flatten(1).norm(dim=1)
        if bool((err > sn.e + 1e-9 * (1 + ex.J.flatten(1).norm(dim=1))).any()):
            st["v_cert"] += 1
        # control: the same recursion with the vanilla eta that ignores I^perp
        st["e_naive"] = rho * st["e_naive"] + n_naive
        if bool((err > st["e_naive"] + 1e-9 * (1 + ex.J.flatten(1).norm(dim=1))).any()):
            st["v_naive"] += 1
            st.setdefault("first_v", t)
        st["S_prev"] = sn.S.clone()

    st["S_prev"] = torch.zeros(B, cell.n_state, cell.p_cell, dtype=DT)
    stream(cell, readW, xs, ys, {"ex": ex, "sn": sn}, per_step=check)
    frac = sum(st["frac"]) / len(st["frac"])
    ok = record("R004a[gru] eta_t == ||Ahat_t S_{t-1} + I_t^perp||_F exactly",
                st["d_eta"] < 1e-10, f"max |dev| = {st['d_eta']:.3e}")
    ok &= record("R004b[gru] r=0 certificate valid (covers the cross-block term)",
                 st["v_cert"] == 0, f"violations {st['v_cert']}/{T}, "
                                    f"mean ||I^perp||/eta = {frac:.3f}")
    ok &= record("R004c[gru] control: eta without I^perp is NOT a valid bound",
                 st["v_naive"] > 0,
                 f"violations {st['v_naive']}/{T}, first at t={st.get('first_v')} "
                 "(a positive count is the expected result: dropping I^perp "
                 "makes the recursion an under-count)")
    return ok


# ---------------------------------------------------------------------------
# R005 / R006  baselines
# ---------------------------------------------------------------------------
def r005(kind, n=8, m=3, B=2, T=30):
    cell, readW, xs, ys = make_setup(kind, n=n, m=m, B=B, T=T, seed=11)
    ex = ExactRTRL(cell, B)
    algos = {"ex": ex, "uoro": UORO(cell, B), "rflo": RFLO(cell, B)}
    torch.manual_seed(123)
    acc = stream(cell, readW, xs, ys, algos)
    shp = all(tuple(v.shape) == (cell.n, cell.p_cell) for v in acc.values())
    ok = record(f"R005a[{kind}] UORO / RFLO land in the (n, p_cell) layout", shp,
                f"shapes {[tuple(v.shape) for v in acc.values()]}")
    # UORO is unbiased, so averaging many independent replicas must converge to
    # the exact gradient; one replica only has to be finite and correlated.
    cs = {}
    for k in ("uoro", "rflo"):
        g, ge = acc[k].flatten(), acc["ex"].flatten()
        cs[k] = float((g * ge).sum() / (g.norm() * ge.norm()))
    ok &= record(f"R005b[{kind}] estimates finite and positively aligned",
                 all(math.isfinite(v) and v > 0 for v in cs.values()),
                 f"cos(uoro) = {cs['uoro']:.3f}, cos(rflo) = {cs['rflo']:.3f}")
    # UORO unbiasedness over replicas (the noise is in step_state's signs)
    reps = []
    for s in range(24):
        cell2, readW2, xs2, ys2 = make_setup(kind, n=n, m=m, B=B, T=T, seed=11)
        torch.manual_seed(1000 + s)
        reps.append(stream(cell2, readW2, xs2, ys2, {"u": UORO(cell2, B)})["u"])
    mean = torch.stack(reps).mean(0)
    c_mean = float((mean.flatten() * acc["ex"].flatten()).sum()
                   / (mean.norm() * acc["ex"].norm()))
    ok &= record(f"R005c[{kind}] UORO mean over 24 replicas aligns with exact",
                 c_mean > 0.6, f"cos = {c_mean:.3f} (unbiased, so -> 1 with replicas)")
    return ok


def r006(kind, n=6, m=3, B=2):
    cell, _, _, _ = make_setup(kind, n=n, m=m, B=B, T=2, seed=0)
    try:
        KFRTRL(cell, B)
        return record(f"R006[{kind}] KF-RTRL refuses to construct", False,
                      "it constructed")
    except NotImplementedError as exc:
        return record(f"R006[{kind}] KF-RTRL refuses to construct", True,
                      "NotImplementedError: " + str(exc).split(":")[0][:40])


# ---------------------------------------------------------------------------
def main():
    ok = True
    for kind in ("gru", "lstm"):
        print(f"\n=== {kind.upper()}: R002 exact endpoint ===")
        ok &= r002(kind)
        print(f"=== {kind.upper()}: R003 certificate validity ===")
        ok &= r003(kind)
    print("\n=== GRU: R004 r=0 endpoint covers I_t^perp ===")
    ok &= r004()
    for kind in ("gru", "lstm"):
        print(f"\n=== {kind.upper()}: R005 / R006 baselines ===")
        ok &= r005(kind)
        ok &= r006(kind)

    n_ok = sum(1 for _, o, _ in RESULTS if o)
    print("\n" + "=" * 74)
    print(f"{'PASS' if ok else 'FAIL'}  {n_ok}/{len(RESULTS)} checks")
    for tag, o, note in RESULTS:
        if not o:
            print(f"  FAILED: {tag}  {note}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
