"""M5 sanity: exact eligibility traces of OnlineLRU / RTU vs autograd-through-unroll,
plus TMaze scripted-policy checks.
Run: python -m code.tests.test_m5_traces (repo root) or python -m tests.test_m5_traces (code/)."""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skrtrl.diag_cells import OnlineLRU, RTU
from skrtrl.envs import TMaze

DEV = "cpu"
torch.manual_seed(0)


def lru_unroll(cell, xs, dones):
    en = torch.exp(cell.nu)
    rr = torch.exp(-en)
    lr_, li = rr * torch.cos(cell.theta), rr * torch.sin(cell.theta)
    gam = torch.sqrt((1.0 - rr * rr).clamp_min(1e-12))
    B = xs[0].shape[0]
    ur = torch.zeros(B, cell.n)
    ui = torch.zeros(B, cell.n)
    for x, d in zip(xs, dones):
        keep = (~d).float().unsqueeze(1)
        ur, ui = ur * keep, ui * keep
        bx = x @ cell.Bin.T
        ur, ui = lr_ * ur - li * ui + gam * bx, li * ur + lr_ * ui
    return ur, ui


def rtu_unroll(cell, xs, dones):
    en = torch.exp(cell.nu)
    rr = torch.exp(-en)
    th = torch.exp(cell.tl)
    g_, ph = rr * torch.cos(th), rr * torch.sin(th)
    gam = torch.sqrt((1.0 - rr * rr).clamp_min(1e-12))
    B = xs[0].shape[0]
    h1 = torch.zeros(B, cell.n)
    h2 = torch.zeros(B, cell.n)
    for x, d in zip(xs, dones):
        keep = (~d).float().unsqueeze(1)
        h1, h2 = h1 * keep, h2 * keep
        z1 = g_ * h1 - ph * h2 + gam * (x @ cell.W1.T)
        z2 = g_ * h2 + ph * h1 + gam * (x @ cell.W2.T)
        h1, h2 = torch.relu(z1), torch.relu(z2)
    return h1, h2


def check_cell(name, cell, unroll, recur_params, T=25, B=3, m=4):
    xs = [torch.randn(B, m) for _ in range(T)]
    dones = [torch.rand(B) < 0.1 for _ in range(T)]
    dones[0] = torch.zeros(B, dtype=torch.bool)
    cell.begin(B)
    for x, d in zip(xs, dones):
        cell.advance(x, d)
        cell.commit()
    w1, w2 = torch.randn(B, cell.n), torch.randn(B, cell.n)
    # trace-predicted grads (sum over batch to match autograd); loss on raw state leaves
    cell.features()
    la, lb = cell._leaves
    (w1 * la + w2 * lb).sum().backward()
    cell.backward_grads(scale=float(B))      # mean(0) * B = sum(0)
    pred = {k: getattr(cell, k).grad.clone() for k in recur_params}
    for k in recur_params:
        getattr(cell, k).grad = None
    # autograd reference
    a, b = unroll(cell, xs, dones)
    ((w1 * a).sum() + (w2 * b).sum()).backward()
    ok = True
    for k in recur_params:
        ref = getattr(cell, k).grad
        err = (pred[k] - ref).norm() / ref.norm().clamp_min(1e-12)
        print(f"  {name}.{k}: rel err {err:.2e}")
        ok &= err < 1e-4
    assert ok, f"{name} trace mismatch"


def check_lru_features_match():
    """OnlineLRU features() applies out+relu to [Re u, Im u] consistently with advance()."""
    cell = OnlineLRU(4, 8, device=DEV)
    cell.begin(2)
    x = torch.randn(2, 4)
    f_adv = cell.advance(x, torch.zeros(2, dtype=torch.bool))
    cell.commit()
    with torch.no_grad():
        f_now = cell.features()
    assert torch.allclose(f_adv, f_now), "advance/features mismatch"
    print("  lru features consistent")


def check_tmaze():
    env = TMaze(8, length=5, device=DEV, seed=0)
    obs = env.reset()
    assert obs.shape == (8, 4) and (obs.sum(1) == 1).all()
    ret = torch.zeros(8)
    goal = env.goal.clone()
    for t in range(5):                       # walk corridor
        obs, r, done = env.step(torch.zeros(8, dtype=torch.long))
        ret += r
        assert not done.any()
    assert (obs.argmax(1) == 3).all(), "should be at junction"
    acts = goal + 1                          # turn to the cued side
    obs, r, done = env.step(acts)
    ret += r
    assert done.all() and (r == 4.0).all()
    assert torch.allclose(ret, torch.full((8,), 4.0 - 0.1 * 5))
    assert (obs.argmax(1) <= 1).all(), "auto-reset should show start obs"
    # wrong turn
    obs, r, done = env.step(2 - env.goal)    # 2-goal: goal0->2(right)=wrong
    # at start cell, non-forward = bump (-0.1), not a turn
    assert (r == -0.1).all() and not done.any()
    # timeout
    env2 = TMaze(4, length=3, device=DEV, seed=1)
    env2.reset()
    for t in range(6):
        obs, r, done = env2.step(torch.ones(4, dtype=torch.long))  # never move
    assert done.all(), "timeout at 2N"
    print("  tmaze ok (optimal return, aliasing, auto-reset, timeout)")


if __name__ == "__main__":
    print("LRU traces vs autograd:")
    check_cell("lru", OnlineLRU(4, 8, device=DEV), lru_unroll, ["nu", "theta", "Bin"])
    print("RTU traces vs autograd:")
    check_cell("rtu", RTU(4, 8, device=DEV), rtu_unroll, ["nu", "tl", "W1", "W2"])
    check_lru_features_match()
    print("TMaze:")
    check_tmaze()
    print("ALL OK")
