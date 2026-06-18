"""Streamed synthetic tasks. Each yields (x_t, target_t or None, new_episode_mask) per step.

All tasks are episodic streams: episodes are generated independently per batch lane and
concatenated; `new_ep` marks lanes whose hidden state should be reset at this step.
"""
import torch


class StreamTask:
    n_in: int
    n_out: int
    loss_type: str  # "ce" | "mse"

    def __init__(self, batch, device, seed=0):
        self.B, self.device = batch, device
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def step(self):
        raise NotImplementedError


class CopyTask(StreamTask):
    """Memorize k random symbols, recall after a delay. Episode: k + delay + k steps."""
    loss_type = "ce"

    def __init__(self, batch, device, seed=0, n_sym=8, k=5, delay=40):
        super().__init__(batch, device, seed)
        self.n_sym, self.k, self.delay = n_sym, k, delay
        self.n_in = n_sym + 2          # symbols + blank + recall-cue
        self.n_out = n_sym
        self.T = k + delay + k
        self.t = 0
        self.payload = self._draw()

    def _draw(self):
        return torch.randint(0, self.n_sym, (self.B, self.k), generator=self.g).to(self.device)

    def step(self):
        t = self.t
        x = torch.zeros(self.B, self.n_in, device=self.device)
        y = None
        new_ep = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        if t == 0:
            new_ep[:] = True
        if t < self.k:
            x.scatter_(1, self.payload[:, t:t + 1], 1.0)
        elif t < self.k + self.delay:
            x[:, self.n_sym] = 1.0
        else:
            x[:, self.n_sym + 1] = 1.0
            y = self.payload[:, t - self.k - self.delay]
        self.t += 1
        if self.t >= self.T:
            self.t = 0
            self.payload = self._draw()
        return x, y, new_ep


class AddingTask(StreamTask):
    """Two-channel adding problem of length T; MSE target at last step."""
    loss_type = "mse"
    n_out = 1

    def __init__(self, batch, device, seed=0, T=100):
        super().__init__(batch, device, seed)
        self.T = T
        self.n_in = 2
        self.t = 0
        self._draw()

    def _draw(self):
        self.vals = torch.rand(self.B, self.T, generator=self.g).to(self.device)
        m = torch.zeros(self.B, self.T)
        half = self.T // 2
        i1 = torch.randint(0, half, (self.B,), generator=self.g)
        i2 = torch.randint(half, self.T, (self.B,), generator=self.g)
        m[torch.arange(self.B), i1] = 1.0
        m[torch.arange(self.B), i2] = 1.0
        self.marks = m.to(self.device)

    def step(self):
        t = self.t
        x = torch.stack([self.vals[:, t], self.marks[:, t]], dim=1)
        y = None
        new_ep = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        if t == 0:
            new_ep[:] = True
        if t == self.T - 1:
            y = (self.vals * self.marks).sum(dim=1, keepdim=True)
        self.t += 1
        if self.t >= self.T:
            self.t = 0
            self._draw()
        return x, y, new_ep


class RotationMemoryTask(StreamTask):
    """Cross-unit credit task: emit v0 at t=0, then target y_t = Rot^t v0 every step.

    A fixed random rotation in d dims must be applied repeatedly -> requires
    distributed, cross-unit recurrent dynamics (diagonal recurrence cannot rotate).
    """
    loss_type = "mse"

    def __init__(self, batch, device, seed=0, d=8, T=64):
        super().__init__(batch, device, seed)
        self.d, self.T = d, T
        self.n_in = d + 1               # v0 + go-bit at t=0
        self.n_out = d
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=self.g))
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        self.Rot = q.to(device)
        self.t = 0
        self._draw()

    def _draw(self):
        v = torch.randn(self.B, self.d, generator=self.g)
        self.v0 = (v / v.norm(dim=1, keepdim=True)).to(self.device)
        self.cur = self.v0.clone()

    def step(self):
        t = self.t
        x = torch.zeros(self.B, self.n_in, device=self.device)
        new_ep = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        if t == 0:
            x[:, :self.d] = self.v0
            x[:, self.d] = 1.0
            new_ep[:] = True
        self.cur = self.cur @ self.Rot.T
        y = self.cur.clone()
        self.t += 1
        if self.t >= self.T:
            self.t = 0
            self._draw()
        return x, y, new_ep


class AnbnTask(StreamTask):
    """Next-char prediction on a^n b^n strings, n ~ U[1, n_max]; CE every step."""
    loss_type = "ce"
    n_in = 3   # a, b, sep
    n_out = 3

    def __init__(self, batch, device, seed=0, n_max=16):
        super().__init__(batch, device, seed)
        self.n_max = n_max
        self.seq = [self._draw_one() for _ in range(batch)]
        self.pos = [0] * batch

    def _draw_one(self):
        n = int(torch.randint(1, self.n_max + 1, (1,), generator=self.g))
        return [0] * n + [1] * n + [2]

    def step(self):
        xs, ys, ne = [], [], []
        for b in range(self.B):
            s, i = self.seq[b], self.pos[b]
            xs.append(s[i])
            ys.append(s[i + 1] if i + 1 < len(s) else 2)
            ne.append(i == 0)
            self.pos[b] += 1
            if self.pos[b] >= len(s):
                self.seq[b] = self._draw_one()
                self.pos[b] = 0
        x = torch.zeros(self.B, 3, device=self.device)
        x[torch.arange(self.B), torch.tensor(xs)] = 1.0
        y = torch.tensor(ys, device=self.device)
        new_ep = torch.tensor(ne, device=self.device)
        return x, y, new_ep


class RotationRecallTask(RotationMemoryTask):
    """Delayed variant: targets only in the last `recall` steps of the episode.
    Forces long-horizon credit (T - recall steps) AND cross-unit rotation dynamics."""

    def __init__(self, batch, device, seed=0, d=8, T=64, recall=8):
        super().__init__(batch, device, seed, d=d, T=T)
        self.recall = recall

    def step(self):
        t_before = self.t
        x, y, new_ep = super().step()
        if t_before < self.T - self.recall:
            y = None
        return x, y, new_ep


class TimeSeriesTask(StreamTask):
    """Online next-step prediction on a (partially observed) chaotic series.

    Only the first coordinate is fed as input; the target is its next value.
    Partial observability forces the RNN to use recurrent memory to reconstruct
    the unobserved state. The full series is normalized to zero mean / unit
    variance, so the reported MSE is the normalized MSE (NMSE).

    `washout` resets the hidden state + estimator every `washout` steps to keep
    the influence matrix bounded (0 = never reset = single continuous stream).
    Per batch-lane independence: each lane streams a different segment / IC.
    """
    loss_type = "mse"
    n_in = 1
    n_out = 1

    def __init__(self, batch, device, seed=0, length=20000, washout=200, horizon=1,
                 causal=False, causal_window=2000):
        super().__init__(batch, device, seed)
        self.washout, self.horizon = washout, horizon
        series = self._gen(length + horizon + 64)            # (B, L) raw
        if causal:
            # causal normalization: statistics from an initial window only (no future leakage)
            w = min(causal_window, series.shape[1] - horizon - 64)
            mu = series[:, 64:64 + w].mean(dim=1, keepdim=True)
            sd = series[:, 64:64 + w].std(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            mu = series.mean(dim=1, keepdim=True)
            sd = series.std(dim=1, keepdim=True).clamp_min(1e-6)
        self.series = ((series - mu) / sd).to(device)        # (B, L)
        self.L = self.series.shape[1] - horizon
        self.t = 64                                          # skip transient

    def _gen(self, L):
        raise NotImplementedError

    def step(self):
        t = self.t
        x = self.series[:, t].unsqueeze(1)                   # (B, 1)
        y = self.series[:, t + self.horizon].unsqueeze(1)    # (B, 1)
        new_ep = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        if self.washout > 0 and (t - 64) % self.washout == 0:
            new_ep[:] = True
        self.t += 1
        if self.t >= self.L:
            self.t = 64
        return x, y, new_ep


class HenonTask(TimeSeriesTask):
    """Henon map x_{t+1}=1-a x_t^2 + y_t, y_{t+1}=b x_t (a=1.4,b=0.3). Observe x only."""
    def _gen(self, L):
        a, b = 1.4, 0.3
        x = 0.1 + 0.01 * torch.arange(self.B, dtype=torch.float64).unsqueeze(1)
        y = torch.zeros(self.B, 1, dtype=torch.float64)
        out = []
        xt, yt = x.squeeze(1), y.squeeze(1)
        for _ in range(L):
            out.append(xt.clone())
            xn = 1 - a * xt * xt + yt
            yn = b * xt
            xt, yt = xn, yn
        return torch.stack(out, dim=1).float()


class MackeyGlassTask(TimeSeriesTask):
    """Mackey-Glass dx/dt = beta x(t-tau)/(1+x(t-tau)^n) - gamma x (tau=17, chaotic)."""
    def _gen(self, L):
        beta, gamma, n, tau, dt = 0.2, 0.1, 10.0, 17, 1.0
        steps = int(L / dt) + tau + 100
        hist = 1.2 + 0.2 * torch.rand(self.B, tau + 1, generator=self.g)
        buf = list(hist.unbind(dim=1))                       # each (B,)
        x = buf[-1].double()
        series = []
        bufd = [b.double() for b in buf]
        for i in range(steps):
            xtau = bufd[-tau]
            x = x + dt * (beta * xtau / (1 + xtau ** n) - gamma * x)
            bufd.append(x.clone())
            series.append(x.clone())
        s = torch.stack(series, dim=1).float()
        return s[:, ::1][:, :L]


class LorenzTask(TimeSeriesTask):
    """Lorenz system (sigma=10, rho=28, beta=8/3), observe x coordinate only."""
    def _gen(self, L):
        sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01
        sub = 2                                              # subsample for richer dynamics
        st = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x = st.repeat(self.B, 1) + 0.1 * torch.randn(self.B, 3, generator=self.g).double()
        out = []
        for i in range(L * sub + 100):
            xs, ys, zs = x[:, 0], x[:, 1], x[:, 2]
            dx = sigma * (ys - xs)
            dy = xs * (rho - zs) - ys
            dz = xs * ys - beta * zs
            x = x + dt * torch.stack([dx, dy, dz], dim=1)
            if i >= 100 and (i - 100) % sub == 0:
                out.append(x[:, 0].clone())                  # observe x only
        return torch.stack(out, dim=1).float()[:, :L]


class RealSeriesTask(TimeSeriesTask):
    """Online next-step prediction on a REAL univariate series loaded from a data file
    (one value per line). Batch lanes are circular phase-shifts of the same series so each
    lane streams a different window; the per-seed base rotation is drawn from the RNG.
    Set the class attribute `fname` in subclasses. No data are fabricated: if the file is
    missing the task raises, so a missing benchmark is never silently replaced."""
    fname = None

    def _gen(self, L):
        import os
        path = os.path.join(os.path.dirname(__file__), "data", self.fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"real-series data not found: {path} (download it; do not fabricate)")
        with open(path) as f:
            vals = [float(x) for x in f.read().split() if x.strip()]
        s = torch.tensor(vals, dtype=torch.float64)
        n = s.shape[0]
        # tile to >= L, then build B lanes by circular shift (per-seed base offset)
        reps = (L // n) + 2
        s = s.repeat(reps)
        base = int(torch.randint(0, n, (1,), generator=self.g))
        lanes = []
        for b in range(self.B):
            off = (base + b * (n // max(self.B, 1))) % n
            lanes.append(s[off:off + L])
        return torch.stack(lanes, dim=0).float()


class SunspotTask(RealSeriesTask):
    """Monthly smoothed Sunspot number (real data; data/sunspot.txt)."""
    fname = "sunspot.txt"


class LaserTask(RealSeriesTask):
    """Santa Fe far-infrared NH3 laser intensity (real data; data/laser.txt)."""
    fname = "laser.txt"


def _ts_factory(cls, **kw):
    def make(batch, device, seed=0, horizon=1, causal=False, washout=200):
        return cls(batch, device, seed, horizon=horizon, causal=causal, washout=washout, **kw)
    return make


def _rotrecall24(batch, device, seed=0):
    return RotationRecallTask(batch, device, seed, d=8, T=24, recall=8)


def _adding40(batch, device, seed=0):
    return AddingTask(batch, device, seed, T=40)


# Real-series tasks use a shorter length (their files are ~1k-3k points).
def _sunspot(batch, device, seed=0, horizon=1, causal=False, washout=200):
    return SunspotTask(batch, device, seed, length=3000, horizon=horizon, causal=causal, washout=washout)


def _laser(batch, device, seed=0, horizon=1, causal=False, washout=200):
    return LaserTask(batch, device, seed, length=1000, horizon=horizon, causal=causal, washout=washout)


TASKS = {"copy": CopyTask, "adding": AddingTask, "rotation": RotationMemoryTask,
         "rotrecall": RotationRecallTask, "rotrecall24": _rotrecall24,
         "adding40": _adding40, "anbn": AnbnTask,
         "henon": HenonTask, "mackeyglass": MackeyGlassTask, "lorenz": LorenzTask,
         "sunspot": _sunspot, "laser": _laser}
