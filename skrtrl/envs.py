"""Batched POMDP environments for online RL (M5). Vectorized over B lanes, auto-reset."""
import torch


class TMaze:
    """Bakker (2001)-style T-maze: length-N corridor, B independent lanes.

    obs (one-hot, 4): [start+goal-left, start+goal-right, corridor, junction].
    The goal cue is visible only at the start cell; corridor observations are aliased,
    so solving the task requires carrying the cue for >= N steps.
    actions: 0=forward, 1=left, 2=right.
    reward: +4.0 correct turn at junction, -0.4 wrong turn, -0.1 per step otherwise.
    Episode ends on a turn at the junction or at timeout 2N; finished lanes auto-reset
    (the returned obs for those lanes is the first obs of the new episode).
    """
    n_obs, n_actions = 4, 3

    def __init__(self, batch, length=10, device="cuda", seed=0):
        self.B, self.N, self.device = batch, length, device
        self.timeout = 2 * length
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.pos = torch.zeros(batch, dtype=torch.long, device=device)
        self.t = torch.zeros(batch, dtype=torch.long, device=device)
        self.goal = self._draw(batch)          # 0 = left, 1 = right

    def _draw(self, k):
        return torch.randint(0, 2, (k,), generator=self.g).to(self.device)

    def _obs(self):
        idx = self.goal.clone()                # start cell shows the goal side
        idx[self.pos > 0] = 2                  # aliased corridor
        idx[self.pos >= self.N] = 3            # junction
        o = torch.zeros(self.B, self.n_obs, device=self.device)
        o[torch.arange(self.B, device=self.device), idx] = 1.0
        return o

    def reset(self):
        self.pos.zero_()
        self.t.zero_()
        self.goal = self._draw(self.B)
        return self._obs()

    def step(self, actions):
        """actions (B,) long -> (obs (B, n_obs), reward (B,), done (B,) bool)."""
        junc = self.pos >= self.N
        fwd = actions == 0
        self.pos = torch.where(~junc & fwd, self.pos + 1, self.pos)  # bump into walls = stay
        turn = junc & ~fwd
        correct = turn & ((actions - 1) == self.goal)
        r = torch.full((self.B,), -0.1, device=self.device)
        r[turn] = -0.4
        r[correct] = 4.0
        self.t += 1
        done = turn | (self.t >= self.timeout)
        if done.any():
            self.pos[done] = 0
            self.t[done] = 0
            self.goal[done] = self._draw(int(done.sum()))
        return self._obs(), r, done
