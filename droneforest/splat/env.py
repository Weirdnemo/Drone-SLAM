"""DroneForestEnv flying inside a splat scene: same dynamics / observation / reward, world = SplatWorld, start/goal from planned free-space pairs."""
import numpy as np
import torch
from droneforest.env import DroneForestEnv
from .world import SplatWorld
from . import plan as P


class SplatDroneEnv(DroneForestEnv):
    def __init__(self, proxy_path, num_envs=1, device="cpu", pairs=None, seed=0, radius=0.35, **kw):  # pairs: precomputed (start, goal, path) triples
        self.world = SplatWorld(proxy_path, num_envs, device, radius)
        px = P.Proxy(proxy_path); self.px = px
        pr = pairs if pairs is not None else P.sample_pairs(px, 64, np.random.default_rng(seed))
        if not pr:
            raise RuntimeError("no feasible start/goal pairs in this scene (check --up/--scale, or lower the clearance)")
        self.pairs = pr
        self._S = torch.tensor(np.stack([p[0] for p in pr]), dtype=torch.float32, device=device); self._G = torch.tensor(np.stack([p[1] for p in pr]), dtype=torch.float32, device=device)
        super().__init__(num_envs, device, seed=seed, **kw)

    def reset_idx(self, idx):
        self.forest = self.world
        m, dev = idx.numel(), self.device
        k = torch.randint(len(self.pairs), (m,), device=dev)
        start, goal = self._S[k], self._G[k]
        yaw = torch.atan2(goal[:, 1] - start[:, 1], goal[:, 0] - start[:, 0]) + (torch.rand(m, device=dev) - 0.5) * 0.4
        self.start[idx], self.goal[idx] = start, goal
        self.q.reset(idx, start, yaw)
        self.t[idx] = 0; self.prev_a[idx] = 0; self.est_bias[idx] = 0; self.q.wind[idx] = 0
        self.d_prev[idx] = (goal - start).norm(dim=-1)

    def _oob(self, pos, gz):
        return ((pos < self.world.lo - 4.0) | (pos > self.world.hi + 4.0)).any(-1)
