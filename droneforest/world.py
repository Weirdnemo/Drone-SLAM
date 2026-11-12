"""Per-env procedural forest + batched analytic raycasting, all in torch tensors.

Each tree = vertical trunk cylinder (radius rt, from ground zg to zg+h) + spherical canopy
(radius rc, centred at zg+h). Inactive trees are parked far away so no masking is needed.
"""
import math
import torch
from . import terrain

FAR = 1000.0
ARENA_X = 22.0   # half-extent along the flight direction
ARENA_Y = 14.0


class Forest:
    def __init__(self, n, max_trees, device):
        self.n, self.T, self.device = n, max_trees, device
        self.terr = torch.zeros(n, terrain.K, 4, device=device)
        z = lambda: torch.full((n, max_trees), FAR, device=device)
        self.tx, self.ty = z(), z()
        self.tzg = torch.zeros(n, max_trees, device=device)
        self.th = torch.ones(n, max_trees, device=device)
        self.trt = torch.full((n, max_trees), 0.3, device=device)
        self.trc = torch.full((n, max_trees), 2.0, device=device)

    # ------------------------------------------------------------------ generation
    def regenerate(self, idx, level, start_xy, goal_xy, terrain_amp=1.0):
        """Re-roll terrain + trees for envs `idx`. `level` in [0,1] sets how many trees are active."""
        m, T, dev = idx.numel(), self.T, self.device
        self.terr[idx] = terrain.sample_terrain(m, dev, terrain_amp)
        n_act = torch.round(level * T).long()                       # (m,)
        t = torch.rand(m, T, device=dev)
        base = start_xy[:, None, :] + (goal_xy - start_xy)[:, None, :] * t[..., None]
        corridor = base + torch.stack([2.0 * torch.randn(m, T, device=dev),
                                       3.5 * torch.randn(m, T, device=dev)], -1)
        uniform = torch.stack([(2 * torch.rand(m, T, device=dev) - 1) * ARENA_X,
                               (2 * torch.rand(m, T, device=dev) - 1) * ARENA_Y], -1)
        use_corr = (torch.arange(T, device=dev)[None, :] % 2 == 0)[..., None]
        xy = torch.where(use_corr, corridor, uniform)
        xy[..., 0] = xy[..., 0].clamp(-ARENA_X, ARENA_X)
        xy[..., 1] = xy[..., 1].clamp(-ARENA_Y, ARENA_Y)
        # keep start / goal clear
        d_s = (xy - start_xy[:, None, :]).norm(dim=-1)
        d_g = (xy - goal_xy[:, None, :]).norm(dim=-1)
        active = (torch.arange(T, device=dev)[None, :] < n_act[:, None]) & (d_s > 3.5) & (d_g > 3.5)
        far = torch.full_like(xy[..., 0], FAR)
        tx = torch.where(active, xy[..., 0], far)
        ty = torch.where(active, xy[..., 1], far)
        zg = terrain.height(self.terr[idx], xy[..., 0], xy[..., 1])
        self.tx[idx], self.ty[idx], self.tzg[idx] = tx, ty, zg
        self.th[idx] = 5.0 + 5.0 * torch.rand(m, T, device=dev)
        self.trt[idx] = 0.2 + 0.25 * torch.rand(m, T, device=dev)
        self.trc[idx] = 1.6 + 1.4 * torch.rand(m, T, device=dev)

    def ground(self, idx, x, y):
        return terrain.height(self.terr[idx], x, y)

    # ------------------------------------------------------------------ collisions
    def collide(self, pos, radius=0.25):
        """pos (E,3) for ALL envs -> bool (E,)."""
        gz = terrain.height(self.terr, pos[:, 0:1], pos[:, 1:2])[:, 0]
        hit = pos[:, 2] < gz + radius
        dx, dy = self.tx - pos[:, None, 0], self.ty - pos[:, None, 1]
        d2 = dx * dx + dy * dy
        zt = self.tzg + self.th
        in_trunk = (d2 < (self.trt + radius) ** 2) & (pos[:, None, 2] > self.tzg) & (pos[:, None, 2] < zt)
        dz = pos[:, None, 2] - zt
        in_canopy = (d2 + dz * dz) < (self.trc + radius) ** 2
        return hit | (in_trunk | in_canopy).any(-1)

    def clearance(self, idx, pos):
        """Exact distance from pos (m,3) to the nearest obstacle surface (ground, trunks, canopies)."""
        gz = terrain.height(self.terr[idx], pos[:, 0:1], pos[:, 1:2])[:, 0]
        d_ground = pos[:, 2] - gz
        cx, cy, zg, h = self.tx[idx], self.ty[idx], self.tzg[idx], self.th[idx]
        rt, rc = self.trt[idx], self.trc[idx]
        dxy = ((cx - pos[:, None, 0]) ** 2 + (cy - pos[:, None, 1]) ** 2).sqrt()
        zt = zg + h
        d_canopy = (dxy ** 2 + (pos[:, None, 2] - zt) ** 2).sqrt() - rc
        out_z = (zg - pos[:, None, 2]).clamp(min=0) + (pos[:, None, 2] - zt).clamp(min=0)
        d_trunk = (((dxy - rt).clamp(min=0)) ** 2 + out_z ** 2).sqrt()
        return torch.minimum(d_ground, torch.minimum(d_canopy, d_trunk).min(-1).values)

    # ------------------------------------------------------------------ raycasting
    def raycast(self, idx, pos, dirs, max_range, ground_steps=20):
        """pos (m,3), dirs (m,R,3) unit -> (t_tree (m,R), t_ground (m,R)), clamped to max_range."""
        m, R, _ = dirs.shape
        inf = torch.tensor(float("inf"), device=self.device)
        cx, cy = self.tx[idx], self.ty[idx]
        zg, h = self.tzg[idx], self.th[idx]
        rt, rc = self.trt[idx], self.trc[idx]
        T = cx.shape[1]

        o = pos[:, None, None, :]                       # (m,1,1,3)
        d = dirs[:, :, None, :]                         # (m,R,1,3)
        # ---- canopy spheres
        c = torch.stack([cx, cy, zg + h], -1)[:, None]  # (m,1,T,3)
        oc = o - c
        b = (d * oc).sum(-1)                            # (m,R,T)
        cc = (oc * oc).sum(-1) - (rc ** 2)[:, None]
        disc = b * b - cc
        t_s = -b - torch.sqrt(disc.clamp(min=0))
        t_s = torch.where((disc > 0) & (t_s > 0), t_s, inf)
        # ---- trunks (vertical finite cylinders)
        dxy = d[..., :2]
        ocxy = oc[..., :2]
        a2 = (dxy * dxy).sum(-1)                        # (m,R,1)
        b2 = (dxy * ocxy).sum(-1)
        c2 = (ocxy * ocxy).sum(-1) - (rt ** 2)[:, None]
        disc2 = b2 * b2 - a2 * c2
        t_c = (-b2 - torch.sqrt(disc2.clamp(min=0))) / a2.clamp(min=1e-6)
        zhit = o[..., 2] + t_c * d[..., 2]
        ok = (disc2 > 0) & (t_c > 0) & (a2 > 1e-6) & (zhit > zg[:, None]) & (zhit < (zg + h)[:, None])
        t_c = torch.where(ok, t_c, inf)
        t_tree = torch.minimum(t_s, t_c).min(-1).values.clamp(max=max_range)

        # ---- ground (ray marching on the analytic heightfield)
        ts = torch.linspace(0, max_range, ground_steps, device=self.device)
        p = pos[:, None, None, :] + dirs[:, :, None, :] * ts[None, None, :, None]   # (m,R,S,3)
        f = p[..., 2] - terrain.height(self.terr[idx], p[..., 0], p[..., 1])         # (m,R,S)
        neg = f < 0
        first = neg.int().argmax(-1)                                                  # (m,R)
        anyneg = neg.any(-1)
        i1 = first
        i0 = (first - 1).clamp(min=0)
        f1 = f.gather(-1, i1[..., None])[..., 0]
        f0 = f.gather(-1, i0[..., None])[..., 0]
        t1, t0 = ts[i1], ts[i0]
        frac = (f0 / (f0 - f1).clamp(min=1e-6)).clamp(0, 1)
        t_g = torch.where(first == 0, torch.zeros_like(t1), t0 + (t1 - t0) * frac)
        t_ground = torch.where(anyneg, t_g, torch.full_like(t_g, max_range))
        return t_tree, t_ground
