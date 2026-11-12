"""SplatWorld: the Forest interface (ground / collide / clearance / raycast) backed by a signed distance field built from a Gaussian-splat scene.

One static scene shared by every env. Lidar = sphere tracing through the trilinearly-interpolated SDF, collisions = SDF < drone radius.
"""
import numpy as np
import torch
import torch.nn.functional as F


class SplatWorld:
    def __init__(self, proxy_path, n_envs, device, radius=0.35):
        self.radius = radius
        d = np.load(proxy_path); self.device = torch.device(device); self.n, self.B, self.T = n_envs, 0, 0
        shape = tuple(int(s) for s in d["shape"]); self.vs = float(d["vs"]); self.floor = float(d["floor"])
        self.lo = torch.tensor(d["origin"], dtype=torch.float32, device=self.device); self.hi = torch.tensor(d["hi"], dtype=torch.float32, device=self.device)
        sdf = torch.tensor(d["sdf"].astype(np.float32)).permute(2, 1, 0).contiguous()                    # (Z,Y,X) for grid_sample (x,y,z) order
        self.sdf = sdf[None, None].to(self.device); self.shape = shape
        self.centre = 0.5 * (self.lo + self.hi); self.half = 0.5 * (self.hi - self.lo)

    def sdf_at(self, p):
        """p (...,3) world metres -> signed distance (...), continuing linearly outside the grid."""
        sh = p.shape[:-1]; q = p.reshape(-1, 3)
        u = ((q - self.lo) / self.vs - 0.5) / (torch.tensor(self.shape, device=self.device, dtype=torch.float32) - 1) * 2 - 1       # voxel centres at +-1 (align_corners)
        v = F.grid_sample(self.sdf, u.clamp(-1, 1).reshape(1, -1, 1, 1, 3), mode="bilinear", padding_mode="border", align_corners=True).reshape(-1)
        out = ((q - self.centre).abs() - self.half).clamp(min=0).norm(dim=-1)
        return (v + out).reshape(sh)

    # ---- Forest interface
    def regenerate(self, *a, **k):
        pass

    def ground(self, idx, x, y):
        return torch.full_like(x, self.floor)

    def collide(self, pos, radius=None):
        radius = self.radius if radius is None else radius
        return (self.sdf_at(pos) < radius) | (pos[:, 2] < self.floor + radius)

    def clearance(self, idx, pos):
        return torch.minimum(self.sdf_at(pos), pos[:, 2] - self.floor)

    def raycast(self, idx, pos, dirs, max_range, steps=72):
        m, R, _ = dirs.shape
        o = pos[:, None, :].expand(m, R, 3); t = torch.zeros(m, R, device=self.device); hit = torch.zeros(m, R, dtype=torch.bool, device=self.device)
        for _ in range(steps):
            d = self.sdf_at(o + dirs * t[..., None])
            newly = (d < 0.25 * self.vs) & ~hit
            hit = hit | newly
            t = torch.where(hit, t, (t + torch.clamp(d, min=0.4 * self.vs) * 0.9).clamp(max=max_range))
        t_tree = torch.where(hit, t, torch.full_like(t, max_range))
        dz = dirs[..., 2]; tg = torch.where(dz < -1e-4, (self.floor - o[..., 2]) / dz.clamp(max=-1e-4), torch.full_like(t, max_range))
        gxy = o[..., :2] + dirs[..., :2] * tg[..., None]                                                    # the floor only exists inside the captured footprint
        tg = torch.where(((gxy >= self.lo[:2]) & (gxy <= self.hi[:2])).all(-1), tg, torch.full_like(tg, max_range))
        return t_tree.clamp(max=max_range), tg.clamp(min=0, max=max_range)
