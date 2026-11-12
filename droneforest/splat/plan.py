"""Free-space planning on the proxy: connected free region, A* with a clearance margin, shortcut smoothing, start/goal pair sampling."""
import heapq
import numpy as np
from scipy import ndimage as ndi


class Proxy:
    def __init__(self, path):
        d = np.load(path); self.shape = tuple(int(s) for s in d["shape"]); self.origin = d["origin"].astype(np.float64); self.vs = float(d["vs"])
        self.floor = float(d["floor"]); self.hi = d["hi"].astype(np.float64); self.topz = d["topz"] if "topz" in d.files else None
        self.sdf = d["sdf"].astype(np.float32); self.occ = np.unpackbits(d["occ"])[: int(np.prod(self.shape))].reshape(self.shape).astype(bool)

    def idx(self, p):
        return np.floor((np.asarray(p) - self.origin) / self.vs).astype(int)

    def top_at(self, x, y):
        """height of the highest occupied surface (roof, canopy, ground) at x,y"""
        if self.topz is None:
            return self.floor
        i = int(np.clip(np.floor((x - self.origin[0]) / self.vs), 0, self.shape[0] - 1)); j = int(np.clip(np.floor((y - self.origin[1]) / self.vs), 0, self.shape[1] - 1))
        return float(self.topz[i, j])

    def world(self, ijk):
        return self.origin + (np.asarray(ijk) + 0.5) * self.vs

    def sdf_at(self, pts):
        c = (np.asarray(pts, dtype=np.float64) - self.origin) / self.vs - 0.5
        return ndi.map_coordinates(self.sdf, c.T, order=1, mode="nearest")

    def free_mask(self, clear=0.6, zmin=0.8, zmax=5.0):
        z = self.origin[2] + (np.arange(self.shape[2]) + 0.5) * self.vs
        band = ((z >= zmin) & (z <= zmax))[None, None, :]
        return (self.sdf > clear) & band


def astar(free, start, goal, cost_scale=1.0):
    """free: bool (X,Y,Z) on the planning grid; start/goal index tuples. 26-connected. Returns list of index tuples or None."""
    start, goal = tuple(start), tuple(goal)
    if not (free[start] and free[goal]):
        return None
    nb = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
    nbc = [float(np.sqrt(a * a + b * b + c * c)) for a, b, c in nb]
    h = lambda p: ((p[0] - goal[0]) ** 2 + (p[1] - goal[1]) ** 2 + (p[2] - goal[2]) ** 2) ** 0.5
    openq = [(h(start), 0.0, start)]; g = {start: 0.0}; came = {}
    X, Y, Z = free.shape
    while openq:
        _, gc, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]; path.append(cur)
            return path[::-1]
        if gc > g.get(cur, 1e18):
            continue
        for (a, b, c), w in zip(nb, nbc):
            n = (cur[0] + a, cur[1] + b, cur[2] + c)
            if 0 <= n[0] < X and 0 <= n[1] < Y and 0 <= n[2] < Z and free[n]:
                ng = gc + w
                if ng < g.get(n, 1e18):
                    g[n] = ng; came[n] = cur; heapq.heappush(openq, (ng + h(n), ng, n))
    return None


def plan(px, start, goal, clear=0.6, stride=2, zmin=0.8, zmax=5.0):
    """World-space path (K,3) from start to goal with SDF > clear, smoothed and resampled every ~0.5 m; None if no path."""
    fm = px.free_mask(clear, zmin, zmax)[::stride, ::stride, ::stride]
    to_g = lambda p: tuple(np.clip(px.idx(p) // stride, 0, np.array(fm.shape) - 1))
    s, g = to_g(start), to_g(goal)
    for q in (s, g):                                                                                     # snap endpoints to the nearest free coarse cell
        pass
    def snap(c):
        if fm[c]:
            return c
        r = 3
        lo = np.maximum(np.array(c) - r, 0); hi = np.minimum(np.array(c) + r + 1, fm.shape)
        sub = np.argwhere(fm[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        if not len(sub):
            return None
        sub = sub + lo; return tuple(sub[np.argmin(((sub - c) ** 2).sum(1))])
    s, g = snap(s), snap(g)
    if s is None or g is None:
        return None
    ids = astar(fm, s, g)
    if ids is None:
        return None
    pts = np.array([px.world(np.array(i) * stride + stride // 2) for i in ids]); pts[0], pts[-1] = start, goal
    out = [pts[0]]; i = 0                                                                               # greedy line-of-sight shortcuts
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1:
            seg = np.linspace(pts[i], pts[j], max(int(np.linalg.norm(pts[j] - pts[i]) / (px.vs * 0.5)), 2))
            if px.sdf_at(seg).min() > clear * 0.9:
                break
            j -= 1
        out.append(pts[j]); i = j
    out = np.array(out); res = [out[0]]
    for a, b in zip(out[:-1], out[1:]):
        n = max(int(np.linalg.norm(b - a) / 0.5), 1); res.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.array(res)


def snap_free(px, p, clear, r=4.0, zlo=0.3, zhi=80.0):
    """Nearest point to p (within r metres) whose SDF >= clear and height in [zlo,zhi]; None if there is none."""
    fm = px.free_mask(clear, zlo, zhi); c = px.idx(p); k = int(r / px.vs) + 1
    lo = np.maximum(c - k, 0); hi = np.minimum(c + k + 1, fm.shape); sub = np.argwhere(fm[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    if not len(sub):
        return None
    sub = sub + lo; w = px.world(sub); return w[np.argmin(((w - p) ** 2).sum(1))]


def sample_pairs(px, n, rng, min_d=10.0, max_d=40.0, clear=1.2, tries=800, start_z=(1.2, 3.5), goal_z=(1.2, 3.5), path_zmax=None):
    """Random (start, goal, path) triples in the largest connected free region (heights above the floor within the given bands),
    with a feasible A* path that keeps `clear` metres from everything and is not absurdly long."""
    zlo = min(start_z[0], goal_z[0]); zhi = path_zmax if path_zmax is not None else max(start_z[1], goal_z[1]) + 1.0
    fm = px.free_mask(clear, max(zlo - 0.4, 0.5), zhi); lab, k = ndi.label(fm, structure=np.ones((3, 3, 3)))
    if k == 0:
        return []
    sizes = ndi.sum(fm, lab, index=np.arange(1, k + 1)); main = lab == (1 + int(np.argmax(sizes)))
    zc = px.origin[2] + (np.arange(px.shape[2]) + 0.5) * px.vs
    band = lambda b: ((zc >= b[0]) & (zc <= b[1]))[None, None, :]
    cs, cg = np.argwhere(main & band(start_z)), np.argwhere(main & band(goal_z))
    if not len(cs) or not len(cg):
        return []
    out = []
    for _ in range(tries):
        a, b = cs[rng.integers(len(cs))], cg[rng.integers(len(cg))]
        s, g = px.world(a), px.world(b); d = np.linalg.norm(s - g)
        if not (min_d <= d <= max_d):
            continue
        p = plan(px, s, g, clear=clear, zmin=max(zlo - 0.4, 0.5), zmax=zhi)
        if p is None or np.linalg.norm(np.diff(p, axis=0), axis=1).sum() > 2.5 * d:
            continue
        out.append((s, g, p))
        if len(out) >= n:
            break
    return out
