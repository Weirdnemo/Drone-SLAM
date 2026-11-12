"""Stand-in Gaussian-splat scene for testing the pipeline (NOT a real capture): a procedural outdoor clearing with trees, walls and crates,
written as flat surface-aligned Gaussians with noisy colours. Use it to check alignment/collision before pointing the pipeline at a real scan.

  python -m splat.demo_scene --out splat_viewer/data/demo_scene.ply
"""
import argparse
import numpy as np
from . import io as sio


def _qz_to(n):
    """quaternion (w,x,y,z) rotating +z onto unit vectors n (N,3)"""
    z = np.array([0, 0, 1.0]); c = n @ z; ax = np.cross(np.broadcast_to(z, n.shape), n); l = np.linalg.norm(ax, axis=1)
    w = np.sqrt(np.maximum((1 + c) / 2, 0)); s = np.where(l > 1e-6, np.sqrt(np.maximum((1 - c) / 2, 0)) / np.maximum(l, 1e-9), 0)
    q = np.concatenate([w[:, None], ax * s[:, None]], 1)
    flip = (l < 1e-6) & (c < 0); q[flip] = [0, 1, 0, 0]
    return q.astype(np.float32)


def build(seed=0, density=1.0):
    rng = np.random.default_rng(seed); P, N, C = [], [], []
    def add(p, n, col, jit=0.06):
        P.append(p); N.append(n); C.append(np.clip(np.asarray(col) + rng.normal(0, jit, (len(p), 3)), 0, 1))
    k = lambda n: int(n * density)
    # ground with gentle undulation
    n = k(36000); x, y = rng.uniform(-13, 13, n), rng.uniform(-9, 9, n); z = 0.05 * np.sin(x / 3) * np.cos(y / 2.5) + rng.normal(0, 0.01, n)
    add(np.stack([x, y, z], 1), np.tile([0, 0, 1.0], (n, 1)), np.array([0.32, 0.4, 0.2]), 0.07)
    # trees: trunk cylinder + canopy sphere
    trees = [(-4, 3), (-1, -4), (2.5, 4.5), (5, -2.5), (7.5, 5), (-6.5, -5), (0, 0.5)]
    for tx, ty in trees:
        h, r, rc = rng.uniform(3.5, 5.5), rng.uniform(0.22, 0.35), rng.uniform(1.6, 2.4); n = k(5000); a, zz = rng.uniform(0, 6.28, n), rng.uniform(0, h, n)
        add(np.stack([tx + r * np.cos(a), ty + r * np.sin(a), zz], 1), np.stack([np.cos(a), np.sin(a), 0 * a], 1), np.array([0.35, 0.24, 0.15]), 0.05)
        n = k(9000); v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        add(np.array([tx, ty, h]) + rc * v * rng.uniform(0.92, 1.0, (n, 1)), v, np.array([0.12, 0.38, 0.14]), 0.08)
    # boxes: walls and crates
    boxes = [(-2, -0.5, 0.2, 2.2, 1.4), (4, 1.0, 2.0, 0.25, 1.1), (1.5, -6.0, 0.25, 1.8, 1.8), (-8, 4.5, 1.2, 0.6, 0.7), (8.5, -4, 0.7, 0.7, 1.0), (-5.5, 6.5, 1.6, 0.25, 1.5)]
    for cx, cy, hx, hy, hz in boxes:
        for ax in range(3):
            for sgn in (-1, 1):
                h = np.array([hx, hy, hz]); a1, a2 = [i for i in range(3) if i != ax]; n = k(int(4000 * (h[a1] * h[a2]) ** 0.5 + 800))
                p = np.zeros((n, 3)); p[:, ax] = sgn * h[ax]; p[:, a1] = rng.uniform(-h[a1], h[a1], n); p[:, a2] = rng.uniform(-h[a2], h[a2], n); nn = np.zeros((n, 3)); nn[:, ax] = sgn
                add(p + np.array([cx, cy, hz]), nn, np.array([0.55, 0.52, 0.5]) if hz < 1.2 else np.array([0.6, 0.35, 0.3]), 0.05)
    p, nrm, col = np.concatenate(P), np.concatenate(N), np.concatenate(C)
    n = len(p); s = rng.uniform(0.05, 0.1, n)
    return {"pos": p.astype(np.float32), "scale": np.stack([s, s, 0.2 * s], 1).astype(np.float32), "quat": _qz_to(nrm.astype(np.float64)),
            "opacity": rng.uniform(0.75, 1.0, n).astype(np.float32), "color": col.astype(np.float32)}


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--out", default="splat_viewer/data/demo_scene.ply"); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--density", type=float, default=1.0); a.add_argument("--tilt", action="store_true", help="store it y-down and 0.4x scale like a raw COLMAP capture (to test --up/--scale)")
    o = a.parse_args(); g = build(o.seed, o.density)
    if o.tilt:
        g = sio.transform(g, [[1, 0, 0], [0, 0, -1], [0, 1, 0]], 0.4, (3, 1, -2))
    sio.write_ply(o.out, g); print("wrote", o.out, len(g["pos"]), "gaussians")
