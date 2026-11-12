"""Procedural irregular asteroids: star-shaped radius field
    r(u) = R0 * ellipsoid(u) * waist(u) * ridge(u) * (1 + sum of bumps/craters/boulders)

Analytic, so the identical function is evaluated in numpy (sim) and in JavaScript (viewer).  Units: km.

Three body types, loosely modelled on well-studied targets:
  rubble     spinning-top rubble pile (Bennu/Ryugu-like): equatorial ridge, many boulders, a few craters
  peanut     contact binary (Itokawa-like): two lobes joined by a neck
  elongated  stretched, cratered body (Eros-like)
"""
import numpy as np

# s: ellipsoid semi-axes (relative), pinch=(depth, width) along x, ridge=(height, width) about the equator,
# n: smooth bumps, nb: boulders (narrow positive bumps), nc: craters, mu in km^3/s^2, period in hours, j2: oblateness proxy
KINDS = {
    "rubble":    dict(R0=0.245, s=(1.00, 0.95, 0.90), pinch=(0.00, 0.4), ridge=(0.13, 0.38), n=22, nb=10, nc=4, mu=4.9e-9,  period=4.296, j2=0.015, name="Rubble pile (Bennu-like)"),
    "peanut":    dict(R0=0.270, s=(1.00, 0.52, 0.47), pinch=(0.30, 0.30), ridge=(0.00, 0.4), n=20, nb=8,  nc=3, mu=4.0e-9,  period=12.13, j2=0.040, name="Contact binary (Itokawa-like)"),
    "elongated": dict(R0=0.420, s=(1.00, 0.42, 0.40), pinch=(0.08, 0.55), ridge=(0.00, 0.4), n=22, nb=8,  nc=5, mu=1.9e-8,  period=5.27,  j2=0.060, name="Elongated (Eros-like)"),
}


class Asteroid:
    def __init__(self, seed=3, kind="rubble"):
        if kind == "random":
            kind = list(KINDS)[seed % len(KINDS)]
        K = KINDS[kind]
        rng = np.random.default_rng(seed)
        self.seed, self.kind, self.R0, self.name = seed, kind, K["R0"], K["name"]
        self.mu, self.period_h, self.j2 = K["mu"], K["period"], K["j2"]
        self.s = np.array(K["s"]) * rng.uniform(0.95, 1.06, 3)
        self.pinch, self.ridge = tuple(K["pinch"]), tuple(K["ridge"])
        n, nb, nc = K["n"], K["nb"], K["nc"]
        c = rng.normal(size=(n + nb + nc, 3)); c /= np.linalg.norm(c, axis=1, keepdims=True)
        a = np.concatenate([rng.normal(0, 0.06, n), rng.uniform(0.025, 0.06, nb), -np.full(nc, 0.20)])       # smooth bumps, boulders, craters
        k = np.concatenate([rng.uniform(8, 45, n), rng.uniform(140, 320, nb), np.full(nc, 90.0)])
        self.c, self.a, self.k = c, a, k
        self.nb, self.nc, self.n = nb, nc, n
        self.Rb = self.R0 * self.s.max() * 1.7                   # bounding sphere
        self.Req = self.R0 * self.s.max()                        # longest semi-axis (orbit radii are multiples of this)

    def radius(self, u):
        e = 1.0 / np.sqrt(((u / self.s) ** 2).sum(-1))
        b = (self.a * np.exp(self.k * (u @ self.c.T - 1.0))).sum(-1)
        w = 1.0 - self.pinch[0] * np.exp(-((u[..., 0] / self.pinch[1]) ** 2))
        g = 1.0 + self.ridge[0] * np.exp(-((u[..., 2] / self.ridge[1]) ** 2))
        return self.R0 * e * w * g * np.maximum(1.0 + b, 0.55)

    def f(self, P):                                              # <0 inside
        r = np.linalg.norm(P, axis=-1)
        return r - self.radius(P / np.maximum(r, 1e-9)[..., None])

    def features(self):
        """Named surface features the viewer pins and announces when the map reaches them."""
        out = []
        n, nb = self.n, self.nb
        for i in range(self.nc):
            j = n + nb + i; u = self.c[j]; out.append({"name": f"Crater {i + 1}", "kind": "crater", "u": u.round(5).tolist()})
        order = np.argsort(-self.a[n:n + nb])[:3]
        for i, o in enumerate(order):
            u = self.c[n + o]; out.append({"name": f"Boulder {i + 1}", "kind": "boulder", "u": u.round(5).tolist()})
        return out

    def params(self):
        return {"kind": self.kind, "name": self.name, "R0": self.R0, "s": self.s.tolist(), "pinch": list(self.pinch), "ridge": list(self.ridge),
                "c": self.c.round(5).tolist(), "a": self.a.round(5).tolist(), "k": self.k.round(3).tolist(),
                "mu": self.mu, "period_h": self.period_h, "j2": self.j2, "Req": self.Req, "features": self.features()}


def march(f, o, d, Rb, M=64, nb=10):
    """First crossing of f<0 along rays o + t d (single origin o (3,), dirs d (N,3)) inside the bounding sphere. NaN = miss."""
    N = len(d); b = d @ o; disc = b * b - (o @ o - Rb * Rb)
    sq = np.sqrt(np.maximum(disc, 0)); t0 = np.maximum(-b - sq, 0.0); t1 = -b + sq
    T = t0[:, None] + (t1 - t0)[:, None] * np.linspace(0, 1, M)[None, :]
    F = f(o + T[..., None] * d[:, None, :])
    neg = F < 0; hit = neg.any(1) & (disc > 0)
    i1 = neg.argmax(1); i0 = np.maximum(i1 - 1, 0); ar = np.arange(N)
    lo, hi = T[ar, i0], T[ar, i1]
    for _ in range(nb):
        mid = 0.5 * (lo + hi); m = f(o + mid[:, None] * d) < 0
        hi = np.where(m, mid, hi); lo = np.where(m, lo, mid)
    t = 0.5 * (lo + hi); t[~hit] = np.nan
    return t
