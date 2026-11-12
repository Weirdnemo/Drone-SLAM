"""Asteroid proximity-operations mapping + SLAM-style position correction.

A spacecraft orbits a rotating irregular asteroid, scans it with a wide-FOV lidar, and builds a lat/lon radial shape model.
Its position estimate drifts (dead-reckoned navigation); the SLAM mode registers each scan against the map built so far and
corrects the drift. Orbits are planned either from a fixed script or actively (next-best orbit plane from the map's gaps).
Units km, s.
"""
import math
import numpy as np
from .shape import Asteroid, march

FOV, NRAY, MAXR = math.radians(30), 21, 3.0
MINC = 3                          # hits needed before a cell counts as observed
PHASE_R = [4.3, 3.7, 3.3, 3.0, 2.8, 2.6]          # orbit radii as multiples of the body's longest semi-axis
PHASE_NAMES = ["Approach & survey", "Global survey", "Detailed mapping I", "Detailed mapping II", "Close proximity I", "Close proximity II"]


def rotz(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def circ_state(r, inc, raan, u, MU):
    ci, si, co, so, cu, su = math.cos(inc), math.sin(inc), math.cos(raan), math.sin(raan), math.cos(u), math.sin(u)
    p = r * np.array([co * cu - so * ci * su, so * cu + co * ci * su, si * su])
    v = math.sqrt(MU / r) * np.array([-co * su - so * ci * cu, -so * su + co * ci * cu, si * cu])
    return np.concatenate([p, v])


def accel(p, R0, MU, J2):
    r = np.linalg.norm(p); z2 = (p[2] / r) ** 2
    a = -MU * p / r ** 3
    k = -1.5 * J2 * MU * R0 ** 2 / r ** 5
    return a + k * np.array([p[0] * (1 - 5 * z2), p[1] * (1 - 5 * z2), p[2] * (3 - 5 * z2)])


def rk4(x, dt, R0, MU, J2):
    f = lambda s: np.concatenate([s[3:], accel(s[:3], R0, MU, J2)])
    k1 = f(x); k2 = f(x + 0.5 * dt * k1); k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def lidar_dirs(p):
    b = -p / np.linalg.norm(p); e1 = np.cross([0, 0, 1.0], b)
    if np.linalg.norm(e1) < 1e-3: e1 = np.cross([1.0, 0, 0], b)
    e1 /= np.linalg.norm(e1); e2 = np.cross(b, e1)
    g = np.tan(np.linspace(-FOV / 2, FOV / 2, NRAY)); A, B = np.meshgrid(g, g)
    d = b + A.ravel()[:, None] * e1 + B.ravel()[:, None] * e2
    return d / np.linalg.norm(d, axis=1, keepdims=True)


class RadialMap:
    """Star-shaped shape model on a lat/lon grid (per-cell mean radius of lidar hits, body frame)."""
    def __init__(self, nlat=45, nlon=90):
        self.nlat, self.nlon = nlat, nlon; self.dlat, self.dlon = math.pi / nlat, 2 * math.pi / nlon
        self.sum = np.zeros((nlat, nlon)); self.cnt = np.zeros((nlat, nlon))
        lat = -math.pi / 2 + (np.arange(nlat) + 0.5) * self.dlat; lon = -math.pi + (np.arange(nlon) + 0.5) * self.dlon
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        self.centres = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1)
        self.w = np.cos(LA)                                      # area weight

    def _ij(self, u):
        lat = np.arcsin(np.clip(u[..., 2], -1, 1)); lon = np.arctan2(u[..., 1], u[..., 0])
        return lat, lon

    def add(self, pb):
        r = np.linalg.norm(pb, axis=1); lat, lon = self._ij(pb / r[:, None])
        i = np.clip(((lat + math.pi / 2) / self.dlat).astype(int), 0, self.nlat - 1); j = ((lon + math.pi) / self.dlon).astype(int) % self.nlon
        np.add.at(self.sum, (i, j), r); np.add.at(self.cnt, (i, j), 1)

    def grid(self):
        g = np.full_like(self.sum, np.nan); m = self.cnt >= MINC; g[m] = self.sum[m] / self.cnt[m]; return g

    def filled(self, R0):
        g = self.grid(); obs = ~np.isnan(g); out = np.where(obs, g, np.nanmean(g) if obs.any() else R0)
        for _ in range(40):
            up = np.vstack([out[:1], out[:-1]]); dn = np.vstack([out[1:], out[-1:]])
            avg = 0.25 * (up + dn + np.roll(out, 1, 1) + np.roll(out, -1, 1)); out = np.where(obs, g, avg)
        return out

    def interp(self, u, G):
        lat, lon = self._ij(u); fi = (lat + math.pi / 2) / self.dlat - 0.5; fj = (lon + math.pi) / self.dlon - 0.5
        i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int); a = fi - i0; b = fj - j0
        i0c, i1c = np.clip(i0, 0, self.nlat - 1), np.clip(i0 + 1, 0, self.nlat - 1); j0m, j1m = j0 % self.nlon, (j0 + 1) % self.nlon
        return (1 - a) * (1 - b) * G[i0c, j0m] + (1 - a) * b * G[i0c, j1m] + a * (1 - b) * G[i1c, j0m] + a * b * G[i1c, j1m]

    def observed_at(self, u):
        lat, lon = self._ij(u); i = np.clip(((lat + math.pi / 2) / self.dlat).astype(int), 0, self.nlat - 1); j = ((lon + math.pi) / self.dlon).astype(int) % self.nlon
        return self.cnt[i, j] >= MINC


def correct_position(mp, R0, Rb, o_b, d_b, r_meas, G):
    """Registers one scan against the map; returns the body-frame position correction (3,) (Gauss-Newton, 2 iterations)."""
    fmap = lambda P: np.linalg.norm(P, axis=-1) - mp.interp(P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-9), G)
    o = o_b.copy(); eps = 0.002
    for _ in range(2):
        t0 = march(fmap, o, d_b, Rb, M=48, nb=8); res = r_meas - t0
        hitp = o + np.nan_to_num(t0)[:, None] * d_b
        ok = ~np.isnan(res) & (np.abs(np.nan_to_num(res)) < 0.05)
        ok[ok] &= mp.observed_at(hitp[ok] / np.linalg.norm(hitp[ok], axis=1, keepdims=True))
        if ok.sum() < 40: break
        J = np.stack([(march(fmap, o + eps * e, d_b, Rb, M=48, nb=8) - t0) / eps for e in np.eye(3)], 1)
        Jo, ro = J[ok], res[ok]; fin = np.isfinite(Jo).all(1) & np.isfinite(ro); Jo, ro = Jo[fin], ro[fin]
        if len(ro) < 40: break
        x = np.linalg.solve(Jo.T @ Jo + 1e-3 * len(ro) * np.eye(3), Jo.T @ ro)
        o = o + np.clip(x, -0.02, 0.02)
    return o - o_b


def metrics(mp, ast, vol_true, eval_dirs, eval_true):
    G = mp.grid(); obs = ~np.isnan(G)
    cov = float((mp.w * obs).sum() / mp.w.sum())
    Gn = G.copy(); val = mp.interp(eval_dirs, Gn); ok = ~np.isnan(val)
    rms = float(np.sqrt(np.mean((val[ok] - eval_true[ok]) ** 2)) * 1000) if ok.sum() > 20 else float("nan")
    Gf = mp.filled(ast.R0); est = mp.interp(mp.centres, Gf); vol = float((est ** 3 / 3 * mp.w * mp.dlat * mp.dlon).sum())
    return cov, rms, 100 * (vol - vol_true) / vol_true


def plan_orbit(mp, R0, rng, n_cand=9, samples=300):
    """Active planner: inclination whose sub-satellite latitude distribution covers the most unobserved (area-weighted) cells."""
    G = mp.grid(); unobs = np.isnan(G) * mp.w; lat = np.arcsin(mp.centres[..., 2])
    best, bi = -1, 60.0
    for inc in np.linspace(15, 90, n_cand):
        phi = np.arcsin(math.sin(math.radians(inc)) * np.sin(rng.uniform(0, 2 * math.pi, samples)))
        sc = (unobs[..., None] * (np.abs(lat[..., None] - phi) < math.radians(9))).sum()
        if sc > best: best, bi = sc, inc
    return bi


def run(ast, mode="slam", policy="active", hours_per_orbit=None, scan_dt=180.0, seed=1, drift=3e-3, record=False, snap_every=12, log=None):
    """mode: perfect | drift | slam.  policy: scripted | active.  drift in m/sqrt(h) -> km/sqrt(h) given as km."""
    rng = np.random.default_rng(seed); prng = np.random.default_rng(seed + 100)
    mp = RadialMap(); Rb = ast.Rb
    fine = RadialMap(60, 120).centres.reshape(-1, 3); ed = np.random.default_rng(0).normal(size=(6000, 3)); ed /= np.linalg.norm(ed, axis=1, keepdims=True)
    et = ast.radius(ed); gridt = ast.radius(mp.centres); vol_true = float((gridt ** 3 / 3 * mp.w * mp.dlat * mp.dlon).sum())
    MU, J2, OMEGA = ast.mu, ast.j2, 2 * math.pi / (ast.period_h * 3600)
    radii = [f * ast.Req for f in PHASE_R]; script = [(90, 0), (50, 60), (130, 120), (75, 30), (110, 150), (25, 90)]
    t, theta0, bias, corr, nscan = 0.0, 0.3, np.zeros(3), np.zeros(3), 0
    rec = {"frames": [], "snaps": [], "snap_t": [], "phases": []}; birth = np.full((mp.nlat, mp.nlon), -1.0); curve = []
    for ph, r in enumerate(radii):
        inc_d, raan_d = script[ph] if policy == "scripted" else (plan_orbit(mp, ast.R0, rng), rng.uniform(0, 360))
        if policy == "active" and ph % 2 == 1: inc_d = 180 - inc_d                   # alternate prograde / retrograde
        x = circ_state(r, math.radians(inc_d), math.radians(raan_d), rng.uniform(0, 6.28), MU)
        rec["phases"].append({"name": PHASE_NAMES[ph], "t0": round(t, 1), "r_km": round(r, 4), "inc": round(inc_d, 1)})
        T = 2 * math.pi * math.sqrt(r ** 3 / MU) * (hours_per_orbit or 1.0)
        for _ in range(int(T / scan_dt)):
            for _ in range(6): x = rk4(x, scan_dt / 6, ast.R0, MU, J2)
            t += scan_dt; theta = theta0 + OMEGA * t; R = rotz(theta)
            bias = bias + drift * math.sqrt(scan_dt / 3600) * prng.normal(size=3)
            p_true = x[:3]; d = lidar_dirs(p_true)
            tr = march(ast.f, R.T @ p_true, d @ R, Rb, M=64, nb=10)
            rm = tr + (0.0005 + 0.001 * tr) * rng.normal(size=len(tr)); rm[rng.random(len(tr)) < 0.02] = np.nan; rm[(rm > MAXR)] = np.nan
            p_est = p_true + (bias if mode != "perfect" else 0)
            if mode == "slam":
                if nscan % 4 == 0 and (mp.cnt >= MINC).mean() > 0.05:
                    okr = ~np.isnan(rm); xb = correct_position(mp, ast.R0, Rb, R.T @ (p_est + corr), d[okr] @ R, rm[okr], mp.filled(ast.R0))
                    corr = corr + 0.7 * (R @ xb)
                p_est = p_est + corr
            ok = ~np.isnan(rm)
            hits = p_est + rm[ok][:, None] * d[ok]; mp.add(hits @ R)                  # body frame = R^T hit
            new = (mp.cnt >= MINC) & (birth < 0); birth[new] = t / 3600
            nscan += 1
            pose_err = float(np.linalg.norm(p_est - p_true) * 1000); nav_err = float(np.linalg.norm(bias) * 1000) if mode != "perfect" else 0.0
            if nscan % 10 == 0:
                c, rms, ve = metrics(mp, ast, vol_true, ed, et); curve.append([round(t / 3600, 3), round(c, 4), round(rms, 2), round(ve, 3), round(pose_err, 2), round(nav_err, 2)])
                if log and nscan % 200 == 0: log(f"  t={t/3600:6.1f} h  cov={c*100:5.1f}%  rms={rms:6.1f} m  vol err={ve:+5.1f}%  pose err={pose_err:5.1f} m (nav only {nav_err:5.1f})")
            if record:
                rec["frames"].append([round(t, 1), *np.round(p_true, 5).tolist(), *np.round(p_est, 5).tolist(), round(theta % (2 * math.pi), 4), round(pose_err, 2), round(nav_err, 2), ph, *np.round(p_true + bias, 5).tolist()])
                if nscan % snap_every == 0:
                    g = mp.grid(); rec["snaps"].append(np.nan_to_num(np.round(g * 1000)).astype(int).ravel().tolist()); rec["snap_t"].append(round(t, 1))
    for i, ph_ in enumerate(rec["phases"]): ph_["t1"] = rec["phases"][i + 1]["t0"] if i + 1 < len(rec["phases"]) else round(t, 1)
    g = mp.grid(); final = metrics(mp, ast, vol_true, ed, et)
    if record: rec["snaps"].append(np.nan_to_num(np.round(g * 1000)).astype(int).ravel().tolist()); rec["snap_t"].append(round(t, 1))
    rec["birth"] = np.round(birth, 3).ravel().tolist()
    return {"curve": curve, "final": final, "rec": rec, "map": mp}
