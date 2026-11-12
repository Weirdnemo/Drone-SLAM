"""Collision proxy for a Gaussian-splat scene.

Splats have no hard geometry, so the sim needs a stand-in: orient/scale the capture to a z-up metric frame (floor at z=0), drop floaters,
voxelise the opaque Gaussians into an occupancy grid, clean it (remove small blobs, close pinholes) and take a signed distance field.
The *same transformed splats* are written for the viewer, so what you see and what the lidar hits line up.
"""
import numpy as np
from scipy import ndimage as ndi
from . import io as sio

UP = {"z": np.eye(3), "y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]), "-y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]]),
      "-z": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]]),
      "x": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.0]]), "-x": np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0.0]])}


def guess_up(g):
    """thinnest axis of the opaque splats; the sign comes from the side the long tail (trees, roofs, walls) sticks out to"""
    gg = sio.subset(g, g["opacity"] > 0.3); p = gg["pos"]; ext = np.percentile(p, 99, axis=0) - np.percentile(p, 1, axis=0)
    ax = int(np.argmin(ext)); c = p[:, ax]; skew = float(np.mean(((c - np.median(c)) / (c.std() + 1e-9)) ** 3)); return ("" if skew > 0 else "-") + "xyz"[ax], ext


def orient(g, up="z", scale=1.0, floor=None, trim=0.5, margin=0.5, cut_above=None):
    """-> (splats in a z-up metric frame, info). `up` names which axis of the capture points up ('y' = y-up, '-y' = COLMAP-style y-down)."""
    g = sio.transform(g, UP[up], scale)
    keep = g["opacity"] > 0.05; g = sio.subset(g, keep)
    lo, hi = np.percentile(g["pos"], trim, axis=0) - margin, np.percentile(g["pos"], 100 - trim, axis=0) + margin
    g = sio.subset(g, ((g["pos"] >= lo) & (g["pos"] <= hi)).all(1))                                    # floaters outside the trimmed box
    z = g["pos"][:, 2]
    fz = float(np.percentile(z, 1.0)) if floor is None else float(floor)
    c = (np.percentile(g["pos"][:, :2], 1, axis=0) + np.percentile(g["pos"][:, :2], 99, axis=0)) / 2
    g = sio.transform(g, np.eye(3), 1.0, (-c[0], -c[1], -fz))
    if cut_above is not None:                                                                           # drop everything higher than this above the floor (floaters in the sky)
        g = sio.subset(g, g["pos"][:, 2] <= cut_above)
    return g, {"floor_shift": fz, "centre_shift": c.tolist()}


def voxelise(g, vs=0.2, min_opacity=0.15, thr=0.4, min_blob=30, close=True, fill_down=False):
    p = g["pos"].astype(np.float64); m = g["opacity"] > min_opacity; p, op = p[m], g["opacity"][m]
    rad = np.clip(2.0 * g["scale"][m].max(1), 0, 3 * vs)                                              # ~2 sigma of the largest axis: conservative
    lo = p.min(0) - 3 * vs; shape = np.ceil((p.max(0) + 3 * vs - lo) / vs).astype(int)
    while shape.prod() > 12e6:                                                                         # keep the grid manageable
        vs *= 1.25; lo = p.min(0) - 3 * vs; shape = np.ceil((p.max(0) + 3 * vs - lo) / vs).astype(int)
    ijk = np.floor((p - lo) / vs).astype(int); kk = np.minimum(np.round(rad / vs).astype(int), 3)                                                 # big splats cover neighbouring voxels, small ones only their own
    dens = np.zeros(shape, dtype=np.float32)
    for k in range(4):
        sel = kk == k
        if not sel.any():
            continue
        offs = [(a, b, c) for a in range(-k, k + 1) for b in range(-k, k + 1) for c in range(-k, k + 1) if a * a + b * b + c * c <= k * k + 0.5]
        for a, b, c in offs:
            q = ijk[sel] + (a, b, c); ok = ((q >= 0) & (q < shape)).all(1)
            np.add.at(dens, (q[ok, 0], q[ok, 1], q[ok, 2]), op[sel][ok])
    occ = dens >= thr
    lab, n = ndi.label(occ, structure=np.ones((3, 3, 3)))
    if n:
        sizes = ndi.sum(occ, lab, index=np.arange(1, n + 1)); occ = np.isin(lab, 1 + np.nonzero(sizes >= min_blob)[0])
    if close:
        occ = ndi.binary_closing(occ, structure=ndi.generate_binary_structure(3, 1), iterations=1) | occ
    if fill_down:                                                                                       # aerial captures only see top surfaces: make everything under them solid
        any_ = occ.any(2); top = occ.shape[2] - 1 - np.argmax(occ[:, :, ::-1], axis=2)
        occ = occ | (any_[:, :, None] & (np.arange(occ.shape[2])[None, None, :] <= top[:, :, None]))
    return occ, lo, vs


def sdf_from_occ(occ, vs):
    out = ndi.distance_transform_edt(~occ) * vs; inn = ndi.distance_transform_edt(occ) * vs
    return np.where(occ, -(inn - 0.5 * vs), out - 0.5 * vs).astype(np.float32)


def build(scene_path, out_prefix, up="z", scale=1.0, floor=None, vs=0.2, thr=0.4, min_blob=30, max_gauss=None, fill_down=False, min_opacity=0.15, cut_above=None, log=print):
    g = sio.load(scene_path); log(f"loaded {len(g['pos'])} Gaussians from {scene_path}")
    if up == "auto":
        up, ext = guess_up(g); log(f"--up auto: central extent per axis {np.round(ext, 1).tolist()} -> thinnest axis, using --up={up}")
    g, info = orient(g, up, scale, floor, cut_above=cut_above)
    log(f"after orient/trim: {len(g['pos'])}  extent {np.round(g['pos'].max(0) - g['pos'].min(0), 1).tolist()} m (x,y,z)")
    occ, lo, vs = voxelise(g, vs, min_opacity=min_opacity, thr=thr, min_blob=min_blob, fill_down=fill_down); sdf = sdf_from_occ(occ, vs)
    log(f"grid {occ.shape} @ {vs:.2f} m, occupied {int(occ.sum())} voxels ({100 * occ.mean():.1f}%)")
    if max_gauss and len(g["pos"]) > max_gauss:                                                         # thin the viewer file only; the proxy keeps everything
        order = np.argsort(-(g["scale"].prod(1) * g["opacity"]))[:max_gauss]; gv = sio.subset(g, np.sort(order))
    else:
        gv = g
    sio.write_splat(out_prefix + ".splat", gv)
    has = occ.any(2); top = occ.shape[2] - 1 - np.argmax(occ[:, :, ::-1], axis=2); topz = np.where(has, lo[2] + (top + 1) * vs, 0.0).astype(np.float32)   # highest occupied surface per column
    np.savez_compressed(out_prefix + "_proxy.npz", topz=topz, sdf=sdf.astype(np.float16), occ=np.packbits(occ), shape=np.array(occ.shape), origin=lo, vs=vs, floor=0.0,
                        hi=lo + np.array(occ.shape) * vs)
    log(f"wrote {out_prefix}.splat ({len(gv['pos'])} Gaussians) and {out_prefix}_proxy.npz")
    return out_prefix + "_proxy.npz"


def inspect(scene_path, out_png=None, log=print):
    """Print what is in a splat file and suggest --up / --scale; optionally save side/top preview images."""
    g = sio.load(scene_path); g = sio.subset(g, g["opacity"] > 0.3); p = g["pos"]
    lo, hi = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0); ext = hi - lo
    log(f"{len(p)} Gaussians with opacity>0.3; extent of the central 98% per axis (x,y,z): {np.round(ext, 2).tolist()} scene units")
    ax = int(np.argmin(ext)); c = p[:, ax]; m = np.median(c); skew = float(np.mean(((c - m) / (c.std() + 1e-9)) ** 3))
    names = "xyz"; sign = 1 if skew > 0 else -1
    sug = names[ax] if sign > 0 else "-" + names[ax]
    log(f"thinnest axis is {names[ax]} (typical for a flat ground-with-objects scene: ground is flat, things stick up on one side)")
    log(f"  tail of the distribution points to {'+' if sign > 0 else '-'}{names[ax]}, so the suggested up-axis is  --up={sug}" + ("   (supported: z, y, -y, -z)" if sug not in UP else ""))
    log("  check the preview image: the ground should be a flat band at the bottom with trees/houses rising from it. If it is upside down use the opposite sign.")
    gg, _ = orient(g, sug if sug in UP else "z", 1.0)
    e2 = gg["pos"].max(0) - gg["pos"].min(0); log(f"with --up={sug if sug in UP else 'z'} the scene spans {np.round(e2, 1).tolist()} (x,y,z) scene units.")
    log("scale: pick something you know in the real world (e.g. a house is ~12 m long, a driveway ~4 m wide), measure it in these units in the preview, and set --scale = real_metres / scene_units.")
    if out_png:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sel = np.random.default_rng(0).choice(len(gg["pos"]), min(len(gg["pos"]), 200000), replace=False); q = gg["pos"][sel]; col = gg["color"][sel]
        fig, a = plt.subplots(1, 2, figsize=(12, 5)); a[0].scatter(q[:, 0], q[:, 1], s=0.3, c=col); a[0].set_title("top view (x,y)"); a[0].set_aspect("equal")
        a[1].scatter(q[:, 0], q[:, 2], s=0.3, c=col); a[1].set_title("side view (x,z) - ground should be at the bottom"); a[1].set_aspect("equal")
        for x in a: x.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(out_png, dpi=110); log("wrote preview", out_png)
