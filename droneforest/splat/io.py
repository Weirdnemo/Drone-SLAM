"""Gaussian-splat file I/O: standard 3DGS .ply (binary little endian) and the compact antimatter15 .splat format.

A scene is a dict of numpy arrays: pos (N,3), scale (N,3) in metres-ish scene units, quat (N,4) as (w,x,y,z), opacity (N,) in [0,1],
color (N,3) in [0,1].
"""
import numpy as np

SH_C0 = 0.28209479177387814
_PLY = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "char": "i1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4"}


def read_ply(path):
    with open(path, "rb") as f:
        head = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("not a PLY file")
            head.append(line.decode("latin1").strip())
            if head[-1] == "end_header":
                break
        if not any("binary_little_endian" in h for h in head):
            raise ValueError("only binary_little_endian PLY is supported (convert compressed/ASCII PLY first, e.g. with splat-transform)")
        n = int(next(h for h in head if h.startswith("element vertex")).split()[2])
        props = [h.split() for h in head if h.startswith("property")]
        data = np.fromfile(f, dtype=np.dtype([(p[2], _PLY[p[1]]) for p in props]), count=n)
    need = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"]
    miss = [k for k in need if k not in data.dtype.names]
    if miss:
        raise ValueError(f"PLY is not a standard 3DGS file, missing {miss}")
    q = np.stack([data[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    return {"pos": np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float32),
            "scale": np.exp(np.stack([data[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32), "quat": q,
            "opacity": (1 / (1 + np.exp(-data["opacity"].astype(np.float64)))).astype(np.float32),
            "color": np.clip(0.5 + SH_C0 * np.stack([data[f"f_dc_{i}"] for i in range(3)], 1), 0, 1).astype(np.float32)}


def write_ply(path, g):
    n = len(g["pos"]); names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    op = np.clip(g["opacity"], 1e-4, 1 - 1e-4)
    cols = [g["pos"][:, 0], g["pos"][:, 1], g["pos"][:, 2], np.zeros(n), np.zeros(n), np.zeros(n), *((g["color"][:, i] - 0.5) / SH_C0 for i in range(3)),
            np.log(op / (1 - op)), *np.log(np.maximum(g["scale"], 1e-6)).T, *g["quat"].T]
    arr = np.stack(cols, 1).astype("<f4")
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + "".join(f"property float {k}\n" for k in names) + "end_header\n").encode())
        f.write(arr.tobytes())


_SPLAT = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])


def read_splat(path):
    d = np.fromfile(path, dtype=_SPLAT)
    q = (d["rot"].astype(np.float32) - 128) / 128; q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    return {"pos": d["pos"].copy(), "scale": d["scale"].copy(), "quat": q, "opacity": d["rgba"][:, 3] / 255.0, "color": d["rgba"][:, :3] / 255.0}


def write_splat(path, g):
    n = len(g["pos"]); d = np.zeros(n, dtype=_SPLAT)
    d["pos"], d["scale"] = g["pos"], g["scale"]
    d["rgba"][:, :3] = np.clip(np.round(g["color"] * 255), 0, 255); d["rgba"][:, 3] = np.clip(np.round(g["opacity"] * 255), 0, 255)
    d["rot"] = np.clip(np.round(g["quat"] * 128 + 128), 0, 255)
    order = np.argsort(-(g["scale"].prod(1) * g["opacity"]))                     # big, opaque splats first (as in the reference converter)
    d[order].tofile(path)


def load(path):
    p = str(path).lower()
    if p.endswith(".ply"):
        return read_ply(path)
    if p.endswith(".splat"):
        return read_splat(path)
    raise ValueError("expected a .ply (3DGS) or .splat file")


def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]; bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2; q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1) * 2; q = [0, 0, 0, 0]
        q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = 0.25 * s; q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q, dtype=np.float32); return q / np.linalg.norm(q)


def transform(g, R, s=1.0, t=(0, 0, 0)):
    """Rigid transform + uniform scale applied to the Gaussians (positions, sizes and orientations)."""
    out = dict(g); R = np.asarray(R, dtype=np.float64)
    out["pos"] = (s * g["pos"].astype(np.float64) @ R.T + np.asarray(t)).astype(np.float32)
    out["scale"] = (g["scale"] * s).astype(np.float32)
    out["quat"] = qmul(np.broadcast_to(mat_to_quat(R), g["quat"].shape), g["quat"]).astype(np.float32)
    return out


def subset(g, m):
    return {k: v[m] for k, v in g.items()}
