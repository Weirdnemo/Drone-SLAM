"""Interactive flight planner for the splat viewer (serves the project AND plans/flies on demand, no re-running run_splat.py).

  python plan_server.py            # then open http://localhost:8000/splat_viewer/
In the viewer: 'Plan a flight' folder -> place start / goal in 3D (click the scene), set heights, preview the collision-free route, press Fly it.
Needs the scene to have been processed once with run_splat.py (data/<scene>_proxy.npz + .splat).
"""
import argparse, json, os, sys, threading, time
import numpy as np
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import torch
import run_splat as rs
from splat import plan as splan, io as sio
from splat.env import SplatDroneEnv

DATA = "splat_viewer/data"; CACHE = {}; LOCK = threading.Lock(); ARGS = None


def scene(name):
    if name not in CACHE:
        pf = f"{DATA}/{name}_proxy.npz"
        if not os.path.exists(pf):
            raise FileNotFoundError(f"{pf} not found: run run_splat.py on this scene once first")
        from scipy.spatial import cKDTree
        gs = sio.read_splat(f"{DATA}/{name}.splat"); CACHE[name] = dict(pf=pf, px=splan.Proxy(pf), tree=cKDTree(gs["pos"][gs["opacity"] > 0.4]))
    return CACHE[name]


def resolve(sc, q):
    """snap start/goal to free space and plan; returns (start, goal, path, messages) or raises ValueError"""
    px = sc["px"]; margin = float(q.get("margin", 1.2)); radius = float(q.get("radius", 0.35)); msgs = []; pts = []
    for nm in ("start", "goal"):
        p = np.array(q[nm], float)
        if not ((p[:2] >= px.origin[:2]) & (p[:2] <= px.hi[:2])).all():
            raise ValueError(f"{nm} is outside the scene")
        r = splan.snap_free(px, p, margin, r=4.0, zlo=0.3, zhi=px.hi[2] + 30)
        tight = False
        if r is None:
            r = splan.snap_free(px, p, radius + 0.3, r=4.0, zlo=0.3, zhi=px.hi[2] + 30); tight = True
        if r is None:
            raise ValueError(f"{nm} is inside solid geometry; move it up or sideways")
        if np.linalg.norm(r - p) > 0.05:
            msgs.append(f"{nm} moved {np.linalg.norm(r - p):.1f} m to free space" + (" (tight spot)" if tight else ""))
        pts.append(r)
    ceil = float(q.get("ceiling") or 0) or max(pts[0][2], pts[1][2]) + 8.0; path = None
    for m in (margin, margin * 0.75, margin * 0.5):
        path = splan.plan(px, pts[0], pts[1], clear=max(m, radius + 0.25), zmin=max(min(pts[0][2], pts[1][2]) - 0.4, 0.5), zmax=ceil)
        if path is not None:
            if m < margin: msgs.append(f"route needs reduced clearance {max(m, radius + 0.25):.2f} m")
            break
    if path is None:
        raise ValueError("no collision-free route: raise the ceiling, lower the margin, or move the points")
    return pts[0], pts[1], path, msgs


def do_plan(q):
    sc = scene(q["scene"]); s, g, path, msgs = resolve(sc, q)
    return dict(ok=True, start=rs.r3(s), goal=rs.r3(g), path=[rs.r3(p) for p in path[::2]], length=round(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), 1), msgs=msgs)


def do_fly(q):
    sc = scene(q["scene"]); s, g, path, msgs = resolve(sc, q); t0 = time.time()
    env = SplatDroneEnv(sc["pf"], 1, "cpu", pairs=[(s, g, path)], realism=True, drift=float(q.get("drift", 0.12)), wind_sigma=0.4, seed=int(q.get("seed", 0)), radius=float(q.get("radius", 0.35)))
    env.follow_soft = float(q.get("margin", 1.2)) + 0.1
    out, fr, est, scn, ld, clr = rs.fly(env, path, None, float(q.get("max_time", 120)))
    dmin = float(sc["tree"].query(np.array([f[:3] for f in fr]))[0].min()) if fr else 0.0
    if out == "success" and dmin < float(q.get("splat_clear", 0.5)):
        out = f"too close ({dmin:.2f} m to a splat)"
    if out != "success" and not fr:
        return dict(ok=False, error=out, msgs=msgs)
    ep = {"outcome": out, "start": rs.r3(s), "goal": rs.r3(g), "path": [rs.r3(p) for p in path[::2]], "frames": fr, "est": est, "scan": scn, "lidar_scan": ld, "min_clearance": round(clr, 2), "min_splat_dist": round(dmin, 2), "custom": True}
    return dict(ok=True, outcome=out, episode=ep, msgs=msgs, wall=round(time.time() - t0, 1))


class H(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *a):
        if "/api/" not in (a[0] if a else ""): super().log_message(fmt, *a)

    def end_headers(self):                       # never let the browser reuse an old rollout / splat after run_splat.py is re-run
        self.send_header("Cache-Control", "no-store"); super().end_headers()

    def _json(self, obj, code=200):
        b = json.dumps(obj, separators=(",", ":")).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/api/ping"): return self._json({"ok": True})
        return super().do_GET()

    def do_POST(self):
        try:
            q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with LOCK:
                if self.path.startswith("/api/plan"): return self._json(do_plan(q))
                if self.path.startswith("/api/fly"): return self._json(do_fly(q))
            self._json({"ok": False, "error": "unknown endpoint"}, 404)
        except (ValueError, FileNotFoundError) as e:
            self._json({"ok": False, "error": str(e)})
        except Exception as e:
            import traceback; traceback.print_exc(); self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


def main():
    global ARGS
    p = argparse.ArgumentParser(); p.add_argument("--port", type=int, default=8000); ARGS = p.parse_args()
    print(f"open  http://localhost:{ARGS.port}/splat_viewer/   (Ctrl+C to stop)")
    ThreadingHTTPServer(("0.0.0.0", ARGS.port), H).serve_forever()


if __name__ == "__main__":
    main()
