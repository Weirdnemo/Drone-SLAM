"""Fly the lidar drone inside a Gaussian-splat scene.

  python run_splat.py --scene my_scan.ply --up -y --scale 1.0     # your capture (3DGS .ply or .splat); --up = which axis of the capture points up
  python run_splat.py --demo                                      # stand-in procedural splat scene (not a real capture) to test the pipeline
  python run_splat.py --scene s.ply ... --start 5 -3 4 --goal 40 12 6      # your own start / end point (metres, z above the floor); add --agl to make z "above the local roof/canopy"
View:  python -m http.server 8000  ->  http://localhost:8000/splat_viewer/

Steps: orient/trim the splats to a z-up metric frame -> voxelise to an occupancy grid + SDF (collision proxy) -> sample start/goal pairs with an A* path ->
fly each pair with the heuristic flyer (or a trained policy) through the torch sim with realism (noise, drift, wind, latency) -> export poses, lidar ranges
and the transformed splat for the viewer.
"""
import argparse, json, math, os, time
import numpy as np
import torch
from droneforest.env import OBS_DIM, ACT_DIM, MAX_RANGE, local_ray_dirs, yaw_rotate
from droneforest.heuristic import heuristic_action
from droneforest.dynamics import quat_yaw
from splat import proxy as sproxy, plan as splan, io as sio, demo_scene
from splat.env import SplatDroneEnv
from splat.follow import follow_action

r3 = lambda x: [round(float(v), 3) for v in x]


LID_AZ, LID_EL, LID_RANGE, LID_STRIDE = 90, np.radians(np.linspace(-45, 10, 10)), 20.0, 3


def lidar_dirs():
    """360 deg mapping lidar: 90 azimuths x 10 elevation channels, index = iaz*10 + iel, body-yaw frame."""
    az = np.linspace(0, 2 * np.pi, LID_AZ, endpoint=False); a, e = np.meshgrid(az, LID_EL, indexing="ij")
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1).reshape(-1, 3).astype(np.float32)


def fly(env, path, policy, max_t, look=7, verbose=True):
    ids = torch.zeros(1, dtype=torch.long); dt = env.dt_phys * env.decim; W = env.world
    P = torch.tensor(path, dtype=torch.float32); start = P[0].clone()
    env.q.reset(ids, start[None], torch.tensor([math.atan2(float(P[min(5, len(P) - 1), 1] - P[0, 1]), float(P[min(5, len(P) - 1), 0] - P[0, 0]))]))
    env.est_bias.zero_(); env.prev_a.zero_(); env._compute_obs(ids, fresh=True)
    frames, est, scans, lid, idx, best, last_prog, outcome, min_clr = [], [], [], [], 0, 0, 0, "timeout", 1e9; LD = torch.tensor(lidar_dirs())[None]; ne = 0
    noise = lambda t: (t + (0.01 + 0.006 * t) * torch.randn_like(t)).clamp(0, MAX_RANGE)
    for step in range(int(max_t / dt)):
        pos = env.q.pos[0]; j0 = max(idx - 2, 0)
        idx = j0 + int((P[j0:j0 + 30] - pos).norm(dim=-1).argmin()); env.goal[0] = P[min(idx + look, len(P) - 1)]
        if step % 2 == 0:
            yaw = quat_yaw(env.q.quat)
            tt, tg = W.raycast(ids, env.q.pos, yaw_rotate(local_ray_dirs(env.device)[None], yaw), MAX_RANGE)
            r_ = torch.minimum(tt, tg); sc = torch.where(r_ >= MAX_RANGE - 1e-3, r_, noise(r_))[0]            # misses stay exactly at max range
            frames.append(r3(env.q.pos[0].tolist() + env.q.quat[0].tolist() + [env.q.thrust_acc[0].item(), float(yaw[0])]))
            est.append(r3(env.est_bias[0].tolist())); scans.append([int(round(float(v) * 100)) for v in sc])
            if ne % LID_STRIDE == 0:                                                                     # dense 360 deg mapping scan
                t2, g2 = W.raycast(ids, env.q.pos, yaw_rotate(LD, yaw), LID_RANGE, steps=110); r2 = torch.minimum(t2, g2)[0]
                r2 = torch.where(r2 >= LID_RANGE - 1e-3, r2, (r2 + (0.02 + 0.004 * r2) * torch.randn_like(r2)).clamp(0.05, LID_RANGE))
                lid.append([int(round(float(v) * 100)) for v in r2])
            ne += 1
        with torch.no_grad():
            a = follow_action(env) if policy is None else policy(env.obs).clamp(-1, 1)
        env.q.wind += (-env.q.wind / 2.0) * dt + 0.4 * (2 * dt / 2.0) ** 0.5 * torch.randn_like(env.q.wind)
        env.est_bias += env.drift * dt ** 0.5 * torch.randn_like(env.est_bias)
        for _ in range(env.decim):
            env.q.step(a, env.dt_phys)
        env.prev_a = a.clone(); env._compute_obs(ids)
        min_clr = min(min_clr, float(W.clearance(ids, env.q.pos)[0]))
        if W.collide(env.q.pos).item() or env._oob(env.q.pos, None).item():
            outcome = "collision"; break
        if idx > best + 1:
            best, last_prog = idx, step
        if step - last_prog > 12 / dt:
            outcome = "stalled"; break
        if idx >= len(P) - 3 and (P[-1] - env.q.pos[0]).norm() < 1.5:
            outcome = "success"; break
    return outcome, frames, est, scans, lid, min_clr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", help="3DGS .ply or .splat capture"); p.add_argument("--demo", action="store_true", help="use the stand-in procedural splat scene")
    p.add_argument("--up", default="z", choices=list(sproxy.UP) + ["auto"], help="axis of the capture that points up (COLMAP/3DGS captures are usually -y)")
    p.add_argument("--scale", type=float, default=1.0, help="multiply scene units to metres"); p.add_argument("--floor", type=float, default=None, help="floor height in scene units after orienting (default: 1st percentile)")
    p.add_argument("--voxel", type=float, default=0.2); p.add_argument("--thr", type=float, default=0.4, help="min opacity-weighted splat count per occupied voxel (lower = more conservative / thicker obstacles)")
    p.add_argument("--min-opacity", type=float, default=0.15, help="splats fainter than this are ignored by the collision proxy"); p.add_argument("--cut-above", type=float, default=None, help="ignore splats higher than this (m above floor), e.g. a sky dome")
    p.add_argument("--max-gauss", type=int, default=1500000, help="cap on Gaussians written for the viewer (largest/most opaque kept)"); p.add_argument("--name", default="scene")
    p.add_argument("--inspect", action="store_true", help="only print what is in the file and suggest --up/--scale; writes <out-dir>/<name>_inspect.png")
    p.add_argument("--fill-down", action="store_true", help="aerial/drone captures: treat everything below the top surface as solid (canopies, roofs have no undersides)")
    p.add_argument("--zmin", type=float, default=1.2, help="random pairs: lowest start/goal height above the floor (m)"); p.add_argument("--zmax", type=float, default=3.5, help="random pairs: highest start/goal height (m)")
    p.add_argument("--start-z", type=float, nargs=2, metavar=("LO", "HI"), help="random pairs: start height band (overrides --zmin/--zmax)"); p.add_argument("--goal-z", type=float, nargs=2, metavar=("LO", "HI"), help="random pairs: goal height band")
    p.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "Z"), help="exact start point in metres (oriented frame: x,y on the map, z above the floor; click the heat map in the viewer to read one)")
    p.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "Z"), help="exact end point, same frame"); p.add_argument("--agl", action="store_true", help="--start/--goal z means metres ABOVE THE LOCAL SURFACE (roof/canopy/ground), not above the floor")
    p.add_argument("--ceiling", type=float, default=None, help="highest altitude the planner may use (default: highest of start/goal + 8 m)")
    p.add_argument("--radius", type=float, default=0.35, help="drone collision radius (m)"); p.add_argument("--margin", type=float, default=1.2, help="planner keeps this far from every surface (m)")
    p.add_argument("--splat-clear", type=float, default=0.5, help="independent check: a flight that gets closer than this to any opaque splat centre is rejected")
    p.add_argument("--min-d", type=float, default=10.0); p.add_argument("--max-d", type=float, default=40.0)
    p.add_argument("--episodes", type=int, default=4); p.add_argument("--policy", default="heuristic"); p.add_argument("--max-time", type=float, default=90)
    p.add_argument("--drift", type=float, default=0.05); p.add_argument("--seed", type=int, default=1); p.add_argument("--out-dir", default="splat_viewer/data")
    a = p.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    if a.demo:
        src = os.path.join(a.out_dir, "demo_scene.ply")
        if not os.path.exists(src):
            sio.write_ply(src, demo_scene.build(0, 1.0))
        a.scene, a.name = src, "demo"
    if not a.scene:
        p.error("give --scene FILE or --demo")
    prefix = os.path.join(a.out_dir, a.name)
    if a.inspect:
        sproxy.inspect(a.scene, prefix + "_inspect.png"); return
    pf = sproxy.build(a.scene, prefix, a.up, a.scale, a.floor, a.voxel, a.thr, max_gauss=a.max_gauss, fill_down=a.fill_down, min_opacity=a.min_opacity, cut_above=a.cut_above)
    rng = np.random.default_rng(a.seed); px = splan.Proxy(pf); lo_, hi_ = px.origin, px.hi
    print(f"scene box (oriented, metres): x {lo_[0]:.1f}..{hi_[0]:.1f}  y {lo_[1]:.1f}..{hi_[1]:.1f}  z 0..{hi_[2]:.1f}")
    if a.start or a.goal:
        assert a.start and a.goal, "give both --start and --goal"
        pts = []
        for nm, q in (("start", a.start), ("goal", a.goal)):
            q = np.array(q, float)
            if a.agl:
                q[2] += px.top_at(q[0], q[1])
            if not ((q[:2] >= lo_[:2]) & (q[:2] <= hi_[:2])).all():
                raise SystemExit(f"--{nm} {q[:2].round(1).tolist()} is outside the scene (x {lo_[0]:.1f}..{hi_[0]:.1f}, y {lo_[1]:.1f}..{hi_[1]:.1f})")
            r = splan.snap_free(px, q, a.margin, r=4.0, zlo=0.3, zhi=hi_[2] + 30)
            if r is None:
                r = splan.snap_free(px, q, a.radius + 0.3, r=4.0, zlo=0.3, zhi=hi_[2] + 30); why = "tight spot, reduced clearance"
                if r is None:
                    raise SystemExit(f"--{nm} {q.round(1).tolist()}: inside solid geometry and no free space within 4 m. Pick a higher z or use --agl.")
            else:
                why = ""
            if np.linalg.norm(r - q) > 0.05:
                print(f"  --{nm} {q.round(2).tolist()} was inside/too close to geometry -> moved to {r.round(2).tolist()} {why}")
            pts.append(r)
        ceil = a.ceiling if a.ceiling is not None else max(pts[0][2], pts[1][2]) + 8.0; path = None
        for m in (a.margin, a.margin * 0.75, a.margin * 0.5):
            path = splan.plan(px, pts[0], pts[1], clear=max(m, a.radius + 0.25), zmin=max(min(pts[0][2], pts[1][2]) - 0.4, 0.5), zmax=ceil)
            if path is not None:
                print(f"  planned a {len(path) * 0.5:.0f} m route (clearance {max(m, a.radius + 0.25):.2f} m, ceiling {ceil:.1f} m)"); break
        if path is None:
            raise SystemExit("no collision-free route between those points. Raise --ceiling (fly over things), lower --margin, or pick other points.")
        pairs = [(pts[0], pts[1], path)]
    else:
        sz, gz = (tuple(a.start_z) if a.start_z else (a.zmin, a.zmax)), (tuple(a.goal_z) if a.goal_z else (a.zmin, a.zmax))
        pairs = splan.sample_pairs(px, max(a.episodes * 3, 8), rng, min_d=a.min_d, max_d=a.max_d, clear=a.margin, start_z=sz, goal_z=gz, path_zmax=a.ceiling)
        print(f"{len(pairs)} feasible start/goal pairs (start z {sz[0]}..{sz[1]} m, goal z {gz[0]}..{gz[1]} m)")
    if not pairs and not (a.start or a.goal):                                                              # scene smaller than --min-d: scale the distances to the scene
        diag = float(np.hypot(*(px.hi[:2] - px.origin[:2]))); print(f"  no pair {a.min_d:.0f}-{a.max_d:.0f} m apart fits in this {diag:.0f} m wide scene: retrying with {0.25 * diag:.0f}-{0.9 * diag:.0f} m")
        pairs = splan.sample_pairs(px, max(a.episodes * 3, 8), rng, min_d=0.25 * diag, max_d=0.9 * diag, clear=a.margin, start_z=sz, goal_z=gz, path_zmax=a.ceiling)
    assert pairs, "no free space found: check --up / --scale (the scene must be in metres, z-up), or lower --margin / raise --ceiling"
    env = SplatDroneEnv(pf, 1, "cpu", pairs=pairs, realism=True, drift=a.drift, wind_sigma=0.4, seed=a.seed, radius=a.radius)
    env.follow_soft = a.margin + 0.1
    from scipy.spatial import cKDTree
    gs = sio.read_splat(prefix + ".splat"); tree = cKDTree(gs["pos"][gs["opacity"] > 0.4])
    ext = np.percentile(gs["pos"], 98, axis=0) - np.percentile(gs["pos"], 2, axis=0)
    print(f"written {prefix}.splat extent x {ext[0]:.1f} y {ext[1]:.1f} z {ext[2]:.1f} m (z must be the up axis)")
    if ext[2] > 0.9 * max(ext[0], ext[1]):
        print("  WARNING: the scene is as tall as it is wide: --up is probably wrong (use --inspect, e.g. --up=-y for COLMAP captures, --up=y, --up=-z)")               # independent of the SDF: raw opaque splat centres
    policy = None
    if a.policy != "heuristic":
        from droneforest.ppo import ActorCritic
        ck = torch.load(a.policy, map_location="cpu"); ac = ActorCritic(OBS_DIM, ACT_DIM, critic_dim=ck.get("critic_dim", OBS_DIM)); ac.load_state_dict(ck["ac"]); policy = lambda o: ac.actor(o)
    eps, tried, wins = [], 0, 0
    for s, g, path in pairs:
        t0 = time.time(); out, fr, est, sc, ld, clr = fly(env, path, policy, a.max_time); tried += 1
        dmin = float(tree.query(np.array([f[:3] for f in fr]))[0].min()) if fr else 0.0
        if out == "success" and dmin < a.splat_clear:
            out = f"too close ({dmin:.2f} m to a splat)"
        print(f"  flight {tried}: {out:9s} {len(fr) * 0.04:5.1f}s  SDF clearance {clr:4.2f} m  nearest splat {dmin:4.2f} m  ({time.time() - t0:.0f}s wall)", flush=True)
        if out == "success":
            wins += 1
            eps.append({"outcome": out, "start": r3(s), "goal": r3(g), "path": [r3(q) for q in path[::2]], "frames": fr, "est": est, "scan": sc, "lidar_scan": ld, "min_clearance": round(clr, 2), "min_splat_dist": round(dmin, 2)})
        if len(eps) >= a.episodes:
            break
    dirs = local_ray_dirs(torch.device("cpu")).tolist()
    meta = {"scene": {"splat": f"data/{a.name}.splat", "lo": r3(px.origin), "hi": r3(px.hi), "floor": 0.0, "voxel": px.vs, "source": os.path.basename(a.scene)},
            "dt": 0.04, "max_range": MAX_RANGE, "lidar": {"az": LID_AZ, "el_deg": [round(float(v), 3) for v in np.degrees(LID_EL)], "stride": LID_STRIDE, "max_range": LID_RANGE},
            "heightmap": {"x0": float(px.origin[0]), "y0": float(px.origin[1]), "cell": px.vs, "nx": int(px.shape[0]), "ny": int(px.shape[1]), "z": [int(round(float(v) * 100)) for v in px.topz.reshape(-1)]},
            "radius": a.radius, "ray_dirs": [[round(v, 5) for v in d] for d in dirs], "drift_m_per_sqrt_s": a.drift, "policy": a.policy,
            "success_rate": f"{wins}/{tried}", "episodes": eps}
    out = os.path.join(a.out_dir, f"{a.name}_rollout.json"); json.dump(meta, open(out, "w"), separators=(",", ":")); print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB", f"({wins}/{tried} flights succeeded)")
    idxp = os.path.join(a.out_dir, "index.json"); cur = json.load(open(idxp)) if os.path.exists(idxp) else []
    cur = [{"id": a.name, "name": a.name, "file": f"data/{a.name}_rollout.json"}] + [c for c in cur if c["id"] != a.name]; json.dump(cur, open(idxp, "w"), indent=1)


if __name__ == "__main__":
    main()
