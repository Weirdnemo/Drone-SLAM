"""Long procedural SLAM mission: forest -> collapsed districts -> forest ... flown autonomously.

  python mission.py --seed 7 --length 520                 # heuristic flyer
  python mission.py --seed 7 --policy runs/forest/model.pt  # learned policy (flies toward a moving look-ahead waypoint)

World = seeded biomes along a wandering route. Forest biomes get broadleaf/conifer trees; disaster biomes get ruined
buildings (broken walls + rubble + fallen slabs), wrecked vehicles, snapped stumps, toppled logs, poles, fires, and
survivors to find. The route corridor is kept (mostly) clear so the flight is feasible; seeds are retried until the
drone completes it. Writes viewer/data/mission.json (schema v2: extra boxes/path/fires/pois/biomes/species).
"""
import argparse, json, math, os, time
import numpy as np
import torch
from droneforest import terrain
from droneforest.env import DroneForestEnv, OBS_DIM, ACT_DIM, MAX_RANGE
from droneforest.heuristic import heuristic_action
from droneforest.world import Forest, FAR
from droneforest.dynamics import quat_yaw

T_MAX, B_MAX = 700, 460


def r3(x):
    return [round(float(v), 3) for v in x]


def route_y(x, ph, amp):
    return amp * (0.65 * np.sin(x / 70 + ph[0]) + 0.35 * np.sin(x / 31 + ph[1]))


def build_world(seed, length):
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 6.28, 2)
    amp = rng.uniform(16, 26)
    px = np.arange(0, length + 1e-6, 1.0)
    py = route_y(px, ph, amp)
    P = np.stack([px, py], -1)

    # biomes
    segs, x, kind = [], 0.0, "forest"
    first = rng.uniform(70, 110)
    while x < length:
        ln = first if not segs else rng.uniform(85, 140)
        x1 = min(length, x + ln)
        segs.append({"x0": round(x, 1), "x1": round(x1, 1), "kind": kind})
        x, kind = x1, ("disaster" if kind == "forest" else "forest")
    if segs[-1]["x1"] - segs[-1]["x0"] < 40 and len(segs) > 1:
        segs[-2]["x1"] = segs[-1]["x1"]; segs.pop()
    biome_at = lambda xx: next((s["kind"] for s in segs if s["x0"] <= xx <= s["x1"]), segs[-1]["kind"])
    dpath = lambda xy: np.sqrt(((P[None] - np.asarray(xy)[:, None]) ** 2).sum(-1)).min(1)

    torch.manual_seed(seed)
    terr = terrain.sample_terrain(1, "cpu", 0.8)
    gz = lambda xx, yy: terrain.height(terr, torch.as_tensor(np.atleast_1d(xx), dtype=torch.float32)[None],
                                        torch.as_tensor(np.atleast_1d(yy), dtype=torch.float32)[None])[0].numpy()

    trees, boxes, fires, pois = [], [], [], []
    def add_tree(x, y, kind):
        z = float(gz(x, y)[0])
        if kind == 2:   # snapped stump with debris crown
            trees.append([x, y, z, rng.uniform(2.0, 4.0), rng.uniform(.2, .35), rng.uniform(.5, .9), 2])
        else:
            trees.append([x, y, z, rng.uniform(5, 10), rng.uniform(.2, .45), rng.uniform(1.6, 3.0), kind])
    def add_box(x, y, hx, hy, hz, kind, lift=0.0):
        z = float(gz(x, y)[0])
        boxes.append([x, y, z + hz + lift, hx, hy, hz, kind])
        return boxes[-1]

    for s in segs:
        x0, x1 = s["x0"], s["x1"]
        L = x1 - x0
        if s["kind"] == "forest":
            n = int(L * 60 * 0.032)
            cand = np.stack([rng.uniform(x0, x1, n * 2), route_y(rng.uniform(x0, x1, n * 2), ph, amp) * 0 + 0], -1)
            xs = rng.uniform(x0, x1, n * 2)
            ys = route_y(xs, ph, amp) + rng.normal(0, 14, n * 2)
            d = dpath(np.stack([xs, ys], -1))
            conif = rng.random() < 0.5
            k = 0
            for xx, yy, dd in zip(xs, ys, d):
                if dd < 2.6 or abs(yy) > 52 or k >= n:
                    continue
                add_tree(xx, yy, 1 if rng.random() < (0.7 if conif else 0.2) else 0); k += 1
            mid = x0 + L * rng.uniform(.3, .7)
            my = route_y(mid, ph, amp) + rng.choice([-1, 1]) * rng.uniform(4, 8)
            pois.append({"x": round(mid, 2), "y": round(my, 2), "z": round(float(gz(mid, my)[0]), 2), "type": "hiker"})
        else:
            nb = int(L / 13)
            for _ in range(nb):
                bx = rng.uniform(x0 + 6, x1 - 6)
                side = rng.choice([-1, 1])
                by = route_y(bx, ph, amp) + side * rng.uniform(11, 24)
                w, dpt = rng.uniform(8, 14), rng.uniform(6, 10)
                # 4 broken walls with random gaps / heights
                for (cx, cy, hx, hy) in [(bx, by - dpt / 2, w / 2, .25), (bx, by + dpt / 2, w / 2, .25),
                                         (bx - w / 2, by, .25, dpt / 2), (bx + w / 2, by, .25, dpt / 2)]:
                    if rng.random() < 0.2:
                        continue
                    f = rng.uniform(.35, 1.0)
                    if hx > hy:
                        add_box(cx + rng.uniform(-1, 1), cy, hx * f, hy, rng.uniform(.9, 2.4), 0)
                    else:
                        add_box(cx, cy + rng.uniform(-1, 1), hx, hy * f, rng.uniform(.9, 2.4), 0)
                for _ in range(rng.integers(4, 9)):
                    r = add_box(bx + rng.uniform(-w, w), by + rng.uniform(-dpt, dpt) , rng.uniform(.5, 1.5),
                                rng.uniform(.5, 1.5), rng.uniform(.3, .9), 1)
                    if rng.random() < .18:
                        fires.append([r[0], r[1], r[2] + r[5], round(rng.uniform(.6, 1.2), 2)])
                if rng.random() < .6:
                    add_box(bx + rng.uniform(-3, 3), by + rng.uniform(-2, 2), rng.uniform(2, 3), rng.uniform(1.5, 2.5),
                            rng.uniform(.18, .3), 2, lift=rng.uniform(1.0, 1.8))
                for _ in range(rng.integers(1, 3)):
                    py_ = by + rng.uniform(-dpt, dpt); px_ = bx + rng.uniform(-w, w)
                    pois.append({"x": round(px_, 2), "y": round(py_, 2), "z": round(float(gz(px_, py_)[0]), 2), "type": "survivor"})
            for _ in range(int(L / 12)):       # wrecked vehicles along the road
                vx = rng.uniform(x0 + 3, x1 - 3)
                vy = route_y(vx, ph, amp) + rng.choice([-1, 1]) * rng.uniform(3.2, 7)
                horiz = rng.random() < .5
                r = add_box(vx, vy, 2.1 if horiz else .95, .95 if horiz else 2.1, .75, 3)
                if rng.random() < .45:
                    fires.append([r[0], r[1], r[2] + r[5], round(rng.uniform(.8, 1.4), 2)])
            for _ in range(int(L / 9)):        # toppled logs, poles, stumps
                lx = rng.uniform(x0, x1); ly = route_y(lx, ph, amp) + rng.normal(0, 8)
                if dpath(np.array([[lx, ly]]))[0] < 3.0:
                    continue
                c = rng.random()
                if c < .35:
                    add_box(lx, ly, rng.uniform(1.5, 3), .3, .3, 4) if rng.random() < .5 else add_box(lx, ly, .3, rng.uniform(1.5, 3), .3, 4)
                elif c < .6:
                    add_box(lx, ly, .12, .12, rng.uniform(2.5, 4), 5)
                else:
                    add_tree(lx, ly, 2)
            for _ in range(int(L / 14)):
                lx = rng.uniform(x0, x1); ly = route_y(lx, ph, amp) + rng.normal(0, 9)
                if dpath(np.array([[lx, ly]]))[0] > 3.5:
                    px_ = lx
                    pois.append({"x": round(px_, 2), "y": round(ly, 2), "z": round(float(gz(px_, ly)[0]), 2), "type": "survivor"})
    # keep the route corridor drivable: drop anything whose body overlaps the path
    trees = [t for t in trees if dpath([[t[0], t[1]]])[0] > t[5] + 0.9]
    boxes = [b for b in boxes if dpath([[b[0], b[1]]])[0] > max(b[3], b[4]) + 1.3]
    fires = [f for f in fires if dpath([[f[0], f[1]]])[0] > 1.5]
    trees, boxes = trees[:T_MAX], boxes[:B_MAX]
    for i, p in enumerate(pois):
        p["id"] = i
    pz = gz(P[:, 0], P[:, 1]) + 3.2
    path = np.concatenate([P, pz[:, None]], -1)
    return dict(terr=terr, trees=trees, boxes=boxes, fires=fires, pois=pois, segs=segs, path=path, length=length)


def fly(w, seed, policy, max_t, realism_drift, verbose=True):
    dev = torch.device("cpu")
    env = DroneForestEnv(1, dev, realism=True, drift=realism_drift, wind_sigma=0.4, seed=seed)
    f = Forest(1, T_MAX, dev, max_boxes=B_MAX)
    f.terr = w["terr"].clone()
    for i, t in enumerate(w["trees"]):
        f.tx[0, i], f.ty[0, i], f.tzg[0, i], f.th[0, i], f.trt[0, i], f.trc[0, i] = t[:6]
    for i, b in enumerate(w["boxes"]):
        f.bc[0, i] = torch.tensor(b[:3]); f.bh[0, i] = torch.tensor(b[3:6]); f.bkind[0, i] = int(b[6])
    env.forest = f
    path = torch.tensor(w["path"], dtype=torch.float32)
    start = path[0].clone(); start[2] = float(f.ground(torch.zeros(1, dtype=torch.long), start[None, 0:1], start[None, 1:2])[0, 0]) + 2.5
    env.q.reset(torch.zeros(1, dtype=torch.long), start[None], torch.zeros(1))
    env.est_bias.zero_(); env.prev_a.zero_()
    ids = torch.zeros(1, dtype=torch.long)
    dt = env.dt_phys * env.decim
    frames, est, idx, best, last_prog, t0 = [], [], 0, 0.0, 0, time.time()
    outcome, look = "timeout", 9
    env._compute_obs(ids, fresh=True)
    for step in range(int(max_t / dt)):
        pos = env.q.pos[0]
        j0 = max(idx - 2, 0)
        d = (path[j0:j0 + 40] - pos).norm(dim=-1)
        idx = j0 + int(d.argmin())
        env.goal[0] = path[min(idx + look, len(path) - 1)]
        if step % 2 == 0:
            frames.append(r3(env.q.pos[0].tolist() + env.q.quat[0].tolist() + [env.q.thrust_acc[0].item()]))
            est.append(r3(env.est_bias[0].tolist()))
        with torch.no_grad():
            a = heuristic_action(env, vmax=5.0) if policy is None else policy(env.obs).clamp(-1, 1)
        env.q.wind += (-env.q.wind / 2.0) * dt + 0.4 * (2 * dt / 2.0) ** 0.5 * torch.randn_like(env.q.wind)
        env.est_bias += env.drift * dt ** 0.5 * torch.randn_like(env.est_bias)
        for _ in range(env.decim):
            env.q.step(a, env.dt_phys)
        env.prev_a = a.clone()
        env._compute_obs(ids)
        if f.collide(env.q.pos).item() or env.q.pos[0, 2] - f.ground(ids, env.q.pos[:, 0:1], env.q.pos[:, 1:2])[0, 0] > 30:
            outcome = "collision"; break
        if idx > best + 0.5:
            best, last_prog = idx, step
        if step - last_prog > 20 / dt:
            outcome = "stalled"; break
        if idx >= len(path) - 3 and (path[-1] - env.q.pos[0]).norm() < 4.0:
            outcome = "success"; break
        if verbose and step % 1000 == 0:
            print(f"   t={step*dt:5.1f}s  x={pos[0]:6.1f}/{w['length']:.0f}  ({time.time()-t0:.0f}s wall)", flush=True)
    return outcome, frames, est, dt * 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--length", type=float, default=520)
    p.add_argument("--policy", default="heuristic")
    p.add_argument("--tries", type=int, default=12, help="seeds tried (seed, seed+1, ...) until a clean flight")
    p.add_argument("--drift", type=float, default=0.12)
    p.add_argument("--max-time", type=float, default=240)
    p.add_argument("--out", default="viewer/data/mission.json")
    a = p.parse_args()
    policy = None
    if a.policy != "heuristic":
        from droneforest.ppo import ActorCritic
        ck = torch.load(a.policy, map_location="cpu")
        ac = ActorCritic(OBS_DIM, ACT_DIM, critic_dim=ck.get("critic_dim", OBS_DIM))
        ac.load_state_dict(ck["ac"]); policy = lambda o: ac.actor(o)
    for k in range(a.tries):
        seed = a.seed + k
        w = build_world(seed, a.length)
        print(f"seed {seed}: {len(w['trees'])} trees, {len(w['boxes'])} boxes, {len(w['fires'])} fires, {len(w['pois'])} POIs, "
              f"biomes {[s['kind'][0] for s in w['segs']]}", flush=True)
        outcome, frames, est, edt = fly(w, seed, policy, a.max_time, a.drift)
        print("  ->", outcome, f"({len(frames)*edt:.0f}s of flight)", flush=True)
        if outcome == "success":
            break
    else:
        print("no clean flight found; exporting the last attempt anyway")
    ep = {"scene": "mission", "seed": seed, "terrain": w["terr"][0].tolist(), "trees": [r3(t) for t in w["trees"]],
          "boxes": [r3(b) for b in w["boxes"]], "fires": [r3(f) for f in w["fires"]], "pois": w["pois"],
          "biomes": w["segs"], "path": [r3(q) for q in w["path"][::2].tolist()],
          "start": frames[0][:3], "goal": r3(w["path"][-1].tolist()), "frames": frames, "est": est, "outcome": outcome,
          "stats": {"time": round(len(frames) * edt, 1), "length": w["length"]}}
    meta = {"dt": edt, "drone_radius": 0.25, "max_range": MAX_RANGE, "realism": True, "drift": a.drift, "policy": os.path.basename(os.path.dirname(a.policy)) if a.policy != "heuristic" else "heuristic", "schema": 2}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"meta": meta, "episodes": [ep]}, fh, separators=(",", ":"))
    print("wrote", a.out, f"{os.path.getsize(a.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
