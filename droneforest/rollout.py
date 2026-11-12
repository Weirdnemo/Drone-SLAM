"""Roll out a policy and export trajectories + the exact world for the viewer.

  python rollout.py --policy heuristic --level 0.6            # no training needed
  python rollout.py --policy runs/forest/model.pt --level 0.8 --device cuda
  python rollout.py --policy runs/forest/model.pt --realism   # fly with noise / drift / wind / latency

Each exported episode carries: terrain params, trees, start/goal, per-frame pose, an INS-style
position-estimate drift (`est`, used by the viewer to map with a drifting pose), and summary stats.
Writes viewer/data/rollout.json
"""
import argparse, json, os
import torch
from droneforest.env import DroneForestEnv, OBS_DIM, PRIV_DIM, ACT_DIM, MAX_RANGE
from droneforest.heuristic import heuristic_action
from droneforest.ppo import ActorCritic
from droneforest.world import FAR


def r3(x):
    return [round(v, 3) for v in x]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="heuristic")
    p.add_argument("--level", type=float, default=0.6)
    p.add_argument("--episodes", type=int, default=4, help="episodes to export")
    p.add_argument("--pool", type=int, default=48, help="envs simulated; successes are exported first")
    p.add_argument("--scene", default="forest", choices=["forest", "disaster", "mixed"])
    p.add_argument("--realism", action="store_true", help="noise/dropout/drift/latency/wind in the sim")
    p.add_argument("--drift", type=float, default=0.05, help="position-estimate drift (m / sqrt(s)) exported for the viewer")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="viewer/data/rollout.json")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    dev = torch.device(a.device)
    env = DroneForestEnv(a.pool, dev, init_level=a.level, max_level=1.0, freeze_curriculum=True, seed=a.seed,
                         realism=a.realism, drift=a.drift, scene=a.scene)
    policy = None
    if a.policy != "heuristic":
        ck = torch.load(a.policy, map_location=dev)
        ac = ActorCritic(OBS_DIM, ACT_DIM, critic_dim=ck.get("critic_dim", OBS_DIM)).to(dev)
        ac.load_state_dict(ck["ac"])
        policy = lambda o: ac.actor(o)

    E, f, dt = a.pool, env.forest, env.dt_phys * env.decim
    worlds = []
    for e in range(E):
        act = f.tx[e] < FAR / 2
        trees = torch.stack([f.tx[e], f.ty[e], f.tzg[e], f.th[e], f.trt[e], f.trc[e]], -1)[act]
        worlds.append({"terrain": f.terr[e].tolist(), "trees": [r3(t) for t in trees.tolist()],
                       "boxes": [r3(b) for b in torch.cat([f.bc[e], f.bh[e], f.bkind[e][:, None].float()], -1)[f.bc[e, :, 0] < FAR / 2].tolist()] if f.B else [],
                       "start": r3(env.start[e].tolist()), "goal": r3(env.goal[e].tolist()),
                       "frames": [], "est": [], "outcome": "timeout"})
    alive = torch.ones(E, dtype=torch.bool)
    walk = torch.zeros(E, 3)                                   # viewer-side drift when the sim itself has none
    path = torch.zeros(E); min_clear = torch.full((E,), 1e9); vmax = torch.zeros(E); vsum = torch.zeros(E); nstep = torch.zeros(E)
    ids = torch.arange(E, device=dev)
    prev = env.q.pos.clone()
    with torch.no_grad():
        for _ in range(env.ep_len):
            q = env.q
            bias = env.est_bias if a.realism else walk
            for e in range(E):
                if alive[e]:
                    worlds[e]["frames"].append(r3(q.pos[e].tolist() + q.quat[e].tolist() + [q.thrust_acc[e].item()]))
                    worlds[e]["est"].append(r3(bias[e].tolist()))
            act = heuristic_action(env) if policy is None else policy(env.obs)
            _, _, done, info = env.step(act)
            if not a.realism:
                walk += a.drift * dt ** 0.5 * torch.randn(E, 3)
            m = alive.to(dev)
            step_len = (env.q.pos - prev).norm(dim=-1)
            spd = env.q.vel.norm(dim=-1)
            cl = env.forest.clearance(ids, env.q.pos)
            path += (step_len * m).cpu(); vmax = torch.maximum(vmax, (spd * m).cpu()); vsum += (spd * m).cpu(); nstep += m.cpu().float()
            min_clear = torch.minimum(min_clear, torch.where(m, cl, torch.full_like(cl, 1e9)).cpu())
            prev = env.q.pos.clone()
            for e in range(E):
                if alive[e] and done[e]:
                    worlds[e]["outcome"] = ("success" if info["success"][e] else
                                            "collision" if info["collision"][e] else "timeout")
                    alive[e] = False
            if not alive.any():
                break
    for e in range(E):
        worlds[e]["stats"] = {"time": round(nstep[e].item() * dt, 2), "path": round(path[e].item(), 1),
                              "min_clearance": round(max(min_clear[e].item(), 0), 2),
                              "mean_speed": round(vsum[e].item() / max(nstep[e].item(), 1), 2),
                              "max_speed": round(vmax[e].item(), 2)}
    rank = {"success": 0, "timeout": 1, "collision": 2}
    worlds = sorted(worlds, key=lambda w: (rank[w["outcome"]], -len(w["frames"])))[:a.episodes]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    meta = {"dt": dt, "drone_radius": 0.25, "max_range": MAX_RANGE, "realism": a.realism, "drift": a.drift,
            "policy": os.path.basename(os.path.dirname(a.policy)) if a.policy != "heuristic" else "heuristic"}
    with open(a.out, "w") as fh:
        json.dump({"meta": meta, "episodes": worlds}, fh, separators=(",", ":"))
    for i, w in enumerate(worlds):
        s = w["stats"]
        print(f"episode {i}: {w['outcome']:9s} t={s['time']:5.1f}s path={s['path']:5.1f}m "
              f"min-clearance={s['min_clearance']:.2f}m vmean={s['mean_speed']:.1f} trees={len(w['trees'])}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
