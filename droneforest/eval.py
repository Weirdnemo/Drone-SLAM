"""Benchmark a policy across forest densities (the number to put in a table / paper).

  python eval.py --policy runs/forest/model.pt --device cuda
  python eval.py --policy runs/baseline/model.pt --realism --levels 0.2 0.5 0.8 1.0
  python eval.py --policy heuristic

For each density level it flies `--envs` fresh episodes and reports success / collision / timeout rate,
mean time-to-goal and path length (successes only), mean speed, minimum obstacle clearance and
action smoothness. Results are appended to --out (CSV).
"""
import argparse, csv, os
import torch
from droneforest.env import DroneForestEnv, OBS_DIM, ACT_DIM
from droneforest.heuristic import heuristic_action
from droneforest.ppo import ActorCritic


def run_level(policy, level, envs, device, realism, seed, scene="forest"):
    env = DroneForestEnv(envs, device, init_level=level, max_level=1.0, freeze_curriculum=True, seed=seed, realism=realism, scene=scene)
    E, dt = envs, env.dt_phys * env.decim
    ids = torch.arange(E, device=device)
    alive = torch.ones(E, dtype=torch.bool, device=device)
    outcome = torch.zeros(E, dtype=torch.long, device=device)       # 0 timeout, 1 success, 2 collision
    steps = torch.zeros(E, device=device); path = torch.zeros(E, device=device)
    min_clear = torch.full((E,), 1e9, device=device); jerk = torch.zeros(E, device=device); spd = torch.zeros(E, device=device)
    prev, prev_a = env.q.pos.clone(), torch.zeros(E, 4, device=device)
    with torch.no_grad():
        for _ in range(env.ep_len):
            a = heuristic_action(env) if policy is None else policy(env.obs)
            _, _, done, info = env.step(a)
            m = alive.float()
            steps += m; path += (env.q.pos - prev).norm(dim=-1) * m; spd += env.q.vel.norm(dim=-1) * m
            jerk += (a.clamp(-1, 1) - prev_a).abs().mean(-1) * m
            min_clear = torch.minimum(min_clear, torch.where(alive, env.forest.clearance(ids, env.q.pos), torch.full_like(min_clear, 1e9)))
            prev, prev_a = env.q.pos.clone(), a.clamp(-1, 1)
            outcome = torch.where(alive & info["success"], torch.ones_like(outcome), outcome)
            outcome = torch.where(alive & info["collision"], torch.full_like(outcome, 2), outcome)
            alive = alive & ~done
            if not alive.any():
                break
    succ = outcome == 1
    f = lambda x: x.float().mean().item()
    sel = lambda v: (v[succ].mean().item() if succ.any() else float("nan"))
    return {"level": level, "success": f(succ), "collision": f(outcome == 2), "timeout": f(outcome == 0),
            "time_s": sel(steps * dt), "path_m": sel(path), "mean_speed": (spd / steps.clamp(min=1)).mean().item(),
            "min_clearance_m": min_clear.clamp(max=50).mean().item(), "action_jerk": (jerk / steps.clamp(min=1)).mean().item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="heuristic")
    p.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--scene", default="forest", choices=["forest", "disaster", "mixed"])
    p.add_argument("--realism", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default="runs/eval.csv")
    a = p.parse_args()
    dev = torch.device(a.device)
    policy = None
    if a.policy != "heuristic":
        ck = torch.load(a.policy, map_location=dev)
        ac = ActorCritic(OBS_DIM, ACT_DIM, critic_dim=ck.get("critic_dim", OBS_DIM)).to(dev)
        ac.load_state_dict(ck["ac"])
        policy = lambda o: ac.actor(o)
    rows = [run_level(policy, l, a.envs, dev, a.realism, a.seed, a.scene) for l in a.levels]
    hdr = ["level", "success", "collision", "timeout", "time_s", "path_m", "mean_speed", "min_clearance_m", "action_jerk"]
    print(f"policy={a.policy} realism={a.realism} envs={a.envs}")
    print(" | ".join(f"{h:>14s}" for h in hdr))
    for r in rows:
        print(" | ".join(f"{r[h]:14.3f}" for h in hdr))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["policy", "realism"] + hdr)
        for r in rows:
            w.writerow([a.policy, a.realism] + [round(r[h], 4) for h in hdr])
    print("appended to", a.out)


if __name__ == "__main__":
    main()
