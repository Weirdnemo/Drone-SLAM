"""Training modes (the ablation in one script):

  python train.py --stage teacher                                  # open-field flyer, no trees -> runs/teacher/model.pt
  python train.py --stage forest --teacher runs/teacher/model.pt   # init from teacher + scan-gated imitation
  python train.py --stage forest --teacher runs/teacher/model.pt --il-weight 0   # init only
  python train.py --stage baseline                                 # forest from scratch

Add --realism to train with sensor noise / dropout / INS drift / latency / wind gusts.
The critic is asymmetric (privileged clean state) unless you pass --no-asym.
"""
import argparse, csv, os, time
import torch
from droneforest.env import DroneForestEnv, OBS_DIM, PRIV_DIM, ACT_DIM
from droneforest.ppo import ActorCritic, PPO, load_init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["teacher", "forest", "baseline"], required=True)
    p.add_argument("--teacher", default=None, help="teacher checkpoint (init + imitation target)")
    p.add_argument("--il-weight", type=float, default=0.5)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--max-trees", type=int, default=64)
    p.add_argument("--scene", default="forest", choices=["forest", "disaster", "mixed"])
    p.add_argument("--realism", action="store_true", help="sensor noise, dropout, INS drift, latency, wind")
    p.add_argument("--no-asym", action="store_true", help="symmetric critic (actor obs only)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = a.out or f"runs/{a.stage}"
    os.makedirs(out, exist_ok=True)
    dev = torch.device(a.device)
    use_teacher = a.stage == "forest" and a.teacher
    env = DroneForestEnv(a.envs, dev, max_trees=a.max_trees,
                         max_level=0.0 if a.stage == "teacher" else 1.0,
                         init_level=0.0 if a.stage == "teacher" else 0.1,
                         il_weight=a.il_weight if use_teacher else 0.0, seed=a.seed, realism=a.realism, scene=a.scene)
    cdim = OBS_DIM if a.no_asym else PRIV_DIM
    ac = ActorCritic(OBS_DIM, ACT_DIM, critic_dim=cdim).to(dev)

    if use_teacher:
        ck = torch.load(a.teacher, map_location=dev)
        teacher = ActorCritic(OBS_DIM, ACT_DIM, critic_dim=ck.get("critic_dim", OBS_DIM)).to(dev)
        teacher.load_state_dict(ck["ac"])
        n_ok, n_all = load_init(ac, ck["ac"])
        for q in teacher.parameters():
            q.requires_grad_(False)
        env.teacher = lambda o: teacher.actor(o)
        print(f"[init] teacher {a.teacher}: copied {n_ok}/{n_all} tensors; il_weight={a.il_weight}")

    ppo = PPO(ac)
    H, E = a.horizon, a.envs
    obs, cobs = env.obs.clone(), (env.obs if a.no_asym else env.priv).clone()
    crit = (lambda e: e.obs) if a.no_asym else (lambda e: e.priv)
    log = open(os.path.join(out, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["iter", "reward", "success", "collision", "level", "lr", "std"])
    t0 = time.time()

    for it in range(1, a.iters + 1):
        B = lambda *s: torch.zeros(H, E, *s, device=dev)
        o_b, c_b, a_b, lp_b, v_b, r_b, term_b, boot_b = B(OBS_DIM), B(cdim), B(ACT_DIM), B(), B(), B(), B(), B()
        n_succ = n_col = n_done = 0.0
        for t in range(H):
            act, lp, val, _ = ac.act(obs, cobs)
            nobs, rew, done, info = env.step(act)
            o_b[t], c_b[t], a_b[t], lp_b[t], v_b[t], r_b[t] = obs, cobs, act, lp, val, rew
            term_b[t] = done.float()                       # chain is cut on every reset
            to = info["timeout"]
            if to.any():                                   # truncation: bootstrap from the final observation's value
                with torch.no_grad():
                    cf = info["obs_final"] if a.no_asym else info["priv_final"]
                    boot_b[t] = torch.where(to, ac.critic(cf)[:, 0], torch.zeros_like(rew))
            n_succ += info["success"].sum().item(); n_col += info["collision"].sum().item(); n_done += done.sum().item()
            obs, cobs = nobs, crit(env)
        with torch.no_grad():
            last_v = ac.critic(cobs)[:, 0]
        adv, ret = ppo.gae(r_b, v_b, last_v, term_b, boot_b)
        st = ppo.update(o_b, c_b, a_b, lp_b, adv, ret, v_b)
        nd = max(n_done, 1)
        row = [it, r_b.mean().item(), n_succ / nd, n_col / nd, info["level_mean"].item(), st["lr"], st["std"]]
        w.writerow(row); log.flush()
        if it % 10 == 0 or it == 1:
            print(f"it {it:5d} | rew {row[1]:+.3f} | succ {row[2]:.2f} | coll {row[3]:.2f} | "
                  f"level {row[4]:.2f} | lr {st['lr']:.1e} | std {st['std']:.2f} | {time.time()-t0:.0f}s", flush=True)
        if it % 50 == 0 or it == a.iters:
            torch.save({"ac": ac.state_dict(), "critic_dim": cdim, "realism": a.realism}, os.path.join(out, "model.pt"))
    print("saved", os.path.join(out, "model.pt"))


if __name__ == "__main__":
    main()
