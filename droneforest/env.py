"""DroneForest: fly a quadrotor from a start zone to a goal through a procedural forest.

Observation (97-d):  goal in body frame (3) | body velocity (3) | gravity in body frame (3) |
                     body rates (3) | previous action (4) | height above ground (1) | range scan (80)
Action (4-d):        collective thrust + body rates (see dynamics.py)

Realism options (env(realism=True)), motivated by sim-to-real work on vision-based agile flight
(Loquercio et al. 2021, Kaufmann et al. 2023, Aerial Gym 2025) and GPS-denied navigation:
  * range noise that grows with distance, random dropouts (returns lost -> max range)
  * gyro / velocity noise, and a slowly drifting position estimate (INS-style random walk) that
    corrupts the goal-relative measurement
  * one-step sensing latency, wind gusts (Ornstein-Uhlenbeck acceleration disturbance)
The critic is *asymmetric*: it sees privileged clean state (true pose, wind, clearance), the actor never does.

Training tricks lifted from the Spot-terrain post, adapted to a drone:
  * frozen teacher (an open-field flyer) + scan-gated imitation reward: the student is pulled
    towards the teacher's actions only where no obstacle is near, so it deviates only when needed.
  * per-env adaptive curriculum: every env promotes / demotes its own forest density.
"""
import math
import torch
from . import dynamics as dyn
from .dynamics import Quadrotor, quat_to_rot, quat_yaw
from .world import Forest, ARENA_X, ARENA_Y

N_AZ, N_EL = 16, 5
N_RAYS = N_AZ * N_EL
OBS_DIM = 3 + 3 + 3 + 3 + 4 + 1 + N_RAYS
PRIV_DIM = OBS_DIM + 12            # clean obs + wind(3) + pos(3) + vel(3) + thrust/drag scale(2) + clearance(1)
ACT_DIM = 4
MAX_RANGE = 12.0


def local_ray_dirs(device):
    az = torch.linspace(-math.radians(70), math.radians(70), N_AZ, device=device)
    el = torch.tensor([-35., -15., 0., 15., 35.], device=device) * math.pi / 180
    AZ, EL = torch.meshgrid(az, el, indexing="ij")
    d = torch.stack([torch.cos(EL) * torch.cos(AZ), torch.cos(EL) * torch.sin(AZ), torch.sin(EL)], -1)
    return d.reshape(-1, 3)


def yaw_rotate(v, yaw):
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


class DroneForestEnv:
    dt_phys = 0.01
    decim = 2                       # policy at 50 Hz
    goal_radius = 1.5

    def __init__(self, num_envs=1024, device="cuda", max_trees=64, max_level=1.0, init_level=0.0,
                 il_weight=0.0, ep_len=600, freeze_curriculum=False, seed=0, realism=False,
                 drift=0.05, wind_sigma=0.6):
        torch.manual_seed(seed)
        self.n, self.device = num_envs, torch.device(device)
        self.max_level, self.il_weight, self.ep_len = max_level, il_weight, ep_len
        self.freeze_curriculum = freeze_curriculum
        self.realism, self.drift, self.wind_sigma = realism, drift, wind_sigma
        self.q = Quadrotor(num_envs, self.device)
        self.forest = Forest(num_envs, max_trees, self.device)
        self.local_dirs = local_ray_dirs(self.device)
        self.level = torch.full((num_envs,), float(init_level), device=self.device)
        self.goal = torch.zeros(num_envs, 3, device=self.device)
        self.start = torch.zeros(num_envs, 3, device=self.device)
        self.t = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.prev_a = torch.zeros(num_envs, ACT_DIM, device=self.device)
        self.d_prev = torch.ones(num_envs, device=self.device)
        self.obs = torch.zeros(num_envs, OBS_DIM, device=self.device)
        self.obs_t = torch.zeros(num_envs, OBS_DIM, device=self.device)   # teacher view (no trees in scan)
        self.min_tree = torch.full((num_envs,), MAX_RANGE, device=self.device)
        self.tree_t = torch.full((num_envs, N_RAYS), MAX_RANGE, device=self.device)
        self.dirs_w = torch.zeros(num_envs, N_RAYS, 3, device=self.device)
        self.teacher = None            # callable(obs) -> action
        self.est_bias = torch.zeros(num_envs, 3, device=self.device)      # INS-style position-estimate drift
        self.priv = torch.zeros(num_envs, PRIV_DIM, device=self.device)   # privileged critic input
        self._last = torch.zeros(num_envs, OBS_DIM, device=self.device)   # previous (undelayed) obs
        self.reset_idx(torch.arange(num_envs, device=self.device))
        self._compute_obs(torch.arange(num_envs, device=self.device), fresh=True)

    # ------------------------------------------------------------------ reset
    def reset_idx(self, idx):
        m, dev = idx.numel(), self.device
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(m, device=dev)
        sxy = torch.stack([u(-ARENA_X + 2, -ARENA_X + 4), u(-6, 6)], -1)
        gxy = torch.stack([u(ARENA_X - 5, ARENA_X - 2), u(-8, 8)], -1)
        lvl = self.level[idx]
        self.forest.regenerate(idx, lvl, sxy, gxy)
        gz = self.forest.ground(idx, sxy[:, 0:1], sxy[:, 1:2])[:, 0]
        tz = self.forest.ground(idx, gxy[:, 0:1], gxy[:, 1:2])[:, 0]
        start = torch.cat([sxy, (gz + 2.5)[:, None]], -1)
        goal = torch.cat([gxy, (tz + u(2.0, 4.0))[:, None]], -1)
        self.start[idx], self.goal[idx] = start, goal
        self.q.reset(idx, start, u(-0.4, 0.4))
        self.t[idx] = 0
        self.prev_a[idx] = 0
        self.est_bias[idx] = 0
        self.q.wind[idx] = 0
        self.d_prev[idx] = (goal - start).norm(dim=-1)

    # ------------------------------------------------------------------ observation
    def _compute_obs(self, idx, fresh=False):
        q, dev, m = self.q, self.device, idx.numel()
        pos, quat, vel, om = q.pos[idx], q.quat[idx], q.vel[idx], q.omega[idx]
        R = quat_to_rot(quat)
        yaw = quat_yaw(quat)
        dirs = yaw_rotate(self.local_dirs[None].expand(m, -1, -1), yaw)
        tt, tg = self.forest.raycast(idx, pos, dirs, MAX_RANGE)
        to_b = lambda v: torch.einsum("mji,mj->mi", R, v)
        gz = self.forest.ground(idx, pos[:, 0:1], pos[:, 1:2])[:, 0]
        grav_b = to_b(torch.tensor([0., 0., -1.], device=dev).expand_as(pos))

        def build(tt_, tg_, goal_rel, vel_, om_, grav_):
            height = ((pos[:, 2] - gz) / 6.0).clamp(-1, 3)[:, None]
            head = torch.cat([(to_b(goal_rel) / 20.0).clamp(-3, 3), to_b(vel_) / 5.0, grav_, om_ / 6.0,
                              self.prev_a[idx], height], -1)
            return head, torch.minimum(tt_, tg_) / MAX_RANGE, tg_ / MAX_RANGE

        # clean (privileged) observation
        head_c, scan_c, gscan_c = build(tt, tg, self.goal[idx] - pos, vel, om, grav_b)
        clean = torch.cat([head_c, scan_c], -1)
        if self.realism:
            def noisy_range(t):
                t = t + (0.01 + 0.006 * t) * torch.randn_like(t)             # sigma grows with range
                drop = torch.rand_like(t) < 0.03                              # lost returns
                return torch.where(drop, torch.full_like(t, MAX_RANGE), t).clamp(0, MAX_RANGE)
            tt_n, tg_n = noisy_range(tt), noisy_range(tg)
            goal_rel = self.goal[idx] - (pos + self.est_bias[idx])           # drifting position estimate
            head, scan, gscan = build(tt_n, tg_n, goal_rel, vel + 0.08 * torch.randn_like(vel),
                                      om + 0.05 * torch.randn_like(om), grav_b + 0.02 * torch.randn_like(grav_b))
            new = torch.cat([head, scan], -1)
            teacher_view = torch.cat([head, gscan], -1)
        else:
            new, teacher_view = clean, torch.cat([head_c, gscan_c], -1)
        if self.realism and not fresh:
            self.obs[idx] = self._last[idx]                                   # one-step sensing latency
        else:
            self.obs[idx] = new
        self._last[idx] = new
        self.obs_t[idx] = teacher_view
        self.min_tree[idx] = tt.min(-1).values                                # truth (used for reward / gating)
        self.tree_t[idx] = tt
        self.dirs_w[idx] = dirs
        extra = torch.cat([q.wind[idx] / 3.0, pos / 20.0, vel / 5.0, (q.thr_scale[idx] - 1)[:, None],
                           (q.drag[idx] - 0.15)[:, None], (self.forest.clearance(idx, pos) / 5.0).clamp(0, 2)[:, None]], -1)
        self.priv[idx] = torch.cat([clean, extra], -1)

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = action.clamp(-1, 1)
        all_idx = torch.arange(self.n, device=self.device)
        a_teacher = None
        if self.teacher is not None and self.il_weight > 0:
            with torch.no_grad():
                a_teacher = self.teacher(self.obs_t).clamp(-1, 1)

        if self.realism:
            dt = self.dt_phys * self.decim
            tau = 2.0                                                          # gust correlation time (s)
            self.q.wind += (-self.q.wind / tau) * dt + self.wind_sigma * (2 * dt / tau) ** 0.5 * torch.randn_like(self.q.wind)
            self.est_bias += self.drift * dt ** 0.5 * torch.randn_like(self.est_bias)
        for _ in range(self.decim):
            self.q.step(a, self.dt_phys)
        self.t += 1
        pos = self.q.pos
        self._compute_obs(all_idx)

        # ---- termination
        d_now = (self.goal - pos).norm(dim=-1)
        gz = self.forest.ground(all_idx, pos[:, 0:1], pos[:, 1:2])[:, 0]
        oob = (pos[:, 0].abs() > ARENA_X + 4) | (pos[:, 1].abs() > ARENA_Y + 4) | (pos[:, 2] - gz > 25)
        collision = self.forest.collide(pos) | oob
        success = (d_now < self.goal_radius) & ~collision
        timeout = (self.t >= self.ep_len) & ~collision & ~success
        done = collision | success | timeout

        # ---- reward
        yaw = quat_yaw(self.q.quat)
        bearing = torch.atan2(self.goal[:, 1] - pos[:, 1], self.goal[:, 0] - pos[:, 0])
        r_prog = 2.0 * (self.d_prev - d_now)
        r_yaw = -0.05 * (1 - torch.cos(yaw - bearing))
        r_act = -0.01 * ((a - self.prev_a) ** 2).sum(-1)
        r_rate = -0.003 * (self.q.omega ** 2).sum(-1)
        r_near = -0.3 * (1.5 - self.min_tree).clamp(min=0)
        rew = 0.05 + r_prog + r_yaw + r_act + r_rate + r_near          # +0.05 alive bonus
        if a_teacher is not None:
            gate = ((self.min_tree - 3.0) / 4.0).clamp(0, 1)           # 1 = no obstacle nearby
            rew = rew - self.il_weight * gate * ((a - a_teacher) ** 2).mean(-1)
        rew = rew + 10.0 * success.float() - 10.0 * collision.float()
        self.d_prev = d_now
        self.prev_a = a.clone()

        info = {"success": success, "collision": collision, "timeout": timeout,
                "obs_final": self.obs.clone(), "priv_final": self.priv.clone()}

        # ---- per-env adaptive curriculum + reset
        didx = done.nonzero(as_tuple=True)[0]
        if didx.numel() > 0:
            if not self.freeze_curriculum:
                delta = 0.04 * success[didx].float() - 0.04 * collision[didx].float() - 0.02 * timeout[didx].float()
                self.level[didx] = (self.level[didx] + delta).clamp(0.0, self.max_level)
            self.reset_idx(didx)
            self._compute_obs(didx, fresh=True)
        info["level_mean"] = self.level.mean()
        return self.obs, rew, done, info

    def reset(self):
        self.reset_idx(torch.arange(self.n, device=self.device))
        self._compute_obs(torch.arange(self.n, device=self.device), fresh=True)
        return self.obs
