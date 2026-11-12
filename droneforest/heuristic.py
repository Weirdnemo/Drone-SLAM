"""Hand-written reference flyer (potential-field avoidance + geometric attitude control).

Used (a) to sanity-check the dynamics/sensors without any training and (b) to produce demo
rollouts for the viewer. It reads privileged simulator state, so it is NOT a learned baseline.
"""
import torch
from .dynamics import G, MAX_RATE, quat_to_rot, quat_yaw
from .env import N_AZ, N_EL, MAX_RANGE


def heuristic_action(env, vmax=4.5, steer_gain=3.0, slow=0.4):
    q, dev = env.q, env.device
    E = env.n
    to_goal = env.goal - q.pos
    dist = to_goal.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    v_des = to_goal / dist * (0.8 * dist).clamp(max=vmax)

    t = env.tree_t.reshape(E, N_AZ, N_EL)
    w = ((1 - t / 9.0).clamp(min=0)) ** 2                  # (E,AZ,EL) obstacle proximity weight
    w_az = w.max(-1).values                                 # (E,AZ)
    half = N_AZ // 2
    right, left = w_az[:, :half].sum(-1), w_az[:, half:].sum(-1)
    front = w_az[:, half - 2:half + 2].max(-1).values
    yaw = quat_yaw(q.quat)
    left_u = torch.stack([-torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)], -1)
    steer = (right - left) + 0.3                            # tie-break towards the left
    v_des = v_des * (1 - slow * front)[:, None] + left_u * (steer_gain * steer.clamp(-3, 3))[:, None] * (0.4 + front)[:, None]
    speed = v_des.norm(dim=-1, keepdim=True)
    v_des = v_des * (speed.clamp(max=vmax) / speed.clamp(min=1e-3))

    ids = torch.arange(E, device=dev)
    hdir = v_des[:, :2] / v_des[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-3)
    gz = env.forest.ground(ids, q.pos[:, 0:1], q.pos[:, 1:2])[:, 0]
    for look in (2.0, 4.0):                            # terrain look-ahead along travel direction
        ahead = q.pos[:, :2] + hdir * look
        gz = torch.maximum(gz, env.forest.ground(ids, ahead[:, 0:1], ahead[:, 1:2])[:, 0])
    clearance = q.pos[:, 2] - gz
    need = torch.where(dist[:, 0] < 5.0, torch.full_like(clearance, 1.2), torch.full_like(clearance, 2.2))
    v_des[:, 2] = v_des[:, 2] + ((need - clearance).clamp(min=0) * 2.5)

    a_des = 3.0 * (v_des - q.vel)
    a_des[:, 2] += G
    hor = a_des[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
    a_des[:, :2] = a_des[:, :2] * (hor.clamp(max=0.6 * G) / hor)
    thrust = a_des.norm(dim=-1)
    z_des = a_des / thrust[:, None]

    R = quat_to_rot(q.quat)
    z_b = R[:, :, 2]
    e_b = torch.einsum("mji,mj->mi", R, torch.cross(z_b, z_des, dim=-1))
    rates = 8.0 * e_b[:, :2]
    yaw_des = torch.atan2(v_des[:, 1], v_des[:, 0])
    err = torch.atan2(torch.sin(yaw_des - yaw), torch.cos(yaw_des - yaw))
    rate_z = (2.0 * err).clamp(-3, 3)
    mr = torch.tensor(MAX_RATE, device=dev)
    a = torch.cat([((thrust / G) - 1)[:, None], rates / mr[:2], (rate_z / mr[2])[:, None]], -1)
    return a.clamp(-1, 1)
