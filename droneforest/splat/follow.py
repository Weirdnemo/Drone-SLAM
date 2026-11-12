"""Reference path follower for splat scenes (privileged: reads the SDF, so it is NOT a learned baseline).

Pure pursuit of the planned waypoint + repulsion along the SDF gradient when closer than `soft` metres, then the same geometric attitude
controller as droneforest/heuristic.py.
"""
import torch
from droneforest.dynamics import G, MAX_RATE, quat_to_rot, quat_yaw


def follow_action(env, vmax=3.0, soft=None, push=2.5):
    soft = getattr(env, 'follow_soft', 1.3) if soft is None else soft
    q, W, dev = env.q, env.world, env.device
    to_goal = env.goal - q.pos; dist = to_goal.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    v_des = to_goal / dist * (1.2 * dist).clamp(max=vmax)
    e = 0.15; I = torch.eye(3, device=dev)
    grad = torch.stack([(W.sdf_at(q.pos + e * I[i]) - W.sdf_at(q.pos - e * I[i])) / (2 * e) for i in range(3)], -1)
    grad = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-6); d = W.sdf_at(q.pos)
    v_des = v_des + grad * (push * (soft - d).clamp(min=0))[:, None]
    allowed = vmax * ((d - 0.4) / 1.5).clamp(0.25, 1.0)[:, None]                                       # slow down when close to anything
    sp = v_des.norm(dim=-1, keepdim=True); v_des = v_des * (torch.minimum(sp, allowed) / sp.clamp(min=1e-3))
    a_des = 3.0 * (v_des - q.vel); a_des[:, 2] += G
    hor = a_des[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6); a_des[:, :2] = a_des[:, :2] * (hor.clamp(max=0.6 * G) / hor)
    thrust = a_des.norm(dim=-1); z_des = a_des / thrust[:, None]
    R = quat_to_rot(q.quat); z_b = R[:, :, 2]
    e_b = torch.einsum("mji,mj->mi", R, torch.cross(z_b, z_des, dim=-1)); rates = 8.0 * e_b[:, :2]
    yaw = quat_yaw(q.quat); yaw_des = torch.atan2(v_des[:, 1], v_des[:, 0]); err = torch.atan2(torch.sin(yaw_des - yaw), torch.cos(yaw_des - yaw)); rate_z = (2.0 * err).clamp(-3, 3)
    mr = torch.tensor(MAX_RATE, device=dev)
    return torch.cat([((thrust / G) - 1)[:, None], rates / mr[:2], (rate_z / mr[2])[:, None]], -1).clamp(-1, 1)
