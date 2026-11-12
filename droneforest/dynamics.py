"""Batched quadrotor dynamics in PyTorch with a collective-thrust + body-rate (CTBR) interface.

Action a in [-1, 1]^4:
  a[0]    -> collective thrust acceleration  g * (1 + a0), clamped to [0, 2g]
  a[1:4]  -> body-rate setpoint (roll, pitch, yaw) scaled by MAX_RATE
An inner rate loop (first-order lag) tracks the rate setpoint, as on a real flight controller.
"""
import torch

G = 9.81
MAX_RATE = (6.0, 6.0, 3.0)  # rad/s


def quat_to_rot(q):
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def quat_yaw(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_to_quat(yaw):
    z = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)], dim=-1)


class Quadrotor:
    def __init__(self, n, device):
        self.n, self.device = n, device
        z3 = lambda: torch.zeros(n, 3, device=device)
        self.pos, self.vel, self.omega = z3(), z3(), z3()
        self.wind = z3()                      # disturbance acceleration (m/s^2), set by the env
        self.quat = yaw_to_quat(torch.zeros(n, device=device))
        self.thrust_acc = torch.full((n,), G, device=device)
        self.max_rate = torch.tensor(MAX_RATE, device=device)
        # domain randomisation (per env)
        self.thr_scale = torch.ones(n, device=device)
        self.drag = torch.full((n,), 0.15, device=device)
        self.tau_thr = torch.full((n,), 0.05, device=device)
        self.tau_rate = torch.full((n,), 0.03, device=device)

    def reset(self, idx, pos, yaw):
        m = idx.numel()
        self.pos[idx] = pos
        self.vel[idx] = 0
        self.omega[idx] = 0
        self.quat[idx] = yaw_to_quat(yaw)
        self.thrust_acc[idx] = G
        self.thr_scale[idx] = 1 + 0.15 * (2 * torch.rand(m, device=self.device) - 1)
        self.drag[idx] = 0.05 + 0.25 * torch.rand(m, device=self.device)
        self.tau_thr[idx] = 0.03 + 0.04 * torch.rand(m, device=self.device)
        self.tau_rate[idx] = 0.02 + 0.03 * torch.rand(m, device=self.device)

    def rot(self):
        return quat_to_rot(self.quat)

    def step(self, a, dt):
        a = a.clamp(-1, 1)
        thrust_cmd = (G * (1 + a[:, 0])).clamp(0, 2 * G) * self.thr_scale
        self.thrust_acc = self.thrust_acc + (thrust_cmd - self.thrust_acc) * (dt / self.tau_thr).clamp(max=1)
        rate_cmd = a[:, 1:4] * self.max_rate
        self.omega = self.omega + (rate_cmd - self.omega) * (dt / self.tau_rate).clamp(max=1)[:, None]

        w, x, y, z = self.quat.unbind(-1)
        zb = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)
        acc = zb * self.thrust_acc[:, None] - self.drag[:, None] * self.vel + self.wind
        acc[:, 2] -= G
        self.vel = self.vel + acc * dt
        self.pos = self.pos + self.vel * dt

        ox, oy, oz = self.omega.unbind(-1)
        dq = 0.5 * torch.stack([
            -x * ox - y * oy - z * oz,
            w * ox + y * oz - z * oy,
            w * oy + z * ox - x * oz,
            w * oz + x * oy - y * ox], -1)
        q = self.quat + dq * dt
        self.quat = q / q.norm(dim=-1, keepdim=True)
