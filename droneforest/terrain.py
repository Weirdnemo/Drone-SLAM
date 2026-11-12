"""Analytic procedural terrain: h(x, y) = sum_k a_k sin(kx_k x + ky_k y + ph_k).

Being analytic, the exact same function is evaluated in torch (training) and in
JavaScript (viewer), so the rendered world is bit-for-bit the world the policy flew in.
"""
import math
import torch

K = 5
WAVELENGTHS = (45.0, 28.0, 17.0, 10.0, 6.0)
AMPLITUDES = (2.6, 1.4, 0.7, 0.35, 0.15)


def sample_terrain(n, device, amp=1.0):
    """Returns params of shape (n, K, 4) = (amplitude, kx, ky, phase)."""
    wl = torch.tensor(WAVELENGTHS, device=device)
    am = torch.tensor(AMPLITUDES, device=device) * amp
    ang = torch.rand(n, K, device=device) * 2 * math.pi
    wl = wl * (0.8 + 0.4 * torch.rand(n, K, device=device))
    k = 2 * math.pi / wl
    a = am * (0.7 + 0.6 * torch.rand(n, K, device=device))
    ph = torch.rand(n, K, device=device) * 2 * math.pi
    return torch.stack([a, k * torch.cos(ang), k * torch.sin(ang), ph], dim=-1)


def height(params, x, y):
    """params (n,K,4); x, y (n, ...) -> ground height with the same shape."""
    shp = x.shape
    n = shp[0]
    xf = x.reshape(n, -1, 1)
    yf = y.reshape(n, -1, 1)
    a, kx, ky, ph = (params[:, None, :, i] for i in range(4))
    h = (a * torch.sin(kx * xf + ky * yf + ph)).sum(-1)
    return h.reshape(shp)
