"""GPU feature helpers: Gaussian blur and MIND descriptors (modality-independent)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gauss_kernel1d(sigma: float, device) -> torch.Tensor:
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gblur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (N,C,H,W) tensor with reflect padding."""
    if sigma <= 0:
        return x
    k = gauss_kernel1d(sigma, x.device)
    c = x.shape[1]
    r = k.numel() // 2
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def _shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    p = max(abs(dy), abs(dx))
    xp = F.pad(x, (p, p, p, p), mode="replicate")
    h, w = x.shape[-2:]
    return xp[..., p + dy:p + dy + h, p + dx:p + dx + w]


def mind(x: torch.Tensor, radius: float = 2.0, dilation: int = 2, eps: float = 1e-3) -> torch.Tensor:
    """2D MIND descriptor (Heinrich et al. 2012), 8-neighbourhood. x: (N,1,H,W) -> (N,8,H,W)."""
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    d = torch.cat([gblur((x - _shift(x, oy * dilation, ox * dilation)) ** 2, radius) for oy, ox in offs], 1)
    v = d.mean(1, keepdim=True)
    v = v.clamp(min=eps * v.mean().clamp(min=1e-8), max=1e3 * v.mean().clamp(min=1e-8))
    m = torch.exp(-d / v)
    return m / m.amax(1, keepdim=True).clamp(min=1e-8)
