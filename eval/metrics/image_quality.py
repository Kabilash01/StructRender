"""KB §11 Core Metrics — Image Quality: LPIPS, SSIM, optional PSNR.

PSNR and SSIM are implemented directly (no dependency). LPIPS uses the reference
``lpips`` package (Zhang et al. 2018) when it is installed; there is deliberately
no fallback approximation — a made-up number reported under the name "LPIPS"
would be worse than no number.

Tensors are ``[B, 3, H, W]`` float in ``[0, 1]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Per-image PSNR in dB -> ``[B]``."""
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(data_range**2 / mse)


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    win = (g[:, None] @ g[None, :])[None, None, ...]
    return win.expand(channels, 1, size, size).contiguous()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Per-image mean SSIM (Wang et al. 2004, 11x11 Gaussian window) -> ``[B]``."""
    c = pred.shape[1]
    win = _gaussian_window(window_size, sigma, c, pred.device, pred.dtype)
    pad = window_size // 2

    def filt(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, win, padding=pad, groups=c)

    mu_x, mu_y = filt(pred), filt(target)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = filt(pred * pred) - mu_x2
    sigma_y2 = filt(target * target) - mu_y2
    sigma_xy = filt(pred * target) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    smap = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return smap.flatten(1).mean(dim=1)


_SOBEL_X = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
_LUMA = torch.tensor([0.299, 0.587, 0.114])


def gradient_energy(img: torch.Tensor) -> torch.Tensor:
    """High-pass (squared Sobel gradient) energy of the luma channel -> ``[B]``.

    A blur-sensitive sharpness proxy: smoothing removes high-frequency content,
    so this falls monotonically. That is the failure mode a colour-histogram
    stylisation distance cannot see — a blurred frame keeps its shifted palette
    and so scores *higher* on histogram travel, not lower.

    **Squared** magnitude, not mean absolute magnitude. Measured on this
    project's teacher targets, the mean-|grad| variant is *non-monotone* under
    blur — it goes **up** (1.00 -> 1.04 -> 1.14 -> 1.19 at sigma 0, 0.5, 1.0,
    2.0) because the content is mostly flat 8-bit-quantised regions where the
    gradient is exactly zero, and blurring spreads the few real edges across
    more pixels, raising the mean. Squared energy behaves (1.00 -> 0.98 -> 0.90
    -> 0.65 at the same sigmas). A mean-magnitude sharpness gate here would have
    been not just weak but backwards.
    """
    luma = (img * _LUMA.to(img.device, img.dtype).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    kx = _SOBEL_X.to(img.device, img.dtype).view(1, 1, 3, 3)
    # Replicate padding: zero padding would manufacture a bright frame-wide edge
    # at the image border and put a floor under the energy of a flat image.
    padded = F.pad(luma, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kx)
    gy = F.conv2d(padded, kx.transpose(2, 3))
    return (gx * gx + gy * gy).flatten(1).mean(dim=1)


def lpips_available() -> bool:
    try:
        import lpips  # noqa: F401
    except Exception:
        return False
    return True


class LpipsMetric:
    """Reference LPIPS (AlexNet trunk). Raises if the ``lpips`` package is absent."""

    def __init__(self, net: str = "alex", device: str | torch.device = "cpu"):
        import lpips as _lpips

        self.net = net
        self.model = _lpips.LPIPS(net=net, verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-image LPIPS -> ``[B]``. Inputs in ``[0,1]``, rescaled to ``[-1,1]``."""
        p = pred.to(self.device) * 2.0 - 1.0
        t = target.to(self.device) * 2.0 - 1.0
        return self.model(p, t).flatten()
