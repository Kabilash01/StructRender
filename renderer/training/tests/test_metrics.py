"""Sanity checks on the directly-implemented image-quality metrics (KB §11)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from eval.metrics.image_quality import gradient_energy, lpips_available, psnr, ssim


def _gaussian_blur(t: torch.Tensor, sigma: float) -> torch.Tensor:
    k = int(2 * round(3 * sigma) + 1)
    x = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :]).view(1, 1, k, k).expand(3, 1, k, k)
    return F.conv2d(t, w, padding=k // 2, groups=3)


def test_identical_images_score_at_the_ceiling():
    x = torch.rand(2, 3, 64, 64)
    assert float(psnr(x, x).min()) > 100.0
    assert float(ssim(x, x).min()) > 0.999


def test_known_psnr_value():
    x = torch.zeros(1, 3, 32, 32)
    y = torch.full_like(x, 0.1)  # MSE = 0.01 -> 20 dB
    assert abs(float(psnr(x, y)) - 20.0) < 1e-3


def test_ssim_degrades_with_noise():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    a = ssim(x, (x + 0.02 * torch.randn_like(x)).clamp(0, 1))
    b = ssim(x, (x + 0.20 * torch.randn_like(x)).clamp(0, 1))
    assert float(a) > float(b)
    assert 0.0 <= float(b) <= float(a) <= 1.0


def test_gradient_energy_falls_monotonically_under_blur():
    """The whole point of the sharpness term: it must move the *right way* when
    an image is smoothed. Pinned because the mean-|grad| variant does not — on
    mostly-flat quantised content it rises under blur (see the docstring)."""
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    ratios = [float(gradient_energy(_gaussian_blur(x, s)) / gradient_energy(x)) for s in (0.5, 1.0, 2.0)]
    assert ratios == sorted(ratios, reverse=True), f"not monotone under blur: {ratios}"
    assert ratios[0] < 1.0 and ratios[-1] < ratios[0]


def test_gradient_energy_is_zero_for_a_flat_image():
    assert float(gradient_energy(torch.full((1, 3, 32, 32), 0.4))) < 1e-9


def test_lpips_is_the_reference_implementation_or_absent():
    """Never a home-made stand-in reported under the name LPIPS."""
    if not lpips_available():
        return
    import lpips

    from eval.metrics.image_quality import LpipsMetric

    m = LpipsMetric(device="cpu")
    assert isinstance(m.model, lpips.LPIPS)
    x = torch.rand(1, 3, 64, 64)
    assert float(m(x, x)) < 1e-4
