"""Small numpy image primitives shared by the deterministic teacher backend and
the alignment validator.

Deliberately dependency-free (numpy only): the M3 test path must run on a clean
checkout with no torch / scipy / opencv (KB Decision 31 — the ``deterministic``
backend is the CI teacher).
"""

from __future__ import annotations

import numpy as np

_SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])


def shift(arr: np.ndarray, dy: int, dx: int, *, mode: str = "edge") -> np.ndarray:
    """Shift so that ``out[y, x] == arr[y + dy, x + dx]`` (border per ``mode``:
    ``"edge"`` replicates, ``"zero"`` fills with 0/False)."""
    pad_y = (max(-dy, 0), max(dy, 0))
    pad_x = (max(-dx, 0), max(dx, 0))
    pads = [pad_y, pad_x] + [(0, 0)] * (arr.ndim - 2)
    if mode == "edge":
        padded = np.pad(arr, pads, mode="edge")
    elif mode == "zero":
        fill = False if arr.dtype == bool else 0
        padded = np.pad(arr, pads, mode="constant", constant_values=fill)
    else:  # pragma: no cover - programming error
        raise ValueError(f"bad mode {mode!r}")
    H, W = arr.shape[0], arr.shape[1]
    y0 = pad_y[0] + dy
    x0 = pad_x[0] + dx
    return padded[y0:y0 + H, x0:x0 + W]


def valid_shift_mask(shape: tuple[int, int], dy: int, dx: int) -> np.ndarray:
    """True where ``(y + dy, x + dx)`` is inside the image."""
    H, W = shape
    m = np.zeros((H, W), dtype=bool)
    ys = slice(max(0, -dy), H - max(0, dy))
    xs = slice(max(0, -dx), W - max(0, dx))
    m[ys, xs] = True
    return m


def conv3(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """3x3 correlation with edge-replicated borders. ``img`` is (H, W) float."""
    out = np.zeros_like(img, dtype=np.float64)
    for ky in range(3):
        for kx in range(3):
            k = kernel[ky, kx]
            if k == 0.0:
                continue
            out += k * shift(img, ky - 1, kx - 1)
    return out


def luma(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma of an (H, W, 3) array, in the input's own units."""
    a = rgb.astype(np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def gradient_magnitude(gray01: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sobel gradient of a [0, 1] grayscale image, normalised by the kernel
    weight sum (8) so a value of ``0.04`` means "≈4 % intensity step per pixel".

    Returns ``(magnitude, gx, gy)``.
    """
    gx = conv3(gray01, _SOBEL_X) / 8.0
    gy = conv3(gray01, _SOBEL_Y) / 8.0
    return np.hypot(gx, gy), gx, gy


def nms_thin(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Canny-style non-maximum suppression: keep a pixel only where its gradient
    magnitude is >= both neighbours along the (4-way quantised) gradient
    direction. Without this, a soft painterly edge counts many times over and the
    boundary F-measure is dominated by edge thickness rather than edge position.
    """
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    sector = np.zeros(mag.shape, dtype=np.int8)
    sector[(ang >= 22.5) & (ang < 67.5)] = 1
    sector[(ang >= 67.5) & (ang < 112.5)] = 2
    sector[(ang >= 112.5) & (ang < 157.5)] = 3

    keep = np.zeros(mag.shape, dtype=bool)
    # (dy, dx) pairs of the two neighbours along each gradient sector
    for s, (dy, dx) in enumerate(((0, 1), (1, 1), (1, 0), (1, -1))):
        a = shift(mag, dy, dx)
        b = shift(mag, -dy, -dx)
        keep |= (sector == s) & (mag >= a) & (mag >= b)
    return np.where(keep, mag, 0.0)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a Euclidean disk of the given radius (the ``k px
    tolerance`` of the boundary metrics)."""
    if radius <= 0:
        return mask.copy()
    out = np.zeros_like(mask)
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > r2:
                continue
            out |= shift(mask, dy, dx, mode="zero")
    return out


def bilateral3_step(img01: np.ndarray, sigma_color: float) -> np.ndarray:
    """One 3x3 edge-preserving (bilateral) smoothing pass over an (H, W, 3)
    image in [0, 1]. Pure numpy, so bit-exact across runs."""
    acc = np.zeros_like(img01, dtype=np.float64)
    wsum = np.zeros(img01.shape[:2], dtype=np.float64)
    sc2 = 2.0 * float(sigma_color) ** 2
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neigh = shift(img01, dy, dx)
            spatial = np.exp(-(dy * dy + dx * dx) / 2.0)
            d2 = ((neigh - img01) ** 2).sum(axis=-1)
            w = spatial * np.exp(-d2 / sc2)
            acc += w[..., None] * neigh
            wsum += w
    return acc / wsum[..., None]
