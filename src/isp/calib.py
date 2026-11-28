"""Calibration products derived from Project 1 captures: defect map and lens shading gains.

* Hot pixels come from dark frames (lens cap on, same ISO and exposure as the
  scenes you care about; hot pixels grow with exposure time and temperature).
* Stuck/dead pixels come from the flat field: a site far from its same-colour
  neighbours under uniform light.
* Lens shading is estimated per Bayer channel (R, G1, G2, B) from the flat
  field, smoothed so the gain map carries vignetting and not PRNU or noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import convolve, gaussian_filter

from .groundtruth import quarter_res
from .rawio import pack_rggb, tile_rggb, unpack_rggb

_RING = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)


def normalise(mosaic_dn: np.ndarray, black_rggb, white: float) -> np.ndarray:
    b = tile_rggb(black_rggb, mosaic_dn.shape)
    return (mosaic_dn.astype(np.float32) - b) / (white - b)


# --------------------------------------------------------------------------- #
# Defective pixels
# --------------------------------------------------------------------------- #
def _per_channel_outliers(packed: np.ndarray, k: float, side: str) -> np.ndarray:
    out = np.zeros(packed.shape, dtype=bool)
    for c in range(4):
        x = packed[..., c]
        med = np.median(x)
        mad = np.median(np.abs(x - med)) * 1.4826
        mad = max(mad, 1e-12)
        if side == "high":
            out[..., c] = x > med + k * mad
        else:
            out[..., c] = np.abs(x - med) > k * mad
    return out


def hot_pixels_from_darks(dark_stack_dn: np.ndarray, k: float = 8.0) -> np.ndarray:
    """Full-resolution boolean mask of sites whose mean dark level is a k-sigma outlier (robust)."""
    mean_dark = dark_stack_dn.astype(np.float64).mean(axis=0)
    return unpack_rggb(_per_channel_outliers(pack_rggb(mean_dark), k, "high"))


def stuck_pixels_from_flat(flat_norm: np.ndarray, tolerance: float = 0.3) -> np.ndarray:
    """Sites that differ from the mean of their 8 same-colour neighbours by more than ``tolerance`` (relative)."""
    p = pack_rggb(flat_norm.astype(np.float64))
    bad = np.zeros(p.shape, dtype=bool)
    for c in range(4):
        local = convolve(p[..., c], _RING / _RING.sum(), mode="reflect")
        bad[..., c] = np.abs(p[..., c] / np.maximum(local, 1e-9) - 1.0) > tolerance
    return unpack_rggb(bad)


def correct_defects(mosaic: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked sites with the mean of their unmasked same-colour neighbours.

    Masked values are never used as inputs, so applying this twice gives the
    same result as applying it once (tested). Works in any units.

    Sparse version: visits only the masked sites (0.05 % of the sensor), not
    the whole frame. Same result as ``correct_defects_reference`` up to float
    rounding, including at the border, where out-of-range neighbours are
    reflected back into the image the same way (tested).
    """
    if mask is None or not mask.any():
        return mosaic
    h, w = mosaic.shape
    ys, xs = np.nonzero(mask)
    num = np.zeros(ys.size, dtype=np.float64)
    den = np.zeros(ys.size, dtype=np.float64)
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            if dy == 0 and dx == 0:
                continue
            # Same-colour neighbour two sites away; beyond the edge, reflect as the
            # reference's packed-plane convolution does (packed index -1 -> 0, n -> n-1).
            yy, xx = ys + dy, xs + dx
            yy = np.where(yy < 0, yy + 2, np.where(yy >= h, yy - 2, yy))
            xx = np.where(xx < 0, xx + 2, np.where(xx >= w, xx - 2, xx))
            good = ~mask[yy, xx]
            num += np.where(good, mosaic[yy, xx], 0.0)
            den += good
    out = mosaic.copy()
    keep = den > 0
    out[ys[keep], xs[keep]] = (num[keep] / den[keep]).astype(mosaic.dtype)
    return out


def correct_defects_reference(mosaic: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Whole-frame convolution version of ``correct_defects`` (kept as the tested reference)."""
    if mask is None or not mask.any():
        return mosaic
    p = pack_rggb(mosaic).astype(np.float32)
    pm = pack_rggb(mask)
    good = (~pm).astype(np.float32)
    out = p.copy()
    for c in range(4):
        num = convolve(p[..., c] * good[..., c], _RING, mode="reflect")
        den = convolve(good[..., c], _RING, mode="reflect")
        fill = np.where(den > 0, num / np.maximum(den, 1e-12), p[..., c])
        out[..., c] = np.where(pm[..., c], fill, p[..., c])
    return unpack_rggb(out).astype(mosaic.dtype)


# --------------------------------------------------------------------------- #
# Lens shading
# --------------------------------------------------------------------------- #
def poly_surface(plane: np.ndarray, degree: int = 6, step: int | None = None) -> np.ndarray:
    """Least-squares 2-D polynomial fit to a smooth plane, evaluated on the full grid.

    Unlike Gaussian smoothing, a polynomial is not biased at the image border,
    which is where shading is strongest. Fit on a subsampled grid for speed.
    """
    h, w = plane.shape
    step = step or max(1, min(h, w) // 100)
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    sub = plane[0:h:step, 0:w:step]

    def basis(y, x):
        u, v = 2 * x / (w - 1) - 1, 2 * y / (h - 1) - 1
        return np.stack([u**i * v**j for i in range(degree + 1) for j in range(degree + 1 - i)], axis=-1)

    coef, *_ = np.linalg.lstsq(basis(yy, xx).reshape(-1, (degree + 1) * (degree + 2) // 2), sub.ravel(), rcond=None)
    Y, X = np.mgrid[0:h, 0:w]
    return (basis(Y, X) @ coef).reshape(h, w)


def shading_gain_from_flat(flat_stack_dn: np.ndarray, black_rggb, white: float,
                           sigma_frac: float = 0.01, defect_mask: np.ndarray | None = None,
                           method: str = "poly", degree: int = 6) -> np.ndarray:
    """Per-channel gain map, packed (H/2, W/2, 4), equal to 1 at each channel's brightest point.

    Each channel is normalised to its own peak, so the gains remove fall-off
    but not the flat's colour; colour is white balance's job. ``method`` is
    "poly" (2-D polynomial of ``degree``) or "gauss" (Gaussian smoothing,
    ``sigma_frac`` of the image width; biased near the border). Compare the two
    on your flat and report the residual.
    """
    flat = normalise(flat_stack_dn.mean(axis=0), black_rggb, white)
    if defect_mask is not None:
        flat = correct_defects(flat, defect_mask)
    p = pack_rggb(flat).astype(np.float64)
    sigma = max(1.0, sigma_frac * p.shape[1])
    gain = np.empty_like(p)
    for c in range(4):
        if method == "poly":
            s = poly_surface(p[..., c], degree)
        else:
            s = gaussian_filter(p[..., c], sigma, mode="nearest")
        gain[..., c] = s.max() / np.maximum(s, 1e-6)
    return gain.astype(np.float32)


def lsc_rgb_full(gain_packed: np.ndarray) -> np.ndarray:
    """Packed gains -> (H, W, 3) gains at full resolution (nearest, greens averaged)."""
    g3 = np.stack([gain_packed[..., 0], 0.5 * (gain_packed[..., 1] + gain_packed[..., 2]),
                   gain_packed[..., 3]], axis=-1)
    return np.repeat(np.repeat(g3, 2, axis=0), 2, axis=1)


def lsc_rgb_quarter(gain_packed: np.ndarray) -> np.ndarray:
    """Packed gains -> (H/2, W/2, 3), matching the 2x2 ground-truth construction."""
    return quarter_res(unpack_rggb(gain_packed))


@dataclass
class Calibration:
    lsc_gain_packed: np.ndarray | None = None
    defect_mask: np.ndarray | None = None
    report: dict = field(default_factory=dict)

    def save(self, path) -> None:
        arrays = {}
        if self.lsc_gain_packed is not None:
            arrays["lsc_gain_packed"] = self.lsc_gain_packed
        if self.defect_mask is not None:
            arrays["defect_mask"] = self.defect_mask
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "Calibration":
        d = np.load(path)
        return cls(d["lsc_gain_packed"] if "lsc_gain_packed" in d else None,
                   d["defect_mask"] if "defect_mask" in d else None)
