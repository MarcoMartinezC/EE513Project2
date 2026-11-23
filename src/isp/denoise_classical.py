"""Classical edge-aware denoising: the guided filter (He, Sun & Tang 2013), from scratch.

Applied on packed Bayer by default (before shading and white balance), where
the Poisson-Gaussian model from Project 1 still describes the noise exactly
and neighbouring samples are still independent. After demosaicing, the noise
is spatially correlated and no longer follows the model.

With a ``NoiseModel`` the image is first passed through the generalised
Anscombe transform, so noise has unit variance everywhere and one ``eps``
works across the whole tonal range. Without one, sigma is estimated blindly.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from .noise_model import NoiseModel


def box(x: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1) x (2r+1) window, reflecting at the border."""
    return uniform_filter(x, size=2 * r + 1, mode="reflect")


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Single-channel guided filter. ``eps`` is in the units of ``guide`` squared."""
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    mean_i = box(guide, r)
    mean_p = box(src, r)
    cov_ip = box(guide * src, r) - mean_i * mean_p
    var_i = box(guide * guide, r) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a, r) * guide + box(b, r)


def estimate_sigma(x: np.ndarray) -> float:
    """Blind noise estimate from the MAD of diagonal differences (one plane)."""
    d = (x[1:, 1:] - x[:-1, :-1]) / np.sqrt(2)
    return float(np.median(np.abs(d - np.median(d))) / 0.6745)


def denoise_packed(packed: np.ndarray, r: int = 2, eps_scale: float = 1.0,
                   noise: NoiseModel | None = None, guide: str = "self") -> np.ndarray:
    """Guided-filter each Bayer plane.

    ``guide="self"``: each plane guides itself (an edge-preserving smoother;
    tends to identity as eps -> 0).
    ``guide="mean"``: the mean of the four planes guides all of them. Half the
    noise in the guide and edges kept in register across planes, but the four
    planes sit at different positions inside each 2x2 block, so near Nyquist
    no local linear map takes the mean onto one plane and fine detail is lost
    even at eps -> 0. Which wins is an empirical question for your camera.
    """
    if noise is not None:
        y = noise.gat(packed)
        eps = eps_scale * 1.0  # unit noise variance after the GAT
    else:
        y = packed
        eps = eps_scale * float(np.mean([estimate_sigma(packed[..., c]) for c in range(4)]) ** 2)
    shared = y.mean(axis=-1)
    out = np.stack([guided_filter(shared if guide == "mean" else y[..., c], y[..., c], r, eps)
                    for c in range(4)], axis=-1)
    if noise is not None:
        out = noise.gat_inverse(out)
    return out.astype(np.float32)


def denoise_rgb(rgb: np.ndarray, r: int = 2, eps_scale: float = 1.0) -> np.ndarray:
    """Guided filter on a demosaiced image (used when the ablation moves denoising after demosaic)."""
    guide = rgb.mean(axis=-1)
    eps = eps_scale * estimate_sigma(guide) ** 2 * 3  # guide noise var is ~1/3 of a channel's
    return np.stack([guided_filter(guide, rgb[..., c], r, eps) for c in range(3)], axis=-1)
