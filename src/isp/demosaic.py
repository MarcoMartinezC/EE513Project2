"""Demosaicing: bilinear and Malvar-He-Cutler (2004), both from scratch.

Ported from Exercise 5, where both matched ``make_figures.bilinear_demosaic``
and ``make_figures.mhc_demosaic`` to 5.6e-17 on the full frame.

The dtype of the input is preserved (float32 in the pipeline, float64 when
checking against the course reference).

Border handling: the default is ``mode="mirror"`` (whole-sample symmetric,
d c b | a b c d), which keeps the 2-pixel Bayer period across the edge, so the
pixel beyond a red column is green, as it would be on the sensor. Exercise 5
and ``make_figures.py`` use ``mode="reflect"`` (half-sample, d c b a | a b c d),
which puts a red sample where a green one belongs. Bilinear is immune because
it normalises by the reflected mask, but MHC's cross-channel corrections then
mix the wrong colours in the outer two pixels: a flat colour comes back wrong
by up to 0.34 at the frame edge (tests/test_demosaic.py). Pass
``mode="reflect"`` to reproduce the course reference exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

# --- Bilinear kernels (normalised by the measured-site mask, so borders are exact) ---
_K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 4.0
_K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float64) / 4.0

# --- The four MHC 5x5 filters (eight cases, four patterns), scaled by 1/8 ---
F_G_AT_RB = np.array(
    [[0, 0, -1, 0, 0],
     [0, 0, 2, 0, 0],
     [-1, 2, 4, 2, -1],
     [0, 0, 2, 0, 0],
     [0, 0, -1, 0, 0]], dtype=np.float64) / 8.0

F_RB_AT_G_ROW = np.array(
    [[0, 0, 0.5, 0, 0],
     [0, -1, 0, -1, 0],
     [-1, 4, 5, 4, -1],
     [0, -1, 0, -1, 0],
     [0, 0, 0.5, 0, 0]], dtype=np.float64) / 8.0

F_RB_AT_G_COL = F_RB_AT_G_ROW.T

F_RB_AT_BR = np.array(
    [[0, 0, -1.5, 0, 0],
     [0, 2, 0, 2, 0],
     [-1.5, 0, 6, 0, -1.5],
     [0, 2, 0, 2, 0],
     [0, 0, -1.5, 0, 0]], dtype=np.float64) / 8.0

MHC_FILTERS = {
    "G at R/B": F_G_AT_RB,
    "R/B at G, same-row neighbours": F_RB_AT_G_ROW,
    "R/B at G, same-column neighbours": F_RB_AT_G_COL,
    "R at B / B at R": F_RB_AT_BR,
}


def _masks(h: int, w: int, dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m_r = np.zeros((h, w), dtype)
    m_g = np.zeros((h, w), dtype)
    m_b = np.zeros((h, w), dtype)
    m_r[0::2, 0::2] = 1
    m_g[0::2, 1::2] = 1
    m_g[1::2, 0::2] = 1
    m_b[1::2, 1::2] = 1
    return m_r, m_g, m_b


def _restore_measured(out: np.ndarray, bayer: np.ndarray) -> None:
    out[0::2, 0::2, 0] = bayer[0::2, 0::2]
    out[0::2, 1::2, 1] = bayer[0::2, 1::2]
    out[1::2, 0::2, 1] = bayer[1::2, 0::2]
    out[1::2, 1::2, 2] = bayer[1::2, 1::2]


def bilinear(bayer: np.ndarray, mode: str = "mirror") -> np.ndarray:
    """Bilinear demosaic of an RGGB mosaic. Measured sites are returned unchanged."""
    bayer = np.asarray(bayer)
    dtype = bayer.dtype if np.issubdtype(bayer.dtype, np.floating) else np.float32
    bayer = bayer.astype(dtype, copy=False)
    h, w = bayer.shape
    m_r, m_g, m_b = _masks(h, w, dtype)
    k_rb = _K_RB.astype(dtype)
    k_g = _K_G.astype(dtype)

    out = np.empty((h, w, 3), dtype)
    for ch, (mask, k) in enumerate(((m_r, k_rb), (m_g, k_g), (m_b, k_rb))):
        num = convolve(bayer * mask, k, mode=mode)
        den = convolve(mask, k, mode=mode)
        out[..., ch] = num / den
    _restore_measured(out, bayer)
    return out


def mhc(bayer: np.ndarray, clip: bool = False, mode: str = "mirror") -> np.ndarray:
    """Malvar-He-Cutler gradient-corrected demosaic, as four fixed linear filters.

    ``clip=True`` clips to [0, 1] as the Exercise 5 version did (needed to match
    ``make_figures.mhc_demosaic``). The pipeline leaves it False so that noise
    below zero is not rectified before colour correction.
    """
    bayer = np.asarray(bayer)
    dtype = bayer.dtype if np.issubdtype(bayer.dtype, np.floating) else np.float32
    bayer = bayer.astype(dtype, copy=False)
    out = bilinear(bayer, mode)

    g_est = convolve(bayer, F_G_AT_RB.astype(dtype), mode=mode)
    rb_row = convolve(bayer, F_RB_AT_G_ROW.astype(dtype), mode=mode)
    rb_col = convolve(bayer, F_RB_AT_G_COL.astype(dtype), mode=mode)
    rb_diag = convolve(bayer, F_RB_AT_BR.astype(dtype), mode=mode)

    out[0::2, 0::2, 1] = g_est[0::2, 0::2]  # G at R
    out[1::2, 1::2, 1] = g_est[1::2, 1::2]  # G at B
    out[0::2, 1::2, 0] = rb_row[0::2, 1::2]  # R at G in R row
    out[1::2, 0::2, 0] = rb_col[1::2, 0::2]  # R at G in B row
    out[0::2, 1::2, 2] = rb_col[0::2, 1::2]  # B at G in R row
    out[1::2, 0::2, 2] = rb_row[1::2, 0::2]  # B at G in B row
    out[1::2, 1::2, 0] = rb_diag[1::2, 1::2]  # R at B
    out[0::2, 0::2, 2] = rb_diag[0::2, 0::2]  # B at R

    if clip:
        np.clip(out, 0.0, 1.0, out=out)
    return out


METHODS = {"bilinear": bilinear, "mhc": mhc}


def demosaic(bayer: np.ndarray, method: str = "mhc") -> np.ndarray:
    try:
        return METHODS[method](bayer)
    except KeyError:
        raise ValueError(f"unknown demosaic method {method!r}; choose from {list(METHODS)}") from None


def noise_gain(kernel: np.ndarray) -> float:
    """Sum of squared weights: the factor by which a filter scales white-noise variance."""
    return float(np.sum(np.asarray(kernel) ** 2))
