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


# --------------------------------------------------------------------------- #
# Fast versions (Part C). Same results with mode="mirror", computed only where needed.
#
# The versions above convolve the whole mosaic with every filter and then keep
# a quarter or half of each result; mhc() also runs a full bilinear demosaic
# first and then overwrites every interpolated value. The versions below
# evaluate each filter only on the lattice of sites that uses it, as a sum of
# shifted slices of a mirror-padded mosaic, and skip the discarded work.
# Mirror padding (numpy "reflect" = scipy "mirror") keeps the Bayer phase, so
# bilinear's mask normalisation is 1 everywhere and plain averages are exact.
# Agreement with the reference versions is tested to float rounding
# (tests/test_fast_paths.py).
# --------------------------------------------------------------------------- #
_LATTICES = {"R": (0, 0), "Gr": (0, 1), "Gb": (1, 0), "B": (1, 1)}


def _lattice(padded: np.ndarray, kernel: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> np.ndarray:
    """Correlate ``kernel`` with the mosaic, evaluated only at sites (r0 + 2i, c0 + 2j)."""
    k = kernel.shape[0] // 2
    out = np.zeros((h // 2, w // 2), dtype=padded.dtype)
    for dy, dx in zip(*np.nonzero(kernel)):
        y0 = pad + r0 + dy - k
        x0 = pad + c0 + dx - k
        out += padded.dtype.type(kernel[dy, dx]) * padded[y0:y0 + h:2, x0:x0 + w:2]
    return out


def _start(bayer: np.ndarray):
    bayer = np.asarray(bayer)
    dtype = bayer.dtype if np.issubdtype(bayer.dtype, np.floating) else np.float32
    bayer = bayer.astype(dtype, copy=False)
    h, w = bayer.shape
    padded = np.pad(bayer, 2, mode="reflect")  # numpy "reflect" == scipy "mirror"
    out = np.empty((h, w, 3), dtype)
    _restore_measured(out, bayer)
    return bayer, padded, out, h, w


_AVG_H = np.array([[0, 0, 0], [0.5, 0, 0.5], [0, 0, 0]])
_AVG_V = _AVG_H.T
_AVG_X = np.array([[0.25, 0, 0.25], [0, 0, 0], [0.25, 0, 0.25]])
_AVG_PLUS = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])


def bilinear_fast(bayer: np.ndarray) -> np.ndarray:
    bayer, padded, out, h, w = _start(bayer)
    L = lambda k, site: _lattice(padded, k, *_LATTICES[site], h, w, 2)  # noqa: E731
    out[0::2, 0::2, 1] = L(_AVG_PLUS, "R")   # G at R
    out[1::2, 1::2, 1] = L(_AVG_PLUS, "B")   # G at B
    out[0::2, 1::2, 0] = L(_AVG_H, "Gr")     # R at G, red row
    out[1::2, 0::2, 0] = L(_AVG_V, "Gb")     # R at G, blue row
    out[1::2, 1::2, 0] = L(_AVG_X, "B")      # R at B
    out[0::2, 1::2, 2] = L(_AVG_V, "Gr")     # B at G, red row
    out[1::2, 0::2, 2] = L(_AVG_H, "Gb")     # B at G, blue row
    out[0::2, 0::2, 2] = L(_AVG_X, "R")      # B at R
    return out


def mhc_fast(bayer: np.ndarray, clip: bool = False) -> np.ndarray:
    bayer, padded, out, h, w = _start(bayer)
    L = lambda k, site: _lattice(padded, k, *_LATTICES[site], h, w, 2)  # noqa: E731
    out[0::2, 0::2, 1] = L(F_G_AT_RB, "R")
    out[1::2, 1::2, 1] = L(F_G_AT_RB, "B")
    out[0::2, 1::2, 0] = L(F_RB_AT_G_ROW, "Gr")
    out[1::2, 0::2, 0] = L(F_RB_AT_G_COL, "Gb")
    out[0::2, 1::2, 2] = L(F_RB_AT_G_COL, "Gr")
    out[1::2, 0::2, 2] = L(F_RB_AT_G_ROW, "Gb")
    out[1::2, 1::2, 0] = L(F_RB_AT_BR, "B")
    out[0::2, 0::2, 2] = L(F_RB_AT_BR, "R")
    if clip:
        np.clip(out, 0.0, 1.0, out=out)
    return out


METHODS = {"bilinear": bilinear_fast, "mhc": mhc_fast,
           "bilinear_reference": bilinear, "mhc_reference": mhc}


def demosaic(bayer: np.ndarray, method: str = "mhc") -> np.ndarray:
    """Pipeline entry point: the fast versions; "*_reference" for the convolution versions."""
    try:
        return METHODS[method](bayer)
    except KeyError:
        raise ValueError(f"unknown demosaic method {method!r}; choose from {list(METHODS)}") from None


def noise_gain(kernel: np.ndarray) -> float:
    """Sum of squared weights: the factor by which a filter scales white-noise variance."""
    return float(np.sum(np.asarray(kernel) ** 2))
