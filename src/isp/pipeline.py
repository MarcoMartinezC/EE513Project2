"""The classical pipeline as an ordered list of named stages.

Each stage is ``f(x, ctx, **params) -> x`` where ``x`` is either a mosaic
(H, W) or an RGB image (H, W, 3). Stages that make sense in both domains
(black level, shading, white balance, denoising, clipping) accept both, which
is what lets the ablation move a stage to the wrong side of demosaicing and
measure the cost instead of just crashing.

Default order and why:

    to_unit         DN / white, so every later stage works in [0, ~1]
    black_level     (x - b) / (1 - b); 0 is black, 1 is saturation
    defects         replace hot/stuck sites (needs raw neighbours, so first)
    denoise         on packed Bayer, where the measured noise model holds
    lsc             per-channel shading gains from the flat field
    wb              per-channel gains, before demosaicing (explain why in the report)
    clip_highlights min(x, 1) after the gains, so clipped highlights stay neutral
    demosaic        bilinear or MHC
    ccm             camera RGB -> linear sRGB
    tone            exposure, clip to [0, 1], sRGB transfer curve
    sharpen         unsharp mask on luminance
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter

from . import demosaic as dm
from .calib import correct_defects
from .denoise_classical import denoise_packed, denoise_rgb
from .groundtruth import mosaic_rggb
from .noise_model import NoiseModel
from .rawio import pack_rggb, rggb_to_rgb, tile_rggb, unpack_rggb

Step = tuple[str, dict]

DEFAULT_STEPS: list[Step] = [
    ("to_unit", {}),
    ("black_level", {}),
    ("defects", {}),
    # eps_scale is in units of the noise variance after the GAT. 4 is a starting
    # point from the synthetic camera; tune it on a burst you do not evaluate on.
    ("denoise", {"method": "guided", "r": 2, "eps_scale": 4.0, "guide": "self"}),
    ("lsc", {}),
    ("wb", {}),
    ("clip_highlights", {}),
    ("demosaic", {"method": "mhc"}),
    ("ccm", {}),
    ("tone", {"curve": "srgb", "exposure": 1.0}),
    ("sharpen", {"sigma": 1.0, "amount": 0.5}),
]

# Black level, shading, white balance, demosaic and nothing else (Part C).
MINIMAL_STEPS: list[Step] = [
    ("to_unit", {}),
    ("black_level", {}),
    ("lsc", {}),
    ("wb", {}),
    ("demosaic", {"method": "bilinear"}),
    ("clip_unit", {}),
]


@dataclass
class Context:
    """Everything the stages need besides the image."""

    white: float
    black_rggb: np.ndarray  # (4,) DN at R, G1, G2, B
    wb_gains: np.ndarray | None = None  # (3,) R, G, B
    ccm: np.ndarray | None = None  # (3, 3), rows produce sRGB R, G, B
    lsc_rgb: np.ndarray | None = None  # (H, W, 3) gains at the working resolution
    defect_mask: np.ndarray | None = None  # (H, W) bool at the working resolution
    noise: NoiseModel | None = None
    learned_denoiser: Callable[[np.ndarray], np.ndarray] | None = None  # packed -> packed

    @property
    def black_rgb(self) -> np.ndarray:
        return rggb_to_rgb(self.black_rggb)


def _mosaic(x: np.ndarray) -> bool:
    return x.ndim == 2


def _need(obj, name: str):
    if obj is None:
        raise ValueError(f"stage needs ctx.{name}, which is not set (check the config / calibration)")
    return obj


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def to_unit(x, ctx):
    return x.astype(np.float32) / np.float32(ctx.white)


def black_level(x, ctx):
    if _mosaic(x):
        b = tile_rggb(ctx.black_rggb, x.shape) / np.float32(ctx.white)
    else:
        b = (ctx.black_rgb / ctx.white).astype(np.float32)
    return (x - b) / (1.0 - b)


def defects(x, ctx):
    if ctx.defect_mask is None:
        return x
    if not _mosaic(x):
        raise ValueError("defect correction needs the mosaic; it cannot run after demosaicing")
    return correct_defects(x, ctx.defect_mask)


def denoise(x, ctx, method="guided", r=2, eps_scale=4.0, guide="self"):
    if method == "none":
        return x
    if method == "guided":
        if _mosaic(x):
            return unpack_rggb(denoise_packed(pack_rggb(x), r, eps_scale, ctx.noise, guide))
        return denoise_rgb(x, r, eps_scale)
    if method == "learned":
        if not _mosaic(x):
            raise ValueError("the learned denoiser was trained on packed Bayer")
        return unpack_rggb(_need(ctx.learned_denoiser, "learned_denoiser")(pack_rggb(x)))
    raise ValueError(f"unknown denoise method {method!r}")


def lsc(x, ctx):
    g = _need(ctx.lsc_rgb, "lsc_rgb")
    if g.shape[:2] != x.shape[:2]:
        raise ValueError(f"shading map {g.shape[:2]} does not match image {x.shape[:2]}")
    return x * (mosaic_rggb(g) if _mosaic(x) else g)


def wb(x, ctx):
    g = _need(ctx.wb_gains, "wb_gains").astype(np.float32)
    if _mosaic(x):
        return x * tile_rggb([g[0], g[1], g[1], g[2]], x.shape)
    return x * g


def clip_highlights(x, ctx, ceiling=1.0):
    return np.minimum(x, np.float32(ceiling))


def clip_unit(x, ctx):
    return np.clip(x, 0.0, 1.0)


def demosaic(x, ctx, method="mhc"):
    if not _mosaic(x):
        return x  # already RGB (e.g. the quarter-resolution reference)
    return dm.demosaic(x, method)


def ccm(x, ctx):
    if _mosaic(x):
        raise ValueError("colour correction needs RGB; demosaic first")
    m = _need(ctx.ccm, "ccm").astype(np.float32)
    return x @ m.T


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def srgb_eotf(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def tone(x, ctx, curve="srgb", exposure=1.0):
    x = x * np.float32(exposure)
    if curve == "srgb":
        return srgb_oetf(x)
    if curve == "linear":
        return np.clip(x, 0.0, 1.0)
    raise ValueError(f"unknown tone curve {curve!r}")  # add a global S-curve or a local one here


def sharpen(x, ctx, sigma=1.0, amount=0.5):
    """Unsharp mask on luminance only, so colour edges are not given halos of the wrong hue."""
    if amount == 0:
        return x
    y = x @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    detail = y - gaussian_filter(y, sigma, mode="reflect")
    return np.clip(x + amount * detail[..., None], 0.0, 1.0)


STAGES: dict[str, Callable] = {
    f.__name__: f
    for f in (to_unit, black_level, defects, denoise, lsc, wb, clip_highlights, clip_unit,
              demosaic, ccm, tone, sharpen)
}


# --------------------------------------------------------------------------- #
# Running and editing pipelines
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    image: np.ndarray
    timings: list[tuple[str, float]] = field(default_factory=list)  # seconds per stage
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)


def run(x: np.ndarray, ctx: Context, steps: list[Step] = DEFAULT_STEPS, keep: bool = False,
        stop_before: str | None = None) -> Result:
    """Run ``steps`` on a raw DN mosaic (or an RGB image, for the reference path)."""
    res = Result(image=x)
    for name, params in steps:
        if name == stop_before:
            break
        t0 = time.perf_counter()
        x = STAGES[name](x, ctx, **params)
        res.timings.append((name, time.perf_counter() - t0))
        if keep:
            res.intermediates[name] = x
    res.image = x
    return res


def names(steps: list[Step]) -> list[str]:
    return [n for n, _ in steps]


def without(steps: list[Step], name: str) -> list[Step]:
    if name not in names(steps):
        raise KeyError(name)
    return [s for s in steps if s[0] != name]


def move_after(steps: list[Step], name: str, anchor: str) -> list[Step]:
    """Move stage ``name`` to just after ``anchor``."""
    step = next(s for s in steps if s[0] == name)
    rest = without(steps, name)
    i = names(rest).index(anchor)
    return rest[: i + 1] + [step] + rest[i + 1:]


def with_params(steps: list[Step], name: str, **params) -> list[Step]:
    return [(n, {**p, **params}) if n == name else (n, p) for n, p in steps]
