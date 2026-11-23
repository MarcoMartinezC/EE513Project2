"""Build a pipeline ``Context`` from the config, the calibration products, and the frame geometry."""

from __future__ import annotations

import warnings

import numpy as np

from . import config as C
from .calib import Calibration, lsc_rgb_full, lsc_rgb_quarter
from .groundtruth import mosaic_rggb, quarter_res
from .noise_model import NoiseModel
from .pipeline import Context


def build_context(cfg: dict, calib: Calibration | None, black_rggb, white: float, shape: tuple[int, int],
                  resolution: str = "full", iso: int | None = None, require: tuple[str, ...] = ("wb", "ccm"),
                  crop_origin: tuple[int, int] = (0, 0)) -> Context:
    """``resolution='quarter'`` gives the context for the 2x2 ground-truth / re-mosaic path.

    ``shape`` is the full-resolution mosaic shape being processed and
    ``crop_origin`` its offset into the calibration maps (non-zero for runs on
    a crop). Missing calibration raises for anything listed in ``require``; everything
    else is left as None with a warning, and the stage that needs it will
    raise if it runs.
    """
    black = np.asarray(black_rggb, dtype=np.float64)

    def get(fn, label):
        try:
            return fn(cfg)
        except C.MissingCalibration as e:
            if label in require:
                raise
            warnings.warn(str(e))
            return None

    wb = get(C.wb_gains, "wb")
    ccm = get(C.ccm, "ccm")

    noise = None
    if iso is not None:
        try:
            noise = NoiseModel.from_config(cfg, iso, float(np.mean(black)), white)
        except (KeyError, C.MissingCalibration) as e:
            warnings.warn(f"no noise model ({e}); the guided filter will estimate sigma blindly")

    lsc = mask = None
    if calib is not None and calib.lsc_gain_packed is not None:
        lsc = lsc_rgb_full(calib.lsc_gain_packed) if resolution == "full" else lsc_rgb_quarter(calib.lsc_gain_packed)
    if calib is not None and calib.defect_mask is not None:
        mask = calib.defect_mask
        if resolution == "quarter":
            mask = mosaic_rggb(quarter_res(mask.astype(np.float32))) > 0

    if resolution == "quarter":
        bg = 0.5 * (black[1] + black[2])
        black = np.array([black[0], bg, bg, black[3]])

    # Crop maps to the region being processed (e.g. --crop during development).
    h, w = shape
    y0, x0 = crop_origin
    if resolution == "quarter":
        h, w, y0, x0 = h // 2, w // 2, y0 // 2, x0 // 2
    if lsc is not None:
        lsc = _crop(lsc, y0, x0, h, w)
    if mask is not None:
        mask = _crop(mask, y0, x0, h, w)

    return Context(white=white, black_rggb=black, wb_gains=wb, ccm=ccm, lsc_rgb=lsc,
                   defect_mask=mask, noise=noise)


def _crop(a: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    if a.shape[0] < y0 + h or a.shape[1] < x0 + w:
        raise ValueError(f"calibration map {a.shape[:2]} does not cover the image region "
                         f"origin {(y0, x0)} size {(h, w)}; was it made from the same camera mode?")
    return a[y0:y0 + h, x0:x0 + w]


def center_crop_mosaic(m: np.ndarray, size: int | None) -> tuple[np.ndarray, tuple[int, int]]:
    """Centre crop of a mosaic (or a stack of them) for fast development runs.

    Offsets are multiples of 4, so the CFA phase is preserved at full and at
    quarter resolution. Returns (cropped, (y0, x0)).
    """
    H, W = m.shape[-2:]
    if not size:
        return m, (0, 0)
    s = min(size, H, W) // 4 * 4
    y0 = ((H - s) // 2) // 4 * 4
    x0 = ((W - s) // 2) // 4 * 4
    return m[..., y0:y0 + s, x0:x0 + s], (y0, x0)
