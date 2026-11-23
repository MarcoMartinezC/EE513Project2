"""Image-quality metrics and the uncertainty that goes with them.

Every number in the report needs an uncertainty. Two sources are built in:

* ``mean_std`` over repeated measurements that differ in the thing you care
  about (frames of a burst, seeds, scenes). Prefer this.
* ``psnr_tile_bootstrap`` when you have one image pair: resample image tiles
  and report the spread of PSNR. It captures content variation across the
  frame, not noise-draw variation, so say which one you quoted.
"""

from __future__ import annotations

import numpy as np


def psnr(est: np.ndarray, ref: np.ndarray, peak: float = 1.0, margin: int = 0) -> float:
    if margin:
        est = est[margin:-margin, margin:-margin]
        ref = ref[margin:-margin, margin:-margin]
    mse = float(np.mean((np.asarray(est, np.float64) - np.asarray(ref, np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(peak**2 / mse)


def ssim(est: np.ndarray, ref: np.ndarray, data_range: float = 1.0) -> float:
    from skimage.metrics import structural_similarity

    channel_axis = -1 if est.ndim == 3 else None
    return float(structural_similarity(ref, est, data_range=data_range, channel_axis=channel_axis))


class LPIPS:
    """LPIPS (AlexNet) via torchmetrics, on display-referred sRGB in [0, 1]. Lower is better.

    Downloads its weights on first use. Construct once and reuse.
    """

    def __init__(self, device: str = "cpu"):
        import torch
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.torch = torch
        self.device = device
        self.metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    def __call__(self, est_rgb: np.ndarray, ref_rgb: np.ndarray) -> float:
        t = self.torch

        def to_t(a):
            return t.from_numpy(np.clip(a, 0, 1).astype(np.float32)).permute(2, 0, 1)[None].to(self.device)

        with t.no_grad():
            return float(self.metric(to_t(est_rgb), to_t(ref_rgb)))


def mean_std(values) -> tuple[float, float, int]:
    """Mean, sample standard deviation (ddof=1) and count."""
    v = np.asarray(list(values), dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan"), int(v.size)


def mean_sem(values) -> tuple[float, float]:
    """Mean and standard error of the mean, as a [value, uncertainty] pair."""
    m, s, n = mean_std(values)
    return m, s / np.sqrt(n)


def psnr_tile_bootstrap(est: np.ndarray, ref: np.ndarray, tile: int = 64, n_boot: int = 1000,
                        peak: float = 1.0, seed: int = 0) -> tuple[float, float]:
    """PSNR of the whole image and its bootstrap standard deviation over tiles."""
    h, w = ref.shape[:2]
    th, tw = h // tile, w // tile
    if th * tw < 4:
        raise ValueError("image too small for tile bootstrap")
    sq = (np.asarray(est, np.float64) - np.asarray(ref, np.float64)) ** 2
    sq = sq[: th * tile, : tw * tile]
    per_tile = sq.reshape(th, tile, tw, tile, -1).mean(axis=(1, 3, 4)).ravel()
    rng = np.random.default_rng(seed)
    boots = rng.choice(per_tile, size=(n_boot, per_tile.size), replace=True).mean(axis=1)
    return 10 * np.log10(peak**2 / per_tile.mean()), float(np.std(10 * np.log10(peak**2 / boots), ddof=1))


def tile_errors(est: np.ndarray, ref: np.ndarray, tile: int = 64) -> np.ndarray:
    """MSE per tile, shape (H//tile, W//tile). Use it to find the crops where a method fails."""
    h, w = ref.shape[:2]
    th, tw = h // tile, w // tile
    sq = (np.asarray(est, np.float64) - np.asarray(ref, np.float64)) ** 2
    sq = sq[: th * tile, : tw * tile]
    if sq.ndim == 2:
        sq = sq[..., None]
    return sq.reshape(th, tile, tw, tile, -1).mean(axis=(1, 3, 4))
