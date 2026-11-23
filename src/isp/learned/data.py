"""Synthetic training pairs: clean packed Bayer + noise from your measured Poisson-Gaussian model.

Where the clean images come from is your decision and goes in the report. The
default helper (``clean_from_raw``) takes base-ISO raw frames and bins each
Bayer plane 2x2, which halves the noise and keeps a valid RGGB layout. Never
include the scene you evaluate on (the held-out bursts), or the evaluation
measures memorisation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..noise_model import NoiseModel
from ..rawio import pack_rggb, tile_rggb


def bin2(packed: np.ndarray) -> np.ndarray:
    """Average 2x2 blocks of each plane. Noise std halves; layout stays packed RGGB."""
    h, w, c = packed.shape
    p = packed[: h // 2 * 2, : w // 2 * 2]
    return p.reshape(h // 2, 2, w // 2, 2, c).mean(axis=(1, 3))


def clean_from_raw(mosaic_dn: np.ndarray, black_rggb, white: float, bins: int = 1,
                   defect_mask: np.ndarray | None = None) -> np.ndarray:
    """Raw DN mosaic -> normalised packed Bayer, defect-corrected and binned ``bins`` times."""
    from ..calib import correct_defects

    b = tile_rggb(black_rggb, mosaic_dn.shape)
    x = (mosaic_dn.astype(np.float32) - b) / (white - b)
    if defect_mask is not None:
        x = correct_defects(x, defect_mask)
    p = pack_rggb(x)
    for _ in range(bins):
        p = bin2(p)
    return np.clip(p, 0.0, 1.0).astype(np.float32)


def augment(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flips and transposes of each packed plane.

    Strictly, flipping packed planes moves each colour by one full-resolution
    pixel relative to the others (R and G1 swap sides within the block). For
    denoising, where the planes' noise is independent, that is harmless; if
    you train a joint demosaic-denoise model, use Bayer-preserving augmentation
    instead (Liu et al., CVPR 2019 workshops).
    """
    if rng.random() < 0.5:
        p = p[:, ::-1]
    if rng.random() < 0.5:
        p = p[::-1, :]
    if rng.random() < 0.5:
        p = p.transpose(1, 0, 2)
    return np.ascontiguousarray(p)


class SyntheticPairs(Dataset):
    """Random patches with fresh noise every time.

    ``noise_models`` is a list (one per ISO you train at); each sample picks
    one. Deterministic given ``seed`` and index, so "how many pairs" has a
    definite answer: ``len(dataset)`` per epoch.
    """

    def __init__(self, clean: list[np.ndarray], noise_models: list[NoiseModel], patch: int = 64,
                 length: int = 20000, seed: int = 0, gain_jitter: float = 0.0):
        if not clean:
            raise ValueError("no clean images")
        self.clean = [c for c in clean if min(c.shape[:2]) >= patch]
        if not self.clean:
            raise ValueError(f"all clean images are smaller than the patch size {patch}")
        self.noise_models = noise_models
        self.patch = patch
        self.length = length
        self.seed = seed
        self.gain_jitter = gain_jitter  # e.g. 0.2 = +-20% log-uniform jitter on K, for robustness

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, i))
        img = self.clean[rng.integers(len(self.clean))]
        h, w, _ = img.shape
        y = rng.integers(0, h - self.patch + 1)
        x = rng.integers(0, w - self.patch + 1)
        clean = augment(img[y:y + self.patch, x:x + self.patch], rng)
        nm = self.noise_models[rng.integers(len(self.noise_models))]
        if self.gain_jitter:
            k = nm.gain_e_per_dn * np.exp(rng.uniform(-self.gain_jitter, self.gain_jitter))
            nm = NoiseModel(k, nm.read_noise_dn, nm.black, nm.white)
        noisy = nm.sample(clean, rng)
        sigma = nm.std(noisy).astype(np.float32)  # from the noisy input: that is all inference has

        def t(a):
            return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))

        return t(noisy), t(sigma), t(clean)


def load_pool(path) -> list[np.ndarray]:
    """Images saved by scripts/build_clean_pool.py."""
    d = np.load(path)
    return [d[k] for k in sorted((k for k in d.files if k.startswith("arr_")), key=lambda s: int(s[4:]))]
