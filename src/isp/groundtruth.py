"""The two ground-truth constructions from the Project 2 spec.

1. Denoising: align the burst and average it. The noisy input is one frame,
   never resampled, and the reference is the mean of the *other* frames
   aligned to it (leave-one-out), so the input's own noise is not inside its
   reference. With the input included, the reference carries 1/N of the
   input's noise, which correlates the two and flatters any method that passes
   noise through. Repeating this for every frame gives N scores, and their
   spread is the uncertainty.

2. Demosaicing: collapse each 2x2 block of the *averaged* mosaic to one RGB
   pixel (R, mean of the Gs, B), re-mosaic that, and score demosaicers against it.

Alignment is done on packed Bayer planes (half resolution), estimating one
shift from the mean of the four planes and applying it to every plane. Shifting
a full-resolution mosaic by a fraction of a pixel would mix colours.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .rawio import pack_rggb, unpack_rggb


# --------------------------------------------------------------------------- #
# 2x2 construction for the demosaicing reference (Exercise 5)
# --------------------------------------------------------------------------- #
def quarter_res(mosaic: np.ndarray) -> np.ndarray:
    """(H, W) RGGB mosaic -> (H/2, W/2, 3): red site, mean of the two greens, blue site."""
    p = pack_rggb(mosaic)
    return np.stack([p[..., 0], 0.5 * (p[..., 1] + p[..., 2]), p[..., 3]], axis=-1)


def mosaic_rggb(rgb: np.ndarray) -> np.ndarray:
    """Subsample an (H, W, 3) image into an RGGB mosaic (H and W must be even)."""
    h, w, _ = rgb.shape
    if h % 2 or w % 2:
        raise ValueError(f"image shape must be even to mosaic, got {rgb.shape[:2]}")
    m = np.empty((h, w), dtype=rgb.dtype)
    m[0::2, 0::2] = rgb[0::2, 0::2, 0]
    m[0::2, 1::2] = rgb[0::2, 1::2, 1]
    m[1::2, 0::2] = rgb[1::2, 0::2, 1]
    m[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return m


def even_crop(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h // 2 * 2, : w // 2 * 2]


# --------------------------------------------------------------------------- #
# Burst alignment
# --------------------------------------------------------------------------- #
def estimate_shift(ref: np.ndarray, img: np.ndarray, window: np.ndarray, upsample: int = 100) -> tuple[float, float]:
    """Sub-pixel (dx, dy) such that ``img`` is approximately ``ref`` translated by (dx, dy).

    Upsampled cross-correlation (Guizar-Sicairos et al. 2008, via scikit-image)
    on mean-subtracted, Hann-windowed images, *without* phase normalisation.
    Measured on smooth textures with known Fourier shifts of (0.6, -0.4) px:
    ``cv2.phaseCorrelate`` (what the exercise used) was off by up to ~0.25 px,
    and scikit-image's default ``normalization="phase"`` returned (0.21, -0.07),
    while plain cross-correlation returned (0.60, -0.40). Phase normalisation
    whitens the spectrum and lets the weak high frequencies decide. With
    shifts under a pixel, that bias is most of the signal.
    """
    from skimage.registration import phase_cross_correlation

    a = (ref - ref.mean()) * window
    b = (img - img.mean()) * window
    shift, _, _ = phase_cross_correlation(a, b, upsample_factor=upsample, normalization=None)
    # shift is (row, col) such that img(x) ~ ref(x + shift): img is ref translated by -shift.
    return float(-shift[1]), float(-shift[0])


def shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate an image by (dx, dy) with bicubic interpolation (reflect at borders)."""
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        np.ascontiguousarray(img, dtype=np.float32), m, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT,
    )


@dataclass
class Burst:
    """A burst of packed Bayer frames, never resampled in place, plus their shifts.

    ``shifts[i]`` is frame i's (dx, dy) relative to frame 0, in packed pixels.
    Translations compose, so frame i relative to frame k is shifts[i] - shifts[k],
    and the reference for any frame k can be built without re-running phase
    correlation. Frame k itself is never interpolated, so it is a real noisy
    capture and every frame of the burst can serve as a test input.
    """

    packed: np.ndarray  # (N, H/2, W/2, 4)
    shifts: np.ndarray  # (N, 2)
    min_shift: float = 0.02  # below this (packed px) a frame is not resampled

    @property
    def n(self) -> int:
        return self.packed.shape[0]

    @property
    def margin(self) -> int:
        """Packed pixels at the border affected by resampling; excluded from every score."""
        spread = np.abs(self.shifts[:, None, :] - self.shifts[None, :, :]).max()
        return int(np.ceil(spread)) + 2 if spread >= self.min_shift else 0

    def _aligned(self, i: int, k: int) -> np.ndarray:
        dx, dy = self.shifts[i] - self.shifts[k]
        if i == k or np.hypot(dx, dy) < self.min_shift:
            return self.packed[i]
        return np.stack([shift_image(self.packed[i, ..., c], -dx, -dy) for c in range(4)], axis=-1)

    def aligned_to(self, k: int) -> np.ndarray:
        """(N, h, w, 4): every frame aligned to frame k (frame k untouched)."""
        return np.stack([self._aligned(i, k) for i in range(self.n)])

    def reference(self, k: int, leave_out: bool = True) -> np.ndarray:
        """Mean of the burst aligned to frame k, excluding frame k unless ``leave_out`` is False."""
        acc = np.zeros(self.packed.shape[1:], dtype=np.float64)
        count = 0
        for i in range(self.n):
            if leave_out and i == k:
                continue
            acc += self._aligned(i, k)
            count += 1
        return (acc / count).astype(np.float32)

    def interior(self, img: np.ndarray, scale: int = 1) -> np.ndarray:
        """Crop the resampling margin. ``scale`` = 2 for full-resolution images."""
        m = self.margin * scale
        if m == 0:
            return img
        return img[m:-m, m:-m] if img.ndim == 2 else img[..., m:-m, m:-m, :]

    def map(self, fn) -> "Burst":
        """Same burst with ``fn`` applied to every packed frame (e.g. normalisation)."""
        return Burst(np.stack([fn(p) for p in self.packed]), self.shifts, self.min_shift)

    def defect_corrected(self, mask: np.ndarray | None) -> "Burst":
        """Correct defects in every frame *before* any averaging: resampling would smear them otherwise."""
        if mask is None:
            return self
        from .calib import correct_defects

        return self.map(lambda p: pack_rggb(correct_defects(unpack_rggb(p), mask)))


_CFA_LUMA = np.outer([1, 2, 1], [1, 2, 1]) / 16.0  # zero response at all three CFA chroma carriers


def registration_proxy(mosaic: np.ndarray) -> np.ndarray:
    """Full-resolution luminance proxy of an RGGB mosaic for estimating shifts.

    Why not the packed planes: detail finer than ~4 sensor pixels aliases in
    the half-resolution Bayer planes, and aliased patterns do not move with the
    true shift. On the synthetic scene (zone plate, fine stripes), registering
    the mean of the packed planes was off by up to 0.9 sensor px; this proxy,
    which removes the CFA modulation but keeps full-resolution sampling, was
    within 0.03 px (tests/test_end_to_end_synthetic.py).
    """
    from scipy.ndimage import convolve

    return convolve(mosaic.astype(np.float64), _CFA_LUMA, mode="mirror")


def estimate_shifts(stack: np.ndarray, crop: int | None = 2048) -> np.ndarray:
    """(N, 2) shifts of each (N, H, W) mosaic relative to frame 0, in *sensor* pixels.

    Translation is global, so a central ``crop`` (sensor px) is enough and much
    faster on 24 MP frames. Use ``crop=None`` for the whole frame.
    """
    h, w = stack.shape[1:]
    if crop and min(h, w) > crop:
        y0, x0 = (h - crop) // 2, (w - crop) // 2
        stack = stack[:, y0:y0 + crop, x0:x0 + crop]
        h = w = crop
    window = cv2.createHanningWindow((w, h), cv2.CV_64F)
    ref = registration_proxy(stack[0])
    shifts = np.zeros((stack.shape[0], 2))
    for i in range(1, stack.shape[0]):
        shifts[i] = estimate_shift(ref, registration_proxy(stack[i]), window)
    return shifts


def make_burst(stack: np.ndarray, align: bool = True, min_shift: float = 0.02,
               defect_mask: np.ndarray | None = None) -> Burst:
    """(N, H, W) RGGB mosaics -> Burst. ``align=False`` assumes the frames are already registered.

    Pass ``defect_mask``: hot and stuck pixels are identical in every frame and
    pull the correlation peak towards zero shift. Shifts are estimated on
    corrected copies (see ``registration_proxy``); the stored frames are untouched.
    """
    packed = np.stack([pack_rggb(m) for m in stack]).astype(np.float32)
    if not align:
        return Burst(packed, np.zeros((packed.shape[0], 2)), min_shift)
    if defect_mask is not None:
        from .calib import correct_defects

        stack = np.stack([correct_defects(m.astype(np.float32), defect_mask) for m in stack])
    return Burst(packed, estimate_shifts(stack) / 2.0, min_shift)  # sensor px -> packed px


# --------------------------------------------------------------------------- #
# How clean is the clean reference?
# --------------------------------------------------------------------------- #
def robust_sigma(x: np.ndarray) -> float:
    """MAD / 0.6745: matches the standard deviation for Gaussian data, ignores outliers."""
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.median(np.abs(x - np.median(x))) / 0.6745)


def residual_noise_report(burst: Burst, ks=None) -> dict:
    """Noise in one frame, and the noise left in the leave-one-out reference.

    Per channel, from frame k (never resampled) minus the mean of the other
    frames aligned to it: var(x_k - ref_k) = sigma^2 (1 + 1/(N-1)), so sigma is
    recovered by dividing that factor out. Two estimates are given, the plain
    standard deviation and the robust MAD/0.6745. On a well-aligned static
    scene they agree; if the standard deviation is much larger, edges are
    leaving structure in the residual (misalignment, motion, rolling shutter),
    which is the disagreement the training-without-clean-images exercise found.

    The temporal std across aligned frames is *not* used: resampled frames are
    quieter than real ones (interpolation averages noise), so it reads low.
    The noise is signal dependent, so these are summaries; the fitted
    Poisson-Gaussian curve is the full story.
    """
    ks = [0] if ks is None else list(ks)
    n_ref = burst.n - 1
    factor = np.sqrt(1.0 + 1.0 / n_ref)
    res = np.concatenate([burst.interior(burst.packed[k] - burst.reference(k))[None] for k in ks])
    out = {"n_frames": int(burst.n), "n_in_reference": int(n_ref), "frames_used": ks, "channels": {}}
    for c, name in enumerate(("R", "G1", "G2", "B")):
        r = res[..., c]
        s_std = float(r.std() / factor)
        s_mad = robust_sigma(r) / factor
        out["channels"][name] = {"sigma_single_std": s_std, "sigma_single_mad": s_mad,
                                 "sigma_reference": s_mad / np.sqrt(n_ref)}
    s = float(np.mean([v["sigma_single_mad"] for v in out["channels"].values()]))
    out["sigma_single_mean"] = s
    out["sigma_reference_mean"] = s / np.sqrt(n_ref)
    out["reference_mse_floor"] = (s / np.sqrt(n_ref)) ** 2  # added to any MSE measured against the reference
    return out


@dataclass
class SavedBurst:
    """What ``scripts/build_reference.py`` writes, loaded back."""

    burst: Burst  # packed raw DN, defects NOT corrected (the pipeline does that)
    black_rggb: np.ndarray
    white: float
    crop_origin: tuple[int, int]
    iso: int | None

    def normalised(self) -> Burst:
        """Same burst with black subtracted and divided by the span, per Bayer site."""
        b = np.asarray(self.black_rggb, dtype=np.float32)
        return self.burst.map(lambda p: (p - b) / (self.white - b))


def load_saved_burst(path) -> SavedBurst:
    d = np.load(path)
    iso = int(d["iso"])
    burst = Burst(d["packed_dn"].astype(np.float32), d["shifts"], float(d["min_shift"]))
    return SavedBurst(burst, d["black_rggb"], float(d["white"]), tuple(int(v) for v in d["crop_origin"]),
                      None if iso < 0 else iso)
