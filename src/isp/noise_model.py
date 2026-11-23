"""Poisson-Gaussian sensor noise: synthesis, variance stabilisation, and fitting.

Model in raw DN (after black subtraction), with conversion gain K in e-/DN and
read noise s_r in DN:

    var(DN) = mean(DN) / K + s_r**2

In normalised units x = DN / (white - black) this is var(x) = alpha * x + sigma**2
with alpha = 1 / (K * span) and sigma = s_r / span.

K and s_r depend on ISO. Project 1 measured them; ``fit_from_burst`` measures
them again from each burst, which is a cross-check worth reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseModel:
    gain_e_per_dn: float  # K
    read_noise_dn: float  # s_r
    black: float  # mean black level, DN
    white: float  # saturation, DN

    @property
    def span(self) -> float:
        return self.white - self.black

    @property
    def alpha(self) -> float:
        """Poisson scale in normalised units."""
        return 1.0 / (self.gain_e_per_dn * self.span)

    @property
    def sigma(self) -> float:
        """Read noise in normalised units."""
        return self.read_noise_dn / self.span

    # --- variance and synthesis ------------------------------------------ #
    def variance(self, x):
        """Predicted noise variance at normalised signal level x (numpy or torch)."""
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return self.alpha * x.clamp(min=0) + self.sigma**2
        except ImportError:  # pragma: no cover
            pass
        return self.alpha * np.clip(x, 0, None) + self.sigma**2

    def std(self, x):
        return self.variance(x) ** 0.5

    def sample(self, clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw a noisy observation of a clean normalised image.

        Clipped the way a real sensor clips: at saturation (1.0) and at 0 DN
        absolute, which is ``-black / span`` in normalised units.
        """
        electrons = np.clip(clean, 0, None) / self.alpha
        shot = rng.poisson(electrons).astype(np.float32) * self.alpha
        read = rng.normal(0.0, self.sigma, size=clean.shape).astype(np.float32)
        return np.clip(shot + read, -self.black / self.span, 1.0).astype(np.float32)

    # --- generalised Anscombe transform (Makitalo & Foi 2013) --------------- #
    def gat(self, x: np.ndarray) -> np.ndarray:
        """Variance-stabilising transform: output noise has unit variance."""
        a, s2 = self.alpha, self.sigma**2
        return (2.0 / a) * np.sqrt(np.maximum(a * x + 0.375 * a * a + s2, 0.0))

    def gat_inverse(self, y: np.ndarray) -> np.ndarray:
        """Algebraic inverse. Slightly biased at very low counts; see Makitalo & Foi for the exact unbiased one."""
        a, s2 = self.alpha, self.sigma**2
        return ((a * y / 2.0) ** 2 - 0.375 * a * a - s2) / a

    # --- construction ------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: dict, iso: int, black: float, white: float) -> "NoiseModel":
        """Look up K and s_r for ``iso`` in ``cfg['noise']['iso']``.

        If that ISO is missing but another is present, K is scaled as 1/ISO
        from the nearest measured ISO and read noise is copied, with a warning.
        Read noise in DN does *not* stay constant across ISO on most sensors,
        so measure every ISO you use (``fit_from_burst`` does it from the burst).
        """
        from .config import value

        table = {int(k): v for k, v in (cfg.get("noise", {}).get("iso") or {}).items()}
        if iso in table:
            e = table[iso]
            return cls(value(e["gain_e_per_dn"], f"noise.iso.{iso}.gain_e_per_dn"),
                       value(e["read_noise_dn"], f"noise.iso.{iso}.read_noise_dn"), black, white)
        usable = {k: v for k, v in table.items() if v["gain_e_per_dn"][0] is not None}
        if not usable:
            raise KeyError("no noise model in config: fill noise.iso.<ISO> from Project 1")
        near = min(usable, key=lambda k: abs(np.log(k / iso)))
        import warnings

        warnings.warn(f"no noise model for ISO {iso}; scaling K from ISO {near}. Measure it instead.")
        e = usable[near]
        return cls(e["gain_e_per_dn"][0] * near / iso, e["read_noise_dn"][0], black, white)


def residual_samples(burst, ks, gradient_percentile: float = 50.0):
    """(signal, squared residual) samples from unresampled frames minus their leave-one-out references.

    Why not the temporal variance across aligned frames: aligning a frame by a
    fraction of a pixel interpolates it, and interpolation averages neighbouring
    noise, so resampled frames are quieter than real captures and the variance
    comes out low (by ~20% for bicubic at typical shifts). Frame k is never
    resampled here. Its reference holds 1/(N-1) of the noise variance (a bit
    less, since the reference frames were interpolated), which
    ``fit_from_burst`` divides out. Only the flattest pixels are kept, because
    misalignment at edges adds variance that is not noise.
    """
    mus, r2s = [], []
    for k in ks:
        ref = burst.interior(burst.reference(k))
        x = burst.interior(burst.packed[k])
        lum = ref.mean(axis=-1)
        gy, gx = np.gradient(lum)
        grad = np.hypot(gx, gy)
        keep = grad <= np.percentile(grad, gradient_percentile)
        mus.append(ref[keep].ravel())
        r2s.append(((x - ref)[keep] ** 2).ravel())
    return np.concatenate(mus), np.concatenate(r2s)


def fit_from_burst(burst, black: float, white: float, ks=None, n_bins: int = 40,
                   max_level: float = 0.9) -> tuple[NoiseModel, dict]:
    """Fit var = alpha * mean + sigma^2 from a normalised ``groundtruth.Burst``.

    Returns the model and a dict with [value, uncertainty] pairs from the
    weighted least-squares covariance (statistical only; the 1/(N-1)
    reference correction is uncertain by up to about 1/(N-1) * 30%, which you
    should state alongside).
    """
    ks = list(range(burst.n)) if ks is None else list(ks)
    mu, r2 = residual_samples(burst, ks)
    ref_factor = 1.0 + 1.0 / (burst.n - 1)
    sel = (mu > 0) & (mu < max_level)
    mu, var = mu[sel], r2[sel] / ref_factor
    edges = np.quantile(mu, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(mu, edges) - 1, 0, n_bins - 1)
    bm, bv, bse = [], [], []
    for b in range(n_bins):
        s = idx == b
        if s.sum() < 50:
            continue
        bm.append(mu[s].mean())
        bv.append(var[s].mean())
        bse.append(var[s].std(ddof=1) / np.sqrt(s.sum()))
    bm, bv, bse = map(np.asarray, (bm, bv, bse))
    (a, c), cov = np.polyfit(bm, bv, 1, w=1.0 / np.maximum(bse, 1e-30), cov="unscaled")
    span = white - black
    a_err, c_err = np.sqrt(np.diag(cov))
    gain = 1.0 / (a * span)
    gain_err = gain * a_err / a
    sigma = np.sqrt(max(c, 0.0))
    read_dn = sigma * span
    read_err = (0.5 * c_err / sigma * span) if sigma > 0 else float("nan")
    model = NoiseModel(gain, read_dn, black, white)
    return model, {
        "gain_e_per_dn": [float(gain), float(gain_err)],
        "read_noise_dn": [float(read_dn), float(read_err)],
        "frames_used": ks,
        "bins": {"mean": bm.tolist(), "var": bv.tolist(), "se": bse.tolist()},
    }
