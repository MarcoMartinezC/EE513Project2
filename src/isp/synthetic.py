"""A simulated camera with known constants, for testing the pipeline before (and alongside) real captures.

Everything the pipeline estimates has a true value here: black and white
level, per-channel vignetting, white balance gains, CCM, conversion gain and
read noise per ISO, hot and stuck pixels, and the burst's sub-pixel shifts.
Running the pipeline on it and recovering those values is a verification you
can describe in the decisions-and-verification statement. It does not replace
the real captures: synthetic noise is exactly Poisson-Gaussian and real noise
is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .groundtruth import mosaic_rggb


def _smoothstep_edge(d: np.ndarray, width: float = 0.75) -> np.ndarray:
    """Anti-aliased step: 0 for d << 0, 1 for d >> 0, so sub-pixel shifts change the image smoothly."""
    return np.clip(0.5 + d / (2 * width), 0.0, 1.0)


def scene_linear_srgb(h: int, w: int, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """A procedural test scene in linear sRGB, evaluated at positions shifted by (dx, dy) pixels.

    Contains a zone plate (aliasing), slanted edges (zippering), saturated
    colour disks (false colour), fine stripes, a smooth gradient (noise is easy
    to see) and a specular highlight above 1.0 (clipping).
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx += dx
    yy += dy
    u, v = xx / w, yy / h  # noqa: F841 (v used below)
    img = np.empty((h, w, 3))
    base = 0.08 + 0.25 * u + 0.05 * np.sin(6 * v)
    img[:] = base[..., None] * np.array([1.0, 0.95, 0.85])

    # Zone plate, left third.
    cx, cy, r0 = 0.17 * w, 0.5 * h, 0.15 * min(h, w)
    rr = (xx - cx) ** 2 + (yy - cy) ** 2
    zp = 0.5 + 0.45 * np.cos(np.pi * rr / (2.2 * r0))
    inside = _smoothstep_edge(r0 - np.sqrt(rr))
    img = img * (1 - inside[..., None]) + (0.05 + 0.6 * zp)[..., None] * inside[..., None]

    def box(x0, x1, y0, y1):
        """Anti-aliased rectangle mask (fractions of width/height). Hard masks would snap to whole
        pixels and not move with sub-pixel shifts, which pins burst registration at zero."""
        return (_smoothstep_edge(xx - x0 * w) * _smoothstep_edge(x1 * w - xx)
                * _smoothstep_edge(yy - y0 * h) * _smoothstep_edge(y1 * h - yy))[..., None]

    # Slanted black/white edge, middle.
    d = (xx - 0.49 * w) * np.cos(0.09) + (yy - 0.3 * h) * np.sin(0.09)
    edge = (0.03 + 0.8 * _smoothstep_edge(d))[..., None]
    m = box(0.36, 0.62, 0.1, 0.55)
    img = img * (1 - m) + edge * m

    # Fine near-Nyquist stripes, lower middle.
    stripes = (0.2 + 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * xx / 3.3 + 0.4 * yy / h)))[..., None]
    m = box(0.36, 0.62, 0.62, 0.9)
    img = img * (1 - m) + stripes * m

    # Saturated colour disks, right.
    for (fx, fy), col in [((0.74, 0.25), (0.65, 0.02, 0.02)), ((0.88, 0.25), (0.02, 0.45, 0.04)),
                          ((0.74, 0.7), (0.02, 0.05, 0.6)), ((0.88, 0.7), (0.7, 0.55, 0.02))]:
        rr = np.sqrt((xx - fx * w) ** 2 + (yy - fy * h) ** 2)
        m = _smoothstep_edge(0.09 * min(h, w) - rr)[..., None]
        img = img * (1 - m) + np.array(col) * m

    # Specular highlight that will clip.
    rr = (xx - 0.81 * w) ** 2 + (yy - 0.47 * h) ** 2
    img += 3.0 * np.exp(-rr / (2 * (0.012 * min(h, w)) ** 2))[..., None]
    return img.astype(np.float32)


@dataclass
class SyntheticCamera:
    black: float = 256.0
    white: float = 4095.0
    base_iso: int = 100
    gain_base: float = 3.0  # e-/DN at base ISO
    read_base: float = 1.8  # DN at base ISO
    read_slope: float = 0.004  # extra DN of read noise per ISO step above base
    wb_gains: tuple = (2.1, 1.0, 1.55)  # what the pipeline must multiply by
    ccm: np.ndarray = field(default_factory=lambda: np.array(
        [[1.62, -0.48, -0.14], [-0.22, 1.48, -0.26], [0.03, -0.52, 1.49]]))
    vignette: tuple = (0.55, 0.45, 0.62)  # relative fall-off at the corners, R, G, B
    hot_fraction: float = 2e-4
    hot_dn: float = 300.0
    n_stuck: int = 6
    seed: int = 7

    def __post_init__(self):
        self._defects_for = None

    # --- per-ISO noise ---------------------------------------------------- #
    def gain(self, iso: int) -> float:
        return self.gain_base * self.base_iso / iso

    def read(self, iso: int) -> float:
        return self.read_base + self.read_slope * (iso - self.base_iso)

    # --- fixed pattern ------------------------------------------------------ #
    def defects(self, shape):
        if self._defects_for != shape:
            rng = np.random.default_rng(self.seed)
            self.hot = rng.random(shape) < self.hot_fraction
            stuck = np.zeros(shape, dtype=bool)
            idx = rng.choice(shape[0] * shape[1], self.n_stuck, replace=False)
            stuck.flat[idx] = True
            self.stuck = stuck & ~self.hot
            self._defects_for = shape
        return self.hot, self.stuck

    def vignette_rgb(self, h: int, w: int) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        r2 = r2 / 2.0  # 1.0 at the corners
        return np.stack([1 - f * r2 for f in self.vignette], axis=-1).astype(np.float32)

    # --- capture ------------------------------------------------------------ #
    def capture(self, scene_srgb_lin: np.ndarray, iso: int, rng: np.random.Generator,
                exposure: float = 1.0) -> np.ndarray:
        """Scene in linear sRGB -> raw mosaic in integer-valued DN (float32)."""
        h, w, _ = scene_srgb_lin.shape
        cam = scene_srgb_lin @ np.linalg.inv(self.ccm).T  # white-balanced camera RGB
        cam = cam / np.array(self.wb_gains)  # unbalanced sensor response
        cam = np.clip(cam, 0, None) * self.vignette_rgb(h, w) * exposure
        signal_dn = mosaic_rggb(cam.astype(np.float32)) * (self.white - self.black)
        k = self.gain(iso)
        electrons = rng.poisson(np.clip(signal_dn, 0, None) * k)
        dn = electrons / k + rng.normal(0, self.read(iso), size=signal_dn.shape) + self.black
        hot, stuck = self.defects(signal_dn.shape)
        dn = dn + hot * self.hot_dn
        dn = np.where(stuck, self.white, dn)
        return np.clip(np.round(dn), 0, self.white).astype(np.float32)

    def truth(self) -> dict:
        return {
            "black_level_dn": self.black, "white_level_dn": self.white,
            "wb_gains_rgb": list(self.wb_gains), "ccm": self.ccm.tolist(),
            "vignette_corner_falloff_rgb": list(self.vignette),
        }

    def npz_fields(self) -> dict:
        return {"black_rggb": np.full(4, self.black), "white": self.white, "cfa": "RGGB"}
