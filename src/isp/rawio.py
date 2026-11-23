"""Raw file loading and Bayer bookkeeping.

Everything downstream assumes an RGGB mosaic with red at (0, 0). This module
reads the CFA pattern from the file (never assumes it) and crops by one row
and/or column when needed, exactly as in Exercise 5 item 1.

Besides camera raw files (anything rawpy reads, e.g. .NEF), ``load_raw`` also
accepts ``.npz`` files written by ``isp.synthetic``. Those let the whole
pipeline run before real captures exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

RGGB_ORDER = ("R", "G1", "G2", "B")


@dataclass
class RawFrame:
    mosaic: np.ndarray  # (H, W) float32 raw DN, RGGB-aligned, even H and W
    black_rggb: np.ndarray  # (4,) black level in DN at the R, G1, G2, B sites
    white: float  # saturation level in DN
    cfa: str  # native 2x2 pattern as read from the file, e.g. "RGGB"
    offset: tuple[int, int]  # (row, col) crop applied so that (0, 0) is red
    path: str = ""
    as_shot_wb: np.ndarray | None = None  # (3,) camera's as-shot gains, G = 1. Reference only.

    @property
    def black_rgb(self) -> np.ndarray:
        return rggb_to_rgb(self.black_rggb)


# --------------------------------------------------------------------------- #
# Bayer helpers
# --------------------------------------------------------------------------- #
def rggb_to_rgb(v4) -> np.ndarray:
    """Collapse a per-site (R, G1, G2, B) quantity to (R, G, B) by averaging the greens."""
    v4 = np.asarray(v4, dtype=np.float64)
    return np.array([v4[0], 0.5 * (v4[1] + v4[2]), v4[3]])


def tile_rggb(v4, shape, dtype=np.float32) -> np.ndarray:
    """Tile four per-site values over an RGGB mosaic of the given (even) shape."""
    h, w = shape
    if h % 2 or w % 2:
        raise ValueError(f"mosaic shape must be even, got {shape}")
    cell = np.array([[v4[0], v4[1]], [v4[2], v4[3]]], dtype=dtype)
    return np.tile(cell, (h // 2, w // 2))


def pack_rggb(m: np.ndarray) -> np.ndarray:
    """(H, W) RGGB mosaic -> (H/2, W/2, 4) planes in order R, G1, G2, B."""
    return np.stack([m[0::2, 0::2], m[0::2, 1::2], m[1::2, 0::2], m[1::2, 1::2]], axis=-1)


def unpack_rggb(p: np.ndarray) -> np.ndarray:
    """Inverse of ``pack_rggb``."""
    h, w, c = p.shape
    if c != 4:
        raise ValueError("packed Bayer must have 4 channels")
    m = np.empty((2 * h, 2 * w), dtype=p.dtype)
    m[0::2, 0::2] = p[..., 0]
    m[0::2, 1::2] = p[..., 1]
    m[1::2, 0::2] = p[..., 2]
    m[1::2, 1::2] = p[..., 3]
    return m


def cfa_alignment(raw_pattern, color_desc: str) -> tuple[str, tuple[int, int]]:
    """Return the native CFA string and the (row, col) offset that puts red at (0, 0).

    ``raw_pattern`` is rawpy's 2x2 array of indices into ``color_desc`` (e.g. "RGBG").
    Raises if the pattern is not a Bayer pattern with blue diagonal to red.
    """
    raw_pattern = np.asarray(raw_pattern)
    letters = [[color_desc[raw_pattern[i, j]] for j in range(2)] for i in range(2)]
    cfa = "".join(letters[0] + letters[1])
    for r in range(2):
        for c in range(2):
            if letters[r][c] == "R":
                if letters[(r + 1) % 2][(c + 1) % 2] != "B":
                    raise ValueError(f"not a Bayer pattern: {cfa}")
                return cfa, (r, c)
    raise ValueError(f"no red site in CFA pattern {cfa}")


def _black_to_rggb(black) -> np.ndarray:
    b = np.atleast_1d(np.asarray(black, dtype=np.float64))
    if b.size == 1:
        return np.repeat(b, 4)
    if b.size == 4:
        return b
    raise ValueError("black level must be a scalar or four values (R, G1, G2, B)")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(path, black=None, white=None) -> RawFrame:
    """Load a raw frame and align it to RGGB.

    ``black`` / ``white`` override the values stored in the file. Pass your
    Project 1 measurements here (``config.raw.black_level_dn`` etc.) so the
    pipeline uses what you measured rather than what the manufacturer wrote.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        return _load_npz(path, black, white)

    import rawpy  # imported lazily so tests can run without it

    with rawpy.imread(str(path)) as raw:
        m = raw.raw_image_visible.astype(np.float32)
        pattern = np.array(raw.raw_pattern)
        desc = raw.color_desc.decode("ascii")
        black_by_index = np.array(raw.black_level_per_channel, dtype=np.float64)
        white_file = float(raw.white_level)
        cam_wb = np.array(raw.camera_whitebalance[:3], dtype=np.float64)

    cfa, (r, c) = cfa_alignment(pattern, desc)
    m = m[r:, c:]
    h, w = m.shape
    m = np.ascontiguousarray(m[: h // 2 * 2, : w // 2 * 2])

    # rawpy orders black levels by colour index, not by position: map to sites.
    file_black = np.array(
        [black_by_index[pattern[(i + r) % 2, (j + c) % 2]] for i in range(2) for j in range(2)]
    )
    black_rggb = _black_to_rggb(black) if black is not None else file_black
    as_shot = cam_wb / cam_wb[1] if cam_wb[1] > 0 else None

    return RawFrame(
        mosaic=m,
        black_rggb=black_rggb,
        white=float(white) if white is not None else white_file,
        cfa=cfa,
        offset=(r, c),
        path=str(path),
        as_shot_wb=as_shot,
    )


def _load_npz(path: Path, black, white) -> RawFrame:
    d = np.load(path)
    black_rggb = _black_to_rggb(black) if black is not None else d["black_rggb"].astype(np.float64)
    return RawFrame(
        mosaic=d["mosaic"].astype(np.float32),
        black_rggb=black_rggb,
        white=float(white) if white is not None else float(d["white"]),
        cfa=str(d["cfa"]) if "cfa" in d else "RGGB",
        offset=(0, 0),
        path=str(path),
        as_shot_wb=d["as_shot_wb"] if "as_shot_wb" in d else None,
    )


def load_stack(paths, black=None, white=None) -> tuple[np.ndarray, RawFrame]:
    """Load several frames of the same shape. Returns (N, H, W) DN and the first frame's metadata."""
    paths = list(paths)
    if not paths:
        raise FileNotFoundError("no raw files given")
    frames = [load_raw(p, black, white) for p in paths]
    shapes = {f.mosaic.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"frames differ in shape: {shapes}")
    cfas = {f.cfa for f in frames}
    if len(cfas) != 1:
        raise ValueError(f"frames differ in CFA pattern: {cfas}")
    return np.stack([f.mosaic for f in frames]), frames[0]


def find_raws(directory, patterns=("*.NEF", "*.nef", "*.DNG", "*.dng", "*.npz")) -> list[Path]:
    """Sorted list of raw files in a directory."""
    directory = Path(directory)
    found: set[Path] = set()
    for pat in patterns:
        found.update(directory.glob(pat))
    return sorted(found)
