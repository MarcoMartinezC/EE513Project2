"""Small helpers shared by the scripts in ``scripts/``."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .calib import Calibration


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="config/camera.yaml", help="calibration config (default: %(default)s)")
    return p


def load(args) -> tuple[dict, Calibration | None]:
    cfg = C.load_config(args.config)
    calib_path = derived(cfg) / "calib.npz"
    calib = Calibration.load(calib_path) if calib_path.exists() else None
    if calib is None:
        print(f"[note] no {calib_path}; run scripts/calibrate.py first for shading and defect maps")
    return cfg, calib


def derived(cfg) -> Path:
    p = C.resolve(cfg, cfg["paths"]["derived"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def results(cfg) -> Path:
    p = C.resolve(cfg, cfg["paths"]["results"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def iso_from_path(path) -> int | None:
    m = re.search(r"iso[_-]?(\d+)", str(path), re.IGNORECASE)
    return int(m.group(1)) if m else None


def save_png(path, rgb: np.ndarray, bits: int = 8) -> None:
    """Save an RGB image in [0, 1] as PNG (8 or 16 bit)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scale, dtype = (255, np.uint8) if bits == 8 else (65535, np.uint16)
    img = np.round(np.clip(rgb, 0, 1) * scale).astype(dtype)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_rgb(path) -> np.ndarray:
    """Read a JPEG/PNG as RGB float in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / scale


def fmt(pair, digits: int = 2) -> str:
    v, u = pair
    return f"{v:.{digits}f} ± {u:.{digits}f}" if u is not None and np.isfinite(u) else f"{v:.{digits}f}"
