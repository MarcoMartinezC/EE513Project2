"""Calibration config: Project 1 constants, each as [value, uncertainty]."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


class MissingCalibration(KeyError):
    pass


def load_config(path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(Path(path).resolve())
    return cfg


def value(pair, name: str = "value"):
    """Return the value of a [value, uncertainty] pair, refusing nulls with a useful message."""
    if pair is None:
        raise MissingCalibration(f"{name} is not set in the config")
    v = pair[0] if isinstance(pair, (list, tuple)) and len(pair) == 2 and not isinstance(pair[0], list) else pair
    if v is None:
        raise MissingCalibration(f"{name} is null in the config; fill it in from Project 1")
    return v


def black_rggb(cfg: dict) -> np.ndarray | None:
    pair = cfg["raw"].get("black_level_dn")
    if pair is None or pair[0] is None:
        return None
    b = np.atleast_1d(np.asarray(pair[0], dtype=np.float64))
    return np.repeat(b, 4) if b.size == 1 else b


def white_level(cfg: dict) -> float | None:
    pair = cfg["raw"].get("white_level_dn")
    return None if pair is None or pair[0] is None else float(pair[0])


def wb_gains(cfg: dict) -> np.ndarray:
    g = cfg["white_balance"]["gains_rgb"]
    return np.array([value(g[i], f"white_balance.gains_rgb[{i}]") for i in range(3)], dtype=np.float64)


def ccm(cfg: dict) -> np.ndarray:
    m = cfg["color_correction"]["matrix"]
    if m is None:
        raise MissingCalibration("color_correction.matrix is null; paste your Project 1 CCM")
    m = np.asarray(m, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"CCM must be 3x3, got {m.shape}")
    return m


def resolve(cfg: dict, rel: str) -> Path:
    """Resolve a path from the config relative to ``root`` (itself relative to the config file)."""
    p = Path(rel)
    if p.is_absolute():
        return p
    root = Path(cfg["_path"]).parent / cfg.get("root", ".")
    return (root / p).resolve()
