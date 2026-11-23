"""Collect measured numbers and fill results2.json from the course template.

The template is attached to the Project 2 page as ``results2.template.json``;
copy it to the repository root. The schema is the instructor's, so this module
never invents keys: ``fill_template`` refuses any key that is not in the
template.

Rules (same as Project 1): every measured entry is a [value, uncertainty]
pair, ``null`` for anything not measured, and no invented uncertainties.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def pair(value, uncertainty) -> list:
    """Validated [value, uncertainty]. Use None for an unmeasured uncertainty only if the template allows it."""
    if value is None:
        raise ValueError("use None for the whole entry, not for the value alone")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"value is not finite: {value}")
    if uncertainty is None:
        return [v, None]
    u = float(uncertainty)
    if not math.isfinite(u) or u < 0:
        raise ValueError(f"uncertainty must be finite and non-negative, got {uncertainty}")
    return [v, u]


def save_json(path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def _default(o):
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _set(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            raise KeyError(f"{dotted!r}: {k!r} is not in the template")
        d = d[k]
    if keys[-1] not in d:
        raise KeyError(f"{dotted!r} is not in the template")
    d[keys[-1]] = value


def fill_template(template_path, out_path, updates: dict[str, Any]) -> dict:
    """Copy the template, set ``updates`` (dotted keys -> [value, unc] or other values), write it."""
    with open(template_path) as f:
        data = json.load(f)
    for k, v in updates.items():
        _set(data, k, v)
    save_json(out_path, data)
    return data
