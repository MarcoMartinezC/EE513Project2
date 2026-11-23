"""Generate a complete synthetic dataset (calibration, bursts, detail scene) with known constants.

    python scripts/make_synthetic_data.py --out data/synthetic

Then run any script with ``--config data/synthetic/camera.yaml``. Every
constant the pipeline estimates has its true value in ``truth.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from isp.synthetic import SyntheticCamera, scene_linear_srgb


def save(path: Path, mosaic, cam: SyntheticCamera, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mosaic=mosaic.astype(np.uint16), **cam.npz_fields(), **extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--size", type=int, nargs=2, default=[768, 1024], metavar=("H", "W"))
    ap.add_argument("--isos", type=int, nargs="+", default=[100, 1600, 6400])
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--max-shift", type=float, default=0.6, help="burst jitter, sensor pixels")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out)
    h, w = args.size
    cam = SyntheticCamera()
    rng = np.random.default_rng(args.seed)

    # Dark frames (lens cap): zero scene at the highest ISO.
    for k in range(8):
        save(out / f"calib/dark/dark_{k:02d}.npz", cam.capture(np.zeros((h, w, 3), np.float32), max(args.isos), rng), cam)
    # Flat field: a uniform neutral surface at base ISO.
    flat_scene = np.full((h, w, 3), 0.55, np.float32)
    for k in range(8):
        save(out / f"calib/flat/flat_{k:02d}.npz", cam.capture(flat_scene, cam.base_iso, rng), cam)

    # Bursts: sub-pixel jitter, the first frame unshifted.
    shifts = {}
    for iso in args.isos:
        sh = [(0.0, 0.0)] + [tuple(rng.uniform(-args.max_shift, args.max_shift, 2)) for _ in range(args.frames - 1)]
        shifts[iso] = sh
        for k, (dx, dy) in enumerate(sh):
            scene = scene_linear_srgb(h, w, dx, dy)
            save(out / f"burst/iso{iso}/frame_{k:02d}.npz", cam.capture(scene, iso, rng), cam)
        print(f"burst ISO {iso}: {args.frames} frames")

    # Detail scene: mirrored content, so the clean pool built from it is not the burst scene pixel for pixel.
    # (On the synthetic camera that is the best available; with real captures use different scenes.)
    detail = np.ascontiguousarray(scene_linear_srgb(h, w)[::-1, ::-1])
    save(out / "detail/detail.npz", cam.capture(detail, cam.base_iso, rng), cam)

    hot, stuck = cam.defects((h, w))
    np.savez_compressed(out / "truth_maps.npz", hot=hot, stuck=stuck, vignette_rgb=cam.vignette_rgb(h, w))
    truth = cam.truth()
    truth["noise"] = {iso: {"gain_e_per_dn": cam.gain(iso), "read_noise_dn": cam.read(iso)} for iso in args.isos}
    truth["burst_shifts_sensor_px"] = {iso: s for iso, s in shifts.items()}
    truth["n_hot"], truth["n_stuck"] = int(hot.sum()), int(stuck.sum())
    (out / "truth.json").write_text(json.dumps(truth, indent=2))

    cfg = {
        "root": ".",
        "camera": {"make": "synthetic", "model": "SyntheticCamera"},
        "raw": {"black_level_dn": [cam.black, 0.0], "white_level_dn": [cam.white, 0.0]},
        "paths": {"dark_frames": "calib/dark", "flat_frames": "calib/flat", "bursts": "burst",
                  "detail": "detail", "detector": "detector", "derived": "derived", "results": "results"},
        "calibration": {"defects_k_sigma": 8.0, "stuck_tolerance": 0.3, "lsc_sigma_frac": 0.01},
        "white_balance": {"gains_rgb": [[float(g), 0.0] for g in cam.wb_gains]},
        "color_correction": {"matrix": cam.ccm.tolist(), "uncertainty": None},
        "noise": {"iso": {iso: {"gain_e_per_dn": [cam.gain(iso), 0.0], "read_noise_dn": [cam.read(iso), 0.0]}
                          for iso in args.isos}},
    }
    (out / "camera.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out}/ (config: {out}/camera.yaml, true constants: {out}/truth.json)")


if __name__ == "__main__":
    main()
