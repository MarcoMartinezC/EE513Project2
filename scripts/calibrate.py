"""Derive the defect map (dark frames + flat field) and per-channel lens shading gains.

    python scripts/calibrate.py --config config/camera.yaml

Writes data/derived/calib.npz and results/calibration.json.
"""

import json
from pathlib import Path

import numpy as np

from isp import calib as K
from isp import cliutil
from isp import config as C
from isp.rawio import find_raws, load_stack, pack_rggb


def main():
    ap = cliutil.parser(__doc__)
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    ccfg = cfg.get("calibration", {})
    black, white = C.black_rggb(cfg), C.white_level(cfg)

    report: dict = {}
    dark_paths = find_raws(C.resolve(cfg, cfg["paths"]["dark_frames"]))
    flat_paths = find_raws(C.resolve(cfg, cfg["paths"]["flat_frames"]))

    hot = stuck = None
    if dark_paths:
        darks, meta = load_stack(dark_paths, black, white)
        hot = K.hot_pixels_from_darks(darks, k=ccfg.get("defects_k_sigma", 8.0))
        report["dark_frames"] = len(dark_paths)
        report["hot_pixels"] = int(hot.sum())
        report["hot_fraction"] = float(hot.mean())
        # Dark level per channel is a check on the black level in the config.
        dmean = pack_rggb(darks.mean(axis=0))[~pack_rggb(hot)].reshape(-1)
        report["dark_mean_dn"] = float(dmean.mean())
        report["black_level_config_dn"] = meta.black_rggb.tolist()
    else:
        print("[warn] no dark frames found; defect map will only contain stuck pixels from the flat")

    gain = None
    if flat_paths:
        flats, meta = load_stack(flat_paths, black, white)
        flat_norm = K.normalise(flats.mean(axis=0), meta.black_rggb, meta.white)
        stuck = K.stuck_pixels_from_flat(flat_norm, ccfg.get("stuck_tolerance", 0.3))
        report["flat_frames"] = len(flat_paths)
        report["stuck_pixels"] = int(stuck.sum())
        mask = stuck if hot is None else (stuck | hot)
        gain = K.shading_gain_from_flat(flats, meta.black_rggb, meta.white,
                                        ccfg.get("lsc_sigma_frac", 0.01), defect_mask=mask,
                                        method=ccfg.get("lsc_method", "poly"), degree=ccfg.get("lsc_degree", 6))
        h, w, _ = gain.shape
        corners = gain[[0, 0, -1, -1], [0, -1, 0, -1]]  # (4 corners, 4 channels)
        report["lsc_max_gain_per_channel"] = dict(zip(("R", "G1", "G2", "B"), gain.reshape(-1, 4).max(0).tolist()))
        report["lsc_mean_corner_gain_per_channel"] = dict(zip(("R", "G1", "G2", "B"), corners.mean(0).tolist()))
        # Flat-field peak level: under 0.3 or over 0.9 of saturation and the map is noisy or clipped.
        report["flat_peak_level"] = float(np.percentile(flat_norm, 99.5))
    else:
        print("[warn] no flat frames found; lens shading correction will be unavailable")

    mask = None
    if hot is not None or stuck is not None:
        mask = np.zeros_like(hot if hot is not None else stuck)
        for m in (hot, stuck):
            if m is not None:
                mask |= m
        report["defect_pixels_total"] = int(mask.sum())

    out = cliutil.derived(cfg) / "calib.npz"
    K.Calibration(gain, mask, report).save(out)

    # On the synthetic camera, compare with the truth.
    truth_maps = Path(cfg["_path"]).parent / "truth_maps.npz"
    if truth_maps.exists() and mask is not None and gain is not None:
        t = np.load(truth_maps)
        true_bad = t["hot"] | t["stuck"]
        report["synthetic_check"] = {
            "defects_found_of_true": [int((mask & true_bad).sum()), int(true_bad.sum())],
            "false_defects": int((mask & ~true_bad).sum()),
        }
        true_gain = 1.0 / t["vignette_rgb"]
        true_gain = true_gain / true_gain.min(axis=(0, 1))
        est = K.lsc_rgb_full(gain)
        report["synthetic_check"]["lsc_max_rel_error"] = float(np.abs(est / true_gain - 1).max())

    (cliutil.results(cfg) / "calibration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
