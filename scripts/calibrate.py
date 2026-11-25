"""Derive the defect map (dark frames + flat field) and per-channel lens shading gains.

    python scripts/calibrate.py --config config/camera.yaml

Writes data/derived/calib.npz and results/calibration.json.

Shading flat checks (all in the JSON and printed):
* the level of every flat frame (0.4-0.8 of full scale is the useful range);
* the EXIF focal length, aperture and ISO of every flat, compared with
  camera.focal_length_mm / camera.aperture_f in the config;
* a rotation check: the flats are split in file order into a first and a
  second half (by default, first 4 upright, last 4 rotated 180 degrees about
  the lens axis). Lens shading turns with the camera; unevenness of the light
  does not. So the shading maps from the two halves should agree, and their
  difference measures how uneven the light was. The final map uses all
  frames, which cancels most of a linear gradient. Figure:
  results/lsc_rotation_check.png.
"""

import json
from pathlib import Path

import numpy as np

from isp import calib as K
from isp import cliutil
from isp import config as C
from isp.rawio import find_raws, load_stack, pack_rggb

PLANES = ("R", "G1", "G2", "B")


def exif_summary(path) -> dict | None:
    """Focal length, aperture, ISO and shutter from EXIF (None if exifread is missing or a tag is absent)."""
    try:
        import exifread
    except ImportError:
        return None
    with open(path, "rb") as fh:
        tags = exifread.process_file(fh, details=False)

    def num(key):
        if key not in tags:
            return None
        v = tags[key].values[0]
        return float(v.num) / float(v.den) if hasattr(v, "num") else float(v)

    return {"focal_mm": num("EXIF FocalLength"), "f_number": num("EXIF FNumber"),
            "iso": num("EXIF ISOSpeedRatings"), "exposure_s": num("EXIF ExposureTime")}


def rotation_check(flats, black, white, mask, split, method, degree, sigma_frac, fig_path):
    """Shading maps from flats[:split] and flats[split:]; their relative difference, per plane."""
    kw = dict(sigma_frac=sigma_frac, defect_mask=mask, method=method, degree=degree)
    ga = K.shading_gain_from_flat(flats[:split], black, white, **kw)
    gb = K.shading_gain_from_flat(flats[split:], black, white, **kw)
    rel = gb / ga - 1.0
    h, w, _ = rel.shape
    inner = rel[h // 20: -h // 20, w // 20: -w // 20]  # the outer 5 % is where both fits are least certain
    out = {"split": [int(split), int(len(flats) - split)], "per_plane": {}}
    for c, p in enumerate(PLANES):
        out["per_plane"][p] = {"max_abs": float(np.abs(rel[..., c]).max()),
                               "rms": float(np.sqrt(np.mean(rel[..., c] ** 2))),
                               "max_abs_inner_90pct": float(np.abs(inner[..., c]).max())}
    out["max_abs_all_planes"] = max(v["max_abs"] for v in out["per_plane"].values())
    out["rms_all_planes"] = float(np.sqrt(np.mean(rel ** 2)))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    g = 0.5 * (ga + gb)
    im = ax[0].imshow(g[..., 1], cmap="viridis")
    ax[0].set_title("shading gain, G1 (all frames)")
    fig.colorbar(im, ax=ax[0], fraction=0.035)
    lim = max(np.abs(rel[..., 1]).max(), 1e-4)
    im = ax[1].imshow(100 * rel[..., 1], cmap="RdBu_r", vmin=-100 * lim, vmax=100 * lim)
    ax[1].set_title("second half / first half - 1, G1 (%)")
    fig.colorbar(im, ax=ax[1], fraction=0.035)
    mid = g.shape[0] // 2
    for c, p in enumerate(PLANES):
        ax[2].plot(g[mid, :, c], label=p, lw=1)
    ax[2].set_title("gain along the middle row")
    ax[2].set_xlabel("column (packed px)")
    ax[2].legend()
    for a in ax[:2]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    return out


def main():
    ap = cliutil.parser(__doc__)
    ap.add_argument("--rotation-split", type=int, default=None,
                    help="number of upright flats before the rotated ones (default: half of them)")
    ap.add_argument("--no-rotation-check", action="store_true")
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
        levels = [float(np.mean(K.normalise(f, meta.black_rggb, meta.white)[f.shape[0] // 2 - 256:f.shape[0] // 2 + 256,
                                                                            f.shape[1] // 2 - 256:f.shape[1] // 2 + 256]))
                  for f in flats]
        report["flat_files"] = [p.name for p in flat_paths]
        report["flat_centre_levels"] = levels
        for p, lv in zip(flat_paths, levels):
            if not 0.4 <= lv <= 0.8:
                print(f"[warn] {p.name}: centre level {lv:.3f} is outside 0.4-0.8 of full scale")
        exifs = [exif_summary(p) for p in flat_paths]
        if exifs[0] is not None:
            report["flat_exif"] = exifs
            for key, cfg_key in (("focal_mm", "focal_length_mm"), ("f_number", "aperture_f")):
                vals = {round(e[key], 1) for e in exifs if e[key] is not None}
                if len(vals) > 1:
                    print(f"[warn] flats differ in {key}: {sorted(vals)}")
                want = cfg.get("camera", {}).get(cfg_key)
                if want is not None and vals and any(abs(v - want) > 0.05 * want for v in vals):
                    print(f"[warn] flats were shot at {key} {sorted(vals)}, config says {want}: "
                          "the shading map only applies at the scene settings")
        else:
            print("[note] install exifread to check the flats' focal length and aperture")
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
        if not args.no_rotation_check and len(flats) >= 4:
            split = args.rotation_split or len(flats) // 2
            rc = rotation_check(flats, meta.black_rggb, meta.white, mask, split, ccfg.get("lsc_method", "poly"),
                                ccfg.get("lsc_degree", 6), ccfg.get("lsc_sigma_frac", 0.01),
                                cliutil.results(cfg) / "lsc_rotation_check.png")
            report["lsc_rotation_check"] = rc
            print(f"rotation check ({rc['split'][0]} upright vs {rc['split'][1]} rotated): shading maps differ by "
                  f"{100 * rc['rms_all_planes']:.2f} % rms, {100 * rc['max_abs_all_planes']:.2f} % at worst")
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