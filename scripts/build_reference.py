"""Register one burst and save it for the scoring scripts; report how clean the reference is.

    python scripts/build_reference.py --burst data/burst/iso3200 [--crop 2048]

Writes data/derived/burst_iso3200.npz (packed frames in raw DN, unresampled,
plus their shifts) and results/reference_iso3200.json with the shifts, the
noise in one frame, the residual noise in the leave-one-out average, and a
Poisson-Gaussian fit from the burst to compare with your Project 1 numbers.

Run once with --no-align as well and compare the residual noise: if alignment
does not lower the frame-difference noise, your tripod was steady enough and
interpolation only blurs the reference.
"""

import json

import numpy as np

from isp import cliutil
from isp import config as C
from isp.calib import correct_defects
from isp.context import center_crop_mosaic
from isp.groundtruth import make_burst, residual_noise_report
from isp.noise_model import fit_from_burst
from isp.rawio import find_raws, load_stack, pack_rggb, unpack_rggb


def main():
    ap = cliutil.parser(__doc__)
    ap.add_argument("--burst", required=True, help="folder with >= 16 raw frames of one ISO")
    ap.add_argument("--iso", type=int, help="ISO (default: parsed from the folder name)")
    ap.add_argument("--crop", type=int, default=0, help="centre crop (sensor px) for quick runs; 0 = full frame")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--min-shift", type=float, default=0.02, help="packed px below which frames are not resampled")
    ap.add_argument("--tag", default=None, help="name for the outputs (default: iso<ISO>)")
    args = ap.parse_args()

    cfg, calib = cliutil.load(args)
    iso = args.iso or cliutil.iso_from_path(args.burst)
    tag = args.tag or (f"iso{iso}" + ("_noalign" if args.no_align else "") if iso else "burst")
    paths = find_raws(args.burst)
    if len(paths) < 16:
        print(f"[warn] only {len(paths)} frames; the spec asks for at least 16")
    stack, meta = load_stack(paths, C.black_rggb(cfg), C.white_level(cfg))
    stack, origin = center_crop_mosaic(stack, args.crop)
    h, w = stack.shape[-2:]

    b = meta.black_rggb.astype(np.float32)
    mask = None
    if calib is not None and calib.defect_mask is not None:
        y0, x0 = origin
        mask = calib.defect_mask[y0:y0 + h, x0:x0 + w]
    else:
        print("[warn] no defect map: fixed-pattern hot pixels will bias the shift estimates towards zero")
    burst = make_burst(stack, align=not args.no_align, min_shift=args.min_shift, defect_mask=mask)

    # Statistics on a normalised, defect-corrected copy (hot pixels are not noise).

    def clean(p):
        q = (p - b) / (meta.white - b)
        return pack_rggb(correct_defects(unpack_rggb(q), mask)) if mask is not None else q

    norm = burst.map(clean)
    ks = list(range(0, norm.n, max(1, norm.n // 4)))  # a few frames are plenty for statistics
    noise = residual_noise_report(norm, ks)
    fitted, fit_report = fit_from_burst(norm, float(np.mean(b)), meta.white, ks=ks)

    sensor_shift = 2 * np.hypot(*burst.shifts[1:].T)
    report = {
        "burst": str(args.burst), "iso": iso, "frames": len(paths), "crop_origin": list(origin),
        "mosaic_shape": [h, w], "aligned": not args.no_align,
        "shift_vs_frame0_sensor_px": {"median": float(np.median(sensor_shift)), "max": float(sensor_shift.max())},
        "margin_packed_px": burst.margin,
        "noise_normalised": noise,
        "noise_fit_from_burst": {k: v for k, v in fit_report.items()},
    }
    try:
        entry = cfg["noise"]["iso"][iso]
        report["noise_config"] = {k: entry[k] for k in ("gain_e_per_dn", "read_noise_dn")}
    except (KeyError, TypeError):
        report["noise_config"] = None

    np.savez(cliutil.derived(cfg) / f"burst_{tag}.npz", packed_dn=burst.packed, shifts=burst.shifts,
             min_shift=burst.min_shift, black_rggb=meta.black_rggb, white=meta.white,
             crop_origin=np.array(origin), iso=iso if iso else -1)
    (cliutil.results(cfg) / f"reference_{tag}.json").write_text(json.dumps(report, indent=2))

    s1, sr = noise["sigma_single_mean"], noise["sigma_reference_mean"]
    pair = noise["channels"]["G1"]
    print(f"{tag}: {len(paths)} frames, shift vs frame 0: median {report['shift_vs_frame0_sensor_px']['median']:.3f}"
          f" px, max {report['shift_vs_frame0_sensor_px']['max']:.3f} px (sensor)")
    print(f"noise in one frame {s1:.5f}; left in the {noise['n_in_reference']}-frame reference {sr:.5f} "
          f"(normalised; G1 std {pair['sigma_single_std']:.5f} vs MAD {pair['sigma_single_mad']:.5f})")
    print(f"noise fit from burst: K = {cliutil.fmt(fit_report['gain_e_per_dn'], 3)} e-/DN, "
          f"read = {cliutil.fmt(fit_report['read_noise_dn'])} DN;  config: {report['noise_config']}")
    print(f"-> {cliutil.derived(cfg)}/burst_{tag}.npz")


if __name__ == "__main__":
    main()
