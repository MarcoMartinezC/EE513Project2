"""Stage ablation: remove or misplace one stage at a time and measure the PSNR cost.

    python scripts/ablation.py --reference data/derived/burst_iso3200.npz [--frames 8]

Ground truth and comparison (a decision to defend in the report):

* Reference image: the leave-one-out burst average aligned to frame k, defect
  corrected, taken through the 2x2 construction (so no demosaicing and no
  denoising are needed), then through the correct pipeline in RGB:
  black level, shading, white balance, highlight clip, CCM, tone curve.
* Test image: frame k alone, through the same 2x2 construction, re-mosaicked,
  then through each pipeline variant starting from raw DN.
* Both are compared in display-referred sRGB, before sharpening (peak 1.0).
  Sharpening is a perceptual choice that PSNR punishes by design; report it
  separately if you want to.
* Repeating over k frames gives a mean and a standard deviation per variant.
* Denoiser strength is tuned separately for each placement (on packed Bayer,
  and after demosaicing) on one frame that is then left out of the scores.
  Otherwise "denoise after demosaic" compares two different strengths, not two
  placements.

Two things this construction cannot see: anything that only happens at full
resolution (it runs at the 2x2 resolution), and the noise in the green
channel is halved by the 2x2 mean. Say so next to the table.
"""

import json

import numpy as np

from isp import cliutil
from isp import pipeline as P
from isp.context import build_context
from isp.groundtruth import load_saved_burst, mosaic_rggb, quarter_res
from isp.metrics import mean_std, psnr
from isp.rawio import unpack_rggb

BASE = P.without(P.DEFAULT_STEPS, "sharpen")

REFERENCE_STEPS = [("to_unit", {}), ("black_level", {}), ("lsc", {}), ("wb", {}), ("clip_highlights", {}),
                   ("ccm", {}), ("tone", {"curve": "srgb", "exposure": 1.0})]


EPS_GRID = (0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256)


def variants(eps_raw: float = 4.0, eps_rgb: float = 4.0) -> dict[str, list]:
    base = P.with_params(BASE, "denoise", eps_scale=eps_raw)
    late = P.with_params(P.move_after(BASE, "denoise", "demosaic"), "denoise", eps_scale=eps_rgb)
    return _variants(base, late)


def _variants(BASE, late) -> dict[str, list]:
    v = {
        "full pipeline": BASE,
        "no black level": P.without(BASE, "black_level"),
        "black level after demosaic": P.move_after(BASE, "black_level", "demosaic"),
        "no defect correction": P.without(BASE, "defects"),
        "no denoising": P.with_params(BASE, "denoise", method="none"),
        "denoise after demosaic": late,
        "no lens shading": P.without(BASE, "lsc"),
        "WB after demosaic": P.move_after(P.move_after(BASE, "wb", "demosaic"), "clip_highlights", "wb"),
        "no highlight clip": P.without(BASE, "clip_highlights"),
        "bilinear instead of MHC": P.with_params(BASE, "demosaic", method="bilinear"),
        "no colour correction": P.without(BASE, "ccm"),
    }
    return v


def tune(test_in, ref_out, ctx, crop) -> tuple[float, float, dict]:
    """Best eps_scale for each denoiser placement, by PSNR on one frame."""
    grid = {"raw": {}, "rgb": {}}
    for eps in EPS_GRID:
        v = variants(eps, eps)
        grid["raw"][eps] = psnr(P.run(test_in, ctx, v["full pipeline"]).image, ref_out, margin=crop)
        grid["rgb"][eps] = psnr(P.run(test_in, ctx, v["denoise after demosaic"]).image, ref_out, margin=crop)
    best = {k: max(g, key=g.get) for k, g in grid.items()}
    return best["raw"], best["rgb"], grid


def main():
    ap = cliutil.parser(__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--frames", type=int, default=8, help="how many frames of the burst to use as inputs")
    ap.add_argument("--border", type=int, default=4, help="extra border excluded from PSNR (px)")
    ap.add_argument("--no-tune", action="store_true", help="skip denoiser strength tuning (eps_scale = 4)")
    args = ap.parse_args()
    cfg, calib = cliutil.load(args)

    sb = load_saved_burst(args.reference)
    burst = sb.burst
    h2, w2 = burst.packed.shape[1:3]
    full_shape = (2 * h2, 2 * w2)
    ctx = build_context(cfg, calib, sb.black_rggb, sb.white, full_shape, "quarter", iso=sb.iso,
                        crop_origin=sb.crop_origin)
    full_mask = None
    if calib is not None and calib.defect_mask is not None:
        y0, x0 = sb.crop_origin
        full_mask = calib.defect_mask[y0:y0 + full_shape[0], x0:x0 + full_shape[1]]

    crop = burst.margin + args.border

    clean = burst.defect_corrected(full_mask)  # reference side only; the test input keeps its defects

    def pair_for(k):
        ref_dn = unpack_rggb(clean.reference(k))
        ref_out = P.run(quarter_res(ref_dn), ctx, REFERENCE_STEPS).image
        return mosaic_rggb(quarter_res(unpack_rggb(burst.packed[k]))), ref_out

    tune_k = burst.n - 1
    if args.no_tune:
        eps_raw = eps_rgb = 4.0
        grid = None
    else:
        eps_raw, eps_rgb, grid = tune(*pair_for(tune_k), ctx, crop)
        print(f"tuned on frame {tune_k}: eps_scale {eps_raw} on packed Bayer, {eps_rgb} after demosaic")
    candidates = [k for k in range(burst.n) if args.no_tune or k != tune_k]
    ks = np.array(candidates)[np.unique(np.linspace(0, len(candidates) - 1, min(args.frames, len(candidates)))
                                        .round().astype(int))]
    chosen = variants(eps_raw, eps_rgb)
    scores: dict[str, list[float]] = {name: [] for name in chosen}
    for k in ks:
        test_in, ref_out = pair_for(k)
        for name, steps in chosen.items():
            out = P.run(test_in, ctx, steps).image
            scores[name].append(psnr(out, ref_out, peak=1.0, margin=crop))
        print(f"frame {k}: full {scores['full pipeline'][-1]:.2f} dB")

    table = {}
    full = np.array(scores["full pipeline"])
    full_mse = 10 ** (-full / 10)
    for name, vals in scores.items():
        v = np.array(vals)
        m, sd, n = mean_std(v)
        d = v - full  # paired: same input frames, so frame-to-frame changes (e.g. light drift) cancel
        dm, dsd, _ = mean_std(d)
        added = 10 ** (-v / 10) - full_mse  # MSE this variant adds on top of the full pipeline's
        am, asd, _ = mean_std(added)
        table[name] = {"psnr_db": [m, sd], "delta_vs_full_db": [dm, dsd],
                       "added_mse": [am, asd], "n_frames": n, "per_frame": vals}
    report = {"reference": args.reference, "iso": sb.iso, "frames_used": ks.tolist(),
              "compared_in": "display sRGB before sharpening, peak 1.0",
              "uncertainty": "std over input frames (ddof=1); delta and added MSE are paired per frame",
              "excluded_border_px": crop,
              "denoiser_eps_scale": {"packed": eps_raw, "after_demosaic": eps_rgb, "tuned_on_frame": None if
                                     args.no_tune else tune_k, "grid_psnr_db": grid},
              "variants": table}
    tag = f"iso{sb.iso}" if sb.iso else "burst"
    out_dir = cliutil.results(cfg)
    (out_dir / f"ablation_{tag}.json").write_text(json.dumps(report, indent=2))

    lines = [f"| Variant (ISO {sb.iso}) | PSNR (dB) | Δ vs full (dB, paired) | Added MSE |", "|---|---|---|---|"]
    for name, row in sorted(table.items(), key=lambda kv: -kv[1]["psnr_db"][0]):
        dm, dsd = row["delta_vs_full_db"]
        am, asd = row["added_mse"]
        dtxt = "0 (reference)" if name == "full pipeline" else f"{dm:+.2f} ± {dsd:.2f}"
        atxt = "—" if name == "full pipeline" else f"{am:.2e} ± {asd:.1e}"
        lines.append(f"| {name} | {cliutil.fmt(row['psnr_db'])} | {dtxt} | {atxt} |")
    md = "\n".join(lines)
    (out_dir / f"ablation_{tag}.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()