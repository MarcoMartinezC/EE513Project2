"""Classical vs learned denoising on real burst pairs (Part B).

    python scripts/eval_denoisers.py --reference data/derived/burst_iso3200.npz \\
        --model all=runs/dncnn_all/model.pt --model only3200=runs/dncnn_3200/model.pt [--lpips]

For each input frame k: noisy = frame k (never resampled), reference = mean of
the other frames aligned to k. Methods: the noisy input itself, the guided
filter, and every --model given. Metrics:

* raw domain: PSNR and SSIM on packed Bayer (normalised, peak 1.0);
* display domain: PSNR, SSIM and (with --lpips) LPIPS after the identical
  downstream pipeline (shading, WB, clip, MHC, CCM, sRGB curve; no sharpening)
  applied to every method and to the reference.

Mean and standard deviation over input frames. The guided filter's strength is
tuned on the last frame of the burst, which is then excluded from scoring.

Also writes a failure-crop figure: the tiles where the learned model does
worst relative to the guided filter, with a detail ratio (high-frequency
energy of the output over that of the reference). Above 1 suggests texture
that is not in the reference (fabricated); below 1, detail that was erased.
Look at the crops before believing either label.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace

from isp import cliutil
from isp import pipeline as P
from isp.context import build_context
from isp.denoise_classical import denoise_packed
from isp.groundtruth import load_saved_burst
from isp.metrics import LPIPS, mean_std, psnr, ssim, tile_errors
from isp.rawio import unpack_rggb

DOWNSTREAM = [("lsc", {}), ("wb", {}), ("clip_highlights", {}), ("demosaic", {"method": "mhc"}),
              ("ccm", {}), ("tone", {"curve": "srgb", "exposure": 1.0})]
EPS_GRID = (0.5, 1, 2, 4, 8, 16, 32, 64)


def detail_ratio(est_rgb, ref_rgb) -> float:
    y = lambda a: a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return float(np.mean(laplace(y(est_rgb)) ** 2) / max(np.mean(laplace(y(ref_rgb)) ** 2), 1e-12))


def main():
    ap = cliutil.parser(__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--model", action="append", default=[], help="label=path/to/model.pt (or .onnx); repeatable")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--tile", type=int, default=96, help="tile size for the failure crops (sensor px)")
    args = ap.parse_args()
    cfg, calib = cliutil.load(args)

    sb = load_saved_burst(args.reference)
    burst = sb.normalised()
    full_shape = (2 * burst.packed.shape[1], 2 * burst.packed.shape[2])
    ctx = build_context(cfg, calib, sb.black_rggb, sb.white, full_shape, "full", iso=sb.iso,
                        crop_origin=sb.crop_origin)
    if ctx.noise is None:
        raise SystemExit(f"no noise model for ISO {sb.iso} in the config; the comparison needs one")
    # Defects are fixed in every frame before averaging, for inputs and references alike:
    # hot pixels are not what this comparison is about.
    burst = burst.defect_corrected(ctx.defect_mask)

    from isp.learned.infer import OnnxDenoiser, TorchDenoiser

    learned = {}
    for spec in args.model:
        label, path = spec.split("=", 1) if "=" in spec else (Path(spec).parent.name, spec)
        learned[label] = OnnxDenoiser(path, ctx.noise) if path.endswith(".onnx") else TorchDenoiser(path, ctx.noise)

    m = burst.margin
    inner = lambda a, s=1: a[m * s:-m * s, m * s:-m * s] if m else a

    # Tune the guided filter on the last frame, then leave it out.
    tune_k = burst.n - 1
    noisy_t, ref_t = burst.packed[tune_k], burst.reference(tune_k)
    grid = {e: psnr(inner(denoise_packed(noisy_t, 2, e, ctx.noise)), inner(ref_t)) for e in EPS_GRID}
    eps = max(grid, key=grid.get)
    print(f"guided filter eps_scale = {eps} (tuned on frame {tune_k})")

    lp = LPIPS() if args.lpips else None
    ks = np.unique(np.linspace(0, burst.n - 2, min(args.frames, burst.n - 1)).round().astype(int))
    methods = ["noisy", "guided", *learned]
    metrics: dict = {mth: {k: [] for k in ("raw_psnr", "raw_ssim", "rgb_psnr", "rgb_ssim", "rgb_lpips",
                                            "detail_ratio")} for mth in methods}
    worst = None
    check = []  # measured vs predicted std of (noisy - reference)
    for k in ks:
        noisy, ref = burst.packed[k], burst.reference(k)
        predicted = float(np.sqrt(np.mean(ctx.noise.variance(inner(ref))) * (1 + 1 / (burst.n - 1))))
        check.append((float(np.std(inner(noisy - ref))), predicted))
        outs = {"noisy": noisy, "guided": denoise_packed(noisy, 2, eps, ctx.noise)}
        for label, fn in learned.items():
            outs[label] = fn(noisy)
        ref_rgb = P.run(unpack_rggb(ref), ctx, DOWNSTREAM).image
        rgbs = {}
        for mth, out in outs.items():
            rgb = P.run(unpack_rggb(out), ctx, DOWNSTREAM).image
            rgbs[mth] = rgb
            r = metrics[mth]
            r["raw_psnr"].append(psnr(inner(out), inner(ref)))
            r["raw_ssim"].append(ssim(inner(out), inner(ref)))
            r["rgb_psnr"].append(psnr(inner(rgb, 2), inner(ref_rgb, 2)))
            r["rgb_ssim"].append(ssim(inner(rgb, 2), inner(ref_rgb, 2)))
            r["detail_ratio"].append(detail_ratio(inner(rgb, 2), inner(ref_rgb, 2)))
            if lp is not None:
                r["rgb_lpips"].append(lp(inner(rgb, 2), inner(ref_rgb, 2)))
        print(f"frame {k}: " + "  ".join(f"{mth} {metrics[mth]['rgb_psnr'][-1]:.2f}" for mth in methods) + " dB (display)")
        if learned and worst is None:
            worst = (k, {mth: inner(v, 2) for mth, v in rgbs.items()}, inner(ref_rgb, 2))

    summary = {mth: {name: (list(mean_std(v)[:2]) if v else None) for name, v in r.items()}
               for mth, r in metrics.items()}
    measured, predicted = np.mean(check, axis=0)
    print(f"reference check: std(noisy - reference) = {measured:.5f}, noise model predicts {predicted:.5f} "
          f"(ratio {measured / predicted:.2f})")
    if measured > 1.15 * predicted:
        print("[WARNING] the residual is larger than the noise model allows: the reference has errors of its own "
              "(misalignment, resampling of fine detail, motion) or the noise model is wrong for this ISO. "
              "No denoiser can remove that part; it caps every PSNR in the table.")
    report = {"reference": args.reference, "iso": sb.iso, "frames_scored": ks.tolist(),
              "reference_check": {"std_noisy_minus_reference": measured, "predicted_by_noise_model": predicted},
              "guided_eps_scale": eps, "guided_tuning_frame": tune_k, "guided_grid_raw_psnr": grid,
              "uncertainty": "std over input frames (ddof=1)", "peak": 1.0,
              "models": {lbl: s for lbl, s in (sp.split("=", 1) if "=" in sp else (Path(sp).parent.name, sp)
                                               for sp in args.model)},
              "summary": summary, "per_frame": metrics}
    tag = f"iso{sb.iso}"
    res = cliutil.results(cfg)
    (res / f"denoise_{tag}.json").write_text(json.dumps(report, indent=2))

    header = "| Method | raw PSNR (dB) | display PSNR (dB) | display SSIM | LPIPS | detail ratio |"
    lines = [header, "|---|---|---|---|---|---|"]
    for mth in methods:
        s = summary[mth]
        lp_txt = cliutil.fmt(s["rgb_lpips"], 3) if s["rgb_lpips"] else "–"
        lines.append(f"| {mth} | {cliutil.fmt(s['raw_psnr'])} | {cliutil.fmt(s['rgb_psnr'])} | "
                     f"{cliutil.fmt(s['rgb_ssim'], 3)} | {lp_txt} | {cliutil.fmt(s['detail_ratio'], 2)} |")
    (res / f"denoise_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if worst is not None:
        _failure_figure(worst, list(learned)[0], args.tile, res / f"denoise_failures_{tag}.png", sb.iso)


def _failure_figure(worst, label, tile, path, iso, n=3):
    k, rgbs, ref = worst
    e_l = tile_errors(rgbs[label], ref, tile)
    e_g = tile_errors(rgbs["guided"], ref, tile)
    order = np.argsort(-(e_l - e_g).ravel())[:n]
    cols = ["noisy", "guided", label]
    fig, axes = plt.subplots(n, 5, figsize=(16, 3.4 * n), squeeze=False)
    for row, idx in enumerate(order):
        ty, tx = np.unravel_index(idx, e_l.shape)
        sl = (slice(ty * tile, (ty + 1) * tile), slice(tx * tile, (tx + 1) * tile))
        for col, mth in enumerate(cols):
            axes[row, col].imshow(rgbs[mth][sl], interpolation="nearest")
            extra = f"\ndetail ratio {detail_ratio(rgbs[mth][sl], ref[sl]):.2f}" if mth != "noisy" else ""
            axes[row, col].set_title(f"{mth}{extra}", fontsize=9)
        axes[row, 3].imshow(ref[sl], interpolation="nearest")
        axes[row, 3].set_title(f"reference  tile ({ty},{tx})", fontsize=9)
        diff = np.abs(rgbs[label][sl] - ref[sl]).mean(-1)
        axes[row, 4].imshow(diff, cmap="magma", vmin=0, vmax=max(diff.max(), 1e-6))
        axes[row, 4].set_title(f"|{label} - ref|", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(f"ISO {iso}, frame {k}: tiles where '{label}' loses most to the guided filter")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"failure crops -> {path}")


if __name__ == "__main__":
    main()
