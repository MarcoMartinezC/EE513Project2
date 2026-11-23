"""Score bilinear and MHC against the 2x2 re-mosaic ground truth.

    python scripts/score_demosaic.py --reference data/derived/burst_iso100.npz
    python scripts/score_demosaic.py --raw data/detail/DSC_0001.NEF   # detail scene, single frame

With --reference the ground truth is built from the averaged burst, as the spec
asks, and the same scoring is repeated on a single frame of that burst so you
can see the noise effect from Exercise 5 items 15-16 on your own data.

White balance is applied before the 2x2 construction by default, because that
is where the pipeline demosaics; --no-wb scores the raw channels instead.
Uncertainties are tile-bootstrap standard deviations (content variation across
the frame), stated as such in the JSON.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

from isp import cliutil
from isp import config as C
from isp import demosaic as dm
from isp.calib import normalise
from isp.groundtruth import load_saved_burst, mosaic_rggb, quarter_res
from isp.metrics import psnr_tile_bootstrap, ssim
from isp.pipeline import srgb_oetf
from isp.rawio import load_raw, tile_rggb, unpack_rggb


def score(gt: np.ndarray, margin: int) -> dict:
    """Re-mosaic ``gt``, demosaic both ways, return metrics and outputs."""
    m = mosaic_rggb(gt)
    out = {}
    for name in ("bilinear", "mhc"):
        est = dm.demosaic(m, name)
        whole = psnr_tile_bootstrap(est, gt)
        inner = psnr_tile_bootstrap(est[margin:-margin, margin:-margin], gt[margin:-margin, margin:-margin])
        out[name] = {
            "psnr_whole_db": list(whole),
            "psnr_interior_db": list(inner),
            "ssim_interior": ssim(est[margin:-margin, margin:-margin], gt[margin:-margin, margin:-margin]),
            "_img": est,
        }
    return out


def crop_figure(gt, bil, mhc, path, size=48, title=""):
    err_b = np.abs(bil - gt).mean(-1)
    err_m = np.abs(mhc - gt).mean(-1)
    gain = uniform_filter(err_b - err_m, size)
    pad = size
    gain[:pad], gain[-pad:], gain[:, :pad], gain[:, -pad:] = 0, 0, 0, 0
    picks = {"MHC helps most": np.unravel_index(np.argmax(gain), gain.shape),
             "MHC hurts most": np.unravel_index(np.argmin(gain), gain.shape)}
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4))
    vmax = max(np.percentile(err_b, 99.9), 1e-6)
    for row, (label, (cy, cx)) in enumerate(picks.items()):
        sl = (slice(cy - size // 2, cy + size // 2), slice(cx - size // 2, cx + size // 2))
        disp = lambda a: srgb_oetf(np.clip(a[sl] / max(gt.max(), 1e-6), 0, 1))
        panels = [(disp(bil), "bilinear"), (disp(mhc), "MHC"), (disp(gt), "reference"),
                  (err_b[sl], f"|bilinear - ref| mean {err_b[sl].mean():.4f}"),
                  (err_m[sl], f"|MHC - ref| mean {err_m[sl].mean():.4f}")]
        for ax, (img, t) in zip(axes[row], panels):
            ax.imshow(img, cmap="magma" if img.ndim == 2 else None, vmin=0, vmax=vmax if img.ndim == 2 else None,
                      interpolation="nearest")
            ax.set_title(t, fontsize=9)
            ax.axis("off")
        axes[row, 0].set_ylabel(label)
        axes[row, 0].text(-0.1, 0.5, f"{label}\n({cy}, {cx})", transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = cliutil.parser(__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--reference", help="aligned burst from build_reference.py")
    src.add_argument("--raw", help="a single raw frame (e.g. the detail scene)")
    ap.add_argument("--no-wb", action="store_true")
    ap.add_argument("--margin", type=int, default=4, help="border excluded for the interior score (px)")
    args = ap.parse_args()
    cfg, calib = cliutil.load(args)

    inputs = {}
    if args.reference:
        sb = load_saved_burst(args.reference)
        mask = None
        if calib is not None and calib.defect_mask is not None:
            y0, x0 = sb.crop_origin
            h, w = 2 * sb.burst.packed.shape[1], 2 * sb.burst.packed.shape[2]
            mask = calib.defect_mask[y0:y0 + h, x0:x0 + w]
        nb = sb.normalised().defect_corrected(mask)
        # All frames averaged (aligned to frame 0) vs frame 0 alone.
        inputs["averaged burst"] = unpack_rggb(nb.interior(nb.reference(0, leave_out=False)))
        inputs["single frame"] = unpack_rggb(nb.interior(nb.packed[0]))
        tag = Path(args.reference).stem.replace("burst_", "")
    else:
        f = load_raw(args.raw, C.black_rggb(cfg), C.white_level(cfg))
        m = normalise(f.mosaic, f.black_rggb, f.white)
        if calib is not None and calib.defect_mask is not None and calib.defect_mask.shape == m.shape:
            from isp.calib import correct_defects

            m = correct_defects(m, calib.defect_mask)
        inputs["single frame"] = m
        tag = Path(args.raw).stem

    wb = None if args.no_wb else C.wb_gains(cfg)
    report = {"source": args.reference or args.raw, "white_balanced": wb is not None, "peak": 1.0,
              "uncertainty": "tile bootstrap std over 64x64 tiles", "margin_px": args.margin, "results": {}}
    res_dir = cliutil.results(cfg)
    for label, m in inputs.items():
        if wb is not None:
            m = m * tile_rggb([wb[0], wb[1], wb[1], wb[2]], m.shape)
        gt = quarter_res(m)
        gt = gt[: gt.shape[0] // 2 * 2, : gt.shape[1] // 2 * 2]
        r = score(gt, args.margin)
        crop_figure(gt, r["bilinear"]["_img"], r["mhc"]["_img"],
                    res_dir / f"demosaic_crops_{tag}_{label.replace(' ', '_')}.png", title=f"{tag}, {label}")
        report["results"][label] = {k: {kk: vv for kk, vv in v.items() if kk != "_img"} for k, v in r.items()}
        b, mh = r["bilinear"]["psnr_interior_db"], r["mhc"]["psnr_interior_db"]
        print(f"{label:15s} interior PSNR  bilinear {cliutil.fmt(b)} dB   MHC {cliutil.fmt(mh)} dB   "
              f"gap {mh[0] - b[0]:+.2f} dB")

    (res_dir / f"demosaic_{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"-> {res_dir}/demosaic_{tag}.json and crop figures")


if __name__ == "__main__":
    main()
