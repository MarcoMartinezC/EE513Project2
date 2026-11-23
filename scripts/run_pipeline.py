"""Raw file in, displayable image out.

    python scripts/run_pipeline.py data/detail/DSC_0001.NEF -o results/detail.png
    python scripts/run_pipeline.py RAW --denoise learned --model runs/dncnn/model.pt --iso 3200
    python scripts/run_pipeline.py RAW --save-stages results/stages/   # before/after of every stage

Prints the time each stage took.
"""

from pathlib import Path

import numpy as np

from isp import cliutil
from isp import config as C
from isp import pipeline as P
from isp.context import build_context, center_crop_mosaic
from isp.rawio import load_raw


def main():
    ap = cliutil.parser(__doc__)
    ap.add_argument("raw")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--iso", type=int, default=None, help="selects the noise model (default: from the path)")
    ap.add_argument("--demosaic", choices=["bilinear", "mhc"], default="mhc")
    ap.add_argument("--denoise", choices=["guided", "learned", "none"], default="guided")
    ap.add_argument("--model", help="checkpoint (.pt) or ONNX file for --denoise learned")
    ap.add_argument("--minimal", action="store_true", help="black level, shading, WB, demosaic only")
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--save-stages", default=None, help="folder for a PNG after every stage")
    ap.add_argument("--bits", type=int, choices=[8, 16], default=8)
    args = ap.parse_args()

    cfg, calib = cliutil.load(args)
    frame = load_raw(args.raw, C.black_rggb(cfg), C.white_level(cfg))
    mosaic, origin = center_crop_mosaic(frame.mosaic, args.crop)
    iso = args.iso or cliutil.iso_from_path(args.raw)
    ctx = build_context(cfg, calib, frame.black_rggb, frame.white, mosaic.shape, "full", iso=iso,
                        crop_origin=origin)

    steps = P.MINIMAL_STEPS if args.minimal else P.DEFAULT_STEPS
    steps = P.with_params(steps, "demosaic", method=args.demosaic)
    if not args.minimal:
        steps = P.with_params(steps, "denoise", method=args.denoise)
    if ctx.lsc_rgb is None:
        print("[warn] no shading map (run scripts/calibrate.py); skipping lens shading correction")
        steps = P.without(steps, "lsc")
    if args.denoise == "learned":
        if not args.model:
            ap.error("--denoise learned needs --model")
        if ctx.noise is None:
            ap.error("the learned denoiser needs a noise model: pass --iso and fill noise.iso in the config")
        from isp.learned.infer import OnnxDenoiser, TorchDenoiser

        ctx.learned_denoiser = (OnnxDenoiser(args.model, ctx.noise) if args.model.endswith(".onnx")
                                else TorchDenoiser(args.model, ctx.noise))

    res = P.run(mosaic, ctx, steps, keep=bool(args.save_stages))

    out = Path(args.out or Path(cliutil.results(cfg)) / (Path(args.raw).stem + ".png"))
    cliutil.save_png(out, res.image, args.bits)
    total = sum(t for _, t in res.timings)
    print(f"{mosaic.shape[1]}x{mosaic.shape[0]} sensor px, ISO {iso}")
    for name, t in res.timings:
        print(f"  {name:16s} {1e3 * t:9.1f} ms  {100 * t / total:5.1f}%")
    print(f"  {'total':16s} {1e3 * total:9.1f} ms  -> {out}")

    if args.save_stages:
        _save_stages(Path(args.save_stages), res, ctx)


def _save_stages(folder: Path, res: P.Result, ctx: P.Context) -> None:
    """Every intermediate as a PNG; mosaics are shown as grey, linear stages with a display gamma."""
    for i, (name, img) in enumerate(res.intermediates.items()):
        view = img if img.ndim == 3 else np.repeat(img[..., None], 3, axis=-1)
        if name not in ("tone", "sharpen"):
            view = P.srgb_oetf(np.clip(view, 0, None) / max(np.percentile(view, 99.5), 1e-6))
        cliutil.save_png(folder / f"{i:02d}_{name}.png", view)
    print(f"stages -> {folder}/")


if __name__ == "__main__":
    main()
