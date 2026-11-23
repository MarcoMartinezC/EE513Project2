"""Run a pretrained detector on several renderings of each detector scene (Part C).

    python scripts/detect_compare.py --scenes data/detector --threshold 0.5

Each scene is a raw file plus the camera's JPEG with the same stem
(scene01.NEF + scene01.JPG). Renderings:

    camera_jpeg   the camera's own JPEG
    ours          the full pipeline (guided denoiser unless --model is given)
    minimal       black level, shading, WB, bilinear demosaic, nothing else (linear light)
    plus one rendering per stage removed from ours, to see which stages help,
    are neutral, or hurt: no_denoise, no_ccm, no_sharpen, linear_tone

Reported per rendering at a fixed confidence threshold: number of detections,
mean confidence, and agreement with the reference rendering (--reference,
camera_jpeg by default): matched boxes (same class, IoU >= 0.5), precision,
recall, F1. None of the renderings is ground truth; the report has to say why
the chosen reference is defensible.
"""

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from isp import cliutil
from isp import config as C
from isp import pipeline as P
from isp.context import build_context
from isp.detect import Detector, agreement, summarise
from isp.metrics import mean_std
from isp.rawio import find_raws, load_raw


def renderings(base: list) -> dict[str, list]:
    return {
        "ours": base,
        "minimal": P.MINIMAL_STEPS,
        "no_denoise": P.with_params(base, "denoise", method="none"),
        "no_ccm": P.without(base, "ccm"),
        "no_sharpen": P.without(base, "sharpen"),
        "linear_tone": P.with_params(base, "tone", curve="linear"),
    }


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def main():
    ap = cliutil.parser(__doc__)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--reference", default="camera_jpeg")
    ap.add_argument("--detector", default="fasterrcnn_resnet50_fpn_v2")
    ap.add_argument("--max-side", type=int, default=1600, help="downscale before detection (speed)")
    ap.add_argument("--model", default=None, help="learned denoiser for 'ours'")
    ap.add_argument("--iso", type=int, default=None)
    args = ap.parse_args()
    cfg, calib = cliutil.load(args)

    scene_dir = Path(args.scenes or C.resolve(cfg, cfg["paths"]["detector"]))
    raws = find_raws(scene_dir)
    det = Detector(args.detector)
    res_dir = cliutil.results(cfg)
    fig_dir = res_dir / "detector"
    fig_dir.mkdir(exist_ok=True)

    per_scene = {}
    for raw_path in raws:
        jpgs = [p for ext in (".JPG", ".jpg", ".jpeg", ".JPEG") if (p := raw_path.with_suffix(ext)).exists()]
        if not jpgs:
            print(f"[skip] {raw_path.name}: no matching JPEG")
            continue
        frame = load_raw(raw_path, C.black_rggb(cfg), C.white_level(cfg))
        ctx = build_context(cfg, calib, frame.black_rggb, frame.white, frame.mosaic.shape, "full",
                            iso=args.iso or cliutil.iso_from_path(raw_path))
        base = P.DEFAULT_STEPS
        if args.model:
            from isp.learned.infer import TorchDenoiser

            ctx.learned_denoiser = TorchDenoiser(args.model, ctx.noise)
            base = P.with_params(base, "denoise", method="learned")

        images = {"camera_jpeg": resize_max(cliutil.load_rgb(jpgs[0]), args.max_side)}
        for name, steps in renderings(base).items():
            images[name] = resize_max(P.run(frame.mosaic, ctx, steps).image, args.max_side)

        dets = {name: det(img).above(args.threshold) for name, img in images.items()}
        ref = dets[args.reference]
        per_scene[raw_path.stem] = {
            name: {**summarise(d), **({"vs_reference": agreement(d, ref, args.iou)} if name != args.reference else {}),
                   "labels": [det.categories[i] for i in d.labels]}
            for name, d in dets.items()
        }
        _overlay(images, dets, det.categories, fig_dir / f"{raw_path.stem}.jpg")
        print(raw_path.stem + ": " + "  ".join(f"{n} {len(d)}" for n, d in dets.items()))

    names = list(next(iter(per_scene.values())).keys()) if per_scene else []
    agg = {}
    for name in names:
        rows = [s[name] for s in per_scene.values()]
        conf = [r["mean_confidence"] for r in rows if np.isfinite(r["mean_confidence"])]
        agg[name] = {"count_per_scene": list(mean_std(r["count"] for r in rows)[:2]),
                     "mean_confidence": list(mean_std(conf)[:2]) if conf else None}
        if name != args.reference:
            agg[name]["f1_vs_reference"] = list(mean_std(r["vs_reference"]["f1"] for r in rows)[:2])
    report = {"detector": args.detector, "threshold": args.threshold, "iou": args.iou,
              "reference": args.reference, "n_scenes": len(per_scene),
              "uncertainty": "std over scenes (ddof=1)", "aggregate": agg, "per_scene": per_scene}
    (res_dir / "detector.json").write_text(json.dumps(report, indent=2))

    lines = [f"| Rendering | detections / scene | mean confidence | F1 vs {args.reference} |", "|---|---|---|---|"]
    for name, a in agg.items():
        f1 = cliutil.fmt(a["f1_vs_reference"]) if "f1_vs_reference" in a else "(reference)"
        conf = cliutil.fmt(a["mean_confidence"], 3) if a["mean_confidence"] else "–"
        lines.append(f"| {name} | {cliutil.fmt(a['count_per_scene'], 1)} | {conf} | {f1} |")
    (res_dir / "detector.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _overlay(images, dets, categories, path, show=("camera_jpeg", "ours", "minimal")):
    fig, axes = plt.subplots(1, len(show), figsize=(6 * len(show), 4.5))
    for ax, name in zip(axes, show):
        img = images[name]
        h, w = img.shape[:2]
        ax.imshow(np.clip(img, 0, 1))
        for (x0, y0, x1, y1), s, l in zip(dets[name].boxes, dets[name].scores, dets[name].labels):
            ax.add_patch(plt.Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h, fill=False, lw=1.5, ec="yellow"))
            ax.text(x0 * w, y0 * h, f"{categories[l]} {s:.2f}", color="yellow", fontsize=7, va="bottom")
        ax.set_title(f"{name}: {len(dets[name])} detections")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
