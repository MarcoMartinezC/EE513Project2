"""End-to-end time of the whole pipeline on one full-resolution frame, per stage (Part C).

    python scripts/benchmark_pipeline.py data/detail/DSC_0001.NEF --runs 5 [--denoise learned --model ...]

Mean and standard deviation per stage over --runs runs after one discarded
warm-up. File decoding is timed separately from the pipeline.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np

from isp import cliutil
from isp import config as C
from isp import pipeline as P
from isp.context import build_context
from isp.rawio import load_raw


def main():
    ap = cliutil.parser(__doc__)
    ap.add_argument("raw")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--iso", type=int, default=None)
    ap.add_argument("--denoise", choices=["guided", "learned", "none"], default="guided")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg, calib = cliutil.load(args)

    t0 = time.perf_counter()
    frame = load_raw(args.raw, C.black_rggb(cfg), C.white_level(cfg))
    t_load = time.perf_counter() - t0
    iso = args.iso or cliutil.iso_from_path(args.raw)
    ctx = build_context(cfg, calib, frame.black_rggb, frame.white, frame.mosaic.shape, "full", iso=iso)
    steps = P.with_params(P.DEFAULT_STEPS, "denoise", method=args.denoise)
    if args.denoise == "learned":
        from isp.learned.infer import OnnxDenoiser, TorchDenoiser

        ctx.learned_denoiser = (OnnxDenoiser(args.model, ctx.noise) if args.model.endswith(".onnx")
                                else TorchDenoiser(args.model, ctx.noise))

    P.run(frame.mosaic, ctx, steps)  # warm-up, discarded
    per_stage: dict[str, list[float]] = {}
    totals = []
    for _ in range(args.runs):
        r = P.run(frame.mosaic, ctx, steps)
        for name, t in r.timings:
            per_stage.setdefault(name, []).append(1e3 * t)
        totals.append(1e3 * sum(t for _, t in r.timings))

    h, w = frame.mosaic.shape
    report = {"raw": args.raw, "sensor_hw": [h, w], "runs": args.runs, "warmup_discarded": 1,
              "decode_ms_single": 1e3 * t_load,
              "stages_ms": {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in per_stage.items()},
              "total_ms": [float(np.mean(totals)), float(np.std(totals, ddof=1))],
              "hardware": {"machine": platform.machine(), "system": platform.platform()},
              "denoiser": args.denoise}
    out = cliutil.results(cfg) / f"latency_{Path(args.raw).stem}_{args.denoise}.json"
    out.write_text(json.dumps(report, indent=2))

    total = report["total_ms"][0]
    print(f"{w}x{h} sensor px ({h * w / 1e6:.1f} MP), {args.runs} runs, decode {report['decode_ms_single']:.0f} ms")
    for k, (m, s) in sorted(report["stages_ms"].items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:16s} {m:9.1f} ± {s:6.1f} ms  {100 * m / total:5.1f}%")
    print(f"  {'total':16s} {total:9.1f} ± {report['total_ms'][1]:6.1f} ms -> {out}")


if __name__ == "__main__":
    main()
