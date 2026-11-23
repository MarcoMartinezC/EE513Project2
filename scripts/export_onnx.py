"""Export the learned denoiser to ONNX, check it against PyTorch, and time it (Part C).

    python scripts/export_onnx.py runs/dncnn_all/model.pt --sizes 256 512 1024 --runs 20

Latency is the mean and standard deviation over --runs runs after one
discarded warm-up run. Sizes are packed-Bayer sizes (sensor size is twice
that in each direction). Also times a full-resolution frame in tiles if
--full HxW (packed) is given.
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from isp.learned.export import compare_onnx, export_onnx, load_checkpoint, onnx_latency, time_callable


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--full", type=int, nargs=2, metavar=("H", "W"), default=None,
                    help="also time a full frame of this packed size through tiled inference")
    ap.add_argument("--gpu", action="store_true", help="also try the CUDA/CoreML execution provider")
    args = ap.parse_args()

    model = load_checkpoint(args.checkpoint)
    onnx_path = Path(args.out or Path(args.checkpoint).with_suffix(".onnx"))
    export_onnx(model, onnx_path)
    diff = compare_onnx(model, onnx_path)
    report = {
        "onnx": str(onnx_path),
        "max_abs_diff_vs_pytorch": diff,
        "float32_eps": float(np.finfo(np.float32).eps),
        "hardware": {"machine": platform.machine(), "processor": platform.processor(),
                     "system": platform.platform(), "python": platform.python_version()},
        "latency_cpu": [],
    }
    print(f"max |PyTorch - onnxruntime| = {diff:.3e}  (float32 eps {np.finfo(np.float32).eps:.3e})")
    for s in args.sizes:
        r = onnx_latency(onnx_path, model.in_ch, model.noise_map, s, s, args.runs, threads=args.threads)
        report["latency_cpu"].append(r)
        print(f"  packed {s}x{s} (sensor {2 * s}x{2 * s}): {r['mean_ms']:.1f} ± {r['std_ms']:.1f} ms")

    if args.full:
        from isp.learned.infer import OnnxDenoiser
        from isp.noise_model import NoiseModel

        h, w = args.full
        den = OnnxDenoiser(onnx_path, NoiseModel(1.0, 2.0, 0.0, 1000.0), model.noise_map)
        x = np.random.default_rng(0).random((h, w, 4), dtype=np.float32) * 0.5
        r = time_callable(lambda: den(x), n_runs=max(3, args.runs // 5))
        r["packed_hw"] = [h, w]
        report["latency_full_frame_tiled"] = r
        print(f"  full frame packed {h}x{w}, tiled: {r['mean_ms']:.0f} ± {r['std_ms']:.0f} ms")

    if args.gpu:
        import onnxruntime as ort

        avail = ort.get_available_providers()
        for prov in ("CUDAExecutionProvider", "CoreMLExecutionProvider"):
            if prov in avail:
                r = onnx_latency(onnx_path, model.in_ch, model.noise_map, args.sizes[-1], args.sizes[-1],
                                 args.runs, providers=(prov, "CPUExecutionProvider"))
                report.setdefault("latency_accelerated", []).append(r)
                print(f"  {prov}: {r['mean_ms']:.1f} ± {r['std_ms']:.1f} ms")
        if "latency_accelerated" not in report:
            print("  no GPU execution provider available; reporting CPU only")

    out_json = onnx_path.with_name("onnx_report.json")
    out_json.write_text(json.dumps(report, indent=2))
    print(f"-> {onnx_path}, {out_json}")


if __name__ == "__main__":
    main()
