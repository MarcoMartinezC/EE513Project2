"""Part C: ONNX export, PyTorch-vs-onnxruntime agreement, and latency with a spread."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .model import DnCNN


def load_checkpoint(path) -> DnCNN:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = DnCNN(**ck["model_config"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def export_onnx(model: DnCNN, out_path, opset: int = 18) -> Path:
    """Export with dynamic height and width. Inputs are named 'noisy' and (if used) 'sigma'."""
    out_path = Path(out_path)
    model = model.cpu().eval()
    x = torch.rand(1, model.in_ch, 64, 64)
    if model.noise_map:
        args, names = (x, torch.rand_like(x) * 0.02), ["noisy", "sigma"]
    else:
        args, names = (x,), ["noisy"]
    h = torch.export.Dim("h", min=16, max=4096)
    w = torch.export.Dim("w", min=16, max=4096)
    dyn = {"x": {2: h, 3: w}}
    if model.noise_map:
        dyn["sigma"] = {2: h, 3: w}
    torch.onnx.export(model, args, str(out_path), input_names=names, output_names=["denoised"],
                      dynamic_shapes=dyn, opset_version=opset, dynamo=True)
    return out_path


def compare_onnx(model: DnCNN, onnx_path, size: int = 128, seed: int = 0) -> float:
    """Largest absolute difference between PyTorch and onnxruntime on one random input."""
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    x = rng.random((1, model.in_ch, size, size), dtype=np.float32)
    s = (0.01 + 0.02 * rng.random(x.shape)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x), torch.from_numpy(s) if model.noise_map else None).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {"noisy": x, **({"sigma": s} if model.noise_map else {})}
    out = sess.run(None, feeds)[0]
    return float(np.abs(out - ref).max())


def time_callable(fn, n_runs: int = 20, warmup: int = 1) -> dict:
    """Mean and std of wall time in ms over ``n_runs``, after discarding ``warmup`` runs."""
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    t = np.asarray(t)
    return {"mean_ms": float(t.mean()), "std_ms": float(t.std(ddof=1)), "min_ms": float(t.min()),
            "n_runs": n_runs, "warmup_discarded": warmup}


def onnx_latency(onnx_path, in_ch: int, noise_map: bool, height: int, width: int, n_runs: int = 20,
                 providers=("CPUExecutionProvider",), threads: int | None = None) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx_path), so, providers=list(providers))
    x = np.random.default_rng(0).random((1, in_ch, height, width), dtype=np.float32)
    feeds = {"noisy": x, **({"sigma": np.full_like(x, 0.01)} if noise_map else {})}
    res = time_callable(lambda: sess.run(None, feeds), n_runs=n_runs)
    res.update({"input_packed_hw": [height, width], "sensor_hw": [2 * height, 2 * width],
                "providers": sess.get_providers(), "intra_op_threads": threads})
    return res
