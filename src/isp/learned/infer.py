"""Run a trained denoiser (PyTorch or ONNX) on a whole packed Bayer image, in tiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..noise_model import NoiseModel


def tiled(fn, packed: np.ndarray, tile: int = 512, pad: int = 16) -> np.ndarray:
    """Apply ``fn`` ((h, w, 4) -> (h, w, 4)) over overlapping tiles; keeps memory bounded on 24 MP frames.

    Each tile is given up to ``pad`` pixels of real context on every side. No
    artificial padding is added at the image border, so the network sees the
    same zero-padded boundary it would on the whole image, and the result equals
    the whole-image result when ``pad`` covers the receptive field (tested).
    """
    h, w, _ = packed.shape
    if h <= tile and w <= tile:
        return fn(packed)
    out = np.empty_like(packed)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            th, tw = min(tile, h - y), min(tile, w - x)
            y0, x0 = max(0, y - pad), max(0, x - pad)
            y1, x1 = min(h, y + th + pad), min(w, x + tw + pad)
            res = fn(packed[y0:y1, x0:x1])
            out[y:y + th, x:x + tw] = res[y - y0:y - y0 + th, x - x0:x - x0 + tw]
    return out


class TorchDenoiser:
    def __init__(self, checkpoint, noise: NoiseModel, device: str = "cpu", tile: int = 512):
        import torch

        from .model import DnCNN

        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model = DnCNN(**ck["model_config"]).to(device).eval()
        self.model.load_state_dict(ck["state_dict"])
        self.noise = noise
        self.device = device
        self.tile = tile
        self.pad = ck["model_config"]["depth"] + 4
        self.torch = torch

    def _run(self, p: np.ndarray) -> np.ndarray:
        t = self.torch
        x = t.from_numpy(np.ascontiguousarray(p.transpose(2, 0, 1)))[None].to(self.device)
        with t.no_grad():
            sigma = self.noise.std(x) if self.model.noise_map else None
            y = self.model(x, sigma)
        return y[0].cpu().numpy().transpose(1, 2, 0)

    def __call__(self, packed: np.ndarray) -> np.ndarray:
        return tiled(self._run, packed.astype(np.float32), self.tile, self.pad)


class OnnxDenoiser:
    def __init__(self, onnx_path, noise: NoiseModel, noise_map: bool = True, tile: int = 512, pad: int = 16,
                 providers=("CPUExecutionProvider",)):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(Path(onnx_path)), providers=list(providers))
        self.noise = noise
        self.noise_map = noise_map
        self.tile = tile
        self.pad = pad

    def _run(self, p: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(p.transpose(2, 0, 1))[None].astype(np.float32)
        feeds = {"noisy": x}
        if self.noise_map:
            feeds["sigma"] = self.noise.std(x).astype(np.float32)
        return self.sess.run(None, feeds)[0][0].transpose(1, 2, 0)

    def __call__(self, packed: np.ndarray) -> np.ndarray:
        return tiled(self._run, packed.astype(np.float32), self.tile, self.pad)
