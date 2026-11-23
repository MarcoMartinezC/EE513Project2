"""Training loop for the packed-Bayer denoiser.

Sized so a CPU run finishes well under an hour: patches of 64x64 packed pixels
(128x128 of sensor), batch 16, a few thousand steps. Losses go to a CSV next to
the checkpoint so the loss curve can be plotted without rerunning.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..noise_model import NoiseModel
from .data import SyntheticPairs
from .model import DnCNN, count_parameters, pick_device


def train(clean: list[np.ndarray], noise_models: list[NoiseModel], out_dir, steps: int = 3000,
          batch: int = 16, patch: int = 64, lr: float = 1e-3, width: int = 32, depth: int = 8,
          noise_map: bool = True, seed: int = 0, device: str | None = None, gain_jitter: float = 0.0,
          log_every: int = 100) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    dev = torch.device(device) if device else pick_device()

    ds = SyntheticPairs(clean, noise_models, patch=patch, length=steps * batch, seed=seed,
                        gain_jitter=gain_jitter)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, drop_last=True)
    model = DnCNN(width=width, depth=depth, noise_map=noise_map).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log_path = out_dir / "loss.csv"
    t0 = time.time()
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        # loss_ratio = loss / MSE of the noisy input: 1.0 means the network has learned nothing.
        w.writerow(["step", "loss", "noisy_mse", "seconds"])
        model.train()
        recent = []
        for step, (noisy, sigma, target) in enumerate(dl):
            noisy, sigma, target = noisy.to(dev), sigma.to(dev), target.to(dev)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(noisy, sigma if noise_map else None), target)
            loss.backward()
            opt.step()
            sched.step()
            base = torch.nn.functional.mse_loss(noisy, target).item()
            recent = (recent + [loss.item() / max(base, 1e-20)])[-log_every:]
            w.writerow([step, f"{loss.item():.8g}", f"{base:.8g}", f"{time.time() - t0:.2f}"])
            if step % log_every == 0 or step == steps - 1:
                print(f"step {step:6d}  loss {loss.item():.3e}  loss/noisy {np.mean(recent):.3f}  "
                      f"{time.time() - t0:7.1f}s", flush=True)

    model.eval()
    alive = model.alive_fraction(noisy, sigma if noise_map else None)
    if min(alive) < 0.01:
        print(f"[WARNING] a ReLU layer is dead (alive fractions {[round(a, 3) for a in alive]}); the network "
              "cannot learn past this. Retrain with another --seed or a lower --lr.")
    if np.mean(recent) > 0.95:
        print(f"[WARNING] final loss is {np.mean(recent):.2f} of the noisy input's MSE: almost nothing was learned.")

    meta = {
        "model_config": model.config(),
        "parameters": count_parameters(model),
        "steps": steps, "batch": batch, "patch_packed_px": patch, "lr": lr, "seed": seed,
        "pairs_seen": steps * batch,
        "n_clean_images": len(clean),
        "noise_models": [{"gain_e_per_dn": m.gain_e_per_dn, "read_noise_dn": m.read_noise_dn}
                         for m in noise_models],
        "gain_jitter": gain_jitter,
        "device": str(dev),
        "train_seconds": time.time() - t0,
        "final_loss_over_noisy_mse": float(np.mean(recent)),
        "relu_alive_fractions": alive,
    }
    torch.save({"state_dict": model.cpu().state_dict(), "model_config": model.config(), "meta": meta},
               out_dir / "model.pt")
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
