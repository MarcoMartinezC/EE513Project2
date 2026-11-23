"""Train the packed-Bayer denoiser on synthetic Poisson-Gaussian noise from your measured model.

    python scripts/train_denoiser.py --isos 100 800 3200 --steps 3000 --out runs/dncnn_all
    python scripts/train_denoiser.py --isos 3200 --out runs/dncnn_3200   # single-level, for the mismatch study

The noise models come from ``noise.iso`` in the config (your Project 1 values).
Writes model.pt, loss.csv, train_meta.json and loss.png into --out.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from isp import cliutil
from isp import config as C
from isp.learned.data import load_pool
from isp.learned.train import train
from isp.noise_model import NoiseModel


def main():
    ap = cliutil.parser(__doc__)
    ap.add_argument("--pool", default=None, help="clean pool (default: data/derived/clean_pool.npz)")
    ap.add_argument("--isos", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--no-noise-map", action="store_true")
    ap.add_argument("--gain-jitter", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg, _ = cliutil.load(args)

    pool = load_pool(args.pool or cliutil.derived(cfg) / "clean_pool.npz")
    black = float(np.mean(C.black_rggb(cfg))) if C.black_rggb(cfg) is not None else None
    white = C.white_level(cfg)
    if black is None or white is None:
        raise SystemExit("set raw.black_level_dn and raw.white_level_dn in the config")
    models = [NoiseModel.from_config(cfg, iso, black, white) for iso in args.isos]

    meta = train(pool, models, args.out, steps=args.steps, batch=args.batch, patch=args.patch, lr=args.lr,
                 width=args.width, depth=args.depth, noise_map=not args.no_noise_map, seed=args.seed,
                 device=args.device, gain_jitter=args.gain_jitter)
    meta["isos"] = args.isos
    out = Path(args.out)
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))

    loss = np.loadtxt(out / "loss.csv", delimiter=",", skiprows=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(loss[:, 0], loss[:, 1], lw=0.6, alpha=0.5, label="per step")
    k = max(1, len(loss) // 50)
    ax.semilogy(loss[k - 1:, 0], np.convolve(loss[:, 1], np.ones(k) / k, mode="valid"), label=f"mean of {k}")
    ax.set_xlabel("step")
    ax.semilogy(loss[:, 0], loss[:, 2], lw=0.6, alpha=0.4, color="grey", label="noisy input MSE")
    ax.set_ylabel("MSE (normalised units)")
    ax.set_title(f"Training loss, ISOs {args.isos}, {meta['parameters']:,} parameters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=130)
    print(f"{meta['parameters']:,} parameters, {meta['pairs_seen']:,} pairs, "
          f"{meta['train_seconds']:.0f} s on {meta['device']} -> {out}/model.pt")


if __name__ == "__main__":
    main()
