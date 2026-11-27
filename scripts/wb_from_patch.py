"""White balance gains from a white or grey patch in a raw frame (e.g. the white paper in the burst scene).

    python scripts/wb_from_patch.py data/burst/iso100/DSC_xxxx.NEF --preview    # grid image to locate the patch
    python scripts/wb_from_patch.py data/burst/iso100/DSC_xxxx.NEF --box 1200 1500 3000 3400

--box is ROW0 ROW1 COL0 COL1 in sensor pixels (as on the preview grid). The
patch is black-subtracted, shading-corrected (if calib.npz exists), and
collapsed 2x2 to [R, mean(G1, G2), B], as in Project 1; gains are G/R and G/B.
Several frames can be given and are averaged. Prints a white_balance block for
the config, with the spread across frames as the uncertainty.
"""

from pathlib import Path

import numpy as np

from isp import cliutil
from isp import config as C
from isp.calib import lsc_rgb_quarter, normalise
from isp.groundtruth import quarter_res
from isp.rawio import load_raw


def main():
    ap = cliutil.parser(__doc__)
    ap.add_argument("raws", nargs="+")
    ap.add_argument("--box", type=int, nargs=4, metavar=("ROW0", "ROW1", "COL0", "COL1"))
    ap.add_argument("--preview", action="store_true", help="save a downscaled image with a coordinate grid")
    args = ap.parse_args()
    cfg, calib = cliutil.load(args)

    if args.preview:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        f = load_raw(args.raws[0], C.black_rggb(cfg), C.white_level(cfg))
        q = quarter_res(normalise(f.mosaic, f.black_rggb, f.white))
        q = q / max(np.percentile(q, 99.5), 1e-6)
        h, w = f.mosaic.shape
        fig, ax = plt.subplots(figsize=(14, 14 * h / w))
        ax.imshow(np.clip(q, 0, 1) ** (1 / 2.2), extent=(0, w, h, 0))
        ax.set_xticks(range(0, w, 250), minor=True)
        ax.set_yticks(range(0, h, 250), minor=True)
        ax.grid(True, which="both", color="yellow", alpha=0.35, lw=0.5)
        ax.set_xticks(range(0, w + 1, 500))
        ax.set_yticks(range(0, h + 1, 500))
        ax.set_title(f"{Path(args.raws[0]).name}: sensor pixel coordinates (rows down, columns across)")
        out = cliutil.results(cfg) / f"wb_preview_{Path(args.raws[0]).stem}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"preview -> {out}  (colours are not white balanced)")
        if not args.box:
            return

    if not args.box:
        ap.error("give --box ROW0 ROW1 COL0 COL1 (use --preview to find it)")
    r0, r1, c0, c1 = args.box
    gains, levels = [], []
    for p in args.raws:
        f = load_raw(p, C.black_rggb(cfg), C.white_level(cfg))
        m = normalise(f.mosaic, f.black_rggb, f.white)
        rgb = quarter_res(m)
        if calib is not None and calib.lsc_gain_packed is not None and calib.lsc_gain_packed.shape[:2] == rgb.shape[:2]:
            rgb = rgb * lsc_rgb_quarter(calib.lsc_gain_packed)
        patch = rgb[r0 // 2:r1 // 2, c0 // 2:c1 // 2]
        mean = patch.reshape(-1, 3).mean(0)
        if patch.max() > 0.95:
            print(f"[warn] {Path(p).name}: patch has values near clipping; choose a darker area")
        gains.append([mean[1] / mean[0], 1.0, mean[1] / mean[2]])
        levels.append(mean)
        print(f"{Path(p).name}: patch mean R {mean[0]:.4f}  G {mean[1]:.4f}  B {mean[2]:.4f}  "
              f"-> gains R {gains[-1][0]:.4f}  B {gains[-1][2]:.4f}")
    g = np.array(gains)
    unc = (g.max(0) - g.min(0)) / 2 if len(g) > 1 else np.zeros(3)
    m = g.mean(0)
    print("\nwhite_balance:")
    print(f"  # from {len(g)} frame(s), box rows {r0}-{r1}, cols {c0}-{c1}; uncertainty = half-spread over frames")
    print(f"  gains_rgb: [[{m[0]:.4f}, {unc[0]:.4f}], [1.0, 0.0], [{m[2]:.4f}, {unc[2]:.4f}]]")


if __name__ == "__main__":
    main()