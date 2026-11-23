"""Collect clean training images for the learned denoiser.

    python scripts/build_clean_pool.py data/detector/*.NEF --bins 1 --exclude-burst data/burst

Each raw frame is black-subtracted, normalised, defect-corrected, packed to
RGGB planes, and binned 2x2 ``--bins`` times (each bin halves the noise std).
Use base-ISO frames. Do not include any scene from the bursts you evaluate on;
``--exclude-burst`` refuses files that live under that folder.

Record in the report: how many images, their ISO, the binning, and the noise
left in them relative to the noise you add.
"""

from pathlib import Path

import numpy as np

from isp import cliutil
from isp import config as C
from isp.learned.data import clean_from_raw
from isp.rawio import load_raw


def main():
    ap = cliutil.parser(__doc__)
    ap.add_argument("raws", nargs="+")
    ap.add_argument("--bins", type=int, default=1)
    ap.add_argument("--exclude-burst", default=None, help="folder of evaluation bursts; files under it are refused")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg, calib = cliutil.load(args)

    excluded = Path(args.exclude_burst).resolve() if args.exclude_burst else None
    images, names = [], []
    for p in map(Path, args.raws):
        if excluded and excluded in p.resolve().parents:
            raise SystemExit(f"refusing {p}: it is part of the evaluation bursts")
        f = load_raw(p, C.black_rggb(cfg), C.white_level(cfg))
        mask = calib.defect_mask if calib is not None and calib.defect_mask is not None and \
            calib.defect_mask.shape == f.mosaic.shape else None
        images.append(clean_from_raw(f.mosaic, f.black_rggb, f.white, args.bins, mask))
        names.append(str(p))
        print(f"{p.name}: packed {images[-1].shape[:2]}")

    out = Path(args.out) if args.out else cliutil.derived(cfg) / "clean_pool.npz"
    np.savez_compressed(out, *images, names=np.array(names), bins=args.bins)
    px = sum(i.shape[0] * i.shape[1] for i in images)
    print(f"{len(images)} images, {px / 1e6:.1f} M packed pixels -> {out}")


if __name__ == "__main__":
    main()
