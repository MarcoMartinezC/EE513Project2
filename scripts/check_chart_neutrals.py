"""Grey row of a ColorChecker frame after this pipeline's black level and white balance.

    python scripts/check_chart_neutrals.py ../images/color_charts/DSC_1404.NEF

Uses camchar.sample_colorchecker from Project 1 to find the chart. R/G and B/G
should be close to 1 on every grey patch if black level, CFA alignment and the
white balance gains are right.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EE513Project1_fix"))  # camchar.py
import camchar as cc  # noqa: E402

from isp import cliutil, config as C, pipeline as P  # noqa: E402
from isp.context import build_context  # noqa: E402
from isp.groundtruth import quarter_res  # noqa: E402
from isp.rawio import load_raw  # noqa: E402

NAMES = ["white 9.5", "neutral 8", "neutral 6.5", "neutral 5", "neutral 3.5", "black 2"]

ap = cliutil.parser(__doc__)
ap.add_argument("raw")
args = ap.parse_args()
cfg, calib = cliutil.load(args)
f = load_raw(args.raw, C.black_rggb(cfg), C.white_level(cfg))
ctx = build_context(cfg, calib, f.black_rggb, f.white, f.mosaic.shape, require=("wb",))
m = P.run(f.mosaic, ctx, [("to_unit", {}), ("black_level", {}), ("wb", {})]).image
means, boxes, score = cc.sample_colorchecker(quarter_res(m), 0.5)
print(f"chart found (neutral-row score {score:.3f})")
for i, name in zip(range(18, 24), NAMES):
    r, g, b = means[i]
    print(f"patch {i + 1:2d} {name:<12} R/G {r / g:.3f}   B/G {b / g:.3f}")