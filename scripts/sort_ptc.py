"""Sort a folder of flat-pair or dark-pair captures into pairs (and clipped frames), using EXIF and raw levels.

    python scripts/sort_ptc.py data/flatfields              # dry run: print what it would do
    python scripts/sort_ptc.py data/flatfields --apply      # copy into data/calib/ptc/ and data/calib/sat/
    python scripts/sort_ptc.py data/darks --kind dark --apply   # copy into data/calib/darkpairs/

Frames are taken in file-number order. Consecutive frames with the same ISO,
shutter and aperture form a run. In each run:

* frames whose centre is (almost) all at the white level are clipped frames;
* the rest are paired in order, (1st, 2nd), (3rd, 4th), ..., and a pair is kept
  only if its two centre levels agree within --tol (relative; for --kind dark,
  within --abs-tol of full scale, since dark levels are ~0). A frame that
  does not match its neighbour is skipped and the next one tried. So runs of
  3+ frames (extra shots) are handled, and a mismatched frame is dropped.

Nothing is moved or deleted: --apply copies. It also writes manifest.csv
(data/calib/ptc/ or data/calib/darkpairs/) listing every kept pair, which the
fit reads so that pairing never depends on folder order.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

BLACK, WHITE = 1008.26, 16383.0


@dataclass
class Frame:
    path: Path
    iso: int
    exposure_s: float
    f_number: float
    focal_mm: float
    level: float  # centre 512x512 mean, as a fraction of (white - black)
    at_white: float  # fraction of centre pixels at the file's maximum
    role: str = "skip"

    @property
    def key(self):
        return (self.iso, round(self.exposure_s, 6), round(self.f_number, 1))


def _num(tag) -> float:
    v = tag.values[0]
    return float(Fraction(v.num, v.den)) if hasattr(v, "num") else float(v)


def read_frame(path: Path) -> Frame:
    import exifread
    import rawpy

    with open(path, "rb") as fh:
        tags = exifread.process_file(fh, details=False)
    with rawpy.imread(str(path)) as raw:
        r = raw.raw_image_visible
        h, w = r.shape
        c = r[h // 2 - 256:h // 2 + 256, w // 2 - 256:w // 2 + 256].astype(np.float64)
        white = float(raw.white_level)
    return Frame(path=path, iso=int(_num(tags["EXIF ISOSpeedRatings"])),
                 exposure_s=_num(tags["EXIF ExposureTime"]), f_number=_num(tags["EXIF FNumber"]),
                 focal_mm=_num(tags["EXIF FocalLength"]),
                 level=(c.mean() - BLACK) / (WHITE - BLACK), at_white=float((c >= white).mean()))


def assign(frames: list[Frame], tol: float = 0.01, clip_fraction: float = 0.5,
           abs_tol: float = 0.0) -> list[tuple[Frame, Frame]]:
    """Set each frame's role ('sat', 'pairN-a/b', 'skip') and return the kept pairs. Pure logic, tested."""
    pairs: list[tuple[Frame, Frame]] = []
    i = 0
    while i < len(frames):
        j = i
        while j + 1 < len(frames) and frames[j + 1].key == frames[i].key:
            j += 1
        run = frames[i:j + 1]
        unclipped = []
        for f in run:
            if f.at_white >= clip_fraction:
                f.role = "sat"
            else:
                unclipped.append(f)
        k = 0
        while k + 1 < len(unclipped):
            a, b = unclipped[k], unclipped[k + 1]
            mean = 0.5 * (a.level + b.level)
            diff = abs(a.level - b.level)
            if diff <= abs_tol or (mean > 0 and diff / mean <= tol):
                n = len(pairs) + 1
                a.role, b.role = f"pair{n}-a", f"pair{n}-b"
                pairs.append((a, b))
                k += 2
            else:
                a.role = "skip (level mismatch)"
                k += 1
        i = j + 1
    return pairs


def shutter_str(t: float) -> str:
    return f"1/{round(1 / t)}" if t < 1 else f"{t:g}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--first", type=int, default=None, help="lowest file number to include, e.g. 1494")
    ap.add_argument("--last", type=int, default=None, help="highest file number to include, e.g. 1570")
    ap.add_argument("--kind", choices=["flat", "dark"], default="flat")
    ap.add_argument("--tol", type=float, default=0.01, help="max relative level difference within a pair")
    ap.add_argument("--abs-tol", type=float, default=None,
                    help="max absolute level difference (fraction of full scale); default 0.0005 (~8 DN) for darks")
    ap.add_argument("--apply", action="store_true", help="copy files into data/calib/ptc and data/calib/sat")
    ap.add_argument("--dest", default="data/calib")
    args = ap.parse_args()

    def number(p: Path) -> int:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return int(digits) if digits else -1

    paths = sorted((p for p in Path(args.folder).iterdir() if p.suffix.lower() == ".nef"), key=number)
    paths = [p for p in paths if (args.first is None or number(p) >= args.first)
             and (args.last is None or number(p) <= args.last)]
    frames = [read_frame(p) for p in paths]
    abs_tol = args.abs_tol if args.abs_tol is not None else (0.0005 if args.kind == "dark" else 0.0)
    pairs = assign(frames, args.tol, abs_tol=abs_tol)

    lvl = ">10.6f" if args.kind == "dark" else ">8.4f"
    print(f"{'file':<14}{'ISO':>6}{'shutter':>9}{'f/':>6}{'mm':>5}{'level':>{10 if args.kind == 'dark' else 8}}"
          f"{'at white':>10}  role")
    for f in frames:
        print(f"{f.path.name:<14}{f.iso:>6}{shutter_str(f.exposure_s):>9}{f.f_number:>6.1f}{f.focal_mm:>5.0f}"
              f"{f.level:{lvl}}{f.at_white:>10.3f}  {f.role}")

    print("\nSummary per ISO:")
    for iso in sorted({f.iso for f in frames}):
        ps = [p for p in pairs if p[0].iso == iso]
        sats = [f for f in frames if f.iso == iso and f.role == "sat"]
        if args.kind == "dark":
            by_t: dict[str, int] = {}
            for a, _ in ps:
                by_t[shutter_str(a.exposure_s)] = by_t.get(shutter_str(a.exposure_s), 0) + 1
            light = [f for f in frames if f.iso == iso and f.level > 0.002]
            print(f"  ISO {iso:>5}: {len(ps):2d} pairs ({', '.join(f'{n} at {t}' for t, n in by_t.items())})"
                  + (f"   [{len(light)} frame(s) above 0.002: light leak?]" if light else ""))
            continue
        lv = [0.5 * (a.level + b.level) for a, b in ps]
        rng = f"levels {min(lv):.4f} to {max(lv):.4f}" if lv else "no pairs"
        top_ok = "" if not lv or max(lv) >= 0.6 else "   [top pair below 0.6]"
        print(f"  ISO {iso:>5}: {len(ps):2d} pairs, {rng}, {len(sats)} clipped frame(s){top_ok}")
        if not sats:
            print(f"             no clipped frame: white level unconfirmed at ISO {iso}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to copy the files.")
        return
    dest = Path(args.dest)
    sub = "darkpairs" if args.kind == "dark" else "ptc"
    rows = []
    for n, (a, b) in enumerate(pairs, 1):
        d = dest / sub / f"iso{a.iso}"
        d.mkdir(parents=True, exist_ok=True)
        for f in (a, b):
            shutil.copy2(f.path, d / f.path.name)
        rows.append([a.iso, a.exposure_s, a.f_number, a.focal_mm, a.path.name, b.path.name,
                     f"{a.level:.7f}", f"{b.level:.7f}"])
    for f in frames:
        if f.role == "sat":
            d = dest / "sat" / f"iso{f.iso}"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f.path, d / f.path.name)
    (dest / sub).mkdir(parents=True, exist_ok=True)
    with open(dest / sub / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iso", "exposure_s", "f_number", "focal_mm", "file_a", "file_b", "level_a", "level_b"])
        w.writerows(rows)
    print(f"\nCopied {2 * len(pairs)} pair frames and {sum(f.role == 'sat' for f in frames)} clipped frames "
          f"into {dest}/; manifest at {dest}/{sub}/manifest.csv")


if __name__ == "__main__":
    main()