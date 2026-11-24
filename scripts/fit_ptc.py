"""Conversion gain and read noise per ISO from the sorted flat pairs and dark pairs.

    python scripts/fit_ptc.py                      # reads data/calib/ptc and data/calib/darkpairs

For each pair, on a central patch of each Bayer plane (R, G1, G2, B):
  signal = mean of the two frames minus black,   variance = 1/2 Var(A - B)
The difference cancels fixed pattern (PRNU, dust, shading), as in Project 1.
Pairs with any clipped pixel in the patch are excluded (clipping removes variance).

Read noise comes directly from the dark pairs (1/2 Var(A - B), no light), and
the 1/60 s pairs are reported separately as a check that exposure time does
not matter at these speeds.

Gain is fitted three ways, all on var = mean / K + read^2 (DN):
  ols      unweighted straight line, free intercept (Project 1's reported method)
  weighted weights 1/SE(var), SE = var * sqrt(2 / (n - 1)), free intercept
  fixed    weighted, intercept fixed to the dark-pair read noise squared
The four Bayer planes share the same pixel electronics, so their spread is a
check on the fit; the quoted uncertainty is that half-spread or the fit's own
error, whichever is larger. Measured read noise includes the 1/12 DN^2 of
rounding to whole DN, as it should: that noise is really in the raw file. Writes results/ptc.json, results/ptc_iso<ISO>.png and
prints config entries ready to paste.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from isp import cliutil
from isp import config as C
from isp.rawio import load_raw, pack_rggb

PLANES = ("R", "G1", "G2", "B")


def read_manifest(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def pair_stats(folder: Path, row: dict, black: np.ndarray, white: float, patch: int):
    """Per-plane (signal, variance, n, clipped) for one pair."""
    a = load_raw(folder / row["file_a"], black, white)
    b = load_raw(folder / row["file_b"], black, white)
    pa, pb = pack_rggb(a.mosaic).astype(np.float64), pack_rggb(b.mosaic).astype(np.float64)
    h, w, _ = pa.shape
    sl = (slice(h // 2 - patch // 2, h // 2 + patch // 2), slice(w // 2 - patch // 2, w // 2 + patch // 2))
    out = []
    for c in range(4):
        x, y = pa[sl + (c,)], pb[sl + (c,)]
        clipped = bool((x >= white).any() or (y >= white).any())
        d = x - y
        out.append((0.5 * (x.mean() + y.mean()) - black[c], 0.5 * d.var(ddof=1), d.size, clipped))
    return out


def fit_line(m, v, n, intercept=None):
    """Return dict with slope (1/K), intercept, and their uncertainties for the three methods."""
    m, v, n = map(np.asarray, (m, v, n))
    se = v * np.sqrt(2.0 / (n - 1))
    res = {}
    (s, c), cov = np.polyfit(m, v, 1, cov=True)
    res["ols"] = (s, np.sqrt(cov[0, 0]), c, np.sqrt(cov[1, 1]))
    (s, c), cov = np.polyfit(m, v, 1, w=1 / se, cov="unscaled")
    res["weighted"] = (s, np.sqrt(cov[0, 0]), c, np.sqrt(cov[1, 1]))
    if intercept is not None:
        w2 = 1 / se**2
        s = np.sum(w2 * m * (v - intercept)) / np.sum(w2 * m * m)
        res["fixed"] = (s, 1 / np.sqrt(np.sum(w2 * m * m)), intercept, 0.0)
    return res


def main():
    ap = cliutil.parser(__doc__)
    ap.add_argument("--calib", default="data/calib")
    ap.add_argument("--patch", type=int, default=256, help="patch size per Bayer plane (packed px)")
    ap.add_argument("--max-level", type=float, default=1.0,
                    help="ignore pairs above this fraction of full scale (to test whether the top pairs bend the fit)")
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    black = C.black_rggb(cfg)
    white = C.white_level(cfg)
    calib = Path(args.calib)
    res_dir = cliutil.results(cfg)

    flats = read_manifest(calib / "ptc" / "manifest.csv")
    darks = read_manifest(calib / "darkpairs" / "manifest.csv")
    report = {"patch_packed_px": args.patch, "max_level": args.max_level, "black_rggb": black.tolist(),
              "white": white, "iso": {}}
    yaml_lines = ["noise:", "  iso:"]

    for iso in sorted({int(r["iso"]) for r in flats}):
        # ---- dark pairs: read noise per plane, split by shutter ----
        dk = {}
        for r in (r for r in darks if int(r["iso"]) == iso):
            t = float(r["exposure_s"])
            st = pair_stats(calib / "darkpairs" / f"iso{iso}", r, black, white, args.patch)
            dk.setdefault(t, []).append(([s[1] for s in st], st[0][2]))
        read = {}  # shutter -> (read noise per plane, uncertainty per plane), DN
        for t, vals in dk.items():
            v = np.array([x for x, _ in vals])  # (pairs, planes) variances
            n_pix = vals[0][1]
            sig = np.sqrt(v.mean(0))
            if len(v) > 1:  # empirical: spread of per-pair estimates
                unc = np.sqrt(v).std(0, ddof=1) / np.sqrt(len(v))
            else:  # one pair: statistical error of a standard deviation from n_pix samples
                unc = sig / np.sqrt(2 * (n_pix - 1))
            read[t] = (sig, unc)
        t_short = min(read) if read else None

        # ---- flat pairs ----
        rows = []
        for r in (r for r in flats if int(r["iso"]) == iso):
            st = pair_stats(calib / "ptc" / f"iso{iso}", r, black, white, args.patch)
            rows.append((r, st))

        per_plane = {}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for c, name in enumerate(PLANES):
            top = args.max_level * (white - black[c])
            keep = [(st[c][0], st[c][1], st[c][2]) for _, st in rows if not st[c][3] and 0 < st[c][0] <= top]
            dropped = sum(st[c][3] for _, st in rows)
            m, v, n = map(np.array, zip(*keep))
            rd = read[t_short][0][c] if t_short is not None else None
            fits = fit_line(m, v, n, rd**2 if rd is not None else None)
            per_plane[name] = {
                "pairs_used": len(m), "pairs_clipped": int(dropped),
                "fits": {k: {"K_e_per_dn": 1 / f[0], "K_unc": f[1] / f[0] ** 2,
                             "intercept_dn2": f[2], "intercept_unc": f[3],
                             "read_from_intercept_dn": float(np.sqrt(max(f[2], 0)))}
                         for k, f in fits.items()},
                "var_over_mean": (v / m).tolist(), "mean_dn": m.tolist(),
            }
            axes[0].loglog(m, v, "o", ms=4, label=name)
            axes[1].semilogx(m, v / m, "o-", ms=4, label=name)
        xs = np.logspace(np.log10(5), np.log10(white - black.mean()), 50)

        Ks = {k: np.array([per_plane[p]["fits"][k]["K_e_per_dn"] for p in PLANES]) for k in per_plane["G1"]["fits"]}
        rd_short = read[t_short] if t_short is not None else None
        best = "fixed" if "fixed" in Ks else "weighted"
        K_best = Ks[best].mean()
        axes[0].loglog(xs, xs / K_best + (rd_short[0].mean() ** 2 if rd_short else 0), "k-", lw=1,
                       label=f"{best}: K = {K_best:.3f} e-/DN")
        axes[0].set_xlabel("signal above black (DN)")
        axes[0].set_ylabel("1/2 Var(A - B) (DN²)")
        axes[0].legend(fontsize=8)
        axes[0].set_title(f"Photon transfer, ISO {iso}")
        axes[1].set_xlabel("signal above black (DN)")
        axes[1].set_ylabel("variance / mean (DN/e- if shot-noise limited)")
        axes[1].set_title("Should flatten to 1/K at high signal")
        fig.tight_layout()
        fig.savefig(res_dir / f"ptc_iso{iso}{'' if args.max_level >= 1.0 else f'_max{args.max_level:g}'}.png", dpi=130)
        plt.close(fig)

        # Uncertainty: half-spread over the four planes, but never below the fits' own mean error.
        summary = {k: [float(v.mean()), float(max((v.max() - v.min()) / 2,
                                                  np.mean([per_plane[p]["fits"][k]["K_unc"] for p in PLANES])))]
                   for k, v in Ks.items()}
        report["iso"][iso] = {
            "planes": per_plane,
            "K_mean_over_planes_with_half_spread": summary,
            "read_noise_dn_by_shutter": {f"{t:g}": {"per_plane": read[t][0].tolist(), "mean": float(read[t][0].mean()),
                                                     "unc": float(np.sqrt(np.mean(read[t][1] ** 2)))}
                                         for t in read},
        }

        print(f"\nISO {iso}: {len(rows)} flat pairs "
              f"({max(per_plane[p]['pairs_clipped'] for p in PLANES)} excluded for clipping in some plane)")
        for t in sorted(read):
            print(f"  read noise at 1/{round(1 / t)} s: " + "  ".join(f"{p} {x:.3f}" for p, x in zip(PLANES, read[t][0]))
                  + f"  -> mean {read[t][0].mean():.3f} DN")
        for k in Ks:
            print(f"  K ({k:<8}): " + "  ".join(f"{p} {x:.4f}" for p, x in zip(PLANES, Ks[k]))
                  + f"  -> {summary[k][0]:.4f} ± {summary[k][1]:.4f} e-/DN")
        if rd_short is not None:
            unc = float(np.sqrt(np.mean(rd_short[1] ** 2)))
            yaml_lines += [f"    {iso}:",
                           f"      gain_e_per_dn: [{summary[best][0]:.4f}, {summary[best][1]:.4f}]   # {best} fit, "
                           f"mean of R/G1/G2/B",
                           f"      read_noise_dn: [{rd_short[0].mean():.4f}, {unc:.4f}]   # dark pairs at "
                           f"1/{round(1 / t_short)} s"]

    tag = "" if args.max_level >= 1.0 else f"_max{args.max_level:g}"
    (res_dir / f"ptc{tag}.json").write_text(json.dumps(report, indent=2, default=float))
    print("\nConfig entries (check the fits and the figures before pasting):\n")
    print("\n".join(yaml_lines))
    print(f"\n-> {res_dir}/ptc.json and ptc_iso*.png")


if __name__ == "__main__":
    main()