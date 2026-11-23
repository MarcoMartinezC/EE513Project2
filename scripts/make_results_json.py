"""Fill results2.json from the course template and the JSON files in results/.

    python scripts/make_results_json.py --template results2.template.json --out results2.json

The template's keys are the instructor's, so the mapping below is yours to
write once you have the template: each entry maps a dotted key in the
template to a function that pulls a [value, uncertainty] pair out of the
results/ files. Keys you do not map stay null. Unknown keys raise.
"""

import json
from pathlib import Path

from isp import cliutil
from isp.results import fill_template, pair  # noqa: F401  (pair is used in mapping())


def load(res: Path, name: str) -> dict:
    p = res / name
    return json.loads(p.read_text()) if p.exists() else {}


def mapping(res: Path) -> dict:
    """Template key -> [value, uncertainty]. Fill this in against results2.template.json."""
    out = {}
    # Examples (the key names on the left are placeholders; use the template's):
    #
    # abl = load(res, "ablation_iso3200.json")
    # if abl:
    #     v = abl["variants"]["no black level"]["psnr_db"]
    #     out["ablation.no_black_level_psnr_db"] = pair(*v)
    #
    # dn = load(res, "denoise_iso3200.json")
    # if dn:
    #     out["denoise.iso3200.learned.psnr_db"] = pair(*dn["summary"]["all"]["rgb_psnr"])
    return out


def main():
    ap = cliutil.parser(__doc__)
    ap.add_argument("--template", default="results2.template.json")
    ap.add_argument("--out", default="results2.json")
    args = ap.parse_args()
    cfg, _ = cliutil.load(args)
    updates = mapping(cliutil.results(cfg))
    if not updates:
        print("[note] mapping() is empty: edit scripts/make_results_json.py against the template")
    fill_template(args.template, args.out, updates)
    print(f"{len(updates)} entries filled -> {args.out}")


if __name__ == "__main__":
    main()
