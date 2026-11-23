# An image signal processor from raw

EE513 Project 2 · Marco Martinez Corral · camera: Nikon (same body and lens as Project 1)

## Problem

Build the pipeline that turns a raw sensor mosaic into a displayable image,
using only the calibration measured in Project 1. Then measure three things:
what each stage contributes, whether a small learned denoiser trained on my
own sensor's noise model beats a classical one on real noise, and whether any
of it helps an object detector.

## Method

```
raw DN ─► black level ─► defects ─► denoise ─► lens shading ─► white balance ─► clip
       ─► demosaic (bilinear | MHC) ─► colour matrix ─► tone curve ─► unsharp mask ─► sRGB
```

| Stage | Where | Notes |
|---|---|---|
| Black level, saturation | `pipeline.black_level`, `clip_highlights` | per Bayer site; highlights clipped after WB so they stay neutral |
| Defective pixels | `calib.hot_pixels_from_darks`, `stuck_pixels_from_flat` | found from dark frames and the flat; idempotent correction |
| Lens shading | `calib.shading_gain_from_flat` | per channel (R, G1, G2, B), degree-6 polynomial fit |
| White balance | `pipeline.wb` | before demosaicing (see `tests/test_stages.py::test_wb_order_matters_for_mhc`) |
| Demosaicing | `demosaic.bilinear`, `demosaic.mhc` | from scratch; Bayer-phase-preserving borders |
| Classical denoiser | `denoise_classical.denoise_packed` | guided filter on packed Bayer after a variance-stabilising transform |
| Learned denoiser | `learned/` | DnCNN-style, packed Bayer + noise-level map, trained on synthetic Poisson-Gaussian noise |
| Colour, tone, sharpening | `pipeline.ccm`, `tone`, `sharpen` | Project 1 CCM; sRGB curve; unsharp mask on luminance |

Ground truth (`groundtruth.py`):

- **Denoising**: frame *k* of a burst (never resampled) against the mean of
  the other frames aligned to *k* (leave-one-out). Repeating over frames gives
  the uncertainty.
- **Demosaicing**: 2×2 blocks of the averaged burst → R, mean G, B → re-mosaic
  → demosaic → compare.
- **Registration**: plain upsampled cross-correlation on a full-resolution
  [1 2 1] luminance proxy of each mosaic, with defects corrected first. Three
  more obvious choices were biased against known shifts (see
  `groundtruth.estimate_shift`, `registration_proxy`, and `docs/decisions_log.md`).

## Results

*To be filled from `results/`. Every number: what it is measured against, and ± what.*

| Measurement | Value | Against | Uncertainty from |
|---|---|---|---|
| Residual noise in the 15-frame reference, ISO 3200 | | single frame | |
| MHC − bilinear, averaged-burst reference | | 2×2 ground truth | tile bootstrap |
| Cost of skipping black level | | correct pipeline | std over frames |
| Learned vs guided filter, ISO 3200, display PSNR | | leave-one-out burst mean | std over frames |
| ONNX latency, 512×512 packed, CPU | | — | std over 20 runs |
| Full pipeline, one 24 MP frame | | — | std over 5 runs |
| Detector F1, ours vs camera JPEG | | camera JPEG | std over scenes |

Ablation table: `results/ablation_iso*.md`. Denoiser table: `results/denoise_iso*.md`.
Detector table: `results/detector.md`.

## Reproduce

```bash
uv sync --group learning --group dev     # PyTorch, onnxruntime, torchmetrics, pytest
make test                                # unit tests, no data needed
make demo                                # whole chain on a simulated camera, no data needed (~15 min CPU)
make reproduce                           # whole chain on the real captures (see data/README.md)
```

Individual steps (each script has `--help`):

```bash
python scripts/calibrate.py                                   # defect map + shading gains
python scripts/build_reference.py --burst data/burst/iso3200  # align, residual noise, noise fit
python scripts/score_demosaic.py --reference data/derived/burst_iso100.npz
python scripts/ablation.py --reference data/derived/burst_iso3200.npz
python scripts/build_clean_pool.py data/pool/*.NEF --exclude-burst data/burst
python scripts/train_denoiser.py --isos 100 800 3200 --out runs/dncnn_all
python scripts/eval_denoisers.py --reference data/derived/burst_iso3200.npz --model all=runs/dncnn_all/model.pt --lpips
python scripts/export_onnx.py runs/dncnn_all/model.pt --sizes 256 512 1024
python scripts/benchmark_pipeline.py data/detail/DSC_xxxx.NEF
python scripts/detect_compare.py --scenes data/detector
python scripts/run_pipeline.py data/detail/DSC_xxxx.NEF --save-stages results/stages/
```

## Layout

```
config/camera.yaml   Project 1 constants as [value, uncertainty]
src/isp/             the pipeline (rawio, demosaic, calib, pipeline, groundtruth, noise_model, metrics, ...)
src/isp/learned/     model, synthetic pairs, training, tiled inference, ONNX export
scripts/             one script per deliverable; outputs JSON + figures in results/
tests/               pytest; includes an end-to-end check against the simulated camera
docs/decisions_log.md  running log for the decisions-and-verification statement
data/README.md       where the captures go and how they were taken
```

## Tests

`make test` runs unit tests for the core functions, including:
a flat colour demosaics back exactly (both methods, including borders);
measured Bayer sites come back unchanged; defect correction is idempotent;
shading correction flattens its own flat; known sub-pixel shifts are recovered
to 0.05 px; the noise model's sampled variance matches its formula; the GAT
gives unit variance; a short training run beats its noisy input (guards
against dead-ReLU collapse); tiled inference equals whole-image inference;
ONNX matches PyTorch to float32 rounding; burst registration survives detail
that aliases in the Bayer planes; and the full pipeline recovers a simulated
camera's scene and constants. With the course's `make_figures.py`
on the path, the demosaicers are also checked against it.
