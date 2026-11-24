# Decisions and verification log

A running log to write the one-page decisions-and-verification statement from.
Add an entry when you choose between alternatives, when you check something, or
when something surprises you. Dated entries also back up the commit history.

Template:

    ## YYYY-MM-DD  <short title>
    Decision:     what you chose
    Rejected:     what you didn't, and why
    Verified by:  test / known case / comparison / figure (file and number)
    AI use:       tool, what for, and how you checked its output
    Still unsure: ...

---

## 2026-11-17  Starting code (AI-generated, to be verified)

The initial repository structure, modules, scripts and tests were generated
with Claude from my Exercise 5 (demosaicing), training-without-clean-images and
PyTorch on-ramp notebooks and the Project 2 spec. It was committed as one
initial commit; everything after it is mine. Before relying on any part of it,
I check it against my own captures and record the result here.

Design choices made in the starting code that I need to own or change:

- [ ] Denoising runs on packed Bayer, after black level and defects, before
      shading and white balance (the Poisson-Gaussian model holds there).
- [ ] Denoising ground truth is leave-one-out: frame k vs the mean of the
      other frames aligned to k. Frame k is never resampled.
- [ ] Shading gains come from a degree-6 polynomial fit per Bayer channel.
- [ ] The ablation compares in display sRGB, before sharpening, at the 2x2
      ground-truth resolution, with denoiser strength tuned per placement on
      a frame excluded from scoring.
- [ ] The guided filter uses each plane as its own guide, with a
      variance-stabilising transform from the noise model.
- [ ] The learned model takes a noise-level map, so one network can cover
      several ISOs.

Found while testing the starting code on a simulated camera, to check on my data:

- [ ] `mode="reflect"` borders (Exercise 5, `make_figures.py`) break the Bayer
      phase at the frame edge, so MHC gets a flat colour wrong by up to 0.34 in
      the outer two pixels. Bilinear is unaffected. Default is now `mirror`.
- [ ] Burst registration had three separate biases, each found with known
      shifts: (1) `cv2.phaseCorrelate`'s sub-pixel estimate was off by up to
      ~0.25 px; (2) scikit-image's default phase normalisation returned 0.21
      for a true 0.60 px shift on smooth texture; (3) registering the packed
      Bayer planes was off by up to 0.9 px on fine detail, because that detail
      aliases at half resolution. Now: plain upsampled cross-correlation on a
      full-resolution [1 2 1] luminance proxy, with defects corrected first.
      Error is ~0.01 px rms on the synthetic camera.
- [ ] Temporal variance across aligned frames reads ~20 % low, because
      resampling averages noise. The noise fit uses unresampled frame minus
      leave-one-out reference instead.
- [ ] Read noise fitted from a burst is an extrapolated intercept and came out
      5-10 % low on the synthetic camera; the dark-pair value is more trustworthy.
- [ ] The averaged-burst demosaic reference is built from resampled frames,
      which are slightly blurred, and that flatters both demosaicers (about
      3 dB on the synthetic camera at ISO 100, where noise is not the
      explanation). On a steady tripod with shifts below `--min-shift`, nothing
      is resampled. Report the shifts and whether the reference was resampled.
- [ ] Check the denoising reference itself: `eval_denoisers.py` prints the
      std of (noisy - reference) and what the noise model predicts. On the
      synthetic camera at ISO 1600: 0.0217 vs 0.0155 for frame 0, and 0.032 vs
      0.016 averaged over frames 0, 7 and 14. The extra is reference error from
      resampling fine detail; no denoiser can remove it.
- [ ] The ablation's best denoiser strength hit the top of the grid on some
      bursts: PSNR rewards blur. Look at the crops before trusting the tuned value.
- [ ] A small learned model can collapse to the identity (dead last ReLU layer)
      depending on initialisation; `train` now warns, and logs loss relative to
      the noisy input's MSE (1.0 = learned nothing).

## 2026-11-23  My earlier burst analysis used the wrong raw constants

Decision:     Treat the training-without-clean-images results (median shift
              0.77 px, std 0.0081, MAD 0.0020) as unreliable; redo them on the
              new bursts with this pipeline.
Why:          That notebook normalised with black 256 / white 3567, but the
              Z50 writes 14-bit NEFs with black 1008 / white 16383. With
              white = 3567 and np.clip(0, 1), every pixel above ~3567 DN became
              exactly 1.0, flattening bright areas and corrupting the noise
              statistics and the registration. It also used cv2.phaseCorrelate
              on one green plane (see the registration bias above).
Verified by:  Project 1 datasheet: 14-bit, white level 16383, black 1008.26.

## 2026-11-23  Calibration constants: what is reused from Project 1

Decision:     config/camera.yaml takes from Project 1 (results.json, results_extra.json):
              - black level 1008.26 ± 0.01 DN (dark frames extrapolated to 0 s; file says 1008)
              - white level 16383 DN, from the file; to confirm with a clipped frame
              - WB gains (Illuminant 1, mean of charts A-C, ± half-spread):
                R 1.6284 ± 0.0045, B 2.7701 ± 0.0014
              - ISO 100 noise: K = 1/0.2811 = 3.557 ± 0.052 e-/DN;
                read noise 1.4016 ± 0.0320 DN (dark pairs, measured directly in DN)
Units:        Project 1 reports gain in DN/e-; the pipeline uses K in e-/DN.
              Uncertainty: 0.0041 / 0.2811^2 = 0.052.
Rejected:     The OLS-vs-weighted choice is open: the weighted PTC fit
              (0.3139 DN/e-, K = 3.186) has an intercept read noise of
              5.9 ± 0.9 e-, consistent with the dark pairs (4.99 e-), while the
              OLS intercept gives 15.7 e-. That favours the weighted fit; the new
              flat pairs replace both.
Still unsure: White balance is only valid under Illuminant 1 (R gain 1.37 /
              1.28 / 2.61 under Ill. 2 / Ill. 3 / daylight). ISO 100 noise does
              not carry to the bursts: the Z50 switches conversion gain at ISO 400.

## 2026-11-24  Colour matrix: Project 1 matrix reused after normalisation

Decision:     Use project1.ipynb's camera -> linear sRGB (D65) matrix, divided by
              its G-row sum (~20.05) so it no longer carries the chart's exposure;
              mean of charts A-C, uncertainty = half-spread. Valid under
              Illuminant 1 only; replace with a fit from the new chart capture.
Why the first look was wrong: results_extra.json stores camera -> XYZ (D50)
              matrices. Their row sums (19.0 : 19.9 : 16.3) are the D50 white
              point times ~20, not a colour cast.
Verified by:  Rebuilding the sRGB matrix from the XYZ ones (Bradford D50 -> D65)
              matches the notebook's export to 0.008. Same convention as the
              pipeline (bal @ M.T; input black-subtracted, scaled, flat-fielded,
              white-balanced quarter-resolution RGB).
Still unsure: Grey maps to (0.968, 1, 0.983): the fit does not preserve white
              exactly. Decide whether to rescale the rows so it does.
## 2026-11-23  Recapture plan (checked against the datasheet)

Recapture:    flat pairs (70-80 % of full scale) and dark pairs at every burst
              ISO, on both sides of the ISO 400 conversion-gain switch; one
              clipped frame per ISO; an evenly lit shading flat at the chosen zoom
              and aperture; a colour chart under the scene light, white patch at
              ~50-70 % of full scale.
Reuse:        black level; geometric calibration.
Correction:   Another AI chat described the Project 1 flat field as "about 6 %
              of full scale at 16 mm and ISO 400". Those numbers are the colour
              charts (4.6-6.1 %, 16 mm, ISO 400); the flats were 26 mm, ISO 100,
              32 % peak. The recapture decision is unchanged.
Exclude:      DSC_1294 from any dark-frame folder (Project 1 flagged it: mean
              below black).

## 2026-11-24  Environment

Decision:     The course environment, lecture_notebooks/.venv (Python 3.12),
              as required. Makefile: PY ?= ../.venv/bin/python; install with
              `uv pip install -e .`.
Rejected:     `uv run` / `uv sync`, which silently built a separate Project2/.venv
              (no PyTorch: 52 passed, 2 skipped, instead of 61 / 1).
Problem found: `uv pip install -e .` added opencv-python 5.1.0.95 next to the
              course's opencv-contrib-python 5.0.0.93 (both provide cv2).
              Removed opencv-python, reinstalled contrib 5.0.0.93, changed the
              dependency in pyproject.toml to opencv-contrib-python.
Verified by:  `make test` in the course environment: 61 passed, 1 skipped
              (the skip is the make_figures.py comparison); a lecture notebook
              that uses OpenCV still runs.

## 2026-11-24  First real frame through the pipeline

Input:        ../images/color_charts/DSC_1404.NEF (Illuminant 1, ISO 400, 16 mm)
Run:          --denoise none --skip ccm --exposure 10; no shading map, no defect map
Change:       added --skip and --exposure to scripts/run_pipeline.py
Timing:       1.85 s total (single run, includes first-call overhead):
              demosaic (MHC) 1045 ms (57 %), tone 478 ms (26 %), sharpen 239 ms (13 %)
Verified by:  TODO: grey patches neutral? red/blue not swapped? white patch not clipped?
Still unsure: Tone curve cost is 63 M power() calls; a lookup table should cut
              it to milliseconds (Part C). Proper latency numbers come from
              benchmark_pipeline.py (mean ± std over runs).