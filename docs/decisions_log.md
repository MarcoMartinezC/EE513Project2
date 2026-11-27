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
Verified by:  check_chart_neutrals.py on DSC_1404-1406: grey patches 19-24 within 2 % of
              neutral, patch 21 within 0.6 %. Full pipeline with CCM: 1.94 s single run.
Still unsure: Tone curve cost is 63 M power() calls; a lookup table should cut
              it to milliseconds (Part C). Proper latency numbers come from
              benchmark_pipeline.py (mean ± std over runs).

## 2026-11-24  White level confirmed at ISO 100

Measured:     DSC_1485 (ISO 100, 1 s, flat target overexposed): max 16383, centre 512x512
              median 16383, 100 % of centre pixels at 16383.
Result:       white level 16383 DN confirmed empirically at ISO 100 (Project 1 only had the file value).
Still to do:  confirm at ISO 800 and 3200 with the clipped frame of each column.

White level 16383 confirmed at ISO 100, 800 and 3200: pixels at 16383 in DSC_1494-1498 (ISO 100),
DSC_1522-1524 (ISO 800, 53 % of centre), DSC_1568-1569 (ISO 3200, 100 %).

Black level 1008.26 DN holds at ISO 100, 800 and 3200: dark-pair centre means within 0.3 DN of it
(DSC_1601-1642, cap on, 1/4000 and 1/60 s).

## 2026-11-24  Noise model per ISO (flat pairs + dark pairs)

Captures:     flat pairs DSC_1494-1570 (overcast sky, defocused; ISO 100 f/8, ISO 800 f/22,
              ISO 3200 f/40; 1/13-1/2000 s), sorted with scripts/sort_ptc.py; dark pairs DSC_1601-1642
              (cap on, 1/4000 and 1/60 s).
Decision:     K from the fixed fit (weighted, intercept = dark-pair read noise^2) on pairs <= 30 % of full
              scale; read noise directly from dark pairs at 1/4000 s.
Why the cut:  On all pairs, ISO 100 OLS gave 2.70 with green (2.57) and red/blue (2.81) disagreeing, against
              3.01 weighted. With pairs above 30 % excluded, OLS moved to 3.13, planes agreed within 3 %, and
              all fits fell within 3.08-3.14. ISO 800 and 3200 changed < 0.5 %. So the top ISO 100 pairs were
              contaminated (most likely cloud structure shifting between frames), not a sensor curvature.
              Project 1's weighted fit (3.19) agrees within 2-4 %; its OLS (3.56) had the same problem.
Results:      K = 3.078 ± 0.041 / 0.399 ± 0.013 / 0.1004 ± 0.0029 e-/DN at ISO 100 / 800 / 3200.
              Read noise 1.358 / 3.351 / 11.563 DN = 4.2 / 1.34 / 1.16 e-: the input-referred read noise
              drops 3x across the conversion-gain switch at ISO 400.
Verified by:  fit_ptc.py on simulated pairs with known K and read noise (all recovered); read noise at
              1/60 s matches 1/4000 s within 3 % at every ISO; four Bayer planes agree.
Still unsure: B plane K is 4-7 % above the others at ISO 800 and 3200 under every fit (covered by the
              quoted uncertainty). One hot or flickering pixel in the ISO 100 1/60 s R patch (1.57 vs 1.39 DN).

## 2026-11-25  Lens shading map (38 mm, f/5.6)

Captures:     DSC_1682-1685 upright, DSC_1686-1689 rotated 180 deg; overcast sky through paper
              taped over the lens; ISO 100, 1/8 s, centre level 0.53-0.60.
Result:       corner gain R 2.02, G 1.98, B 1.92 (corners at ~0.51 of centre).
Check:        maps from the two orientations differ by 3.9 % rms (10 % max, 8.9 % inside the central
              90 %): the sky's gradient still reaches the sensor through one sheet of paper. The final
              map averages both orientations, which cancels the part that flips with rotation; any
              unevenness that does not flip is not measured by this check.
Rejected:     screen flats (DSC_1645-1653): 1 % of full scale and 4.7 % rms rotation difference.
Uncertainty:  ~4 % on the shading gain away from the centre (rotation difference as an upper bound).

## 2026-11-27  Bursts captured

Settings:     38 mm, f/5.6, single lamp indoors; ISO 100 at 1/6 s, ISO 800 at 1/50 s, ISO 3200 at 1/200 s;
              20 frames each, self-timer, single-frame release.
Levels:       median 0.125 / 0.134 / 0.129, 99th percentile 0.42 / 0.40 / 0.44 of full scale;
              0.15-0.26 % clipped in each, from a small bright highlight in the upper middle-right
              (lamp/specular), not the white reference.
Rejected:     two earlier attempts in data/rejected/: first ~35 % clipped (2.6 stops over), second
              ~7 stops under; one ISO 800/3200 set shot at the old shutter speeds.

## 2026-11-27  Lamp flicker in the burst captures

Found by:     build_reference.py: burst-fitted gain matched the flat pairs at ISO 100 (3.20 vs 3.08) but
              not at ISO 800 (0.217 vs 0.399) or 3200 (0.024 vs 0.100); error grew with shutter speed.
Confirmed:    frame-to-frame centre level variation 0.09 % / 0.69 % / 2.54 % and banding ratio 1.3 / 15.9 /
              5.9 at 1/6 s / 1/50 s / 1/200 s: lamp flicker (120 Hz), averaged out only at slow shutters.
Fix:          reshot all three bursts at 2 s / 1/4 s / 1/15 s (>= 8 flicker cycles) with the lamp further
              away; hot-pixel darks reshot at ISO 3200, 1/15 s.
Verified by:  (fill in after the reshoot: variation and banding ratio; burst-fitted K vs config)

## 2026-11-27  Burst references: noise model cross-check

Bursts:       daylight (no flicker): ISO 100 1 s, ISO 800 1/8 s, ISO 3200 1/30 s, f/5.6, 38 mm, 20 frames each.
              Level variation 0.41 / 0.21 / 0.25 %, banding ratio 1.7 / 1.3 / 1.0. No clipping.
Cross-check:  gain fitted from each burst (frame minus leave-one-out reference) vs flat pairs:
              ISO 800 0.401 vs 0.399 (+0.5 %), ISO 3200 0.101 vs 0.1004 (+0.6 %); read noise within 3 %.
              ISO 100 2.81 vs 3.08 (-8.6 %): consistent with its 0.41 % daylight drift adding variance
              proportional to signal; config values kept.
Shifts:       median 0.4-0.8 px, max 1.4 px (sensor): every reference frame was resampled.
Hot pixels:   11,119 (0.053 %) at ISO 3200, 1/30 s.

White balance (daylight): R 2.3091 ± 0.0012, B 1.5225 ± 0.0011 from the white paper in DSC_2136-2155.
Patch G level rose 0.1862 -> 0.1890 (+1.5 %) across the ISO 100 burst at constant colour ratios:
direct evidence of the daylight drift behind the ISO 100 burst-fitted gain being 8.6 % low.
CCM: Project 1 natural light round (closest gains: 2.61 / 1.39).

| Files | Reason | Deleted |
|---|---|---|
| burst, 1st attempt | ~35 % clipped (2.6 stops over) | 2026-11-27 |
| burst, 2nd attempt | ~7 stops under (p99.9 0.03) | 2026-11-27 |
| burst ISO 800/3200, 3rd attempt | shot at old shutter speeds, ~30 % clipped | 2026-11-27 |
| burst, lamp set (1/6, 1/50, 1/200 s) | lamp flicker: level variation 0.09 / 0.69 / 2.54 %, banding 1.3 / 15.9 / 5.9 | 2026-11-27 |
| DSC_1645-1653, DSC_1657, DSC_1671, DSC_1680 | shading flat tests (screen; underexposed sky) | 2026-11-25 |
| DSC_2222 | detail scene test, p99.9 0.113 (2.6 stops under) | 2026-11-27 |

## 2026-11-27  Demosaic scoring

Burst scene (ISO 100, averaged ref): bilinear 61.96 ± 0.25, MHC 62.05 ± 0.21 dB (gap +0.09, not significant);
  single frame: MHC -0.87 dB. Absolute PSNR is high because the scene sits at ~7 % of full scale (peak 1.0);
  only the gap is compared. Reference frames were resampled (shifts up to 1.4 px).
Detail scene DSC_2223 (single frame, 6 s): bilinear 45.50 ± 0.15, MHC 44.68 ± 0.13 dB (gap -0.82).
  "MHC hurts most" crop: sharp colour edge with coloured fringes. MHC borrows another channel's
  curvature, which overshoots where colours change differently across an edge.
  Clipping near the edge: (check).
Uncertainty: tile bootstrap (64x64 tiles), content variation across the frame.

Crop figure (DSC_2223): MHC wins at a grey/white-to-dark edge (bilinear error concentrates along
the edge; MHC's is noise-like); MHC loses at a grey-to-red edge (bright narrow error band along
the edge: cross-channel overshoot where R changes little but G and B drop), and inside flat red
texture (noise amplified across channels). Not clipping: the bright side is light grey, not saturated.

## 2026-11-27  Colour matrix for daylight

Source:       Project 1 Natural light round, DSC_1420-1422, refitted with project1.ipynb in a scratch
              cell (no Project 1 outputs overwritten). Mean dE00 2.43 ± 0.12, worst 5.67 ± 0.17.
Normalised:   camera -> XYZ (D50) -> Bradford -> linear sRGB (D65), divided by G-row sum (~100,
              the charts' exposure); mean of charts A-C, uncertainty = half-spread.
Mismatch:     chart round WB gains R 2.59 / B 1.39 vs today's scene R 2.31 / B 1.52: a different
              daylight. Colour error on today's scene is not measured (no chart); Project 1's
              transfer table bounds a matrix used under another light at mean dE00 3.5-5.4.
Grey:         maps to (0.959, 1, 0.992); not rescaled.

## 2026-11-27  Ablation at ISO 100 / 800 / 3200

Result:       same ranking at all ISOs: black level (-34 / -29 / -25 dB) >> lens shading (-23 / -17 / -13)
              > colour correction > denoising > WB after demosaic > MHC vs bilinear (-0.2 to -0.3).
              Black level costs ~100x more than the choice of demosaicer.
Across ISO:   dB costs of noise-independent stages shrink with ISO only because the full pipeline's own
              error (noise) grows; added MSE is ~constant (shading 2.5-2.7e-3, black level 4.1-4.4e-2).
              Denoising's cost grows with ISO (-2.9 / -5.5 / -7.1 dB), as expected.
Order:        WB before demosaic and denoise before demosaic both win at every ISO.
Zeros:        highlight clip (no clipped pixels in the bursts); defect correction (0.05 % of pixels,
              invisible in whole-frame PSNR but visible as dots).
Uncertainty:  ISO 100 per-frame PSNR drifts 47.6 -> 48.4 with the daylight; differences are reported
              paired per frame, which cancels the drift.