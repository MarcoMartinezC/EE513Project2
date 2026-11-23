# Decisions and verification log

A running log to write the one-page decisions-and-verification statement from.
Add an entry when you choose between alternatives, when you check something, or
when something surprises you. Dated entries also back up the commit history.

Template:

```
## YYYY-MM-DD  <short title>
Decision:     what you chose
Rejected:     what you didn't, and why
Verified by:  test / known case / comparison / figure (file and number)
AI use:       tool, what for, and how you checked its output
Still unsure: ...
```

---

## Starting scaffold (AI-generated, to be verified)

The initial repository structure, modules, scripts and tests were generated
with Claude from my Exercise 5 (demosaicing), training-without-clean-images and
PyTorch on-ramp notebooks and the Project 2 spec. Before relying on any part of
it, I should check it against my own captures and record the result here.

Design choices made in the scaffold that I need to own or change:

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

Things the scaffold found on the synthetic camera that I should check on my data:

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
      Error is ~0.01 px rms on the synthetic camera. My exercise burst reported
      a median shift of 0.77 px using phaseCorrelate on one green plane; re-measure it.
- [ ] Temporal variance across aligned frames reads ~20% low, because
      resampling averages noise. The noise fit uses unresampled frame minus
      leave-one-out reference instead.
- [ ] Read noise fitted from a burst is an extrapolated intercept and came out
      5-10% low on the synthetic camera; the dark-frame value from Project 1
      is more trustworthy.
- [ ] The averaged-burst demosaic reference is built from resampled frames,
      which are slightly blurred, and that flatters both demosaicers. On the
      synthetic camera (all frames jittered up to 0.7 px) both scored about
      3 dB higher against the average than against a single frame, at ISO 100
      where noise is not the explanation. On a steady tripod with shifts below
      `--min-shift`, nothing is resampled. Report the shifts, and say whether
      the reference was resampled.
- [ ] The ablation's best denoiser strength hit the top of the grid on some
      bursts: PSNR rewards blur. Look at the crops before trusting the tuned value.
- [ ] Check the denoising reference itself: `eval_denoisers.py` prints the
      std of (noisy − reference) and what the noise model predicts. On the
      synthetic camera at ISO 1600: 0.0217 vs 0.0155 for frame 0, and 0.032 vs
      0.016 averaged over frames 0, 7 and 14, which moved further from frame 0. The extra is
      reference error from resampling fine detail, not noise, and it caps
      every PSNR. (Same model, synthetic noise on the reference: learned gained
      5.5 dB; on the real frames, 2.6 dB.)
