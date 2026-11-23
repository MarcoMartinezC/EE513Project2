"""The whole classical pipeline on the simulated camera, against constants it knows."""

import numpy as np
import pytest

from isp import pipeline as P
from isp.calib import hot_pixels_from_darks, lsc_rgb_full, shading_gain_from_flat
from isp.groundtruth import make_burst
from isp.metrics import psnr
from isp.noise_model import NoiseModel, fit_from_burst
from isp.synthetic import SyntheticCamera, scene_linear_srgb


@pytest.fixture(scope="module")
def cam():
    return SyntheticCamera()


def test_pipeline_recovers_scene(cam):
    h, w = 192, 256
    scene = scene_linear_srgb(h, w)
    m = cam.capture(scene, 100, np.random.default_rng(0))
    hot, stuck = cam.defects(m.shape)
    ctx = P.Context(white=cam.white, black_rggb=np.full(4, cam.black), wb_gains=np.array(cam.wb_gains),
                    ccm=cam.ccm, lsc_rgb=1 / cam.vignette_rgb(h, w), defect_mask=hot | stuck,
                    noise=NoiseModel(cam.gain(100), cam.read(100), cam.black, cam.white))
    out = P.run(m, ctx, P.without(P.DEFAULT_STEPS, "sharpen")).image
    ref = P.srgb_oetf(np.clip(scene, 0, 1))
    assert psnr(out, ref, margin=4) > 24  # the zone plate and stripes alias; the rest is close
    wrong = P.run(m, ctx, P.without(P.without(P.DEFAULT_STEPS, "sharpen"), "black_level")).image
    assert psnr(wrong, ref, margin=4) < psnr(out, ref, margin=4) - 3


def test_calibration_recovers_defects_and_shading(cam):
    h, w = 192, 256
    rng = np.random.default_rng(1)
    darks = np.stack([cam.capture(np.zeros((h, w, 3), np.float32), 6400, rng) for _ in range(6)])
    hot_true, stuck_true = cam.defects((h, w))
    bad = hot_true | stuck_true  # stuck-at-white sites are bright in darks too
    hot = hot_pixels_from_darks(darks)
    assert (hot & bad).sum() == bad.sum()
    assert (hot & ~bad).sum() == 0

    flats = np.stack([cam.capture(np.full((h, w, 3), 0.55, np.float32), 100, rng) for _ in range(6)])
    _, stuck = cam.defects((h, w))
    gain = shading_gain_from_flat(flats, [cam.black] * 4, cam.white, defect_mask=hot_true | stuck)
    true = 1 / cam.vignette_rgb(h, w)
    assert np.abs(lsc_rgb_full(gain) / true - 1).max() < 0.05


def test_noise_fit_from_burst_recovers_gain(cam):
    h, w = 256, 256
    rng = np.random.default_rng(2)
    iso = 1600
    shifts = [(0, 0)] + [tuple(rng.uniform(-0.6, 0.6, 2)) for _ in range(11)]
    stack = np.stack([cam.capture(scene_linear_srgb(h, w, dx, dy), iso, rng) for dx, dy in shifts])
    hot, stuck = cam.defects((h, w))
    norm = (stack - cam.black) / (cam.white - cam.black)
    from isp.calib import correct_defects

    norm = np.stack([correct_defects(f.astype(np.float32), hot | stuck) for f in norm])
    _, rep = fit_from_burst(make_burst(norm), cam.black, cam.white, ks=[0, 4, 8])
    assert rep["gain_e_per_dn"][0] == pytest.approx(cam.gain(iso), rel=0.08)


def test_registration_survives_aliasing_detail(cam):
    """Zone plate and near-Nyquist stripes alias in the packed planes; the full-resolution proxy must not care."""
    rng = np.random.default_rng(5)
    true = np.array([(0.0, 0.0)] + [tuple(rng.uniform(-0.7, 0.7, 2)) for _ in range(5)])
    stack = np.stack([cam.capture(scene_linear_srgb(512, 768, dx, dy), 1600, rng) for dx, dy in true])
    hot, stuck = cam.defects(stack.shape[1:])
    est = make_burst(stack, defect_mask=hot | stuck).shifts * 2  # packed -> sensor px
    np.testing.assert_allclose(est, -true, atol=0.05)
