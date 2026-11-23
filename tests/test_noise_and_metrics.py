import json

import numpy as np
import pytest

from isp.denoise_classical import denoise_packed, guided_filter
from isp.metrics import mean_std, psnr, psnr_tile_bootstrap
from isp.noise_model import NoiseModel
from isp.results import fill_template, pair


def test_sampled_variance_matches_model():
    nm = NoiseModel(gain_e_per_dn=0.5, read_noise_dn=4.0, black=256, white=4095)
    rng = np.random.default_rng(0)
    for level in (0.05, 0.3, 0.7):
        x = nm.sample(np.full(200_000, level, np.float32), rng)
        assert x.var() == pytest.approx(nm.variance(level), rel=0.02)
        assert x.mean() == pytest.approx(level, abs=3 * np.sqrt(nm.variance(level) / x.size))


def test_gat_stabilises_variance():
    nm = NoiseModel(0.3, 6.0, 256, 4095)
    rng = np.random.default_rng(1)
    for level in (0.02, 0.2, 0.8):
        y = nm.gat(nm.sample(np.full(100_000, level, np.float32), rng))
        assert y.std() == pytest.approx(1.0, rel=0.05)


def test_gat_inverse_round_trip():
    nm = NoiseModel(1.0, 3.0, 256, 4095)
    x = np.linspace(0.01, 1, 50)
    np.testing.assert_allclose(nm.gat_inverse(nm.gat(x)), x, rtol=1e-6)


def test_guided_filter_preserves_constant_and_tends_to_identity():
    rng = np.random.default_rng(2)
    c = np.full((20, 20), 0.4, np.float32)
    np.testing.assert_allclose(guided_filter(c, c, 2, 0.01), c, atol=1e-6)
    img = rng.random((20, 20)).astype(np.float32)
    np.testing.assert_allclose(guided_filter(img, img, 2, 1e-9), img, atol=1e-3)


def test_guided_denoiser_reduces_error():
    nm = NoiseModel(0.2, 8.0, 256, 4095)
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:64, 0:64]
    clean = np.repeat((0.3 + 0.2 * (xx > 32))[..., None], 4, -1).astype(np.float32)
    noisy = nm.sample(clean, rng)
    out = denoise_packed(noisy, 2, 4.0, nm)
    assert psnr(out, clean) > psnr(noisy, clean) + 3


def test_psnr_known_value():
    a = np.zeros((10, 10))
    assert psnr(a + 0.1, a) == pytest.approx(20.0)


def test_tile_bootstrap_brackets_whole_image_psnr():
    rng = np.random.default_rng(4)
    ref = rng.random((256, 256))
    est = ref + 0.01 * rng.standard_normal(ref.shape)
    p, s = psnr_tile_bootstrap(est, ref, tile=32, n_boot=200)
    assert p == pytest.approx(psnr(est, ref), abs=1e-9)
    assert 0 < s < 0.5


def test_mean_std():
    m, s, n = mean_std([1.0, 2.0, 3.0])
    assert (m, s, n) == (2.0, 1.0, 3)


def test_pair_validation():
    assert pair(1.5, 0.1) == [1.5, 0.1]
    for bad in [(float("nan"), 0.1), (1.0, -0.1), (1.0, float("inf"))]:
        with pytest.raises(ValueError):
            pair(*bad)


def test_fill_template_refuses_unknown_keys(tmp_path):
    tpl = tmp_path / "t.json"
    tpl.write_text(json.dumps({"a": {"b": None}, "c": None}))
    out = tmp_path / "o.json"
    data = fill_template(tpl, out, {"a.b": [1.0, 0.1]})
    assert data["a"]["b"] == [1.0, 0.1] and data["c"] is None
    with pytest.raises(KeyError):
        fill_template(tpl, out, {"a.x": [1.0, 0.1]})
