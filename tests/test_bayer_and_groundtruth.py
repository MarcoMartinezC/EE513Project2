import numpy as np
import pytest

from isp.groundtruth import make_burst, mosaic_rggb, quarter_res, residual_noise_report, shift_image
from isp.rawio import cfa_alignment, pack_rggb, tile_rggb, unpack_rggb


def test_pack_unpack_round_trip():
    m = np.random.default_rng(0).random((10, 14))
    np.testing.assert_array_equal(unpack_rggb(pack_rggb(m)), m)


def test_mosaic_round_trip_of_sampled_sites():
    rgb = np.random.default_rng(1).random((8, 12, 3))
    m = mosaic_rggb(rgb)
    np.testing.assert_array_equal(m[0::2, 0::2], rgb[0::2, 0::2, 0])
    np.testing.assert_array_equal(m[1::2, 1::2], rgb[1::2, 1::2, 2])


def test_quarter_res_of_known_mosaic():
    m = tile_rggb([0.1, 0.4, 0.6, 0.9], (6, 8), dtype=np.float64)
    q = quarter_res(m)
    assert q.shape == (3, 4, 3)
    np.testing.assert_allclose(q[..., 0], 0.1)
    np.testing.assert_allclose(q[..., 1], 0.5)
    np.testing.assert_allclose(q[..., 2], 0.9)


@pytest.mark.parametrize("pattern,desc,expected", [
    ([[0, 1], [3, 2]], "RGBG", ("RGGB", (0, 0))),
    ([[1, 0], [2, 3]], "RGBG", ("GRBG", (0, 1))),
    ([[3, 2], [0, 1]], "RGBG", ("GBRG", (1, 0))),
    ([[2, 3], [1, 0]], "RGBG", ("BGGR", (1, 1))),
])
def test_cfa_alignment(pattern, desc, expected):
    assert cfa_alignment(np.array(pattern), desc) == expected


def test_cfa_alignment_rejects_non_bayer():
    with pytest.raises(ValueError):
        cfa_alignment(np.array([[0, 2], [1, 3]]), "RGBG")  # R and B in the same row


def _textured(h, w, dx=0.0, dy=0.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx, yy = xx + dx, yy + dy
    return 0.5 + 0.2 * np.sin(xx / 5.0) * np.cos(yy / 7.0) + 0.1 * np.sin((xx + 2 * yy) / 11.0)


def _bandlimited(n, d, seed=0):
    """Smooth random texture translated by exactly d = (dx, dy) pixels (Fourier shift): img(x) = base(x + d)."""
    from scipy.ndimage import gaussian_filter

    base = gaussian_filter(np.random.default_rng(seed).standard_normal((n, n)), 2.0)
    ky, kx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    return 0.4 + 0.5 * np.real(np.fft.ifft2(np.fft.fft2(base) * np.exp(2j * np.pi * (kx * d[0] + ky * d[1]))))


def test_shift_estimate_recovers_known_translation():
    """Frames at known sub-pixel offsets: estimated packed-pixel shifts must match to 0.05 px."""
    true = [(0.0, 0.0), (0.6, -0.4), (-1.3, 0.8), (0.1, 0.05)]  # sensor pixels; img(x) = base(x + d)
    stack = [mosaic_rggb(np.stack([_bandlimited(400, d)[40:296, 40:296]] * 3, -1)) for d in true]
    burst = make_burst(np.array(stack))
    for (dx, dy), est in zip(true, burst.shifts):
        np.testing.assert_allclose(est, [-dx / 2, -dy / 2], atol=0.05)


def test_alignment_makes_frames_agree():
    """After aligning frame 1 to frame 0, the two should differ far less than before."""
    stack = np.array([mosaic_rggb(np.stack([_bandlimited(400, d)[40:296, 40:296]] * 3, -1))
                      for d in [(0, 0), (0.8, -0.6)]])
    burst = make_burst(stack)
    m = burst.margin
    before = np.abs(burst.packed[1] - burst.packed[0])[m:-m, m:-m].mean()
    after = np.abs(burst.aligned_to(0)[1] - burst.packed[0])[m:-m, m:-m].mean()
    assert after < 0.2 * before  # bicubic resampling of fine texture leaves ~12% (measured)


def test_reference_leaves_out_the_input_frame():
    rng = np.random.default_rng(3)
    base = np.full((16, 16), 0.5)
    stack = np.stack([base + 0.01 * rng.standard_normal(base.shape) for _ in range(4)])
    burst = make_burst(stack, align=False)
    k = 2
    expected = np.delete(burst.packed, k, axis=0).mean(axis=0)
    np.testing.assert_allclose(burst.reference(k), expected, rtol=1e-6)
    np.testing.assert_allclose(burst.reference(k, leave_out=False), burst.packed.mean(0), rtol=1e-6)


def test_residual_noise_recovers_sigma_on_static_burst():
    rng = np.random.default_rng(4)
    sigma = 0.02
    base = _textured(96, 96)
    stack = np.stack([base + sigma * rng.standard_normal(base.shape) for _ in range(16)])
    rep = residual_noise_report(make_burst(stack, align=False), ks=[0, 5, 10])
    assert rep["sigma_single_mean"] == pytest.approx(sigma, rel=0.05)
    assert rep["sigma_reference_mean"] == pytest.approx(sigma / np.sqrt(15), rel=0.05)


def test_shift_image_identity_at_zero():
    img = np.random.default_rng(5).random((20, 20)).astype(np.float32)
    np.testing.assert_allclose(shift_image(img, 0.0, 0.0), img, atol=1e-6)
