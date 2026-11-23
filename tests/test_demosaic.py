import numpy as np
import pytest

from isp import demosaic as dm
from isp.groundtruth import mosaic_rggb


@pytest.mark.parametrize("method", ["bilinear", "mhc"])
@pytest.mark.parametrize("shape", [(8, 8), (16, 24), (6, 10)])
def test_constant_colour_reproduced_exactly(method, shape):
    """A flat colour must come back exactly, including at the border (reflect padding, normalised weights)."""
    rgb = np.empty(shape + (3,))
    rgb[...] = [0.25, 0.5, 0.75]  # dyadic values: exact in binary
    out = dm.demosaic(mosaic_rggb(rgb), method)
    np.testing.assert_array_equal(out, rgb)


@pytest.mark.parametrize("method", ["bilinear", "mhc"])
def test_measured_sites_unchanged(method):
    m = np.random.default_rng(0).random((32, 48))
    out = dm.demosaic(m, method)
    np.testing.assert_array_equal(out[0::2, 0::2, 0], m[0::2, 0::2])
    np.testing.assert_array_equal(out[0::2, 1::2, 1], m[0::2, 1::2])
    np.testing.assert_array_equal(out[1::2, 0::2, 1], m[1::2, 0::2])
    np.testing.assert_array_equal(out[1::2, 1::2, 2], m[1::2, 1::2])


def test_bilinear_reproduces_linear_ramp_in_interior():
    """Bilinear interpolation is exact for a plane; MHC too (its corrections are Laplacians)."""
    yy, xx = np.mgrid[0:20, 0:20].astype(np.float64)
    ramp = 0.01 * xx + 0.02 * yy
    rgb = np.stack([ramp, ramp + 0.1, ramp + 0.2], -1)
    for method in ("bilinear", "mhc"):
        out = dm.demosaic(mosaic_rggb(rgb), method)
        np.testing.assert_allclose(out[3:-3, 3:-3], rgb[3:-3, 3:-3], atol=1e-12)


def test_dtype_preserved():
    m = np.random.default_rng(1).random((8, 8)).astype(np.float32)
    assert dm.bilinear(m).dtype == np.float32
    assert dm.mhc(m).dtype == np.float32


def test_mhc_filters_sum_to_one():
    for name, k in dm.MHC_FILTERS.items():
        assert k.sum() == pytest.approx(1.0), name


def test_noise_gain_values_from_exercise5():
    assert dm.noise_gain(dm.F_G_AT_RB) == pytest.approx(36 / 64)
    assert dm.noise_gain(dm.F_RB_AT_G_ROW) == pytest.approx(63.5 / 64)
    assert dm.noise_gain(dm.F_RB_AT_BR) == pytest.approx(61 / 64)


def test_reflect_border_breaks_mhc_on_flat_colour():
    """Documents why the default border mode is 'mirror' (see demosaic.py docstring)."""
    rgb = np.empty((8, 10, 3))
    rgb[...] = [0.25, 0.5, 0.75]
    m = mosaic_rggb(rgb)
    assert np.abs(dm.mhc(m, mode="reflect") - rgb).max() > 0.3
    assert np.abs(dm.mhc(m, mode="reflect") - rgb)[2:-2, 2:-2].max() == 0  # interior is fine


def test_matches_course_reference_if_available():
    """make_figures.py and psu_imaging.py are on Canvas; put them on PYTHONPATH to run this."""
    mf = pytest.importorskip("make_figures")
    m = np.random.default_rng(2).random((64, 64))
    assert np.abs(dm.bilinear(m, mode="reflect") - mf.bilinear_demosaic(m)).max() < 1e-12
    assert np.abs(dm.mhc(m, clip=True, mode="reflect") - mf.mhc_demosaic(m)).max() < 1e-12
