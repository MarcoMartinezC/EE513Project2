"""The Part C speed-ups must give the same results as the reference versions."""

import numpy as np
import pytest

from isp import demosaic as dm
from isp.calib import correct_defects, correct_defects_reference
from isp.pipeline import srgb_oetf, srgb_oetf_lut


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-12), (np.float32, 2e-6)])
@pytest.mark.parametrize("shape", [(8, 8), (30, 46), (64, 96)])
def test_fast_demosaic_matches_reference(dtype, tol, shape):
    m = np.random.default_rng(0).random(shape).astype(dtype)
    np.testing.assert_allclose(dm.bilinear_fast(m), dm.bilinear(m), atol=tol)
    np.testing.assert_allclose(dm.mhc_fast(m), dm.mhc(m), atol=tol)
    np.testing.assert_allclose(dm.mhc_fast(m, clip=True), dm.mhc(m, clip=True), atol=tol)


def test_fast_demosaic_is_the_pipeline_default():
    m = np.random.default_rng(1).random((16, 16)).astype(np.float32)
    np.testing.assert_array_equal(dm.demosaic(m, "mhc"), dm.mhc_fast(m))
    np.testing.assert_allclose(dm.demosaic(m, "mhc_reference"), dm.mhc(m))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sparse_defects_match_reference(seed):
    rng = np.random.default_rng(seed)
    m = rng.random((40, 60)).astype(np.float32)
    mask = rng.random(m.shape) < 0.03
    mask[0, 0] = mask[-1, -1] = mask[0, 5] = True  # corners and edges
    mask[10, 10] = mask[12, 10] = mask[10, 12] = True  # clustered same-colour defects
    np.testing.assert_allclose(correct_defects(m, mask), correct_defects_reference(m, mask), atol=1e-6)


def test_sparse_defects_all_neighbours_masked_leaves_value():
    m = np.arange(36, dtype=np.float32).reshape(6, 6)
    mask = np.zeros_like(m, dtype=bool)
    mask[0::2, 0::2] = True  # every red site
    np.testing.assert_array_equal(correct_defects(m, mask), correct_defects_reference(m, mask))


def test_srgb_lut_error_below_one_percent_of_8bit_step():
    x = np.random.default_rng(3).random(1_000_000).astype(np.float32) * 1.2 - 0.1
    err = np.abs(srgb_oetf_lut(x) - srgb_oetf(x)).max()
    assert err < 0.01 / 255
