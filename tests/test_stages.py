import numpy as np
import pytest

from isp import pipeline as P
from isp.calib import correct_defects, hot_pixels_from_darks, shading_gain_from_flat
from isp.rawio import tile_rggb


def ctx(**kw):
    base = dict(white=4095.0, black_rggb=np.array([256.0, 256.0, 256.0, 256.0]),
                wb_gains=np.array([2.0, 1.0, 1.5]), ccm=np.eye(3))
    base.update(kw)
    return P.Context(**base)


def test_black_level_maps_black_to_zero_and_white_to_one():
    c = ctx(black_rggb=np.array([250.0, 256.0, 258.0, 260.0]))
    m = tile_rggb(c.black_rggb, (4, 4)).astype(np.float32)
    x = P.black_level(P.to_unit(m, c), c)
    np.testing.assert_allclose(x, 0.0, atol=1e-6)
    x = P.black_level(P.to_unit(np.full((4, 4), 4095.0, np.float32), c), c)
    np.testing.assert_allclose(x, 1.0, atol=1e-6)


def test_black_level_same_in_both_domains_for_flat_field():
    """Mosaic and RGB versions must agree on a flat field (the ablation depends on it)."""
    c = ctx()
    m = np.full((4, 4), 1000.0, np.float32)
    a = P.black_level(P.to_unit(m, c), c)
    b = P.black_level(P.to_unit(np.full((4, 4, 3), 1000.0, np.float32), c), c)
    np.testing.assert_allclose(a, b[..., 0], rtol=1e-6)


def test_unit_gains_and_identity_ccm_are_identity():
    c = ctx(wb_gains=np.ones(3), ccm=np.eye(3))
    rgb = np.random.default_rng(0).random((6, 6, 3)).astype(np.float32)
    np.testing.assert_array_equal(P.wb(rgb, c), rgb)
    np.testing.assert_allclose(P.ccm(rgb, c), rgb, atol=1e-7)


def test_wb_mosaic_and_rgb_agree_after_demosaic():
    """WB is per-channel scaling and bilinear is linear per channel: order must not matter for bilinear."""
    c = ctx()
    m = np.random.default_rng(1).random((16, 16)).astype(np.float32)
    a = P.demosaic(P.wb(m, c), c, method="bilinear")
    b = P.wb(P.demosaic(m, c, method="bilinear"), c)
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_wb_order_matters_for_mhc():
    """...but not for MHC, whose correction mixes channels. This is the report's 'why WB first'."""
    c = ctx()
    m = np.random.default_rng(2).random((16, 16)).astype(np.float32)
    a = P.demosaic(P.wb(m, c), c, method="mhc")
    b = P.wb(P.demosaic(m, c, method="mhc"), c)
    assert np.abs(a - b).max() > 1e-3


def test_ccm_refuses_mosaic():
    with pytest.raises(ValueError):
        P.ccm(np.zeros((4, 4), np.float32), ctx())


def test_defect_correction_is_idempotent_and_fixes_hot_pixel():
    m = np.full((12, 12), 0.3, np.float32)
    mask = np.zeros_like(m, dtype=bool)
    m[4, 6] = 1.0
    mask[4, 6] = True
    m[5, 5], mask[5, 5] = 0.9, True  # a neighbouring defect of another colour
    once = correct_defects(m, mask)
    twice = correct_defects(once, mask)
    np.testing.assert_array_equal(once, twice)
    assert once[4, 6] == pytest.approx(0.3)
    assert once[5, 5] == pytest.approx(0.3)
    np.testing.assert_array_equal(once[~mask], m[~mask])


def test_hot_pixels_found_in_darks():
    rng = np.random.default_rng(3)
    darks = 256 + 2 * rng.standard_normal((6, 32, 32))
    darks[:, 10, 11] += 200
    mask = hot_pixels_from_darks(darks, k=8)
    assert mask[10, 11] and mask.sum() == 1


def test_shading_correction_flattens_its_own_flat():
    h, w = 64, 96
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    fall = np.stack([1 - f * r2 / 2 for f in (0.5, 0.4, 0.4, 0.6)], -1)  # per packed channel at full res
    level = 0.5 * fall
    mosaic = np.empty((h, w))
    mosaic[0::2, 0::2] = level[0::2, 0::2, 0]
    mosaic[0::2, 1::2] = level[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = level[1::2, 0::2, 2]
    mosaic[1::2, 1::2] = level[1::2, 1::2, 3]
    dn = 256 + mosaic * (4095 - 256)
    gain = shading_gain_from_flat(dn[None], [256] * 4, 4095.0, method="poly", degree=4)
    from isp.calib import normalise
    from isp.rawio import pack_rggb

    flat = pack_rggb(normalise(dn, [256] * 4, 4095.0)) * gain
    for c in range(4):
        assert flat[..., c].std() / flat[..., c].mean() < 0.01


def test_highlight_clip_and_tone_are_idempotent():
    c = ctx()
    x = np.random.default_rng(4).random((8, 8, 3)).astype(np.float32) * 2
    once = P.clip_highlights(x, c)
    np.testing.assert_array_equal(P.clip_highlights(once, c), once)
    t = P.tone(x, c, curve="srgb")
    assert t.min() >= 0 and t.max() <= 1


def test_srgb_curve_round_trip():
    v = np.linspace(0, 1, 101, dtype=np.float32)
    np.testing.assert_allclose(P.srgb_oetf(P.srgb_eotf(v)), v, atol=1e-5)


def test_pipeline_editing_helpers():
    steps = P.DEFAULT_STEPS
    moved = P.move_after(steps, "wb", "demosaic")
    names = P.names(moved)
    assert names.index("wb") == names.index("demosaic") + 1
    assert "lsc" not in P.names(P.without(steps, "lsc"))
    assert dict(P.with_params(steps, "demosaic", method="bilinear"))["demosaic"]["method"] == "bilinear"
