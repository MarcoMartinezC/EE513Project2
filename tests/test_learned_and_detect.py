import numpy as np
import pytest

from isp.detect import Detections, agreement, iou_matrix

torch = pytest.importorskip("torch")

from isp.learned.data import SyntheticPairs, bin2  # noqa: E402
from isp.learned.infer import tiled  # noqa: E402
from isp.learned.model import DnCNN, conv_params, count_parameters  # noqa: E402
from isp.noise_model import NoiseModel  # noqa: E402


def test_parameter_count_matches_hand_formula():
    m = DnCNN(width=32, depth=8, noise_map=True)
    expected = conv_params(8, 32) + 6 * conv_params(32, 32) + conv_params(32, 4)
    assert count_parameters(m) == expected == 58_980


def test_training_learns_something(tmp_path):
    """A short run on smooth images must beat the noisy input (guards against dead-ReLU collapse)."""
    from isp.learned.train import train

    yy, xx = np.mgrid[0:96, 0:96]
    clean = [np.stack([0.3 + 0.2 * np.sin(xx / 9.0 + c) * np.cos(yy / 13.0) for c in range(4)], -1).astype(np.float32)]
    meta = train(clean, [NoiseModel(0.02, 30.0, 256, 4095)], tmp_path, steps=500, batch=8, patch=32,
                 width=16, depth=5, device="cpu", log_every=1000)
    assert meta["final_loss_over_noisy_mse"] < 0.8


def test_dataset_shapes_and_determinism():
    clean = [np.random.default_rng(0).random((80, 80, 4)).astype(np.float32)]
    nm = NoiseModel(0.5, 4.0, 256, 4095)
    ds = SyntheticPairs(clean, [nm], patch=32, length=10, seed=1)
    a, b = ds[3], ds[3]
    assert a[0].shape == (4, 32, 32)
    for u, v in zip(a, b):
        assert torch.equal(u, v)


def test_bin2_halves_noise():
    rng = np.random.default_rng(1)
    p = rng.standard_normal((200, 200, 4))
    assert bin2(p).std() == pytest.approx(0.5, rel=0.05)


def test_tiled_inference_equals_whole_image():
    m = DnCNN(depth=4).eval()

    def fn(p):
        x = torch.from_numpy(np.ascontiguousarray(p.transpose(2, 0, 1)))[None]
        with torch.no_grad():
            return m(x, torch.full_like(x, 0.01))[0].numpy().transpose(1, 2, 0)

    p = np.random.default_rng(2).random((70, 90, 4)).astype(np.float32)
    np.testing.assert_allclose(tiled(fn, p, tile=32, pad=8), fn(p), atol=1e-5)


def test_onnx_export_matches_pytorch(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from isp.learned.export import compare_onnx, export_onnx

    m = DnCNN(depth=4).eval()
    path = export_onnx(m, tmp_path / "m.onnx")
    assert compare_onnx(m, path, size=48) < 1e-5


def _d(boxes, scores, labels):
    return Detections(np.array(boxes, float).reshape(-1, 4), np.array(scores, float), np.array(labels, int))


def test_iou():
    a = np.array([[0, 0, 0.5, 0.5]])
    b = np.array([[0.25, 0.25, 0.75, 0.75], [0, 0, 0.5, 0.5]])
    np.testing.assert_allclose(iou_matrix(a, b), [[0.0625 / 0.4375, 1.0]])


def test_agreement_matches_same_class_only():
    ref = _d([[0, 0, .4, .4], [.5, .5, .9, .9]], [.9, .8], [1, 3])
    test = _d([[0.01, 0, .4, .41], [.5, .5, .9, .9]], [.7, .6], [1, 62])  # second box: wrong class
    r = agreement(test, ref)
    assert r["matched"] == 1 and r["precision"] == 0.5 and r["recall"] == 0.5


def test_agreement_empty_cases():
    empty = _d([], [], [])
    assert agreement(empty, empty)["f1"] == 1.0
    assert agreement(_d([[0, 0, 1, 1]], [.9], [1]), empty)["f1"] == 0.0
