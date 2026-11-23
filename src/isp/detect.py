"""Part C: run a pretrained detector on three renderings of each scene and compare.

Renderings per scene: your pipeline's output, the camera's own JPEG, and the
minimal raw path (black level, shading, white balance, demosaic, nothing else;
``pipeline.MINIMAL_STEPS``). The minimal path is linear light fed straight to a
detector trained on display-referred JPEGs. That mismatch is part of what the
comparison measures, so keep it and say so.

All boxes are stored in normalised [0, 1] coordinates so renderings with
slightly different sizes or crops (the in-camera JPEG usually differs from the
raw's visible area by a few pixels) can be compared. If your JPEG is cropped
rather than scaled relative to the raw, map the boxes accordingly before
matching; check this on one scene by overlaying the two images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detections:
    boxes: np.ndarray  # (n, 4) x0, y0, x1, y1 in [0, 1]
    scores: np.ndarray  # (n,)
    labels: np.ndarray  # (n,) int class ids

    def above(self, threshold: float) -> "Detections":
        k = self.scores >= threshold
        return Detections(self.boxes[k], self.scores[k], self.labels[k])

    def __len__(self) -> int:
        return len(self.scores)


class Detector:
    """torchvision Faster R-CNN (COCO). Weights download on first use."""

    def __init__(self, name: str = "fasterrcnn_resnet50_fpn_v2", device: str = "cpu"):
        import torch
        import torchvision.models.detection as tvd

        weights_enum = {
            "fasterrcnn_resnet50_fpn_v2": tvd.FasterRCNN_ResNet50_FPN_V2_Weights,
            "fasterrcnn_mobilenet_v3_large_fpn": tvd.FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            "retinanet_resnet50_fpn_v2": tvd.RetinaNet_ResNet50_FPN_V2_Weights,
        }[name]
        weights = weights_enum.DEFAULT
        self.model = getattr(tvd, name)(weights=weights, box_score_thresh=0.05).to(device).eval()
        self.categories = weights.meta["categories"]
        self.device = device
        self.torch = torch
        self.name = name

    def __call__(self, rgb: np.ndarray) -> Detections:
        """rgb: (H, W, 3) float in [0, 1] or uint8."""
        t = self.torch
        img = rgb.astype(np.float32) / 255.0 if rgb.dtype == np.uint8 else rgb.astype(np.float32)
        h, w = img.shape[:2]
        x = t.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).to(self.device)
        with t.no_grad():
            out = self.model([x])[0]
        boxes = out["boxes"].cpu().numpy() / np.array([w, h, w, h], dtype=np.float32)
        return Detections(boxes, out["scores"].cpu().numpy(), out["labels"].cpu().numpy())


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def agreement(test: Detections, ref: Detections, iou_threshold: float = 0.5) -> dict:
    """Greedy one-to-one matching (same label, IoU >= threshold), highest scores first.

    Returns matched count, precision/recall/F1 of ``test`` taking ``ref`` as the
    reference. ``ref`` is not ground truth; say which rendering you chose as the
    reference and why.
    """
    iou = iou_matrix(test.boxes, ref.boxes)
    same = test.labels[:, None] == ref.labels[None, :] if len(test) and len(ref) else iou.astype(bool)
    iou = np.where(same, iou, 0.0)
    order = np.argsort(-test.scores)
    used = np.zeros(len(ref), dtype=bool)
    matched = 0
    for i in order:
        if len(ref) == 0:
            break
        cand = np.where(~used, iou[i], 0.0)
        j = int(np.argmax(cand))
        if cand[j] >= iou_threshold:
            used[j] = True
            matched += 1
    p = matched / len(test) if len(test) else float("nan")
    r = matched / len(ref) if len(ref) else float("nan")
    f1 = 2 * p * r / (p + r) if len(test) and len(ref) and (p + r) > 0 else (1.0 if not len(test) and not len(ref) else 0.0)
    return {"matched": matched, "n_test": len(test), "n_ref": len(ref), "precision": p, "recall": r, "f1": f1}


def summarise(dets: Detections) -> dict:
    return {"count": len(dets), "mean_confidence": float(dets.scores.mean()) if len(dets) else float("nan")}
