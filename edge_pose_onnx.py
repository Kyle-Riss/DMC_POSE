"""Minimal NumPy/Pillow postprocessing for the single-class YOLO pose ONNX."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def box_iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        order = remaining[box_iou_one(boxes[current], boxes[remaining]) <= iou_threshold]
    return keep


def decode_pose(
    output: np.ndarray,
    *,
    original_shape: tuple[int, int],
    scale: float,
    pad: tuple[int, int],
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[dict]:
    prediction = np.asarray(output)
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError(f"unexpected output shape: {prediction.shape}")
    if prediction.shape[0] == 56:
        prediction = prediction.T
    if prediction.shape[1] != 56:
        raise ValueError(f"expected 56 pose channels, got {prediction.shape}")
    prediction = prediction[prediction[:, 4] >= confidence]
    if not len(prediction):
        return []
    xywh = prediction[:, :4]
    boxes = np.column_stack((
        xywh[:, 0] - xywh[:, 2] / 2,
        xywh[:, 1] - xywh[:, 3] / 2,
        xywh[:, 0] + xywh[:, 2] / 2,
        xywh[:, 1] + xywh[:, 3] / 2,
    ))
    selected = nms(boxes, prediction[:, 4], iou_threshold)
    height, width = original_shape
    pad_x, pad_y = pad
    results = []
    for index in selected:
        box = boxes[index].copy()
        box[[0, 2]] = np.clip((box[[0, 2]] - pad_x) / scale, 0, width)
        box[[1, 3]] = np.clip((box[[1, 3]] - pad_y) / scale, 0, height)
        keypoints = prediction[index, 5:].reshape(17, 3).copy()
        keypoints[:, 0] = np.clip((keypoints[:, 0] - pad_x) / scale, 0, width)
        keypoints[:, 1] = np.clip((keypoints[:, 1] - pad_y) / scale, 0, height)
        results.append({
            "score": float(prediction[index, 4]),
            "bbox_xyxy": box.tolist(),
            "keypoints_xyc": keypoints.tolist(),
            "visible_keypoints": int(np.count_nonzero(keypoints[:, 2] >= 0.25)),
        })
    return results


class EdgePoseONNX:
    def __init__(self, model_path: str | Path, *, input_size: int = 320):
        import onnxruntime as ort

        self.input_size = int(input_size)
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, rgb: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        from PIL import Image

        height, width = rgb.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = Image.fromarray(rgb).resize(resized, Image.Resampling.BILINEAR)
        pad_x = (self.input_size - resized[0]) // 2
        pad_y = (self.input_size - resized[1]) // 2
        canvas = Image.new("RGB", (self.input_size, self.input_size), (114, 114, 114))
        canvas.paste(image, (pad_x, pad_y))
        tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return tensor, scale, (pad_x, pad_y)

    def infer(self, rgb: np.ndarray, *, confidence: float = 0.25) -> dict:
        started = time.perf_counter()
        tensor, scale, pad = self.preprocess(rgb)
        preprocessed = time.perf_counter()
        output = self.session.run(None, {self.input_name: tensor})[0]
        inferred = time.perf_counter()
        detections = decode_pose(
            output,
            original_shape=rgb.shape[:2],
            scale=scale,
            pad=pad,
            confidence=confidence,
        )
        finished = time.perf_counter()
        return {
            "detections": detections,
            "preprocess_ms": (preprocessed - started) * 1000,
            "inference_ms": (inferred - preprocessed) * 1000,
            "postprocess_ms": (finished - inferred) * 1000,
            "total_ms": (finished - started) * 1000,
        }

