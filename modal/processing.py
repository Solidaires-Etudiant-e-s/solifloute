"""Face detection and blurring, faithful port of the TypeScript SoliFloute pipeline.

This module reproduces the exact preprocessing, model output decoding, NMS and
box-blur behaviour used by ``shared/utils/faceDetectionCore.ts``,
``shared/utils/nms.ts`` and ``shared/utils/imageProcessing.ts`` so that the
Modal cloud inference produces results consistent with the local server and
browser pipelines.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

MAX_STRIDE = 32
DEFAULT_PROBABILITY_THRESHOLD = 0.2
MIN_PROBABILITY_THRESHOLD = 0.1
NMS_THRESHOLD = 0.3
MAX_CANDIDATES = 5000

CENTER_FACE_STRIDE = 4
YU_NET_STRIDES = (8, 16, 32)

MASK_SCALE = 1.3
MIN_BLUR_RADIUS = 2


def create_session(model_path: str, gpu: bool = True):
    """Create an onnxruntime session with the best available providers.

    On GPU: tries TensorRT EP first, then CUDA EP, with full graph optimization.
    Falls back gracefully when a provider's runtime libraries are missing.
    """
    import onnxruntime

    if not gpu:
        return onnxruntime.InferenceSession(model_path)

    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    try:
        available = set(onnxruntime.get_available_providers())
        providers = []
        for name in ("CUDAExecutionProvider", "CPUExecutionProvider"):
            if name in available:
                providers.append(name)
        if not providers:
            providers = None
        return onnxruntime.InferenceSession(
            model_path, sess_options=session_options, providers=providers
        )
    except Exception:  # noqa: BLE001 - provider init can fail for many reasons
        return onnxruntime.InferenceSession(model_path)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def get_padded_input_size(width: int, height: int) -> tuple[int, int]:
    return (
        math.ceil(width / MAX_STRIDE) * MAX_STRIDE,
        math.ceil(height / MAX_STRIDE) * MAX_STRIDE,
    )


def create_model_input_data(rgba: np.ndarray, padded_width: int, padded_height: int) -> np.ndarray:
    """Build the [1, 3, H, W] float32 tensor from raw RGBA bytes (0-255).

    Mirrors ``createModelInputData`` in the TypeScript pipeline: values are
    copied as-is (no normalisation) and the padded region is zero-filled.
    """
    source_height, source_width = rgba.shape[0], rgba.shape[1]
    copy_width = min(padded_width, source_width)
    copy_height = min(padded_height, source_height)
    output = np.zeros((1, 3, padded_height, padded_width), dtype=np.float32)
    region = rgba[:copy_height, :copy_width, :3]
    output[0, 0, :copy_height, :copy_width] = region[:, :, 0]
    output[0, 1, :copy_height, :copy_width] = region[:, :, 1]
    output[0, 2, :copy_height, :copy_width] = region[:, :, 2]
    return output


def intersection_over_union(a: dict, b: dict) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    intersection = width * height
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def hard_non_max_suppression(boxes: list[dict], iou_threshold: float = 0.3, top_k: int = -1) -> list[dict]:
    """Port of ``hardNonMaxSuppression`` from ``shared/utils/nms.ts``."""
    sorted_boxes = sorted(boxes, key=lambda box: box["score"], reverse=True)
    selected: list[dict] = []

    while sorted_boxes:
        candidate = sorted_boxes.pop(0)
        selected.append(candidate)

        if top_k > 0 and len(selected) >= top_k:
            break

        for index in range(len(sorted_boxes) - 1, -1, -1):
            if intersection_over_union(candidate, sorted_boxes[index]) > iou_threshold:
                sorted_boxes.pop(index)

    return selected


def decode_centerface_outputs(outputs: dict[str, np.ndarray], width: int, height: int, probability_threshold: float) -> list[dict]:
    heatmap = outputs["537"]
    scale = outputs["538"]
    offset = outputs["539"]
    feature_width = math.floor(width / CENTER_FACE_STRIDE)
    feature_height = math.floor(height / CENTER_FACE_STRIDE)
    feature_count = feature_width * feature_height

    threshold = max(probability_threshold, MIN_PROBABILITY_THRESHOLD)
    index = np.arange(feature_count)
    rows = index // feature_width
    cols = index % feature_width

    scores = heatmap[:feature_count]
    mask = scores >= threshold
    picked = index[mask]
    if picked.size == 0:
        return []

    height_scale = np.exp(scale[:feature_count][picked]) * CENTER_FACE_STRIDE
    width_scale = np.exp(scale[feature_count:feature_count * 2][picked]) * CENTER_FACE_STRIDE
    offset_y = offset[:feature_count][picked]
    offset_x = offset[feature_count:feature_count * 2][picked]
    row = rows[picked]
    col = cols[picked]

    y1 = (row + offset_y + 0.5) * CENTER_FACE_STRIDE - (height_scale / 2)
    x1 = (col + offset_x + 0.5) * CENTER_FACE_STRIDE - (width_scale / 2)
    x2 = x1 + width_scale
    y2 = y1 + height_scale
    score = scores[picked]

    candidates = [
        {"x1": float(a), "y1": float(b), "x2": float(c), "y2": float(d), "score": float(e)}
        for a, b, c, d, e in zip(x1, y1, x2, y2, score)
    ]
    candidates.sort(key=lambda box: box["score"], reverse=True)
    candidates = candidates[:MAX_CANDIDATES]
    return hard_non_max_suppression(candidates, NMS_THRESHOLD)


def decode_yunet_outputs(outputs: dict[str, np.ndarray], width: int, height: int, probability_threshold: float) -> list[dict]:
    candidates_x1: list[np.ndarray] = []
    candidates_y1: list[np.ndarray] = []
    candidates_x2: list[np.ndarray] = []
    candidates_y2: list[np.ndarray] = []
    candidates_score: list[np.ndarray] = []

    for stride in YU_NET_STRIDES:
        cls = outputs[f"cls_{stride}"]
        obj = outputs[f"obj_{stride}"]
        bbox = outputs[f"bbox_{stride}"]
        columns = math.floor(width / stride)
        rows = math.floor(height / stride)
        grid = columns * rows

        index = np.arange(grid)
        row = index // columns
        column = index % columns

        cls_score = np.clip(cls[:grid], 0, 1)
        obj_score = np.clip(obj[:grid], 0, 1)
        score = np.sqrt(cls_score * obj_score)
        threshold = max(probability_threshold, MIN_PROBABILITY_THRESHOLD)
        mask = score >= threshold
        picked = index[mask]
        if picked.size == 0:
            continue

        bbox_reshaped = bbox[: grid * 4].reshape(grid, 4)
        bx = bbox_reshaped[picked, 0]
        by = bbox_reshaped[picked, 1]
        bw = bbox_reshaped[picked, 2]
        bh = bbox_reshaped[picked, 3]

        center_x = (column[picked] + bx) * stride
        center_y = (row[picked] + by) * stride
        box_width = np.exp(bw) * stride
        box_height = np.exp(bh) * stride

        candidates_x1.append(center_x - (box_width / 2))
        candidates_y1.append(center_y - (box_height / 2))
        candidates_x2.append(center_x + (box_width / 2))
        candidates_y2.append(center_y + (box_height / 2))
        candidates_score.append(score[picked])

    if not candidates_x1:
        return []

    x1 = np.concatenate(candidates_x1)
    y1 = np.concatenate(candidates_y1)
    x2 = np.concatenate(candidates_x2)
    y2 = np.concatenate(candidates_y2)
    score = np.concatenate(candidates_score)

    order = np.argsort(score)[::-1][:MAX_CANDIDATES]
    candidates = [
        {"x1": float(a), "y1": float(b), "x2": float(c), "y2": float(d), "score": float(e)}
        for a, b, c, d, e in zip(x1[order], y1[order], x2[order], y2[order], score[order])
    ]
    return hard_non_max_suppression(candidates, NMS_THRESHOLD, MAX_CANDIDATES)


def map_faces(candidates: Iterable[dict], source_width: int, source_height: int) -> list[dict]:
    faces = []
    for index, candidate in enumerate(candidates):
        x1 = clamp(candidate["x1"], 0, source_width)
        y1 = clamp(candidate["y1"], 0, source_height)
        x2 = clamp(candidate["x2"], 0, source_width)
        y2 = clamp(candidate["y2"], 0, source_height)

        faces.append({
            "id": f"face-{index + 1}",
            "x": round(x1),
            "y": round(y1),
            "width": round(x2 - x1),
            "height": round(y2 - y1),
            "confidence": round(float(candidate["score"]), 4),
        })

    return faces


def detect_faces(
    session,
    rgba: np.ndarray,
    model_type: str,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
) -> list[dict]:
    """Run a detection model over RGBA pixels and return SoliFloute-style faces."""
    source_width = rgba.shape[1]
    source_height = rgba.shape[0]
    padded_width, padded_height = get_padded_input_size(source_width, source_height)
    tensor = create_model_input_data(rgba, padded_width, padded_height)
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    # The TypeScript pipeline reads every output as a flat, row-major array
    # (Float32Array). Flatten so indexing matches it exactly.
    output_map = {out.name: output.reshape(-1) for out, output in zip(session.get_outputs(), outputs)}

    if model_type == "yunet":
        decoded = decode_yunet_outputs(output_map, padded_width, padded_height, probability_threshold)
    else:
        decoded = decode_centerface_outputs(output_map, padded_width, padded_height, probability_threshold)

    return map_faces(decoded, source_width, source_height)


def scale_face_around_center(face: dict, scale: float) -> dict:
    width = max(1, round(face["width"] * scale))
    height = max(1, round(face["height"] * scale))
    return {
        **face,
        "x": round(face["x"] - ((width - face["width"]) / 2)),
        "y": round(face["y"] - ((height - face["height"]) / 2)),
        "width": width,
        "height": height,
    }


def box_blur_horizontal(input_arr: np.ndarray, width: int, height: int, radius: int) -> np.ndarray:
    window_size = (radius * 2) + 1
    pad_left = np.repeat(input_arr[:, :1], radius, axis=1)
    pad_right = np.repeat(input_arr[:, -1:], radius, axis=1)
    padded = np.concatenate([pad_left, input_arr, pad_right], axis=1).astype(np.float64)
    csum = np.concatenate([np.zeros((height, 1, padded.shape[2])), np.cumsum(padded, axis=1)], axis=1)
    starts = np.arange(width)
    ends = starts + window_size
    return np.round((csum[:, ends] - csum[:, starts]) / window_size).astype(input_arr.dtype)


def box_blur_vertical(input_arr: np.ndarray, width: int, height: int, radius: int) -> np.ndarray:
    window_size = (radius * 2) + 1
    pad_top = np.repeat(input_arr[:1, :], radius, axis=0)
    pad_bottom = np.repeat(input_arr[-1:, :], radius, axis=0)
    padded = np.concatenate([pad_top, input_arr, pad_bottom], axis=0).astype(np.float64)
    csum = np.concatenate([np.zeros((1, width, padded.shape[2])), np.cumsum(padded, axis=0)], axis=0)
    starts = np.arange(height)
    ends = starts + window_size
    return np.round((csum[ends, :] - csum[starts, :]) / window_size).astype(input_arr.dtype)


def blur_region(rgba: np.ndarray, width: int, height: int, radius_x: int, radius_y: int) -> np.ndarray:
    if radius_x <= 1 and radius_y <= 1:
        return rgba

    if radius_x <= 1:
        return box_blur_vertical(rgba, width, height, radius_y)

    temporary = box_blur_horizontal(rgba, width, height, radius_x)

    if radius_y <= 1:
        return temporary

    return box_blur_vertical(temporary, width, height, radius_y)


def apply_blur_effects(
    rgba: np.ndarray,
    faces: list[dict],
    excluded_face_ids: Iterable[str] = (),
    blur_intensity: float = 0.5,
) -> np.ndarray:
    """Port of ``applyBlurEffects`` from ``shared/utils/imageProcessing.ts``."""
    output = rgba.copy()
    excluded = set(excluded_face_ids)
    blur_faces = [face for face in faces if face["id"] not in excluded]

    if not blur_faces:
        return output

    intensity = clamp(blur_intensity, 0, 1) if math.isfinite(blur_intensity) else 0.5
    height, width = rgba.shape[0], rgba.shape[1]
    alpha_accum = np.zeros((height, width), dtype=np.float32)
    blurred_accum = np.zeros_like(rgba)

    for face in blur_faces:
        scaled = scale_face_around_center(face, MASK_SCALE)
        left = clamp(math.floor(scaled["x"]), 0, width - 1)
        top = clamp(math.floor(scaled["y"]), 0, height - 1)
        right = clamp(math.ceil(scaled["x"] + scaled["width"]), 0, width)
        bottom = clamp(math.ceil(scaled["y"] + scaled["height"]), 0, height)
        region_width = max(1, right - left)
        region_height = max(1, bottom - top)
        region = rgba[top:bottom, left:right]

        radius_x = max(MIN_BLUR_RADIUS, round((scaled["width"] * intensity) / 2))
        radius_y = max(MIN_BLUR_RADIUS, round((scaled["height"] * intensity) / 2))
        blurred = blur_region(region, region_width, region_height, radius_x, radius_y)

        center_x = scaled["x"] + (scaled["width"] / 2)
        center_y = scaled["y"] + (scaled["height"] / 2)
        radius_ellipse_x = max(1, scaled["width"] / 2)
        radius_ellipse_y = max(1, scaled["height"] / 2)

        gy = (np.arange(top, bottom) + 0.5 - center_y) / radius_ellipse_y
        gx = (np.arange(left, right) + 0.5 - center_x) / radius_ellipse_x
        alpha = (gx[np.newaxis, :] ** 2 + gy[:, np.newaxis] ** 2) <= 1.0

        region_alpha = alpha_accum[top:bottom, left:right]
        update = alpha > region_alpha
        region_alpha[update] = 1.0
        blurred_accum[top:bottom, left:right][update] = blurred[update]

    mask = alpha_accum > 0
    if not mask.any():
        return output

    a = alpha_accum[mask, None]
    src = rgba[mask].astype(np.float32)
    blurred = blurred_accum[mask].astype(np.float32)
    output[mask] = np.round(blurred * a + src * (1 - a)).astype(np.uint8)

    return output
