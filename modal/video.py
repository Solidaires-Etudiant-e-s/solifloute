"""Video processing for the Modal cloud pipeline.

Mirrors the server-side ``server/utils/process-video.ts`` flow: probe the
input, collect face samples per frame, build a temporal face resolver, blur
each frame and re-encode with the original audio track.
"""

from __future__ import annotations

import json
import math
import subprocess
from typing import Optional

import cv2
import numpy as np

from processing import apply_blur_effects, detect_faces
from tracking import create_face_resolver

FRAME_JPEG_QUALITY = 3
MP4_AUDIO_COPY_CODECS = {
    "aac", "mp3", "ac3", "eac3", "alac", "flac", "opus",
    "vorbis", "pcm_s16le", "pcm_s24le", "pcm_s32le",
}


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Commande echouee ({' '.join(command)}): {result.stderr.strip()}")
    return result.stdout


def read_video_metadata(input_path: str) -> dict:
    ffprobe = _run([
        "ffprobe", "-v", "error", "-show_streams",
        "-show_entries", "format=duration", "-of", "json", input_path,
    ])
    parsed = json.loads(ffprobe)
    video_stream = next((s for s in parsed.get("streams", []) if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in parsed.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not video_stream:
        raise RuntimeError("Dimensions video invalides.")
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Dimensions video invalides.")
    fps = 24.0
    avg = str(video_stream.get("avg_frame_rate") or "")
    if "/" in avg:
        numerator_text, denominator_text = avg.split("/", 1)
        numerator = float(numerator_text or "0")
        denominator = float(denominator_text or "1")
        if denominator > 0 and numerator > 0:
            fps = numerator / denominator
    duration = float(parsed.get("format", {}).get("duration") or 0) or None
    nb_frames = int(video_stream.get("nb_frames") or "0")
    frame_count = nb_frames if nb_frames > 0 else (math.ceil((duration or 0) * fps) if duration else 0)
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": max(1, frame_count),
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
    }


def _decode_range_cv2(
    input_path: str,
    start_frame: int,
    end_frame: int,
):
    """Yield BGR frames [start_frame, end_frame) using OpenCV (frame-accurate)."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Impossible de lire la video.")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    try:
        while frame_index < end_frame:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            yield frame_index, frame_bgr
            frame_index += 1
    finally:
        cap.release()


def _decode_range_cuda(
    input_path: str,
    start_frame: int,
    end_frame: int,
    width: int,
    height: int,
    fps: float,
):
    """Yield BGR frames [start_frame, end_frame) decoded on the GPU (NVDEC).

    Uses ``ffmpeg -hwaccel cuda`` piped as raw BGR, which offloads decode from
    the CPU. ``-ss`` seek is approximate for frame boundaries, so this is best
    used for whole-clip decoding or when exact frame alignment is not critical.
    """
    import subprocess

    start_s = start_frame / fps if fps > 0 else 0
    duration_s = (end_frame - start_frame) / fps if fps > 0 else None
    args = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda",
        "-ss", f"{start_s:.6f}",
        "-i", input_path,
        "-an",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1",
    ]
    if duration_s is not None and duration_s > 0:
        args.insert(1, "-t")
        args.insert(2, f"{duration_s:.6f}")
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = width * height * 3
    frame_index = start_frame
    assert process.stdout is not None
    try:
        while frame_index < end_frame:
            chunk = process.stdout.read(frame_bytes)
            if not chunk or len(chunk) < frame_bytes:
                break
            frame_bgr = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
            yield frame_index, frame_bgr
            frame_index += 1
    finally:
        process.stdout.close()
        process.wait()


def detect_faces_on_video(
    input_path: str,
    detection_model: str,
    confidence_threshold: float,
    session,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    gpu_decode: bool = False,
) -> dict:
    """GPU-only stage: decode a frame range and return per-frame face boxes.

    This is the ``detect`` half of the cloud/server split. It runs the ONNX
    detector on the GPU and returns raw face samples plus video metadata, so the
    (CPU) server can run the temporal resolver + blur + re-encode locally.
    ``start_frame``/``end_frame`` allow splitting a long video into segments for
    parallel detection across containers.
    """
    metadata = read_video_metadata(input_path)
    width = metadata["width"]
    height = metadata["height"]
    fps = metadata["fps"]

    total = metadata["frame_count"]
    start = max(0, int(start_frame))
    end = min(total, int(end_frame)) if end_frame is not None else total
    if end <= start:
        return {
            "width": width, "height": height, "fps": fps,
            "frame_count": total, "samples": [],
        }

    detect_func = (
        lambda frame: detect_faces(session, frame, "yunet", confidence_threshold)
        if detection_model == "fast"
        else detect_faces(session, frame, "centerface", confidence_threshold)
    )

    frames = (
        _decode_range_cuda(input_path, start, end, width, height, fps)
        if gpu_decode
        else _decode_range_cv2(input_path, start, end)
    )

    samples: list[dict] = []
    for frame_index, frame_bgr in frames:
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        samples.append({"frameIndex": frame_index, "faces": detect_func(rgba)})

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": total,
        "samples": samples,
    }


def process_video(
    input_path: str,
    output_path: str,
    detection_model: str,
    confidence_threshold: float,
    blur_intensity: float,
    excluded_face_ids: list[str],
    session,
    on_progress: Optional[Callable[[float], None]] = None,
) -> None:
    metadata = read_video_metadata(input_path)
    width = metadata["width"]
    height = metadata["height"]
    fps = metadata["fps"]
    frame_count = metadata["frame_count"]
    audio_codec = metadata["audio_codec"]

    def emit(fraction: float) -> None:
        if on_progress:
            on_progress(max(0.0, min(1.0, fraction)))

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Impossible de lire la video.")

    # Phase 1: collect face samples.
    detect_func = (
        lambda frame: detect_faces(session, frame, "yunet", confidence_threshold)
        if detection_model == "fast"
        else detect_faces(session, frame, "centerface", confidence_threshold)
    )

    def read_rgba() -> Optional[np.ndarray]:
        ok, frame_bgr = cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)

    samples: list[dict] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or frame_count
    frame_index = 0
    while True:
        rgba = read_rgba()
        if rgba is None:
            break
        samples.append({"frameIndex": frame_index, "faces": detect_func(rgba)})
        frame_index += 1
        emit((frame_index / max(1, total_frames)) * 0.5)
    cap.release()

    resolver = create_face_resolver(
        samples,
        gap_frames=2,
        appearance_frames=max(2, round(fps * 0.1)),
        disappearance_frames=max(2, round(fps * 0.1)),
    )

    # Phase 2: blur frames and encode.
    encode_args = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}",
        "-i", "pipe:0",
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "copy" if audio_codec in MP4_AUDIO_COPY_CODECS else "aac",
        "-movflags", "+faststart",
        "-shortest",
        output_path,
    ]
    encoder = subprocess.Popen(encode_args, stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Impossible de relire la video.")

    written = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        faces = resolver(written)
        processed = apply_blur_effects(rgba, faces, excluded_face_ids, blur_intensity)
        assert encoder.stdin is not None
        encoder.stdin.write(processed.tobytes())
        written += 1
        emit(0.5 + (written / max(1, total_frames)) * 0.5)
    cap.release()

    assert encoder.stdin is not None
    encoder.stdin.close()
    return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"L encodage video a echoue (code {return_code}).")
