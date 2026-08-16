"""SoliFloute cloud inference pipeline for Modal (video only).

Deploy with:

    modal deploy modal/app.py

The app exposes FastAPI web function routes:

    POST /prepare-video    upload a video once, cache it in a shared volume, return a videoId
    POST /detect-faces     detect faces in a frame range of a cached video (videoId) or uploaded file
    POST /process-video    blur faces in an uploaded video and return an MP4 (legacy)

The Nuxt server proxies these routes when the user selects the "Cloud" video
processing target. For production you should protect the endpoints with Modal
proxy-token auth (the ``Modal-Key`` / ``Modal-Secret`` headers) and configure
the URLs on the Nuxt side via environment variables.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import modal
from fastapi import File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response

# Helper modules (processing.py, tracking.py, video.py) are baked into the image
# at the filesystem root. Make sure they are importable regardless of the
# container working directory (which differs between `serve` and `deploy`).
sys.path.insert(0, "/")

from processing import create_session  # noqa: E402
from video import detect_faces_on_video, process_video  # noqa: E402

MODEL_DIR = "/models"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_SRC = PROJECT_ROOT / "public" / "models"

DETECTION_MODELS = {
    "fast": {"file": "yunet-1080p.onnx", "model_type": "yunet"},
    "advanced": {"file": "centerface.onnx", "model_type": "centerface"},
}

VIDEO_CACHE_DIR = "/video_cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "numpy",
        "opencv-python-headless",
        "onnxruntime-gpu==1.26.0",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-nvjitlink-cu12",
        "nvidia-cuda-cupti-cu12",
        "pillow",
        "fastapi",
        "python-multipart",
    )
    .run_commands(
        "bash -c 'cp -d /usr/local/lib/python3.12/site-packages/nvidia/*/lib/*.so* /usr/local/lib/ 2>/dev/null; ldconfig'"
    )
    .add_local_file(str(PROJECT_ROOT / "modal" / "processing.py"), remote_path="/processing.py")
    .add_local_file(str(PROJECT_ROOT / "modal" / "tracking.py"), remote_path="/tracking.py")
    .add_local_file(str(PROJECT_ROOT / "modal" / "video.py"), remote_path="/video.py")
    .add_local_dir(str(MODELS_SRC), MODEL_DIR)
)

app = modal.App("solifloute-cloud", image=image)

video_cache = modal.Volume.from_name("solifloute-video-cache", create_if_missing=True)


def _settings_from_payload(payload: dict) -> dict:
    settings = payload.get("settings") or {}
    detection_model = settings.get("detectionModel")
    if detection_model not in DETECTION_MODELS:
        detection_model = "fast"
    model = DETECTION_MODELS[detection_model]

    def to_float(value, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    confidence_threshold = min(1.0, max(0.0, to_float(settings.get("confidenceThreshold"), 0.5)))
    blur_intensity = min(1.0, max(0.0, to_float(settings.get("blurIntensity"), 0.5)))
    excluded = settings.get("excludedFaceIds") or []
    if not isinstance(excluded, list):
        excluded = []
    return {
        "detection_model": detection_model,
        "model_type": model["model_type"],
        "model_path": f"{MODEL_DIR}/{model['file']}",
        "confidence_threshold": confidence_threshold,
        "blur_intensity": blur_intensity,
        "excluded_face_ids": [str(face_id) for face_id in excluded],
    }


def _video_path_for(video_id: str) -> str:
    return os.path.join(VIDEO_CACHE_DIR, f"{video_id}.mp4")


@app.cls(gpu="T4", volumes={VIDEO_CACHE_DIR: video_cache})
class FaceProcessor:
    @modal.enter()
    def load(self) -> None:
        self._sessions: dict[str, object] = {}
        os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)

    def _session(self, model_path: str) -> object:
        if model_path not in self._sessions:
            self._sessions[model_path] = create_session(model_path, gpu=True)
        return self._sessions[model_path]

    @modal.fastapi_endpoint(method="POST")
    def prepare_video(self, file: UploadFile) -> JSONResponse:
        content = file.file.read()
        video_id = hashlib.sha256(content).hexdigest()[:16]
        path = _video_path_for(video_id)
        with open(path, "wb") as handle:
            handle.write(content)
        return JSONResponse(content={"videoId": video_id})

    @modal.fastapi_endpoint(method="POST")
    def detect_faces(
        self,
        file: Optional[UploadFile] = File(None),
        settings: Optional[str] = Form("{}"),
        videoId: Optional[str] = Query(None),
        startFrame: Optional[int] = Query(None),
        endFrame: Optional[int] = Query(None),
        gpuDecode: Optional[bool] = Query(None),
    ) -> JSONResponse:
        payload = {}
        try:
            payload = json.loads(settings or "{}")
        except json.JSONDecodeError:
            payload = {}
        conf = _settings_from_payload({"settings": payload})
        session = self._session(conf["model_path"])

        video_id = payload.get("videoId") or videoId
        if video_id:
            input_path = _video_path_for(video_id)
        elif file is not None:
            input_path = "/tmp/input-video.mp4"
            content = file.file.read()
            with open(input_path, "wb") as handle:
                handle.write(content)
        else:
            return JSONResponse(
                content={"error": "videoId or file is required"},
                status_code=400,
            )

        def to_int(value, fallback: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        start_frame = startFrame if startFrame is not None else to_int(payload.get("startFrame"), 0)
        end_frame = endFrame if endFrame is not None else to_int(payload.get("endFrame"), -1)
        if end_frame < 0:
            end_frame = None
        gpu_decode = gpuDecode if gpuDecode is not None else bool(payload.get("gpuDecode", False))

        result = detect_faces_on_video(
            input_path,
            conf["detection_model"],
            conf["confidence_threshold"],
            session,
            start_frame=start_frame,
            end_frame=end_frame,
            gpu_decode=gpu_decode,
        )
        return JSONResponse(content=result)

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, file: UploadFile, settings: Optional[str] = "{}") -> Response:
        payload = {}
        try:
            payload = json.loads(settings or "{}")
        except json.JSONDecodeError:
            payload = {}
        conf = _settings_from_payload({"settings": payload})
        session = self._session(conf["model_path"])

        input_path = "/tmp/input-video.mp4"
        output_path = "/tmp/output-video.mp4"
        content = file.file.read()
        with open(input_path, "wb") as handle:
            handle.write(content)

        process_video(
            input_path,
            output_path,
            conf["detection_model"],
            conf["confidence_threshold"],
            conf["blur_intensity"],
            conf["excluded_face_ids"],
            session,
        )

        with open(output_path, "rb") as handle:
            return Response(content=handle.read(), media_type="video/mp4")
