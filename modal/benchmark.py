"""SoliFloute detection benchmark on T4 (the chosen GPU).

Measures the per-container detection speed of the detect-only cloud stage with
and without GPU (NVDEC) decode, for both models, then demonstrates the
wall-clock win of splitting the clip into parallel segments.

Run from the project root:

    modal run modal/benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = "/models"

T4_PRICE = 0.59
MODELS = [
    ("yunet", "yunet-1080p.onnx", "fast"),
    ("centerface", "centerface.onnx", "centerface"),
]

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
    )
    .run_commands(
        "bash -c 'cp -d /usr/local/lib/python3.12/site-packages/nvidia/*/lib/*.so* /usr/local/lib/ 2>/dev/null; ldconfig'"
    )
    .add_local_file(str(PROJECT_ROOT / "modal" / "processing.py"), remote_path="/processing.py")
    .add_local_file(str(PROJECT_ROOT / "modal" / "tracking.py"), remote_path="/tracking.py")
    .add_local_file(str(PROJECT_ROOT / "modal" / "video.py"), remote_path="/video.py")
    .add_local_file("/tmp/people-1min.mp4", remote_path="/input.mp4")
    .add_local_dir(str(PROJECT_ROOT / "public" / "models"), MODEL_DIR)
)

app = modal.App("solifloute-benchmark", image=image)


@app.function(gpu="T4")
def benchmark_detect(model: str, gpu_decode: bool) -> dict:
    import time

    sys.path.insert(0, "/")
    from processing import create_session
    from video import detect_faces_on_video

    model_file = dict((name, file) for name, file, _ in MODELS)[model]
    detection_model = dict((name, dm) for name, _, dm in MODELS)[model]
    session = create_session(f"{MODEL_DIR}/{model_file}", gpu=True)

    started = time.perf_counter()
    result = detect_faces_on_video(
        "/input.mp4", detection_model, 0.2, session, gpu_decode=gpu_decode
    )
    elapsed = time.perf_counter() - started
    frames = result["frame_count"]
    return {
        "model": model,
        "gpu_decode": gpu_decode,
        "frames": frames,
        "seconds": round(elapsed, 2),
        "fps": round(frames / elapsed, 2) if elapsed > 0 else 0.0,
    }


@app.function(gpu="T4", max_containers=10)
def detect_segment(model: str, start: int, end: int) -> int:
    import sys

    sys.path.insert(0, "/")
    from processing import create_session
    from video import detect_faces_on_video

    model_file = dict((name, file) for name, file, _ in MODELS)[model]
    detection_model = dict((name, dm) for name, _, dm in MODELS)[model]
    session = create_session(f"{MODEL_DIR}/{model_file}", gpu=True)
    result = detect_faces_on_video(
        "/input.mp4", detection_model, 0.2, session,
        start_frame=start, end_frame=end, gpu_decode=True,
    )
    return len(result["samples"])


@app.local_entrypoint()
def main() -> None:
    print("=== T4 per-container detection (optimized session) ===")
    for model, _, _ in MODELS:
        for gpu_decode in (False, True):
            r = benchmark_detect.remote(model, gpu_decode)
            mode = "cuda" if r["gpu_decode"] else "cv2 "
            print(
                f"  [{model:<11}] decode={mode}  fps={r['fps']:>7.2f}  "
                f"{r['seconds']:>6.1f}s for {r['frames']}f"
            )

    print("\n=== T4 segmented detection (parallel, gpu_decode) ===")
    for model, _, _ in MODELS:
        segments = 10
        total = 1799
        span = total // segments
        jobs = [(model, i * span, (i + 1) * span if i < segments - 1 else total) for i in range(segments)]
        import time
        started = time.perf_counter()
        # Fire all segments in parallel (spawn returns immediately).
        calls = [detect_segment.spawn(*job) for job in jobs]
        done = [call.get() for call in calls]
        elapsed = time.perf_counter() - started
        print(
            f"  [{model:<11}] {segments} parallel segments: "
            f"{elapsed:>6.1f}s wall-clock for {sum(done)}f "
            f"({sum(done) / elapsed:.1f} fps aggregate)"
        )
