import type { DetectionModel } from '../types/faces'

/**
 * Estimated processing-time helper for the video pipeline.
 *
 * fps numbers come from `.data/benchmark-results.json` (`server_fps_reference_user_provided`
 * and the T4 optimized detect-only benchmark `t4_optimized_detect_1080p_1min_1799f`,
 * rounded to ~60 / ~36 fps at 720p for cloud). Resolution scaling is roughly linear
 * in pixels: the table is anchored at 1280x720 and fps is divided proportionally to
 * the pixel area (1080p ≈ half of 720p fps).
 *
 * IMPORTANT — benchmark vs production clarification:
 *
 * The "detect-only" fps numbers (both server and cloud) measure INFERENCE
 * throughput only — they do NOT represent total wall-clock time. Actual end-to-end
 * processing time includes additional phases that are NOT reflected in these fps
 * numbers:
 *
 *   Server pipeline:
 *     1. Frame extraction (decode video to raw RGBA on CPU)  — not in fps
 *     2. Face detection inference (server fps)                — in fps
 *     3. Blur application (CPU, per-frame)                   — not in fps
 *     4. Video encoding (ffmpeg libx264 on CPU)               — not in fps
 *
 *   Cloud pipeline:
 *     1. Frame extraction (decode video to raw RGBA on CPU)  — not in fps
 *     2. Video upload to Modal (full video per segment*)     — not in fps
 *     3. Face detection inference on T4 (cloud fps)           — in fps
 *     4. Blur application (CPU, per-frame)                    — not in fps
 *     5. Video encoding (ffmpeg libx264 on CPU)              — not in fps
 *
 *   * The redundant-upload issue (Fix 1) uploads the full video once per segment.
 *     The benchmark measures detect-only on a pre-loaded video with zero uploads.
 *
 * The "server fps" already covers detect + blur + encode on the server, so the
 * inference-only estimate is close to total wall-clock time for server mode.
 *
 * The "cloud fps" is detect-only on a T4. The server still performs frame extraction,
 * video upload, blur, and encoding locally — all of which add significant time
 * beyond what the benchmark measures. The 60s CLOUD_COLD_START_MS covers container
 * spin-up but does NOT cover upload/encode/blur overhead.
 *
 * Heuristics (documented):
 * - Server fps already covers detect + blur + encode on the server.
 * - Cloud fps is detect-only end-to-end on a T4 (Option B split: the server still
 *   blurs + encodes locally afterwards, which is not included here).
 * - Cloud gets a fixed cold-start allowance (container spin-up dominates short videos).
 * - Client mode is NOT benchmarked: we reuse the conservative server fps as a
 *   placeholder so the estimate is meaningful until live progress arrives.
 */

export interface ProcessingEstimateInput {
  processingMode: 'client' | 'server' | 'cloud'
  detectionModel: DetectionModel
  resolution: {
    width: number
    height: number
  }
  frameCount?: number
  durationSeconds?: number
  fps?: number
}

export interface ProcessingEstimateResult {
  estimatedMs: number
  placeholder: boolean
}

const REFERENCE_720P_PIXELS = 1280 * 720

const FPS_AT_720P: Record<'client' | 'server' | 'cloud', Record<DetectionModel, number>> = {
  server: { fast: 10, advanced: 6 },
  cloud: { fast: 60, advanced: 36 },
  client: { fast: 10, advanced: 6 }
}

const CLOUD_COLD_START_MS = 60_000
const NOMINAL_VIDEO_FPS = 30

function estimateFps(
  processingMode: 'client' | 'server' | 'cloud',
  detectionModel: DetectionModel,
  width: number,
  height: number
) {
  const pixels = Math.max(1, Math.max(1, Math.round(width)) * Math.max(1, Math.round(height)))
  const baseFps = FPS_AT_720P[processingMode][detectionModel]

  return baseFps * (REFERENCE_720P_PIXELS / pixels)
}

export function estimateProcessingTimeMs(input: ProcessingEstimateInput): ProcessingEstimateResult {
  const width = Number.isFinite(input.resolution.width) ? input.resolution.width : 0
  const height = Number.isFinite(input.resolution.height) ? input.resolution.height : 0

  let frameCount = 0

  if (Number.isFinite(input.frameCount) && (input.frameCount ?? 0) > 0) {
    frameCount = Math.round(input.frameCount!)
  } else if (Number.isFinite(input.durationSeconds) && (input.durationSeconds ?? 0) > 0) {
    const sourceFps = Number.isFinite(input.fps) && (input.fps ?? 0) > 0 ? input.fps! : NOMINAL_VIDEO_FPS
    frameCount = Math.max(1, Math.round(input.durationSeconds! * sourceFps))
  }

  if (frameCount <= 0 || width <= 0 || height <= 0) {
    return { estimatedMs: 0, placeholder: true }
  }

  const fps = estimateFps(input.processingMode, input.detectionModel, width, height)
  const inferenceMs = (frameCount / Math.max(0.001, fps)) * 1000
  const coldStartMs = input.processingMode === 'cloud' ? CLOUD_COLD_START_MS : 0

  return {
    estimatedMs: Math.round(inferenceMs + coldStartMs),
    placeholder: input.processingMode === 'client'
  }
}
