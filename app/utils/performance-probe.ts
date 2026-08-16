import type { DetectionModel } from '~~/shared/types/faces'
import { DETECTION_MODELS, getClientModelUrl } from '~~/shared/utils/detectionModels'
import { detectImageData, warmupDetector } from './detect-worker'

const PROBE_WIDTH = 640
const PROBE_HEIGHT = 480
const PROBE_RUNS = 5

const probeFpsCache = new Map<DetectionModel, number>()

export async function probeClientDetectionFps(detectionModel: DetectionModel) {
  if (!import.meta.client) {
    return null
  }

  const cached = probeFpsCache.get(detectionModel)

  if (cached !== undefined) {
    return cached
  }

  const modelUrl = getClientModelUrl(detectionModel)
  const modelType = DETECTION_MODELS[detectionModel].modelType

  try {
    await warmupDetector(modelUrl, modelType)

    const canvas = document.createElement('canvas')
    canvas.width = PROBE_WIDTH
    canvas.height = PROBE_HEIGHT
    const context = canvas.getContext('2d')

    if (!context) {
      return null
    }

    context.fillStyle = '#808080'
    context.fillRect(0, 0, PROBE_WIDTH, PROBE_HEIGHT)
    const imageData = context.getImageData(0, 0, PROBE_WIDTH, PROBE_HEIGHT)
    let bestMs = Number.POSITIVE_INFINITY

    for (let run = 0; run < PROBE_RUNS; run += 1) {
      const startedAt = performance.now()
      await detectImageData(imageData, 0.99, modelUrl, modelType)
      bestMs = Math.min(bestMs, performance.now() - startedAt)
    }

    const fps = bestMs > 0 ? 1000 / bestMs : 0

    if (fps > 0) {
      probeFpsCache.set(detectionModel, fps)
    }

    return fps > 0 ? fps : null
  } catch (cause) {
    console.debug('[solifloute] probe: echec de la mesure de performance client', cause)
    return null
  }
}

export async function estimateClientVideoProcessingMs(
  detectionModel: DetectionModel,
  width: number,
  height: number,
  frameCount: number
) {
  if (width <= 0 || height <= 0 || frameCount <= 0) {
    return 0
  }

  const probeFps = await probeClientDetectionFps(detectionModel)

  if (probeFps === null || probeFps <= 0) {
    return null
  }

  const probePixels = PROBE_WIDTH * PROBE_HEIGHT
  const targetPixels = width * height
  const fpsAtTarget = probeFps * (probePixels / targetPixels)

  return fpsAtTarget > 0
    ? Math.round((frameCount / fpsAtTarget) * 1000)
    : null
}
