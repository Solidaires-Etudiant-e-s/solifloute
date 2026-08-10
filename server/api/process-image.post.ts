import sharp from 'sharp'
import type { DetectionInput, DetectionModel, ProcessImagePayload } from '~~/shared/types/faces'
import { applyBlurEffects } from '~~/shared/utils/imageProcessing'
import { DETECTION_MODELS, getServerModelPath } from '~~/shared/utils/detectionModels'
import { useFaceDetector } from '~~/shared/utils/useFaceDetector'

const MAX_IMAGE_PAYLOAD_BYTES = Number(process.env.PROCESS_MAX_IMAGE_PAYLOAD_BYTES || 80 * 1024 * 1024)

function getDetector(detectionModel: DetectionModel) {
  return useFaceDetector(
    getServerModelPath(detectionModel),
    DETECTION_MODELS[detectionModel].modelType
  )
}

async function decodeImage(imageBase64: string): Promise<DetectionInput> {
  const inputBuffer = Buffer.from(imageBase64, 'base64')
  const { data, info } = await sharp(inputBuffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })

  return {
    data: new Uint8ClampedArray(data),
    width: info.width,
    height: info.height
  }
}

async function detectFaces(payload: ProcessImagePayload) {
  const image = await decodeImage(payload.imageBase64)
  const detector = getDetector(payload.settings.detectionModel)
  return await detector.detectFaces(image, payload.settings.confidenceThreshold)
}

async function blurImage(payload: ProcessImagePayload) {
  const image = await decodeImage(payload.imageBase64)
  const detector = getDetector(payload.settings.detectionModel)
  const result = await detector.detectFaces(image, payload.settings.confidenceThreshold)
  const allFaces = [...result.faces, ...(payload.manualFaces || [])]

  return await sharp(
    Buffer.from(applyBlurEffects(image, allFaces, payload.settings.excludedFaceIds, payload.settings.blurIntensity)),
    {
      raw: {
        width: image.width,
        height: image.height,
        channels: 4
      }
    }
  ).png().toBuffer()
}

export default defineEventHandler(async (event) => {
  const payload = await readBody<ProcessImagePayload & { action?: 'detect' | 'process' }>(event)

  if (!payload?.imageBase64 || !payload?.settings) {
    throw createError({
      statusCode: 400,
      statusMessage: 'Une image encodee en base64 et des reglages sont requis.'
    })
  }

  if (payload.imageBase64.length > MAX_IMAGE_PAYLOAD_BYTES) {
    throw createError({
      statusCode: 413,
      statusMessage: 'L image depasse la taille maximale autorisee.'
    })
  }

  if (payload.action === 'detect') {
    return await detectFaces(payload)
  }

  const output = await blurImage(payload)
  setHeader(event, 'content-type', 'image/png')
  setHeader(event, 'cache-control', 'no-store')
  return output
})
