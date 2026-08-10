/// <reference lib="webworker" />

import type { EditorSettings, Face } from '~~/shared/types/faces'
import { applyBlurEffects } from '~~/shared/utils/imageProcessing'
import type { FaceDetectionModelType } from '~~/shared/utils/useFaceDetector'
import { useFaceDetector } from '~~/shared/utils/useFaceDetector'
import type { DetectSuccess, ProcessSuccess, WorkerErrorResponse, WorkerRequest } from '~/utils/detect-worker'

const messageContext: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope

async function detectFaces(imageData: ImageData, threshold: number, modelUrl: string, modelType: FaceDetectionModelType): Promise<{ faces: Face[], durationMs: number }> {
  const detector = useFaceDetector(modelUrl, modelType)
  return await detector.detectFaces({
    data: imageData.data,
    width: imageData.width,
    height: imageData.height
  }, threshold)
}

async function processImage(imageData: ImageData, settings: EditorSettings, modelUrl: string, modelType: FaceDetectionModelType, manualFaces: Face[] = []) {
  const { faces, durationMs } = await detectFaces(imageData, settings.confidenceThreshold, modelUrl, modelType)
  const processedImageData = new ImageData(
    applyBlurEffects(
      { data: imageData.data, width: imageData.width, height: imageData.height },
      [...faces, ...manualFaces],
      settings.excludedFaceIds,
      settings.blurIntensity
    ),
    imageData.width,
    imageData.height
  )

  return {
    faces,
    processedImageData,
    durationMs
  }
}

function respond(response: DetectSuccess | ProcessSuccess | WorkerErrorResponse) {
  messageContext.postMessage(response)
}

messageContext.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data
  const respondWithId = (payload: Omit<DetectSuccess, 'id'> | Omit<ProcessSuccess, 'id'>) => {
    respond({
      id: request.id,
      ...payload
    })
  }

  try {
    if (request.type === 'detect') {
      const result = await detectFaces(request.imageData, request.threshold, request.modelUrl, request.modelType)
      respondWithId({
        type: 'detect:success',
        ...result
      })
      return
    }

    const result = await processImage(
      request.imageData,
      request.settings,
      request.modelUrl,
      request.modelType,
      request.manualFaces
    )
    respondWithId({
      type: 'process:success',
      ...result
    })
  } catch (error) {
    respond({
      id: request.id,
      type: 'error',
      message: error instanceof Error ? error.message : 'La requete du worker a echoue.'
    })
  }
}
