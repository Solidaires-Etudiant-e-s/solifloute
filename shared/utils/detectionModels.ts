import type { DetectionModel } from '../types/faces'
import type { FaceDetectionModelType } from './useFaceDetector'

export const DETECTION_MODELS: Record<DetectionModel, {
  modelType: FaceDetectionModelType
  fileName: string
}> = {
  fast: {
    modelType: 'yunet',
    fileName: 'yunet-1080p.onnx'
  },
  advanced: {
    modelType: 'centerface',
    fileName: 'centerface.onnx'
  }
}

export function getClientModelUrl(model: DetectionModel) {
  return `/models/${DETECTION_MODELS[model].fileName}`
}

export function getServerModelPath(model: DetectionModel) {
  return `${process.cwd()}/public/models/${DETECTION_MODELS[model].fileName}`
}
