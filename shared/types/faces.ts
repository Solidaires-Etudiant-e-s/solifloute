export interface Face {
  id: string
  x: number
  y: number
  width: number
  height: number
  confidence: number
}

export type ProcessingMode = 'auto' | 'client' | 'server' | 'cloud'

export type DetectionModel = 'fast' | 'advanced'

export interface EditorSettings {
  confidenceThreshold: number
  blurIntensity: number
  processingMode: ProcessingMode
  excludedFaceIds: string[]
  detectionModel: DetectionModel
}

export interface DetectionInput {
  data: Uint8ClampedArray
  width: number
  height: number
}

export interface DetectionResult {
  faces: Face[]
  durationMs: number
}

export interface ProcessImagePayload {
  imageBase64: string
  mimeType: string
  fileName: string
  settings: EditorSettings
  manualFaces?: Face[]
}
