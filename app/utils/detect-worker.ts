import type { EditorSettings, Face } from '~~/shared/types/faces'
import type { FaceDetectionModelType } from '~~/shared/utils/useFaceDetector'

export interface DetectRequestInput {
  type: 'detect'
  imageData: ImageData
  threshold: number
  modelUrl: string
  modelType: FaceDetectionModelType
}

export interface ProcessRequestInput {
  type: 'process'
  imageData: ImageData
  settings: EditorSettings
  manualFaces: Face[]
  modelUrl: string
  modelType: FaceDetectionModelType
}

export type WorkerRequestInput = DetectRequestInput | ProcessRequestInput
export type DetectRequest = DetectRequestInput & { id: number }
export type ProcessRequest = ProcessRequestInput & { id: number }
export type WorkerRequest = WorkerRequestInput & { id: number }

export interface DetectSuccess {
  id: number
  type: 'detect:success'
  faces: Face[]
  durationMs: number
}

export interface ProcessSuccess {
  id: number
  type: 'process:success'
  faces: Face[]
  processedImageData: ImageData
  durationMs: number
}

export interface WorkerErrorResponse {
  id: number
  type: 'error'
  message: string
}

export type WorkerResponse = DetectSuccess | ProcessSuccess | WorkerErrorResponse

interface PendingRequest {
  resolve: (response: WorkerResponse) => void
  reject: (reason: Error) => void
}

let worker: Worker | null = null
let nextRequestId = 1
const pendingRequests = new Map<number, PendingRequest>()

function getWorker() {
  if (worker) {
    return worker
  }

  worker = new Worker(new URL('../workers/imageProcessor.worker.ts', import.meta.url), { type: 'module' })
  worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
    const response = event.data
    const pending = pendingRequests.get(response.id)

    if (!pending) {
      return
    }

    pendingRequests.delete(response.id)

    if (response.type === 'error') {
      pending.reject(new Error(response.message))
      return
    }

    pending.resolve(response)
  }
  worker.onerror = (event) => {
    for (const [id, pending] of pendingRequests) {
      pendingRequests.delete(id)
      pending.reject(new Error(event.message || 'Le worker de traitement a echoue.'))
    }
  }

  return worker
}

function postRequest<T extends WorkerResponse>(request: WorkerRequestInput): Promise<T> {
  const id = nextRequestId++
  const requestWithId = { ...request, id } as WorkerRequest

  return new Promise<T>((resolve, reject) => {
    pendingRequests.set(id, {
      resolve: resolve as PendingRequest['resolve'],
      reject
    })
    getWorker().postMessage(requestWithId)
  })
}

export function detectImageData(
  imageData: ImageData,
  threshold: number,
  modelUrl: string,
  modelType: FaceDetectionModelType
) {
  return postRequest<DetectSuccess>({
    type: 'detect',
    imageData,
    threshold,
    modelUrl,
    modelType
  })
}

export function processImageData(
  imageData: ImageData,
  settings: EditorSettings,
  manualFaces: Face[],
  modelUrl: string,
  modelType: FaceDetectionModelType
) {
  return postRequest<ProcessSuccess>({
    type: 'process',
    imageData,
    settings,
    manualFaces,
    modelUrl,
    modelType
  })
}

export async function warmupDetector(modelUrl: string, modelType: FaceDetectionModelType) {
  const canvas = document.createElement('canvas')
  canvas.width = 4
  canvas.height = 4
  const context = canvas.getContext('2d')

  if (!context) {
    return
  }

  context.fillRect(0, 0, 4, 4)
  await postRequest<DetectSuccess>({
    type: 'detect',
    imageData: context.getImageData(0, 0, 4, 4),
    threshold: 0.99,
    modelUrl,
    modelType
  })
}
