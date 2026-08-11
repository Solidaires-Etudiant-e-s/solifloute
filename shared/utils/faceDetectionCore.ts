import type { DetectionInput, Face } from '../types/faces'
import { hardNonMaxSuppression } from './nms'

export const MAX_STRIDE = 32
export const DEFAULT_PROBABILITY_THRESHOLD = 0.2
export const NMS_THRESHOLD = 0.3

const CENTER_FACE_STRIDE = 4
const YU_NET_STRIDES = [8, 16, 32] as const
export const MIN_PROBABILITY_THRESHOLD = 0.1
export const MAX_CANDIDATES = 5000

interface TensorOutput {
  data: unknown
}

interface DetectionCandidate {
  x1: number
  y1: number
  x2: number
  y2: number
  score: number
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function readFloat(data: Float32Array, index: number) {
  return data[index] ?? 0
}

export function getPaddedInputSize(width: number, height: number) {
  return {
    width: Math.ceil(width / MAX_STRIDE) * MAX_STRIDE,
    height: Math.ceil(height / MAX_STRIDE) * MAX_STRIDE
  }
}

export function createModelInputData(
  input: DetectionInput,
  width: number,
  height: number
) {
  const output = new Float32Array(3 * width * height)
  const { data, width: sourceWidth, height: sourceHeight } = input
  const copyWidth = Math.min(width, sourceWidth)
  const copyHeight = Math.min(height, sourceHeight)

  for (let y = 0; y < copyHeight; y += 1) {
    for (let x = 0; x < copyWidth; x += 1) {
      const sourceIndex = ((y * sourceWidth) + x) * 4
      const targetIndex = (y * width) + x

      output[targetIndex] = data[sourceIndex] ?? 0
      output[(width * height) + targetIndex] = data[sourceIndex + 1] ?? 0
      output[(width * height * 2) + targetIndex] = data[sourceIndex + 2] ?? 0
    }
  }

  return output
}

function decodeCenterFaceOutputs(
  outputs: Record<string, TensorOutput>,
  width: number,
  height: number,
  probabilityThreshold: number
) {
  const heatmap = outputs['537']?.data as Float32Array | undefined
  const scale = outputs['538']?.data as Float32Array | undefined
  const offset = outputs['539']?.data as Float32Array | undefined

  if (!heatmap || !scale || !offset) {
    throw new Error('Le modele de detection de visages a renvoye des sorties inattendues.')
  }

  const featureWidth = Math.floor(width / CENTER_FACE_STRIDE)
  const featureHeight = Math.floor(height / CENTER_FACE_STRIDE)
  const featureCount = featureWidth * featureHeight
  const threshold = Math.max(probabilityThreshold, MIN_PROBABILITY_THRESHOLD)
  const candidates: DetectionCandidate[] = []

  for (let index = 0; index < featureCount; index += 1) {
    const score = readFloat(heatmap, index)

    if (score < threshold) {
      continue
    }

    const heightScale = Math.exp(readFloat(scale, index)) * CENTER_FACE_STRIDE
    const widthScale = Math.exp(readFloat(scale, featureCount + index)) * CENTER_FACE_STRIDE
    const offsetY = readFloat(offset, index)
    const offsetX = readFloat(offset, featureCount + index)
    const row = Math.floor(index / featureWidth)
    const col = index % featureWidth
    const y1 = (row + offsetY + 0.5) * CENTER_FACE_STRIDE - (heightScale / 2)
    const x1 = (col + offsetX + 0.5) * CENTER_FACE_STRIDE - (widthScale / 2)

    candidates.push({
      x1,
      y1,
      x2: x1 + widthScale,
      y2: y1 + heightScale,
      score
    })
  }

  candidates.sort((left, right) => right.score - left.score)
  candidates.length = Math.min(candidates.length, MAX_CANDIDATES)

  return hardNonMaxSuppression(candidates, NMS_THRESHOLD, MAX_CANDIDATES)
}

function mapFaces(candidates: ReturnType<typeof hardNonMaxSuppression>, sourceWidth: number, sourceHeight: number): Face[] {
  return candidates.map((candidate, index) => {
    const x1 = clamp(candidate.x1, 0, sourceWidth)
    const y1 = clamp(candidate.y1, 0, sourceHeight)
    const x2 = clamp(candidate.x2, 0, sourceWidth)
    const y2 = clamp(candidate.y2, 0, sourceHeight)

    return {
      id: `face-${index + 1}`,
      x: Math.round(x1),
      y: Math.round(y1),
      width: Math.round(x2 - x1),
      height: Math.round(y2 - y1),
      confidence: Number(candidate.score.toFixed(4))
    }
  })
}

export function extractFacesFromOutputs(
  outputs: Record<string, TensorOutput>,
  input: DetectionInput,
  probabilityThreshold: number
) {
  const { width, height } = getPaddedInputSize(input.width, input.height)

  return mapFaces(
    decodeCenterFaceOutputs(outputs, width, height, probabilityThreshold),
    input.width,
    input.height
  )
}

function decodeYunetOutputs(
  outputs: Record<string, TensorOutput>,
  width: number,
  height: number,
  probabilityThreshold: number
) {
  const candidates: DetectionCandidate[] = []
  const threshold = Math.max(probabilityThreshold, MIN_PROBABILITY_THRESHOLD)

  for (const stride of YU_NET_STRIDES) {
    const cls = outputs[`cls_${stride}`]?.data as Float32Array | undefined
    const obj = outputs[`obj_${stride}`]?.data as Float32Array | undefined
    const bbox = outputs[`bbox_${stride}`]?.data as Float32Array | undefined

    if (!cls || !obj || !bbox) {
      throw new Error('Le modele de detection de visages a renvoye des sorties inattendues.')
    }

    const columns = Math.floor(width / stride)
    const rows = Math.floor(height / stride)

    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const index = (row * columns) + column
        const clsScore = clamp(readFloat(cls, index), 0, 1)
        const objScore = clamp(readFloat(obj, index), 0, 1)
        const score = Math.sqrt(clsScore * objScore)

        if (score < threshold) {
          continue
        }

        const centerX = (column + readFloat(bbox, (index * 4) + 0)) * stride
        const centerY = (row + readFloat(bbox, (index * 4) + 1)) * stride
        const boxWidth = Math.exp(readFloat(bbox, (index * 4) + 2)) * stride
        const boxHeight = Math.exp(readFloat(bbox, (index * 4) + 3)) * stride

        candidates.push({
          x1: centerX - (boxWidth / 2),
          y1: centerY - (boxHeight / 2),
          x2: centerX + (boxWidth / 2),
          y2: centerY + (boxHeight / 2),
          score
        })
      }
    }
  }

  candidates.sort((left, right) => right.score - left.score)
  candidates.length = Math.min(candidates.length, MAX_CANDIDATES)

  return hardNonMaxSuppression(candidates, NMS_THRESHOLD, MAX_CANDIDATES)
}

export function extractFacesFromYunetOutputs(
  outputs: Record<string, TensorOutput>,
  input: DetectionInput,
  probabilityThreshold: number
) {
  const { width, height } = getPaddedInputSize(input.width, input.height)

  return mapFaces(
    decodeYunetOutputs(outputs, width, height, probabilityThreshold),
    input.width,
    input.height
  )
}
