import { readFile } from 'node:fs/promises'
import { basename } from 'node:path'
import type { EditorSettings } from '~~/shared/types/faces'
import type { FaceSample } from '~~/shared/utils/videoFaceTracking'

const MODAL_VIDEO_URL = process.env.MODAL_VIDEO_URL || ''
const MODAL_DETECT_URL = process.env.MODAL_DETECT_URL || ''
const MODAL_PREPARE_URL = process.env.MODAL_PREPARE_URL || ''
const MODAL_KEY = process.env.MODAL_KEY || ''
const MODAL_SECRET = process.env.MODAL_SECRET || ''
const MODAL_TIMEOUT_MS = Number(process.env.MODAL_TIMEOUT_MS || 30 * 60 * 1000)

const SEGMENT_DURATION_SECONDS = 10
const MAX_SEGMENTS = 10
const SEGMENT_THRESHOLD_SECONDS = 60

export interface CloudDetectionResult {
  samples: FaceSample[]
  frameCount: number
  fps: number
  width: number
  height: number
}

export interface DetectVideoSegmentsOptions {
  frameCount: number
  fps: number
  onSegmentProgress?: (completed: number, total: number) => void
  signal?: AbortSignal
}

interface DetectSegment {
  startFrame: number
  endFrame: number
}

function modalHeaders() {
  const headers: Record<string, string> = {}

  if (MODAL_KEY && MODAL_SECRET) {
    headers['Modal-Key'] = MODAL_KEY
    headers['Modal-Secret'] = MODAL_SECRET
  }

  return headers
}

function detectEndpointUrl() {
  if (MODAL_DETECT_URL) {
    return MODAL_DETECT_URL
  }

  if (!MODAL_VIDEO_URL) {
    return ''
  }

  try {
    const url = new URL(MODAL_VIDEO_URL)
    url.pathname = '/detect-faces'
    return url.toString()
  } catch {
    return MODAL_VIDEO_URL.replace(/\/process-video$/, '/detect-faces')
  }
}

function prepareEndpointUrl() {
  if (MODAL_PREPARE_URL) {
    return MODAL_PREPARE_URL
  }

  if (!MODAL_VIDEO_URL) {
    return ''
  }

  try {
    const url = new URL(MODAL_VIDEO_URL)
    url.pathname = '/prepare-video'
    return url.toString()
  } catch {
    return MODAL_VIDEO_URL.replace(/\/process-video$/, '/prepare-video')
  }
}

function notConfiguredError() {
  return createError({
    statusCode: 501,
    statusMessage: 'Le traitement video Cloud (Modal) n est pas configure sur ce serveur.'
  })
}

function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted) {
    throw new Error('Traitement video annule.')
  }
}

async function postVideoForm(
  url: string,
  inputPath: string,
  settings: unknown,
  signal?: AbortSignal
): Promise<Response> {
  const videoBuffer = await readFile(inputPath)
  const form = new FormData()
  form.append('file', new Blob([videoBuffer]), basename(inputPath))
  form.append('settings', JSON.stringify(settings))

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), MODAL_TIMEOUT_MS)
  const handleAbort = () => controller.abort()
  signal?.addEventListener('abort', handleAbort, { once: true })

  try {
    throwIfAborted(signal)
    return await fetch(url, {
      method: 'POST',
      headers: modalHeaders(),
      body: form,
      signal: controller.signal
    })
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', handleAbort)
  }
}

async function postFormNoFile(
  url: string,
  settings: unknown,
  signal?: AbortSignal
): Promise<Response> {
  const form = new FormData()
  form.append('settings', JSON.stringify(settings))

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), MODAL_TIMEOUT_MS)
  const handleAbort = () => controller.abort()
  signal?.addEventListener('abort', handleAbort, { once: true })

  try {
    throwIfAborted(signal)
    return await fetch(url, {
      method: 'POST',
      headers: modalHeaders(),
      body: form,
      signal: controller.signal
    })
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', handleAbort)
  }
}

export async function prepareVideoOnModal(
  inputPath: string,
  signal?: AbortSignal
): Promise<string> {
  const url = prepareEndpointUrl()

  if (!url) {
    throw notConfiguredError()
  }

  const videoBuffer = await readFile(inputPath)
  const form = new FormData()
  form.append('file', new Blob([videoBuffer]), basename(inputPath))

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), MODAL_TIMEOUT_MS)
  const handleAbort = () => controller.abort()
  signal?.addEventListener('abort', handleAbort, { once: true })

  try {
    throwIfAborted(signal)
    const response = await fetch(url, {
      method: 'POST',
      headers: modalHeaders(),
      body: form,
      signal: controller.signal
    })

    if (!response.ok) {
      throw createError({
        statusCode: response.status,
        statusMessage: `La preparation de la video Cloud a echoue : ${await response.text()}`
      })
    }

    const parsed = await response.json() as { videoId?: string }
    if (!parsed.videoId) {
      throw new Error('Le serveur cloud n a pas renvoye de videoId.')
    }

    return parsed.videoId
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', handleAbort)
  }
}

function parseDetectionPayload(payload: unknown, fallbackFrameCount: number, fallbackFps: number): CloudDetectionResult {
  const parsed = payload as {
    width?: number
    height?: number
    fps?: number
    frame_count?: number
    samples?: Array<{ frameIndex?: number, faces?: Array<{ id?: string, x?: number, y?: number, width?: number, height?: number, confidence?: number }> }>
  }

  return {
    samples: Array.isArray(parsed.samples)
      ? parsed.samples
          .filter(sample => typeof sample.frameIndex === 'number')
          .map(sample => ({
            frameIndex: sample.frameIndex!,
            faces: Array.isArray(sample.faces)
              ? sample.faces.map(face => ({
                  id: String(face.id ?? ''),
                  x: Number(face.x) || 0,
                  y: Number(face.y) || 0,
                  width: Number(face.width) || 0,
                  height: Number(face.height) || 0,
                  confidence: Number(face.confidence) || 0
                }))
              : []
          }))
      : [],
    frameCount: Number(parsed.frame_count) || fallbackFrameCount,
    fps: Number(parsed.fps) || fallbackFps,
    width: Number(parsed.width) || 0,
    height: Number(parsed.height) || 0
  }
}

export async function detectFacesOnModal(
  inputPath: string,
  settings: EditorSettings,
  segment?: DetectSegment,
  signal?: AbortSignal,
  videoId?: string
): Promise<CloudDetectionResult> {
  const url = detectEndpointUrl()

  if (!url) {
    throw notConfiguredError()
  }

  const payload: Record<string, unknown> = { ...settings }

  if (segment) {
    payload.startFrame = segment.startFrame
    payload.endFrame = segment.endFrame
  }

  if (videoId) {
    payload.videoId = videoId
  }

  let response: Response

  if (videoId) {
    response = await postFormNoFile(url, payload, signal)
  } else {
    response = await postVideoForm(url, inputPath, payload, signal)
  }

  if (!response.ok) {
    throw createError({
      statusCode: response.status,
      statusMessage: `La detection Cloud des visages a echoue : ${await response.text()}`
    })
  }

  return parseDetectionPayload(await response.json(), 0, 0)
}

export function computeCloudSegmentCount(frameCount: number, fps: number) {
  const durationSeconds = fps > 0 ? frameCount / fps : 0

  if (durationSeconds <= SEGMENT_THRESHOLD_SECONDS) {
    return 1
  }

  return Math.min(
    MAX_SEGMENTS,
    Math.max(2, Math.ceil(durationSeconds / SEGMENT_DURATION_SECONDS))
  )
}

function buildSegments(frameCount: number, durationSeconds: number): DetectSegment[] {
  const fps = durationSeconds > 0 ? frameCount / durationSeconds : 0
  const segmentCount = computeCloudSegmentCount(frameCount, fps)
  const framesPerSegment = Math.ceil(frameCount / segmentCount)
  const segments: DetectSegment[] = []

  for (let index = 0; index < segmentCount; index += 1) {
    const startFrame = index * framesPerSegment

    if (startFrame >= frameCount) {
      break
    }

    segments.push({
      startFrame,
      endFrame: Math.min(frameCount, startFrame + framesPerSegment)
    })
  }

  return segments
}

export async function detectVideoFacesOnModal(
  inputPath: string,
  settings: EditorSettings,
  options: DetectVideoSegmentsOptions
): Promise<CloudDetectionResult> {
  const { frameCount, fps, signal } = options
  const durationSeconds = fps > 0 ? frameCount / fps : 0
  let segments = buildSegments(frameCount, durationSeconds)

  let videoId: string | undefined

  if (segments.length > 1) {
    try {
      videoId = await prepareVideoOnModal(inputPath, signal)
    } catch (error) {
      if (error instanceof Error && error.message.includes('404')) {
        // /prepare-video is unavailable: we cannot upload once and reuse it, so
        // collapse to a single full-video request instead of re-uploading the
        // whole file once per segment (which would buffer it N times in memory).
        console.warn('[solifloute:modal] /prepare-video endpoint unavailable, collapsing to a single upload')
        segments = [{ startFrame: 0, endFrame: frameCount }]
      } else {
        throw error
      }
    }
  }

  const results = await Promise.all(segments.map(async (segment, index) => {
    throwIfAborted(signal)
    const result = await detectFacesOnModal(inputPath, settings, segment, signal, videoId)
    options.onSegmentProgress?.(index + 1, segments.length)
    return result
  }))

  const samples = results
    .flatMap(result => result.samples)
    .sort((left, right) => left.frameIndex - right.frameIndex)
  const firstResult = results[0]

  return {
    samples,
    frameCount: firstResult?.frameCount || frameCount,
    fps: firstResult?.fps || fps,
    width: firstResult?.width || 0,
    height: firstResult?.height || 0
  }
}
