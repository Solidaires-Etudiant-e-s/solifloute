import { execFile, spawn } from 'node:child_process'
import { once } from 'node:events'
import { mkdir, mkdtemp, open, readFile, rm, stat, unlink, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { promisify } from 'node:util'
import sharp from 'sharp'
import type { DetectionModel, EditorSettings } from '~~/shared/types/faces'
import { createAdaptiveProgress, type AdaptiveProgressReport } from '~~/shared/utils/progress'
import {
  blurVideoFrame,
  createVideoFaceResolver
} from '~~/shared/utils/videoProcessing'
import type { FaceSample } from '~~/shared/utils/videoFaceTracking'
import { DETECTION_MODELS, getServerModelPath } from '~~/shared/utils/detectionModels'
import { estimateProcessingTimeMs } from '~~/shared/utils/estimatedProcessingTime'
import { useFaceDetector } from '~~/shared/utils/useFaceDetector'
import { computeCloudSegmentCount, detectVideoFacesOnModal } from './modal-client'

const execFileAsync = promisify(execFile)
const VIDEO_OUTPUT_ROOT = process.env.PROCESS_VIDEO_OUTPUT_DIR || join(process.cwd(), '.data', 'video-results')
const DEBUG_DUMP_PATH = process.env.PROCESS_VIDEO_DEBUG_DUMP
const CHECKPOINT_INTERVAL_FRAMES = 100
const MP4_AUDIO_COPY_CODECS = new Set([
  'aac',
  'mp3',
  'ac3',
  'eac3',
  'alac',
  'flac',
  'opus',
  'vorbis',
  'pcm_s16le',
  'pcm_s24le',
  'pcm_s32le'
])

export interface VideoJobWorkspace {
  tempRoot: string
  inputPath: string
  outputPath: string
  framesDir: string
}

interface VideoMetadata {
  width: number
  height: number
  fps: number
  frameCount: number
  rotation: number
  audioCodec: string | null
  audioDuration: number | null
  videoDuration: number | null
}

interface ProcessVideoResult {
  outputPath: string
  tempRoot: string
}

type AdaptiveProgress = ReturnType<typeof createAdaptiveProgress>

interface VideoCheckpoint {
  frameCount: number
  width: number
  height: number
  fps: number
  rotation: number
  samples: FaceSample[]
}

async function readVideoCheckpoint(tempRoot: string): Promise<VideoCheckpoint | null> {
  try {
    const raw = await readFile(join(tempRoot, 'checkpoint.json'), 'utf8')
    const parsed = JSON.parse(raw) as VideoCheckpoint

    if (!Array.isArray(parsed.samples) || parsed.samples.length === 0) {
      return null
    }

    return parsed
  } catch {
    return null
  }
}

function createAbortError() {
  return new Error('Traitement video annule.')
}

function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted) {
    throw createAbortError()
  }
}

function getFfmpegPath() {
  return process.env.FFMPEG_PATH || 'ffmpeg'
}

function getFfprobePath() {
  return process.env.FFPROBE_PATH || 'ffprobe'
}

function getInputExtension(fileName: string) {
  const extension = fileName.toLowerCase().split('.').pop()

  if (!extension || extension.length > 5) {
    return '.mp4'
  }

  return `.${extension}`
}

function parseFrameRate(value: string) {
  const [numeratorText, denominatorText = '1'] = value.trim().split('/')
  const numerator = Number(numeratorText)
  const denominator = Number(denominatorText)

  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator <= 0) {
    return 24
  }

  const fps = numerator / denominator
  return Number.isFinite(fps) && fps > 0 ? fps : 24
}

function extractRotation(stream?: Record<string, unknown>) {
  const rotateTag = Number((stream?.tags as Record<string, unknown> | undefined)?.rotate)

  if (Number.isFinite(rotateTag)) {
    return ((rotateTag % 360) + 360) % 360
  }

  const sideData = (stream?.side_data_list as Array<Record<string, unknown>> | undefined) || []
  const displayMatrix = sideData.find(entry => (
    entry.side_data_type === 'Display Matrix'
    || entry.side_data_type === 'displaymatrix'
  ))
  const rotation = Number(displayMatrix?.rotation)

  if (Number.isFinite(rotation)) {
    return ((rotation % 360) + 360) % 360
  }

  return 0
}

function getTransposeFilter(rotation: number) {
  switch (rotation) {
    case 90:
      return 'transpose=2'
    case 180:
      return 'hflip,vflip'
    case 270:
      return 'transpose=1'
    default:
      return ''
  }
}

function getDisplayDimensions(width: number, height: number, rotation: number) {
  const swaps = rotation === 90 || rotation === 270
  return swaps ? { width: height, height: width } : { width, height }
}

function captureTextStream(stream: NodeJS.ReadableStream | null) {
  let output = ''

  if (stream) {
    stream.setEncoding('utf8')
    stream.on('data', (chunk: string) => {
      output += chunk
    })
  }

  return () => output.trim()
}

function waitForProcess(child: ReturnType<typeof spawn>, fallbackMessage: string, readStderr: () => string, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const handleAbort = () => {
      child.kill('SIGKILL')
      reject(createAbortError())
    }

    signal?.addEventListener('abort', handleAbort, { once: true })

    child.once('error', reject)
    child.once('close', (code) => {
      signal?.removeEventListener('abort', handleAbort)

      if (signal?.aborted) {
        reject(createAbortError())
        return
      }

      if (code === 0) {
        resolve()
        return
      }

      reject(new Error(readStderr() || fallbackMessage))
    })
  })
}

async function silenceProcess(
  child: ReturnType<typeof spawn>,
  done: Promise<void>,
  signal: NodeJS.Signals = 'SIGKILL'
) {
  child.kill(signal)
  await done.catch(() => {})
}

async function writeFrame(stdin: NodeJS.WritableStream, frame: Uint8ClampedArray, signal?: AbortSignal) {
  throwIfAborted(signal)
  const buffer = Buffer.from(frame.buffer, frame.byteOffset, frame.byteLength)

  if (stdin.write(buffer)) {
    return
  }

  await once(stdin, 'drain')
  throwIfAborted(signal)
}

export async function getVideoFrameCount(inputPath: string): Promise<number | null> {
  try {
    const metadata = await readVideoMetadata(inputPath)
    return metadata.frameCount
  } catch {
    return null
  }
}

function parseVideoMetadata(stdout: string): VideoMetadata {
  const parsed = JSON.parse(stdout) as {
    streams?: Array<Record<string, unknown>>
    format?: { duration?: string }
  }
  const videoStream = parsed.streams?.find(stream => stream.codec_type === 'video')
  const audioStream = parsed.streams?.find(stream => stream.codec_type === 'audio')
  const codedWidth = Number(videoStream?.width || 0)
  const codedHeight = Number(videoStream?.height || 0)

  if (codedWidth <= 0 || codedHeight <= 0) {
    throw new Error('Dimensions video invalides.')
  }

  const rotation = extractRotation(videoStream)
  const { width, height } = getDisplayDimensions(codedWidth, codedHeight, rotation)
  const fps = parseFrameRate(String(videoStream?.avg_frame_rate || ''))
  const frameCountFromStream = Number(videoStream?.nb_frames || '0')
  const videoDuration = Number(parsed.format?.duration || '0') || null
  const frameCount = frameCountFromStream > 0
    ? frameCountFromStream
    : Math.max(1, Math.ceil((videoDuration || 0) * fps))
  const audioCodec = audioStream?.codec_name ? String(audioStream.codec_name) : null
  const audioDuration = Number(audioStream?.duration || '0') || null

  return {
    width,
    height,
    fps,
    frameCount,
    rotation,
    audioCodec,
    audioDuration,
    videoDuration
  }
}

export async function readVideoMetadata(inputPath: string, signal?: AbortSignal): Promise<VideoMetadata> {
  throwIfAborted(signal)
  const { stdout } = await execFileAsync(getFfprobePath(), [
    '-v',
    'error',
    '-show_streams',
    '-show_entries',
    'format=duration',
    '-of',
    'json',
    inputPath
  ], { signal, maxBuffer: 1024 * 1024 })
  return parseVideoMetadata(stdout)
}

function writeBufferToStream(stream: NodeJS.WritableStream, buffer: Buffer): Promise<void> {
  if (stream.write(buffer)) {
    return Promise.resolve()
  }

  return new Promise<void>((resolve) => {
    const finish = () => {
      stream.removeListener('drain', finish)
      stream.removeListener('close', finish)
      stream.removeListener('error', finish)
      resolve()
    }

    stream.once('drain', finish)
    stream.once('close', finish)
    stream.once('error', finish)
  })
}

export async function readVideoMetadataFromBuffer(buffer: Buffer, signal?: AbortSignal): Promise<VideoMetadata> {
  throwIfAborted(signal)
  const child = spawn(getFfprobePath(), [
    '-v',
    'error',
    '-show_streams',
    '-show_entries',
    'format=duration',
    '-of',
    'json',
    '-i',
    'pipe:0'
  ], { stdio: ['pipe', 'pipe', 'pipe'] })
  const readStdout = captureTextStream(child.stdout)
  const readStderr = captureTextStream(child.stderr)
  const done = waitForProcess(child, 'L analyse des metadonnees video a echoue.', readStderr, signal)

  child.stdin.on('error', () => {})
  await writeBufferToStream(child.stdin, buffer)

  if (!child.stdin.destroyed) {
    child.stdin.end()
  }

  await done
  return parseVideoMetadata(readStdout())
}

async function extractRawRgbaVideo(
  inputPath: string,
  outputPath: string,
  metadata: VideoMetadata,
  onFrame: (frame: number) => void,
  signal?: AbortSignal
) {
  throwIfAborted(signal)
  const args = [
    '-hide_banner',
    '-loglevel',
    'error',
    '-noautorotate',
    '-i',
    inputPath
  ]

  if (metadata.rotation !== 0) {
    args.push('-vf', getTransposeFilter(metadata.rotation))
  }

  args.push(
    '-f',
    'rawvideo',
    '-pix_fmt',
    'rgba',
    outputPath,
    '-progress',
    'pipe:1'
  )

  const child = spawn(getFfmpegPath(), args, {
    stdio: ['ignore', 'pipe', 'pipe']
  })
  const readError = captureTextStream(child.stderr)
  const done = waitForProcess(child, 'L extraction des images video a echoue.', readError, signal)
  let lineBuffer = ''

  child.stdout?.setEncoding('utf8')
  child.stdout?.on('data', (chunk: string) => {
    lineBuffer += chunk
    const lines = lineBuffer.split('\n')
    lineBuffer = lines.pop() ?? ''

    for (const line of lines) {
      const match = /^frame=(\d+)/.exec(line.trim())

      if (match) {
        onFrame(Number(match[1]))
      }
    }
  })

  await done

  const stats = await stat(outputPath)
  const frameBytes = metadata.width * metadata.height * 4
  return Math.floor(stats.size / frameBytes)
}

async function createRawFrameReader(
  rawVideoPath: string,
  metadata: VideoMetadata,
  signal?: AbortSignal
) {
  throwIfAborted(signal)
  const frameBytes = metadata.width * metadata.height * 4
  const handle = await open(rawVideoPath, 'r')

  return {
    frameBytes,

    async readFrame(frameIndex: number): Promise<Uint8ClampedArray> {
      throwIfAborted(signal)
      const offset = frameIndex * frameBytes
      const { buffer, bytesRead } = await handle.read(
        Buffer.allocUnsafe(frameBytes),
        0,
        frameBytes,
        offset
      )

      if (bytesRead !== frameBytes) {
        throw new Error('La lecture d une image video est incoherente.')
      }

      return new Uint8ClampedArray(buffer.buffer, buffer.byteOffset, bytesRead)
    },

    async close() {
      await handle.close()
    }
  }
}

async function readFramePixels(framesDir: string, frameIndex: number, metadata: VideoMetadata, signal?: AbortSignal) {
  throwIfAborted(signal)
  const jpegPath = join(framesDir, `frame-${String(frameIndex + 1).padStart(6, '0')}.jpg`)
  const { data, info } = await sharp(jpegPath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })

  if (info.width !== metadata.width || info.height !== metadata.height) {
    throw new Error('La lecture d une image video est incoherente.')
  }

  return new Uint8ClampedArray(data)
}

async function detectFacesFromPixels(
  pixels: Uint8ClampedArray,
  metadata: VideoMetadata,
  probabilityThreshold: number,
  detectionModel: DetectionModel
) {
  const detector = useFaceDetector(
    getServerModelPath(detectionModel),
    DETECTION_MODELS[detectionModel].modelType
  )
  const detection = await detector.detectFaces({
    data: pixels,
    width: metadata.width,
    height: metadata.height
  }, probabilityThreshold)

  return detection.faces
}

async function writeDebugDump(
  dumpPath: string,
  metadata: VideoMetadata,
  samples: FaceSample[],
  resolveFaces: (frameIndex: number) => Array<{ id: string, x: number, y: number, width: number, height: number, confidence: number, source?: string }>
) {
  const frames: Array<{ frameIndex: number, faces: ReturnType<typeof resolveFaces> }> = []

  for (let frameIndex = 0; frameIndex < metadata.frameCount; frameIndex += 1) {
    frames.push({
      frameIndex,
      faces: resolveFaces(frameIndex)
    })
  }

  await writeFile(dumpPath, JSON.stringify({
    frameCount: metadata.frameCount,
    fps: metadata.fps,
    width: metadata.width,
    height: metadata.height,
    samples: samples.map(sample => ({
      frameIndex: sample.frameIndex,
      faces: sample.faces.map(face => ({ id: face.id, x: face.x, y: face.y, width: face.width, height: face.height, confidence: face.confidence }))
    })),
    frames
  }, null, 2), 'utf8')
}

export async function createVideoJobWorkspace(fileName: string): Promise<VideoJobWorkspace> {
  await mkdir(VIDEO_OUTPUT_ROOT, { recursive: true })
  const tempRoot = await mkdtemp(join(VIDEO_OUTPUT_ROOT, 'job-'))
  const inputExtension = getInputExtension(fileName)

  return {
    tempRoot,
    inputPath: join(tempRoot, `input${inputExtension}`),
    outputPath: join(tempRoot, 'output.mp4'),
    framesDir: join(tempRoot, 'frames')
  }
}

async function encodeBlurredVideo(
  workspace: VideoJobWorkspace,
  metadata: VideoMetadata,
  samples: FaceSample[],
  settings: EditorSettings,
  progressTracker: AdaptiveProgress,
  emitProgress: (report: AdaptiveProgressReport) => void,
  signal?: AbortSignal,
  rawPath?: string
): Promise<ProcessVideoResult> {
  const resolveFaces = createVideoFaceResolver(samples, metadata.fps)

  if (DEBUG_DUMP_PATH) {
    await writeDebugDump(DEBUG_DUMP_PATH, metadata, samples, resolveFaces)
  }

  const encoderArgs = [
    '-hide_banner',
    '-loglevel',
    'error',
    '-f',
    'rawvideo',
    '-pix_fmt',
    'rgba',
    '-s',
    `${metadata.width}x${metadata.height}`,
    '-r',
    String(metadata.fps),
    '-i',
    'pipe:0',
    '-i',
    workspace.inputPath,
    '-map',
    '0:v:0',
    '-map',
    '1:a?',
    '-c:v',
    'libx264',
    '-preset',
    'fast',
    '-pix_fmt',
    'yuv420p',
    '-c:a',
    metadata.audioCodec && MP4_AUDIO_COPY_CODECS.has(metadata.audioCodec) ? 'copy' : 'aac',
    '-movflags',
    '+faststart'
  ]

  if (
    metadata.audioDuration !== null
    && metadata.videoDuration !== null
    && metadata.audioDuration > metadata.videoDuration
  ) {
    encoderArgs.push('-shortest')
  }

  encoderArgs.push(workspace.outputPath)

  const encoder = spawn(getFfmpegPath(), encoderArgs, {
    stdio: ['pipe', 'ignore', 'pipe']
  })
  const readEncoderError = captureTextStream(encoder.stderr)
  const encoderDone = waitForProcess(encoder, 'L encodage video a echoue.', readEncoderError, signal)
  let frameIndex = 0
  progressTracker.setFrames(metadata.frameCount)
  const rawReader = rawPath
    ? await createRawFrameReader(rawPath, metadata, signal)
    : null

  try {
    for (; frameIndex < metadata.frameCount; frameIndex += 1) {
      throwIfAborted(signal)
      const pixels = rawReader
        ? await rawReader.readFrame(frameIndex)
        : await readFramePixels(workspace.framesDir, frameIndex, metadata, signal)
      const processed = blurVideoFrame(
        {
          data: pixels,
          width: metadata.width,
          height: metadata.height
        },
        settings,
        resolveFaces,
        frameIndex
      )

      await writeFrame(encoder.stdin, processed, signal)
      emitProgress(progressTracker.report(frameIndex + 1))
    }

    encoder.stdin.end()
    await encoderDone
    emitProgress(progressTracker.nextPhase())

    await rm(workspace.framesDir, { recursive: true, force: true })
    if (rawPath) {
      await unlink(rawPath).catch(() => {})
    }
    await unlink(workspace.inputPath).catch(() => {})
    await unlink(join(workspace.tempRoot, 'checkpoint.json')).catch(() => {})

    return { outputPath: workspace.outputPath, tempRoot: workspace.tempRoot }
  } catch (cause) {
    encoder.stdin.destroy()
    await silenceProcess(encoder, encoderDone)
    throw cause
  } finally {
    await rawReader?.close()
  }
}

export async function processVideoFromPath(
  workspace: VideoJobWorkspace,
  settings: EditorSettings,
  onProgress?: (progress: number, remainingMs: number | null, message?: string) => void,
  signal?: AbortSignal
): Promise<ProcessVideoResult> {
  throwIfAborted(signal)

  const inputPath = workspace.inputPath
  const checkpointPath = join(workspace.tempRoot, 'checkpoint.json')
  const progressTracker = createAdaptiveProgress([
    {
      frames: 0,
      weight: 0.15,
      label: 'Extraction des images de la video.',
      labelBuilder: (done, total) => total > 0
        ? `Extraction des images (${done}/${total}).`
        : 'Extraction des images de la video.'
    },
    {
      frames: 0,
      weight: 0.45,
      label: 'Detection des visages dans les images.',
      labelBuilder: (done, total) => `Detection des visages (${done}/${total} images).`
    },
    {
      frames: 0,
      weight: 0.4,
      label: 'Application du flou sur les images.',
      labelBuilder: (done, total) => total > 0
        ? `Application du flou (${done}/${total} images).`
        : 'Application du flou sur les images.'
    }
  ])
  const emitProgress = (report: AdaptiveProgressReport) => {
    onProgress?.(report.progress, report.remainingMs, report.message)
  }

  try {
    throwIfAborted(signal)
    const metadata = await readVideoMetadata(inputPath, signal)
    const checkpoint = await readVideoCheckpoint(workspace.tempRoot)
    let samples: FaceSample[]

    if (checkpoint && checkpoint.frameCount > 0) {
      metadata.width = checkpoint.width
      metadata.height = checkpoint.height
      metadata.fps = checkpoint.fps
      metadata.rotation = checkpoint.rotation
      metadata.frameCount = checkpoint.frameCount
      samples = checkpoint.samples
      progressTracker.nextPhase(metadata.frameCount)
    } else {
      await rm(workspace.framesDir, { recursive: true, force: true })
      progressTracker.setFrames(metadata.frameCount)
      const rawPath = join(workspace.tempRoot, 'raw-rgba.bin')
      const frameCount = await extractRawRgbaVideo(
        inputPath,
        rawPath,
        metadata,
        frame => emitProgress(progressTracker.report(frame)),
        signal
      )
      metadata.frameCount = frameCount
      emitProgress(progressTracker.nextPhase(metadata.frameCount))

      if (frameCount <= 0) {
        throw new Error('Aucune image video n a pu etre extraite.')
      }

      samples = []
    }

    const resumeFromFrame = samples.length > 0 ? samples[samples.length - 1]!.frameIndex + 1 : 0
    const rawPath = join(workspace.tempRoot, 'raw-rgba.bin')
    const rawReader = await createRawFrameReader(rawPath, metadata, signal)

    try {
      for (let frameIndex = resumeFromFrame; frameIndex < metadata.frameCount; frameIndex += 1) {
        throwIfAborted(signal)
        const pixels = await rawReader.readFrame(frameIndex)
        const faces = await detectFacesFromPixels(
          pixels,
          metadata,
          settings.confidenceThreshold,
          settings.detectionModel
        )
        samples.push({ frameIndex, faces })

        if (frameIndex % CHECKPOINT_INTERVAL_FRAMES === 0 || frameIndex === metadata.frameCount - 1) {
          await writeFile(checkpointPath, JSON.stringify({
            frameCount: metadata.frameCount,
            width: metadata.width,
            height: metadata.height,
            fps: metadata.fps,
            rotation: metadata.rotation,
            samples
          }), 'utf8')
        }

        emitProgress(progressTracker.report(frameIndex + 1))
      }
    } finally {
      await rawReader.close()
    }

    emitProgress(progressTracker.nextPhase(metadata.frameCount))
    throwIfAborted(signal)

    return await encodeBlurredVideo(
      workspace,
      metadata,
      samples,
      settings,
      progressTracker,
      emitProgress,
      signal,
      rawPath
    )
  } catch (cause) {
    if (signal?.aborted) {
      throw createAbortError()
    }

    await rm(workspace.tempRoot, { recursive: true, force: true })

    if (cause instanceof Error) {
      throw new Error(`Le traitement video serveur a echoue : ${cause.message}`)
    }

    throw new Error('Le traitement video serveur a echoue.')
  }
}

async function detectVideoWithFakeProgress(
  inputPath: string,
  settings: EditorSettings,
  metadata: VideoMetadata,
  segmentCount: number,
  report: (done: number) => void,
  signal?: AbortSignal
) {
  const estimate = estimateProcessingTimeMs({
    processingMode: 'cloud',
    detectionModel: settings.detectionModel,
    resolution: { width: metadata.width, height: metadata.height },
    frameCount: metadata.frameCount
  })
  const fakeDurationMs = estimate.estimatedMs > 0 ? estimate.estimatedMs : 60_000
  const tickMs = 200
  const steps = Math.max(1, Math.round(fakeDurationMs / tickMs))
  const donePerTick = segmentCount / steps
  let fakeDone = 0
  const timer = setInterval(() => {
    fakeDone = Math.min(segmentCount, fakeDone + donePerTick)
    report(fakeDone)
  }, tickMs)

  try {
    return await detectVideoFacesOnModal(inputPath, settings, {
      frameCount: metadata.frameCount,
      fps: metadata.fps,
      signal,
      onSegmentProgress: (completed, _total) => {
        fakeDone = Math.max(fakeDone, completed)
        report(fakeDone)
      }
    })
  } finally {
    clearInterval(timer)
  }
}

export async function processVideoWithCloudDetection(
  workspace: VideoJobWorkspace,
  settings: EditorSettings,
  onProgress?: (progress: number, remainingMs: number | null, message?: string) => void,
  signal?: AbortSignal
): Promise<ProcessVideoResult> {
  throwIfAborted(signal)

  const progressTracker = createAdaptiveProgress([
    {
      frames: 0,
      weight: 0.15,
      label: 'Extraction des images de la video.',
      labelBuilder: (done, total) => total > 0
        ? `Extraction des images (${done}/${total}).`
        : 'Extraction des images de la video.'
    },
    {
      frames: 0,
      weight: 0.5,
      label: 'Calculs dans le cloud.',
      labelBuilder: (done, total) => `Calculs dans le cloud (${Math.min(total, Math.floor(done))}/${total} instance${total > 1 ? 's' : ''}).`
    },
    {
      frames: 0,
      weight: 0.35,
      label: 'Application du flou sur les images.',
      labelBuilder: (done, total) => total > 0
        ? `Application du flou (${done}/${total} images).`
        : 'Application du flou sur les images.'
    }
  ])
  const emitProgress = (report: AdaptiveProgressReport) => {
    onProgress?.(report.progress, report.remainingMs, report.message)
  }

  try {
    throwIfAborted(signal)
    const metadata = await readVideoMetadata(workspace.inputPath, signal)
    await rm(workspace.framesDir, { recursive: true, force: true })
    progressTracker.setFrames(metadata.frameCount)
    const rawPath = join(workspace.tempRoot, 'raw-rgba.bin')
    const frameCount = await extractRawRgbaVideo(
      workspace.inputPath,
      rawPath,
      metadata,
      frame => emitProgress(progressTracker.report(frame)),
      signal
    )
    metadata.frameCount = frameCount

    if (frameCount <= 0) {
      throw new Error('Aucune image video n a pu etre extraite.')
    }

    const segmentCount = computeCloudSegmentCount(frameCount, metadata.fps)
    emitProgress(progressTracker.nextPhase(segmentCount))

    const detection = await detectVideoWithFakeProgress(
      workspace.inputPath,
      settings,
      metadata,
      segmentCount,
      done => emitProgress(progressTracker.report(done)),
      signal
    )
    emitProgress(progressTracker.nextPhase(frameCount))
    throwIfAborted(signal)

    return await encodeBlurredVideo(
      workspace,
      metadata,
      detection.samples,
      settings,
      progressTracker,
      emitProgress,
      signal,
      rawPath
    )
  } catch (cause) {
    if (signal?.aborted) {
      throw createAbortError()
    }

    await rm(workspace.tempRoot, { recursive: true, force: true })

    if (cause instanceof Error) {
      throw new Error(`Le traitement video cloud a echoue : ${cause.message}`)
    }

    throw new Error('Le traitement video cloud a echoue.')
  }
}
