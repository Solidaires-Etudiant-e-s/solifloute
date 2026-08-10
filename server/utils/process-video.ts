import { execFile, spawn } from 'node:child_process'
import { once } from 'node:events'
import { mkdir, mkdtemp, readFile, readdir, rm, unlink, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { promisify } from 'node:util'
import sharp from 'sharp'
import type { DetectionModel, EditorSettings } from '~~/shared/types/faces'
import { createAdaptiveProgress, type AdaptiveProgressReport } from '~~/shared/utils/progress'
import {
  blurVideoFrame,
  collectFaceSamples,
  createVideoFaceResolver
} from '~~/shared/utils/videoProcessing'
import type { FaceSample } from '~~/shared/utils/videoFaceTracking'
import { DETECTION_MODELS, getServerModelPath } from '~~/shared/utils/detectionModels'
import { useFaceDetector } from '~~/shared/utils/useFaceDetector'

const execFileAsync = promisify(execFile)
const VIDEO_OUTPUT_ROOT = process.env.PROCESS_VIDEO_OUTPUT_DIR || join(process.cwd(), '.data', 'video-results')
const DEBUG_DUMP_PATH = process.env.PROCESS_VIDEO_DEBUG_DUMP
const FRAME_JPEG_QUALITY = 3
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

async function countExtractedFrames(framesDir: string) {
  try {
    const files = await readdir(framesDir)
    return files.filter(file => file.endsWith('.jpg')).length
  } catch {
    return 0
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

async function readVideoMetadata(inputPath: string, signal?: AbortSignal): Promise<VideoMetadata> {
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

async function extractFramesToJpeg(
  inputPath: string,
  framesDir: string,
  metadata: VideoMetadata,
  onFrame: (frame: number) => void,
  signal?: AbortSignal
) {
  throwIfAborted(signal)
  await mkdir(framesDir, { recursive: true })
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
    '-q:v',
    String(FRAME_JPEG_QUALITY),
    '-f',
    'image2',
    join(framesDir, 'frame-%06d.jpg'),
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

  const files = await readdir(framesDir)
  return files.filter(file => file.endsWith('.jpg')).length
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

export async function processVideoFromPath(
  workspace: VideoJobWorkspace,
  settings: EditorSettings,
  onProgress?: (progress: number, remainingMs: number | null) => void,
  signal?: AbortSignal
): Promise<ProcessVideoResult> {
  throwIfAborted(signal)

  const inputPath = workspace.inputPath
  const checkpointPath = join(workspace.tempRoot, 'checkpoint.json')
  const progressTracker = createAdaptiveProgress([
    { frames: 0, weight: 0.2 },
    { frames: 0, weight: 0.5 },
    { frames: 0, weight: 0.3 }
  ])
  const emitProgress = (report: AdaptiveProgressReport) => {
    onProgress?.(report.progress, report.remainingMs)
  }

  try {
    throwIfAborted(signal)
    const metadata = await readVideoMetadata(inputPath, signal)
    const checkpoint = await readVideoCheckpoint(workspace.tempRoot)
    const extractedFrames = await countExtractedFrames(workspace.framesDir)
    let samples: FaceSample[]

    if (checkpoint && checkpoint.frameCount > 0 && extractedFrames === checkpoint.frameCount) {
      metadata.width = checkpoint.width
      metadata.height = checkpoint.height
      metadata.fps = checkpoint.fps
      metadata.rotation = checkpoint.rotation
      metadata.frameCount = checkpoint.frameCount
      samples = checkpoint.samples
      progressTracker.nextPhase()
      progressTracker.nextPhase()
    } else {
      await rm(workspace.framesDir, { recursive: true, force: true })
      progressTracker.setFrames(metadata.frameCount)
      const frameCount = await extractFramesToJpeg(
        inputPath,
        workspace.framesDir,
        metadata,
        frame => emitProgress(progressTracker.report(frame)),
        signal
      )
      metadata.frameCount = frameCount
      emitProgress(progressTracker.nextPhase())

      if (frameCount <= 0) {
        throw new Error('Aucune image video n a pu etre extraite.')
      }

      progressTracker.setFrames(frameCount)
      samples = await collectFaceSamples(
        frameCount,
        async (frameIndex) => {
          const pixels = await readFramePixels(workspace.framesDir, frameIndex, metadata, signal)
          return await detectFacesFromPixels(
            pixels,
            metadata,
            settings.confidenceThreshold,
            settings.detectionModel
          )
        },
        framesDone => emitProgress(progressTracker.report(framesDone))
      )
      emitProgress(progressTracker.nextPhase())
      throwIfAborted(signal)

      await writeFile(checkpointPath, JSON.stringify({
        frameCount: metadata.frameCount,
        width: metadata.width,
        height: metadata.height,
        fps: metadata.fps,
        rotation: metadata.rotation,
        samples
      }), 'utf8')
    }

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
      inputPath,
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

    try {
      for (; frameIndex < metadata.frameCount; frameIndex += 1) {
        throwIfAborted(signal)
        const pixels = await readFramePixels(workspace.framesDir, frameIndex, metadata, signal)
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
      await unlink(inputPath).catch(() => {})
      await unlink(checkpointPath).catch(() => {})

      return { outputPath: workspace.outputPath, tempRoot: workspace.tempRoot }
    } catch (cause) {
      encoder.stdin.destroy()
      await silenceProcess(encoder, encoderDone)
      throw cause
    }
  } catch (cause) {
    await rm(workspace.tempRoot, { recursive: true, force: true })

    if (cause instanceof Error) {
      throw new Error(`Le traitement video serveur a echoue : ${cause.message}`)
    }

    throw new Error('Le traitement video serveur a echoue.')
  }
}
