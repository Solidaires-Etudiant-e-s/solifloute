import { execFile } from 'node:child_process'
import { mkdir, readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { promisify } from 'node:util'
import sharp from 'sharp'
import type { Face } from '../../shared/types/faces'
import { useFaceDetector } from '../../shared/utils/useFaceDetector'

const execFileAsync = promisify(execFile)
const MODEL_PATH = `${process.cwd()}/public/models/centerface.onnx`

export interface VideoMetadata {
  width: number
  height: number
  fps: number
  frameCount: number
}

export function readVideoMetadata(inputPath: string): Promise<VideoMetadata> {
  return execFileAsync('ffprobe', [
    '-v', 'error',
    '-select_streams', 'v:0',
    '-show_streams',
    '-show_entries', 'format=duration',
    '-of', 'json',
    inputPath
  ], { maxBuffer: 1024 * 1024 }).then(({ stdout }) => {
    const parsed = JSON.parse(stdout) as {
      streams?: Array<Record<string, unknown>>
      format?: { duration?: string }
    }
    const videoStream = parsed.streams?.find(stream => stream.codec_type === 'video')
    const width = Number(videoStream?.width || 0)
    const height = Number(videoStream?.height || 0)

    if (width <= 0 || height <= 0) {
      throw new Error('Dimensions video invalides.')
    }

    const fps = parseFrameRate(String(videoStream?.avg_frame_rate || ''))
    const frameCountFromStream = Number(videoStream?.nb_frames || '0')
    const videoDuration = Number(parsed.format?.duration || '0') || null
    const frameCount = frameCountFromStream > 0
      ? frameCountFromStream
      : Math.max(1, Math.ceil((videoDuration || 0) * fps))

    return { width, height, fps, frameCount }
  })
}

export async function extractFrames(inputPath: string, framesDir: string, maxFrames?: number): Promise<number> {
  await mkdir(framesDir, { recursive: true })
  const args = [
    '-hide_banner', '-loglevel', 'error',
    '-i', inputPath,
    '-q:v', '3',
    '-f', 'image2'
  ]

  if (maxFrames && maxFrames > 0) {
    args.push('-frames:v', String(maxFrames))
  }

  args.push(join(framesDir, 'frame-%06d.jpg'))

  await execFileAsync('ffmpeg', args, { maxBuffer: 1024 * 1024 })

  const files = await readdir(framesDir)
  return files.filter(file => file.endsWith('.jpg')).length
}

export async function readFrameDimensions(framesDir: string, frameIndex: number): Promise<{ width: number, height: number }> {
  const jpegPath = join(framesDir, `frame-${String(frameIndex + 1).padStart(6, '0')}.jpg`)
  const metadata = await sharp(jpegPath).metadata()
  return { width: metadata.width ?? 0, height: metadata.height ?? 0 }
}

export async function readFramePixels(
  framesDir: string,
  frameIndex: number,
  metadata: VideoMetadata
): Promise<Uint8ClampedArray> {
  const jpegPath = join(framesDir, `frame-${String(frameIndex + 1).padStart(6, '0')}.jpg`)
  const { data, info } = await sharp(jpegPath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })

  if (info.width !== metadata.width || info.height !== metadata.height) {
    throw new Error('Lecture image incoherente.')
  }

  return new Uint8ClampedArray(data)
}

export function detectFacesAtFrame(
  framesDir: string,
  metadata: VideoMetadata,
  confidenceThreshold: number
): (frameIndex: number) => Promise<Face[]> {
  const detector = useFaceDetector(MODEL_PATH)

  return async (frameIndex) => {
    const pixels = await readFramePixels(framesDir, frameIndex, metadata)
    const result = await detector.detectFaces({
      data: pixels,
      width: metadata.width,
      height: metadata.height
    }, confidenceThreshold)
    return result.faces
  }
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
