import type { FFmpeg as BrowserFFmpeg } from '@ffmpeg/ffmpeg'
import classWorkerURL from '@ffmpeg/ffmpeg/worker?url'
import { toBlobURL } from '@ffmpeg/util'
import type { EditorSettings } from '~~/shared/types/faces'
import { createAdaptiveProgress, type AdaptiveProgressReport } from '~~/shared/utils/progress'
import { DETECTION_MODELS, getClientModelUrl } from '~~/shared/utils/detectionModels'
import {
  blurVideoFrame,
  collectFaceSamples,
  createVideoFaceResolver
} from '~~/shared/utils/videoProcessing'
import { detectImageData, warmupDetector } from './detect-worker'

const FFMPEG_CORE_VERSION = '0.12.10'
const FFMPEG_SELF_HOSTED_BASE = '/ffmpeg'
const FFMPEG_CDN_BASE = `https://cdn.jsdelivr.net/npm/@ffmpeg/core@${FFMPEG_CORE_VERSION}/dist/esm`
const FFMPEG_LOAD_TIMEOUT_MS = 30_000
const MAX_BROWSER_VIDEO_DURATION_SECONDS = 60
const MAX_BROWSER_VIDEO_PIXELS = 1920 * 1080
const MAX_BROWSER_PROCESSING_WIDTH = 1280
const MAX_BROWSER_PROCESSING_HEIGHT = 720
const MAX_BROWSER_VIDEO_FRAMES = 1800
const READ_METADATA_TIMEOUT_MS = 15_000
const OUTPUT_JPEG_QUALITY = 0.9
const DEBUG_PREFIX = '[solifloute:browser-video]'

export { MAX_BROWSER_VIDEO_DURATION_SECONDS, MAX_BROWSER_VIDEO_PIXELS }

export function getBrowserProcessingDimensions(width: number, height: number) {
  if (width <= 0 || height <= 0) {
    return { width, height }
  }

  const scale = Math.min(
    1,
    Math.min(MAX_BROWSER_PROCESSING_WIDTH / width, MAX_BROWSER_PROCESSING_HEIGHT / height)
  )

  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale))
  }
}

export interface BrowserVideoProgress {
  progress: number
  message: string
  remainingMs: number | null
}

type BrowserVideoProgressHandler = (progress: BrowserVideoProgress) => void
type BrowserPhaseProgressHandler = (fraction: number, message: string) => void

let ffmpegPromise: Promise<{
  ffmpeg: BrowserFFmpeg
  fetchFile: (file: File) => Promise<Uint8Array>
}> | null = null

function debugLog(message: string, details?: unknown) {
  if (details === undefined) {
    console.info(`${DEBUG_PREFIX} ${message}`)
    return
  }

  console.info(`${DEBUG_PREFIX} ${message}`, details)
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string) {
  return new Promise<T>((resolve, reject) => {
    const timer = window.setTimeout(() => {
      reject(new Error(message))
    }, timeoutMs)

    promise.then(
      (value) => {
        window.clearTimeout(timer)
        resolve(value)
      },
      (error) => {
        window.clearTimeout(timer)
        reject(error)
      }
    )
  })
}

function reportProgress(
  onProgress: BrowserVideoProgressHandler | undefined,
  progress: number,
  message: string,
  remainingMs: number | null = null
) {
  onProgress?.({
    progress: Math.max(0, Math.min(1, progress)),
    message,
    remainingMs
  })
}

function normalizeErrorMessage(error: unknown) {
  if (error instanceof Error && error.message) {
    const message = error.message

    if (/quota|exceeded|allocate|out of memory|grow|no space|errno|fs error|enospc/i.test(message)) {
      return 'Le navigateur n a pas assez de memoire ou d espace de stockage temporaire pour traiter cette video. Utilisez le mode serveur.'
    }

    if (/fetch|cors|network|timeout|expire/i.test(message)) {
      return 'Le telechargement des dependances navigateur a echoue. Verifiez votre connexion ou utilisez le mode serveur.'
    }

    return message
  }

  return String(error)
}

async function getFfmpegCoreAssets() {
  const candidates = [
    {
      label: 'self-hosted',
      js: `${FFMPEG_SELF_HOSTED_BASE}/ffmpeg-core.js`,
      wasm: `${FFMPEG_SELF_HOSTED_BASE}/ffmpeg-core.wasm`
    },
    {
      label: 'CDN',
      js: `${FFMPEG_CDN_BASE}/ffmpeg-core.js`,
      wasm: `${FFMPEG_CDN_BASE}/ffmpeg-core.wasm`
    }
  ]
  let lastError: unknown = null

  for (const candidate of candidates) {
    try {
      const [coreURL, wasmURL] = await withTimeout(
        Promise.all([
          toBlobURL(candidate.js, 'text/javascript'),
          toBlobURL(candidate.wasm, 'application/wasm')
        ]),
        FFMPEG_LOAD_TIMEOUT_MS,
        'Le telechargement des assets FFmpeg du navigateur a expire. Verifiez l acces reseau ou utilisez le mode serveur.'
      )
      return { coreURL, wasmURL, source: candidate.label }
    } catch (error) {
      lastError = error
      debugLog(`ffmpeg core from ${candidate.label} unavailable`, error)
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error('Le runtime FFmpeg du navigateur est indisponible.')
}

async function getBrowserFFmpeg(onProgress?: BrowserPhaseProgressHandler) {
  const isFirstLoad = !ffmpegPromise

  if (!ffmpegPromise) {
    ffmpegPromise = (async () => {
      onProgress?.(
        0.3,
        'Preparation des dependances navigateur. Cela ne se produit que lors de la premiere utilisation sur ce navigateur.'
      )
      debugLog('loading ffmpeg modules')
      const [{ FFmpeg }, { fetchFile: readFileFromUtil }] = await Promise.all([
        import('@ffmpeg/ffmpeg'),
        import('@ffmpeg/util')
      ])
      const ffmpeg = new FFmpeg()
      const { coreURL, wasmURL, source } = await getFfmpegCoreAssets()
      debugLog('ffmpeg core assets ready', { source })

      onProgress?.(
        0.6,
        'Telechargement de FFmpeg WebAssembly. Premiere utilisation uniquement, ensuite le navigateur le garde en cache.'
      )
      await withTimeout(
        ffmpeg.load({
          classWorkerURL,
          coreURL,
          wasmURL
        }),
        FFMPEG_LOAD_TIMEOUT_MS,
        'Le runtime FFmpeg du navigateur ne repond pas. Verifiez l acces reseau ou utilisez le mode serveur.'
      )
      debugLog('ffmpeg core ready')
      onProgress?.(1, 'FFmpeg navigateur est pret.')

      return { ffmpeg, fetchFile: readFileFromUtil }
    })().catch((error) => {
      console.error(`${DEBUG_PREFIX} ffmpeg bootstrap failed`, error)
      ffmpegPromise = null
      throw error
    })
  }

  if (!isFirstLoad) {
    onProgress?.(1, 'FFmpeg navigateur est pret.')
  }

  return await ffmpegPromise
}

export async function readVideoMetadata(file: File): Promise<{ duration: number, width: number, height: number }> {
  const sourceUrl = URL.createObjectURL(file)

  return await new Promise<{ duration: number, width: number, height: number }>((resolve, reject) => {
    let settled = false
    const timer = window.setTimeout(() => {
      if (settled) {
        return
      }
      settled = true
      URL.revokeObjectURL(sourceUrl)
      reject(new Error('La lecture des metadonnees video a expire.'))
    }, READ_METADATA_TIMEOUT_MS)

    const video = document.createElement('video')
    video.preload = 'metadata'
    video.muted = true
    video.playsInline = true

    const cleanup = () => {
      window.clearTimeout(timer)
      video.removeAttribute('src')
      video.load()
      URL.revokeObjectURL(sourceUrl)
    }

    video.onloadedmetadata = () => {
      if (settled) {
        return
      }
      settled = true

      const duration = Number.isFinite(video.duration) ? video.duration : 0
      const width = video.videoWidth
      const height = video.videoHeight

      cleanup()
      resolve({
        duration,
        width,
        height
      })
    }

    video.onerror = () => {
      if (settled) {
        return
      }
      settled = true
      const mediaErrorCode = video.error?.code
      cleanup()

      if (mediaErrorCode === 3 || mediaErrorCode === 4) {
        reject(new Error('Ce navigateur ne peut pas decoder cette video (codec non supporte). Utilisez le mode serveur.'))
        return
      }

      reject(new Error('Impossible de lire la video selectionnee.'))
    }

    video.src = sourceUrl
  })
}

async function writeCanvasToJpeg(
  ffmpeg: BrowserFFmpeg,
  canvas: HTMLCanvasElement,
  path: string,
  quality: number
) {
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((nextBlob) => {
      if (!nextBlob) {
        reject(new Error('Impossible d encoder une image de la video.'))
        return
      }

      resolve(nextBlob)
    }, 'image/jpeg', quality)
  })

  await ffmpeg.writeFile(path, new Uint8Array(await blob.arrayBuffer()))
}

async function deleteFFmpegFile(ffmpeg: BrowserFFmpeg, name: string) {
  try {
    await ffmpeg.deleteFile(name)
  } catch {
    // Missing temporary files are harmless during cleanup.
  }
}

async function deleteDirContents(ffmpeg: BrowserFFmpeg, dir: string) {
  try {
    const entries = await ffmpeg.listDir(dir)

    for (const entry of entries) {
      if (entry.isDir) {
        continue
      }

      await deleteFFmpegFile(ffmpeg, `${dir}/${entry.name}`)
    }

    await ffmpeg.deleteDir(dir)
  } catch {
    // Missing directories are harmless during cleanup.
  }
}

async function runFfmpeg(
  ffmpeg: BrowserFFmpeg,
  args: string[],
  onProgress?: (progress: number) => void,
  onLog?: (type: string, message: string) => void
) {
  const handleProgress = ({ progress }: { progress: number }) => {
    onProgress?.(Math.max(0, Math.min(1, progress)))
  }
  const handleLog = ({ type, message }: { type: string, message: string }) => {
    onLog?.(type, message)
  }

  ffmpeg.on('progress', handleProgress)
  ffmpeg.on('log', handleLog)

  try {
    return await ffmpeg.exec(args)
  } finally {
    ffmpeg.off('progress', handleProgress)
    ffmpeg.off('log', handleLog)
  }
}

function seekVideoTo(video: HTMLVideoElement, time: number) {
  const target = Number.isFinite(time) ? time : 0

  if (Math.abs(video.currentTime - target) < 0.001) {
    return Promise.resolve()
  }

  return new Promise<void>((resolve) => {
    let settled = false
    const finish = () => {
      if (settled) {
        return
      }
      settled = true
      video.removeEventListener('seeked', finish)
      resolve()
    }

    video.addEventListener('seeked', finish)
    video.currentTime = target

    // Hard ceiling: never let a slow/unrelated seek stall the whole pipeline.
    // If 'seeked' does not fire (throttled tab, slow decode), resolve anyway
    // so processing can move on to the next frame.
    window.setTimeout(finish, 2000)
  })
}

function probeVideoFps(video: HTMLVideoElement) {
  return new Promise<number>((resolve) => {
    const requestFrame = (video as HTMLVideoElement & {
      requestVideoFrameCallback?: (callback: (now: number, metadata: unknown) => void) => number
    }).requestVideoFrameCallback

    if (typeof requestFrame !== 'function') {
      resolve(30)
      return
    }

    let frames = 0
    const startedAt = performance.now()
    let settled = false

    const countFrame = () => {
      if (settled) {
        return
      }
      frames += 1
      const elapsed = performance.now() - startedAt

      if (elapsed >= 1200 || video.currentTime >= video.duration) {
        settled = true
        video.pause()
        const seconds = Math.max(0.001, elapsed / 1000)
        resolve(Math.max(1, Math.round(frames / seconds)))
        return
      }

      requestFrame.call(video, countFrame as (now: number, metadata: unknown) => void)
    }

    requestFrame.call(video, countFrame as (now: number, metadata: unknown) => void)
    void video.play()
  })
}

export async function processVideoInBrowser(
  file: File,
  settings: EditorSettings,
  onProgress?: BrowserVideoProgressHandler
) {
  const progressTracker = createAdaptiveProgress([
    { frames: 1, weight: 0.08 },
    { frames: 1, weight: 0.12 },
    { frames: 0, weight: 0.5 },
    { frames: 1, weight: 0.23 },
    { frames: 0, weight: 0.07 }
  ])
  const emit = (report: AdaptiveProgressReport, message: string) => {
    reportProgress(onProgress, report.progress, message, report.remainingMs)
  }
  const emitDeps = (fraction: number, message: string) => {
    reportProgress(onProgress, progressTracker.report(fraction).progress, message, null)
  }

  debugLog('starting browser video processing', {
    fileName: file.name,
    fileSize: file.size
  })
  emitDeps(
    0.1,
    'Chargement des dependances navigateur: modele IA et FFmpeg WebAssembly. Cela ne se produit que lors de la premiere utilisation.'
  )
  const metadata = await readVideoMetadata(file)

  if (metadata.width * metadata.height <= 0) {
    debugLog('browser cannot decode video (zero dimensions)', metadata)
    throw new Error('Ce navigateur ne peut pas decoder cette video (codec non supporte). Utilisez le mode serveur.')
  }

  if (metadata.duration > MAX_BROWSER_VIDEO_DURATION_SECONDS) {
    throw new Error('Cette video est trop longue pour le traitement navigateur. Utilisez le mode serveur.')
  }

  if (metadata.width * metadata.height > MAX_BROWSER_VIDEO_PIXELS) {
    throw new Error('Cette resolution video est trop elevee pour le traitement navigateur. Utilisez le mode serveur.')
  }

  const { ffmpeg, fetchFile: readUploadedFile } = await getBrowserFFmpeg((fraction, message) => {
    emitDeps(fraction, message)
  })
  emitDeps(1, 'Dependances pretes. Analyse de la video.')

  const jobId = crypto.randomUUID().replaceAll('-', '')
  const outputFramesDir = `${jobId}/out`
  const inputName = `${jobId}-input.${file.name.split('.').pop() || 'mp4'}`
  const outputName = `${jobId}-output.mp4`

  debugLog('video metadata ready', {
    width: metadata.width,
    height: metadata.height,
    duration: metadata.duration
  })

  try {
    debugLog('stage: creating ffmpeg dirs')
    await ffmpeg.createDir(jobId)
    await ffmpeg.createDir(outputFramesDir)
    emitDeps(0.5, 'Extraction des images de la video.')
    debugLog('stage: writing input file')
    await ffmpeg.writeFile(inputName, await readUploadedFile(file))

    debugLog('stage: warming up detector')
    await warmupDetector(
      getClientModelUrl(settings.detectionModel),
      DETECTION_MODELS[settings.detectionModel].modelType
    )
    emitDeps(
      1,
      'Chargement du modele IA de detection. Premiere utilisation uniquement, ensuite il est garde en cache.'
    )
    emit(progressTracker.nextPhase(), 'Extraction des images de la video.')

    debugLog('stage: decoding source video')
    const sourceUrl = URL.createObjectURL(file)
    const video = document.createElement('video')
    let frameWidth = 0
    let frameHeight = 0
    let fps = 30
    let frameCount = 0
    let sourceDuration = 0
    video.preload = 'auto'
    video.muted = true
    video.playsInline = true
    video.src = sourceUrl

    const decodeCanvas = document.createElement('canvas')
    const maybeDecodeContext = decodeCanvas.getContext('2d', { willReadFrequently: true })

    if (!maybeDecodeContext) {
      throw new Error('Le contexte canvas 2D est indisponible.')
    }

    const decodeContext = maybeDecodeContext

    try {
      await new Promise<void>((resolve, reject) => {
        video.onloadedmetadata = () => resolve()
        video.onerror = () => reject(new Error('Le navigateur ne peut pas decoder cette video (codec non supporte). Utilisez le mode serveur.'))
      })
      video.onloadedmetadata = null
      video.onerror = null

      const nativeWidth = video.videoWidth
      const nativeHeight = video.videoHeight
      sourceDuration = Number.isFinite(video.duration) ? video.duration : 0

      if (nativeWidth <= 0 || nativeHeight <= 0 || sourceDuration <= 0) {
        throw new Error('Ce navigateur ne peut pas decoder cette video. Utilisez le mode serveur.')
      }

      const processingDimensions = getBrowserProcessingDimensions(nativeWidth, nativeHeight)
      frameWidth = processingDimensions.width
      frameHeight = processingDimensions.height
      fps = await probeVideoFps(video)
      frameCount = Math.max(1, Math.ceil(sourceDuration * fps))

      if (frameCount > MAX_BROWSER_VIDEO_FRAMES) {
        throw new Error('Cette video contient trop d images pour le traitement navigateur. Utilisez le mode serveur.')
      }

      decodeCanvas.width = frameWidth
      decodeCanvas.height = frameHeight

      async function decodeFrameAtFrameIndex(frameIndex: number) {
        const targetTime = Math.min(Math.max(0, sourceDuration - 0.01), frameIndex / fps)
        await seekVideoTo(video, targetTime)
        decodeContext.drawImage(video, 0, 0, frameWidth, frameHeight)
        return decodeContext.getImageData(0, 0, frameWidth, frameHeight)
      }

      async function detectFacesAtFrame(frameIndex: number) {
        const imageData = await decodeFrameAtFrameIndex(frameIndex)
        const result = await detectImageData(
          imageData,
          settings.confidenceThreshold,
          getClientModelUrl(settings.detectionModel),
          DETECTION_MODELS[settings.detectionModel].modelType
        )
        return result.faces
      }

      progressTracker.setFrames(frameCount)
      debugLog('stage: detecting faces', { frameCount })
      const samples = await collectFaceSamples(
        frameCount,
        detectFacesAtFrame,
        framesDone => emit(progressTracker.report(framesDone), 'Detection des visages dans la video.')
      )
      emit(progressTracker.nextPhase(), 'Floutage des images video.')
      debugLog('face samples collected', { sampleCount: samples.length })
      const resolveFaces = createVideoFaceResolver(samples, fps)

      const previewCanvas = document.createElement('canvas')
      previewCanvas.width = frameWidth
      previewCanvas.height = frameHeight
      const previewContext = previewCanvas.getContext('2d')

      if (!previewContext) {
        throw new Error('Le contexte canvas 2D est indisponible.')
      }

      progressTracker.setFrames(frameCount)
      debugLog('stage: blurring frames', { frameCount })

      for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
        if (frameIndex === 0 || frameIndex % 30 === 0) {
          debugLog('processing frame batch', { frameIndex, frameCount })
        }

        const imageData = await decodeFrameAtFrameIndex(frameIndex)
        const processedImageData = new ImageData(
          blurVideoFrame(
            {
              data: imageData.data,
              width: imageData.width,
              height: imageData.height
            },
            settings,
            resolveFaces,
            frameIndex
          ),
          imageData.width,
          imageData.height
        )

        previewContext.putImageData(processedImageData, 0, 0)
        const frameName = `frame-${String(frameIndex + 1).padStart(5, '0')}.jpg`
        await writeCanvasToJpeg(ffmpeg, previewCanvas, `${outputFramesDir}/${frameName}`, OUTPUT_JPEG_QUALITY)
        emit(progressTracker.report(frameIndex + 1), 'Floutage des images video.')
      }
    } finally {
      video.removeAttribute('src')
      video.load()
      URL.revokeObjectURL(sourceUrl)
    }

    debugLog('encoding output video')
    let lastLoggedEncodingBucket = -1
    emit(progressTracker.nextPhase(), 'Encodage de la video finale.')
    debugLog('stage: encoding')

    const encodeExitCode = await runFfmpeg(
      ffmpeg,
      [
        '-y',
        '-framerate',
        String(fps),
        '-i',
        `${outputFramesDir}/frame-%05d.jpg`,
        '-i',
        inputName,
        '-map',
        '0:v:0',
        '-map',
        '1:a?',
        '-c:v',
        'libx264',
        '-preset',
        'ultrafast',
        '-pix_fmt',
        'yuv420p',
        '-c:a',
        'copy',
        '-movflags',
        '+faststart',
        '-shortest',
        outputName
      ],
      (progress) => {
        emit(progressTracker.report(progress), 'Encodage de la video finale.')
        const bucket = Math.floor(progress * 20)

        if (bucket !== lastLoggedEncodingBucket || progress >= 1) {
          lastLoggedEncodingBucket = bucket
          debugLog('ffmpeg encoding progress', { progress })
        }
      },
      (type, message) => {
        if (
          type === 'fferr'
          || message.includes('frame=')
          || message.includes('time=')
          || message.includes('Error')
        ) {
          debugLog('ffmpeg log', { type, message })
        }
      }
    )

    if (encodeExitCode !== 0) {
      debugLog('audio copy failed, retrying with aac re-encode')
      await deleteFFmpegFile(ffmpeg, outputName)

      const retryExitCode = await runFfmpeg(
        ffmpeg,
        [
          '-y',
          '-framerate',
          String(fps),
          '-i',
          `${outputFramesDir}/frame-%05d.jpg`,
          '-i',
          inputName,
          '-map',
          '0:v:0',
          '-map',
          '1:a?',
          '-c:v',
          'libx264',
          '-preset',
          'ultrafast',
          '-pix_fmt',
          'yuv420p',
          '-c:a',
          'aac',
          '-movflags',
          '+faststart',
          '-shortest',
          outputName
        ],
        (progress) => {
          emit(progressTracker.report(progress), 'Encodage de la video finale.')
        }
      )

      if (retryExitCode !== 0) {
        throw new Error(`L encodage video du navigateur a echoue (code ${retryExitCode}).`)
      }
    }

    const data = await ffmpeg.readFile(outputName)
    debugLog('stage: read output', { bytes: data instanceof Uint8Array ? data.byteLength : null })

    if (!(data instanceof Uint8Array)) {
      throw new Error('L encodeur video du navigateur a renvoye un fichier invalide.')
    }

    debugLog('browser video processing completed', { bytes: data.byteLength })
    reportProgress(onProgress, 1, 'Video traitee.', 0)
    return new Blob([data.slice()], { type: 'video/mp4' })
  } catch (error) {
    console.error(`${DEBUG_PREFIX} processing failed`, error)
    throw new Error(normalizeErrorMessage(error))
  } finally {
    await deleteFFmpegFile(ffmpeg, inputName)
    await deleteFFmpegFile(ffmpeg, outputName)
    await deleteDirContents(ffmpeg, outputFramesDir)
    await ffmpeg.deleteDir(jobId).catch(() => {})
  }
}
