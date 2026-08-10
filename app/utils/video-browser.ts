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
const MAX_BROWSER_VIDEO_PIXELS = 1280 * 720
const MAX_BROWSER_VIDEO_FRAMES = 1800
const FRAME_JPEG_QUALITY = 3
const OUTPUT_JPEG_QUALITY = 0.9
const DEBUG_PREFIX = '[solifloute:browser-video]'

export { MAX_BROWSER_VIDEO_DURATION_SECONDS, MAX_BROWSER_VIDEO_PIXELS }

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

    if (/quota|exceeded|allocate|out of memory|grow/i.test(message)) {
      return 'Le navigateur n a pas assez de memoire pour traiter cette video. Utilisez le mode serveur.'
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
    const video = document.createElement('video')
    video.preload = 'metadata'
    video.muted = true
    video.playsInline = true
    video.onloadedmetadata = () => {
      video.removeAttribute('src')
      video.load()
      URL.revokeObjectURL(sourceUrl)
      resolve({
        duration: Number.isFinite(video.duration) ? video.duration : 0,
        width: video.videoWidth,
        height: video.videoHeight
      })
    }
    video.onerror = () => {
      const mediaErrorCode = video.error?.code

      if (mediaErrorCode === 3 || mediaErrorCode === 4) {
        URL.revokeObjectURL(sourceUrl)
        reject(new Error('Ce navigateur ne peut pas decoder cette video (codec non supporte). Utilisez le mode serveur.'))
        return
      }

      URL.revokeObjectURL(sourceUrl)
      reject(new Error('Impossible de lire la video selectionnee.'))
    }
    video.src = sourceUrl
  })
}

async function readFrameAsImageData(
  ffmpeg: BrowserFFmpeg,
  path: string,
  width: number,
  height: number
) {
  const fileData = await ffmpeg.readFile(path)
  const bytes = fileData instanceof Uint8Array
    ? fileData.slice()
    : new TextEncoder().encode(fileData)
  const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }))

  try {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const context = canvas.getContext('2d', { willReadFrequently: true })

    if (!context) {
      throw new Error('Le contexte canvas 2D est indisponible.')
    }

    context.drawImage(bitmap, 0, 0, width, height)
    return context.getImageData(0, 0, width, height)
  } finally {
    bitmap.close()
  }
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

export async function processVideoInBrowser(
  file: File,
  settings: EditorSettings,
  onProgress?: BrowserVideoProgressHandler
) {
  const progressTracker = createAdaptiveProgress([
    { frames: 1, weight: 0.08 },
    { frames: 1, weight: 0.12 },
    { frames: 0, weight: 0.5 },
    { frames: 0, weight: 0.23 },
    { frames: 1, weight: 0.07 }
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

  const { ffmpeg, fetchFile: readUploadedFile } = await getBrowserFFmpeg((fraction, message) => {
    emitDeps(fraction, message)
  })
  emitDeps(1, 'Dependances pretes. Analyse de la video.')

  if (metadata.duration > MAX_BROWSER_VIDEO_DURATION_SECONDS) {
    throw new Error('Cette video est trop longue pour le traitement navigateur. Utilisez le mode serveur.')
  }

  if (metadata.width * metadata.height > MAX_BROWSER_VIDEO_PIXELS) {
    throw new Error('Cette resolution video est trop elevee pour le traitement navigateur. Utilisez le mode serveur.')
  }

  const jobId = crypto.randomUUID().replaceAll('-', '')
  const framesDir = `${jobId}/in`
  const outputFramesDir = `${jobId}/out`
  const inputName = `${jobId}-input.${file.name.split('.').pop() || 'mp4'}`
  const outputName = `${jobId}-output.mp4`

  debugLog('video metadata ready', {
    width: metadata.width,
    height: metadata.height,
    duration: metadata.duration
  })

  try {
    await ffmpeg.createDir(jobId)
    await ffmpeg.createDir(framesDir)
    await ffmpeg.createDir(outputFramesDir)
    emitDeps(0.5, 'Extraction des images de la video.')
    await ffmpeg.writeFile(inputName, await readUploadedFile(file))

    await warmupDetector(
      getClientModelUrl(settings.detectionModel),
      DETECTION_MODELS[settings.detectionModel].modelType
    )
    emitDeps(
      1,
      'Chargement du modele IA de detection. Premiere utilisation uniquement, ensuite il est garde en cache.'
    )
    emit(progressTracker.nextPhase(), 'Extraction des images de la video.')

    const extractionErrors: string[] = []
    const extractionExitCode = await runFfmpeg(
      ffmpeg,
      [
        '-y',
        '-hide_banner',
        '-loglevel',
        'error',
        '-i',
        inputName,
        '-q:v',
        String(FRAME_JPEG_QUALITY),
        '-f',
        'image2',
        `${framesDir}/frame-%05d.jpg`
      ],
      progress => emit(progressTracker.report(progress), 'Extraction des images de la video.'),
      (type, message) => {
        if (type === 'fferr' || message.includes('Error')) {
          extractionErrors.push(message)
          debugLog('ffmpeg extraction log', { type, message })
        }
      }
    )

    if (extractionExitCode !== 0) {
      const lastError = extractionErrors[extractionErrors.length - 1] || ''

      if (/decoder|decode|codec|not implemented|unknown/i.test(lastError)) {
        throw new Error('Le navigateur ne peut pas decoder ce format video (codec non supporte). Utilisez le mode serveur.')
      }

      throw new Error(`L extraction des images video du navigateur a echoue (code ${extractionExitCode}).`)
    }

    const frameEntries = await ffmpeg.listDir(framesDir)
    const frameNames = frameEntries
      .filter(entry => !entry.isDir && entry.name.endsWith('.jpg'))
      .sort()
    const frameCount = frameNames.length

    if (frameCount <= 0) {
      throw new Error('Aucune image video n a pu etre extraite.')
    }

    if (frameCount > MAX_BROWSER_VIDEO_FRAMES) {
      throw new Error('Cette video contient trop d images pour le traitement navigateur. Utilisez le mode serveur.')
    }

    const fps = Math.max(1, frameCount / Math.max(0.001, metadata.duration))
    debugLog('frames extracted', { frameCount, fps })

    async function readFrame(frameIndex: number) {
      return await readFrameAsImageData(
        ffmpeg,
        `${framesDir}/${frameNames[frameIndex]}`,
        metadata.width,
        metadata.height
      )
    }

    async function detectFacesAtFrame(frameIndex: number) {
      const imageData = await readFrame(frameIndex)
      const result = await detectImageData(
        imageData,
        settings.confidenceThreshold,
        getClientModelUrl(settings.detectionModel),
        DETECTION_MODELS[settings.detectionModel].modelType
      )
      return result.faces
    }

    progressTracker.setFrames(frameCount)
    const samples = await collectFaceSamples(
      frameCount,
      detectFacesAtFrame,
      framesDone => emit(progressTracker.report(framesDone), 'Detection des visages dans la video.')
    )
    emit(progressTracker.nextPhase(), 'Floutage des images video.')
    debugLog('face samples collected', { sampleCount: samples.length })
    const resolveFaces = createVideoFaceResolver(samples, fps)

    const previewCanvas = document.createElement('canvas')
    previewCanvas.width = metadata.width
    previewCanvas.height = metadata.height
    const previewContext = previewCanvas.getContext('2d')

    if (!previewContext) {
      throw new Error('Le contexte canvas 2D est indisponible.')
    }

    progressTracker.setFrames(frameCount)

    for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
      if (frameIndex === 0 || frameIndex % 30 === 0) {
        debugLog('processing frame batch', { frameIndex, frameCount })
      }

      const imageData = await readFrame(frameIndex)
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
      await deleteFFmpegFile(ffmpeg, `${framesDir}/${frameNames[frameIndex]}`)
      emit(progressTracker.report(frameIndex + 1), 'Floutage des images video.')
    }

    debugLog('encoding output video')
    let lastLoggedEncodingBucket = -1
    emit(progressTracker.nextPhase(), 'Encodage de la video finale.')

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

    if (!(data instanceof Uint8Array)) {
      throw new Error('L encodeur video du navigateur a renvoye un fichier invalide.')
    }

    debugLog('browser video processing completed', { bytes: data.byteLength })
    emit(progressTracker.nextPhase(), 'Video traitee.')
    return new Blob([data.slice()], { type: 'video/mp4' })
  } catch (error) {
    console.error(`${DEBUG_PREFIX} processing failed`, error)
    throw new Error(normalizeErrorMessage(error))
  } finally {
    await deleteFFmpegFile(ffmpeg, inputName)
    await deleteFFmpegFile(ffmpeg, outputName)
    await deleteDirContents(ffmpeg, outputFramesDir)
    await deleteDirContents(ffmpeg, framesDir)
    await ffmpeg.deleteDir(jobId).catch(() => {})
  }
}
