import type { EditorSettings, Face, ProcessingMode } from '~~/shared/types/faces'
import { applyBlurEffects } from '~~/shared/utils/imageProcessing'
import { DETECTION_MODELS, getClientModelUrl } from '~~/shared/utils/detectionModels'
import { estimateProcessingTimeMs } from '~~/shared/utils/estimatedProcessingTime'
import { useFaceDetector } from '~~/shared/utils/useFaceDetector'
import { fileToBase64, fileToImageData, imageDataToBlob, imageDataToDetectionInput } from '~/utils/image-io'
import { detectImageData, processImageData } from '~/utils/detect-worker'
import { inferProcessingTarget } from '~/utils/machine-profile'
import { estimateClientVideoProcessingMs } from '~/utils/performance-probe'
import { loadAllEntries, saveEntry, type StoredEntry } from '~/utils/entry-store'
import {
  getBrowserProcessingDimensions,
  MAX_BROWSER_VIDEO_DURATION_SECONDS,
  MAX_BROWSER_VIDEO_PIXELS,
  processVideoInBrowser,
  readVideoMetadata
} from '~/utils/video-browser'
import type { BrowserVideoProgress } from '~/utils/video-browser'

const SETTINGS_STORAGE_KEY = 'solifloute:settings:v2'

const DEFAULT_SETTINGS: EditorSettings = {
  confidenceThreshold: 0.2,
  blurIntensity: 0.5,
  processingMode: 'auto',
  excludedFaceIds: [],
  detectionModel: 'fast'
}

type EditorStatus = 'idle' | 'detecting' | 'processing' | 'ready' | 'error' | 'cancelled'
type MediaKind = 'image' | 'video'

interface UploadEntry {
  id: string
  createdAt: number
  fileName: string
  mediaKind: MediaKind
  originalPreviewUrl: string
  processedPreviewUrl: string
  faces: Face[]
  status: EditorStatus
  error: string
  warning: string
  lastDurationMs: number | null
  processingProgress: number | null
  processingMessage: string
  processingQueued: boolean
  estimatedRemainingMs: number | null
  serverJobId: string | null
  settings: EditorSettings
  autoResolvedMode: Exclude<ProcessingMode, 'auto'>
  videoServerOnly: boolean
  isUploading: boolean
}

interface DetectResponse {
  faces: Face[]
  durationMs: number
}

interface VideoJobResponse {
  id: string
  status: 'queued' | 'processing' | 'completed' | 'error' | 'cancelled'
  progress: number
  remainingMs: number | null
  stage: string
  error: string
  queuePosition: number | null
  downloadUrl: string | null
}

function clampProgress(progress: number | null) {
  if (progress === null || !Number.isFinite(progress)) {
    return null
  }

  return Math.max(0, Math.min(1, progress))
}

function isSafariBrowser() {
  if (!import.meta.client) {
    return false
  }

  const userAgent = navigator.userAgent
  const vendor = navigator.vendor

  return (
    userAgent.includes('Safari')
    && vendor.includes('Apple')
    && !/Chrome|CriOS|FxiOS|EdgiOS|OPiOS|OPR|Android/.test(userAgent)
  )
}

function isManualFace(face: Face) {
  return face.id.startsWith('manual:')
}

function loadPersistedSettings(): Partial<EditorSettings> {
  if (!import.meta.client) {
    return {}
  }

  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY)

    if (!raw) {
      return {}
    }

    const parsed = JSON.parse(raw) as Partial<EditorSettings>

    return {
      confidenceThreshold: typeof parsed.confidenceThreshold === 'number'
        ? parsed.confidenceThreshold
        : DEFAULT_SETTINGS.confidenceThreshold,
      blurIntensity: typeof parsed.blurIntensity === 'number'
        ? parsed.blurIntensity
        : DEFAULT_SETTINGS.blurIntensity,
      processingMode: parsed.processingMode === 'client' || parsed.processingMode === 'server' || parsed.processingMode === 'cloud'
        ? parsed.processingMode
        : DEFAULT_SETTINGS.processingMode,
      detectionModel: parsed.detectionModel === 'fast' || parsed.detectionModel === 'advanced'
        ? parsed.detectionModel
        : DEFAULT_SETTINGS.detectionModel
    }
  } catch {
    return {}
  }
}

function persistSettings(settings: Partial<EditorSettings>) {
  if (!import.meta.client) {
    return
  }

  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify({
      confidenceThreshold: settings.confidenceThreshold,
      blurIntensity: settings.blurIntensity,
      processingMode: settings.processingMode,
      detectionModel: settings.detectionModel
    }))
  } catch {
    // Persisting preferences is best-effort.
  }
}

function sanitizeEntrySettings(raw?: Partial<EditorSettings>): EditorSettings {
  const fallback = { ...DEFAULT_SETTINGS, ...loadPersistedSettings() }

  if (!raw) {
    return {
      ...fallback,
      excludedFaceIds: []
    }
  }

  return {
    confidenceThreshold: typeof raw.confidenceThreshold === 'number'
      ? raw.confidenceThreshold
      : fallback.confidenceThreshold,
    blurIntensity: typeof raw.blurIntensity === 'number'
      ? raw.blurIntensity
      : fallback.blurIntensity,
    processingMode: raw.processingMode === 'client' || raw.processingMode === 'server' || raw.processingMode === 'cloud'
      ? raw.processingMode
      : fallback.processingMode,
    excludedFaceIds: Array.isArray(raw.excludedFaceIds)
      ? raw.excludedFaceIds.filter(id => typeof id === 'string')
      : [],
    detectionModel: raw.detectionModel === 'fast' || raw.detectionModel === 'advanced'
      ? raw.detectionModel
      : fallback.detectionModel
  }
}

export function useImageEditor() {
  const entryFiles = new Map<string, File>()
  const entryImageData = new Map<string, ImageData>()
  const entryProcessedBlobs = new Map<string, Blob>()
  const videoRunIds = new Map<string, number>()
  const detectRunIds = new Map<string, number>()
  const serverVideoJobIds = new Map<string, string>()
  const entryInitialEstimates = reactive(new Map<string, number>())
  const entryVideoMetadata = new Map<string, { duration: number, width: number, height: number, frameCount?: number, fps?: number }>()

  const file = shallowRef<File | null>(null)
  const originalImageData = shallowRef<ImageData | null>(null)
  const uploadEntries = ref<UploadEntry[]>([])
  const currentEntryId = ref<string | null>(null)
  const settings = reactive<EditorSettings>({ ...DEFAULT_SETTINGS, ...loadPersistedSettings() })
  const safariVideoModalOpen = ref(false)
  const isSafari = isSafariBrowser()
  const clientDetector = shallowRef<ReturnType<typeof useFaceDetector> | null>(null)
  let clientDetectorModel: DetectionModel | null = null

  function getEntrySettings(entryId: string): EditorSettings {
    const entry = uploadEntries.value.find(candidate => candidate.id === entryId)
    return entry?.settings ?? settings
  }

  function isEntryServerOnly(entryId: string) {
    const entry = uploadEntries.value.find(candidate => candidate.id === entryId)
    return Boolean(entry && (entry.videoServerOnly || (isSafari && entry.mediaKind === 'video')))
  }

  function activeModeForEntry(entryId: string): Exclude<ProcessingMode, 'auto'> {
    const entry = uploadEntries.value.find(candidate => candidate.id === entryId)

    if (!entry) {
      return 'server'
    }

    const requested = entry.settings.processingMode === 'auto' ? entry.autoResolvedMode : entry.settings.processingMode

    // Cloud is a video-only target: never resolve it for images.
    if (requested === 'cloud' && entry.mediaKind !== 'video') {
      return 'server'
    }

    // serverOnly means the browser cannot process this entry: override auto /
    // client choices, but keep an explicit Cloud selection (Cloud runs server-side).
    if (isEntryServerOnly(entryId) && requested !== 'cloud') {
      return 'server'
    }

    return requested
  }

  function getClientDetector(detectionModel: DetectionModel) {
    if (!import.meta.client) {
      return null
    }

    if (!clientDetector.value || clientDetectorModel !== detectionModel) {
      clientDetector.value = useFaceDetector(
        getClientModelUrl(detectionModel),
        DETECTION_MODELS[detectionModel].modelType
      )
      clientDetectorModel = detectionModel
    }

    return clientDetector.value
  }

  const currentEntry = computed(() => (
    currentEntryId.value
      ? uploadEntries.value.find(entry => entry.id === currentEntryId.value) || null
      : null
  ))

  const mediaKind = computed(() => currentEntry.value?.mediaKind ?? null)
  const faces = computed(() => currentEntry.value?.faces ?? [])
  const status = computed(() => currentEntry.value?.status ?? 'idle')
  const error = computed(() => currentEntry.value?.error ?? '')
  const lastDurationMs = computed(() => currentEntry.value?.lastDurationMs ?? null)
  const originalPreviewUrl = computed(() => currentEntry.value?.originalPreviewUrl ?? '')
  const processedPreviewUrl = computed(() => currentEntry.value?.processedPreviewUrl ?? '')
  const processingProgress = computed(() => currentEntry.value?.processingProgress ?? null)
  const processingMessage = computed(() => currentEntry.value?.processingMessage ?? '')
  const estimatedRemainingMs = computed(() => currentEntry.value?.estimatedRemainingMs ?? null)

  function revokeUrl(url: string) {
    if (url) {
      URL.revokeObjectURL(url)
    }
  }

  function nextRunId(map: Map<string, number>, entryId: string) {
    const next = (map.get(entryId) || 0) + 1
    map.set(entryId, next)
    return next
  }

  function currentRunId(map: Map<string, number>, entryId: string) {
    return map.get(entryId) || 0
  }

  function updateEntry(entryId: string, patch: Partial<UploadEntry>) {
    const entry = uploadEntries.value.find(candidate => candidate.id === entryId)

    if (entry) {
      Object.assign(entry, patch)
    }
  }

  function updateCurrentEntry(patch: Partial<UploadEntry>) {
    if (currentEntryId.value) {
      updateEntry(currentEntryId.value, patch)
    }
  }

  function setEntryStatus(entryId: string, nextStatus: EditorStatus, nextError = '') {
    updateEntry(entryId, {
      status: nextStatus,
      error: nextError
    })
    void persistEntry(entryId)
  }

  function setStatus(nextStatus: EditorStatus, nextError = '') {
    if (currentEntryId.value) {
      setEntryStatus(currentEntryId.value, nextStatus, nextError)
    }
  }

  async function persistEntry(entryId: string) {
    const entry = uploadEntries.value.find(candidate => candidate.id === entryId)
    const entryFile = entryFiles.get(entryId)

    if (!entry || !entryFile) {
      return
    }

    const stored: StoredEntry = {
      id: entry.id,
      createdAt: entry.createdAt,
      fileName: entry.fileName,
      mediaKind: entry.mediaKind,
      faces: entry.faces.map(face => ({ ...face })),
      status: entry.status,
      error: entry.error,
      warning: entry.warning,
      lastDurationMs: entry.lastDurationMs,
      serverJobId: entry.serverJobId,
      settings: {
        confidenceThreshold: entry.settings.confidenceThreshold,
        blurIntensity: entry.settings.blurIntensity,
        processingMode: entry.settings.processingMode,
        excludedFaceIds: [...entry.settings.excludedFaceIds],
        detectionModel: entry.settings.detectionModel
      },
      originalBlob: entryFile,
      processedBlob: entryProcessedBlobs.get(entryId) ?? null
    }

    try {
      await saveEntry(stored)
      console.info('[solifloute] entree persistee', entry.fileName, entry.status)
    } catch (cause) {
      console.error('[solifloute] echec de persistance', entry.fileName, cause)
    }
  }

  function updateProgress(
    entryId: string,
    progress: number | BrowserVideoProgress | null,
    message = '',
    remainingMs: number | null = null
  ) {
    const progressValue = progress && typeof progress === 'object' ? progress.progress : progress
    const nextMessage = progress && typeof progress === 'object' ? progress.message : message
    const sourceRemainingMs = progress && typeof progress === 'object' ? progress.remainingMs : remainingMs
    const shouldEstimate = !nextMessage.includes('dependances navigateur')
      && !nextMessage.includes('FFmpeg WebAssembly')
      && !nextMessage.includes('FFmpeg navigateur')
      && !nextMessage.includes('modele IA')
    const normalizedProgress = clampProgress(progressValue)
    const estimatedRemainingMs = shouldEstimate && sourceRemainingMs !== null
      ? Math.max(0, sourceRemainingMs)
      : null

    updateEntry(entryId, {
      processingProgress: normalizedProgress,
      processingMessage: normalizedProgress === null ? '' : nextMessage,
      estimatedRemainingMs
    })
  }

  async function fetchServerVideoMetadata(entryId: string): Promise<{ duration: number, width: number, height: number, frameCount?: number, fps?: number } | null> {
    const entryFile = entryFiles.get(entryId)

    if (!entryFile) {
      return null
    }

    const form = new FormData()
    form.append('file', entryFile)

    updateEntry(entryId, { isUploading: true })

    try {
      const metadata = await $fetch<{ width: number, height: number, duration: number, fps: number, frameCount: number }>('/api/video-metadata', {
        method: 'POST',
        body: form
      })

      if (!metadata || metadata.width <= 0 || metadata.height <= 0 || metadata.duration <= 0) {
        return null
      }

      return metadata
    } catch (cause) {
      console.debug('[solifloute] estimate: echec des metadonnees serveur', { entryId, cause: cause instanceof Error ? cause.message : String(cause) })
      return null
    } finally {
      updateEntry(entryId, { isUploading: false })
    }
  }

  async function setInitialVideoEstimate(entryId: string) {
    const entryFile = entryFiles.get(entryId)

    if (!import.meta.client || !entryFile) {
      console.debug('[solifloute] estimate skipped: pas de client navigateur ou fichier introuvable', { entryId, hasFile: Boolean(entryFile) })
      return
    }

    let metadata: { duration: number, width: number, height: number, frameCount?: number, fps?: number } | null | undefined = entryVideoMetadata.get(entryId)
    let metadataSource: 'cache' | 'lecture' | 'serveur' = 'cache'

    if (!metadata) {
      metadataSource = 'lecture'
      metadata = await readVideoMetadata(entryFile).catch((cause) => {
        console.debug('[solifloute] estimate: echec lecture metadonnees video', { entryId, cause: cause instanceof Error ? cause.message : String(cause) })
        return null
      })
    }

    if (!metadata || metadata.width <= 0 || metadata.height <= 0 || metadata.duration <= 0) {
      // Certaines videos (ex: messages WhatsApp) ne fournissent pas de
      // metadonnees exploitables via l element <video> du navigateur.
      // On interroge alors le serveur (ffprobe) pour obtenir de vraies valeurs.
      console.debug('[solifloute] estimate: metadonnees navigateur inutilisables, fallback serveur', { entryId, metadata })
      const serverMetadata = await fetchServerVideoMetadata(entryId)

      if (!serverMetadata) {
        console.debug('[solifloute] estimate: aucun temps calcule', { entryId })
        return
      }

      metadata = serverMetadata
      metadataSource = 'serveur'
      entryVideoMetadata.set(entryId, serverMetadata)
    }

    console.debug('[solifloute] estimate: metadonnees video', { entryId, metadataSource, metadata })

    const entrySettings = getEntrySettings(entryId)
    const processingMode = activeModeForEntry(entryId)
    let estimatedMs = 0

    if (processingMode === 'client') {
      const frameCount = metadata.frameCount && metadata.frameCount > 0
        ? metadata.frameCount
        : metadata.duration > 0
          ? Math.max(1, Math.round(metadata.duration * 30))
          : 0
      const { width: processWidth, height: processHeight } = getBrowserProcessingDimensions(
        metadata.width,
        metadata.height
      )
      const clientMs = await estimateClientVideoProcessingMs(
        entrySettings.detectionModel,
        processWidth,
        processHeight,
        frameCount
      )

      if (clientMs !== null && clientMs > 0) {
        estimatedMs = clientMs
      }
    }

    if (estimatedMs <= 0) {
      const result = estimateProcessingTimeMs({
        processingMode,
        detectionModel: entrySettings.detectionModel,
        resolution: { width: metadata.width, height: metadata.height },
        frameCount: metadata.frameCount,
        durationSeconds: metadata.frameCount ? undefined : metadata.duration
      })
      estimatedMs = result.estimatedMs
    }

    console.debug('[solifloute] estimate: calcul', {
      entryId,
      processingMode,
      detectionModel: entrySettings.detectionModel,
      width: metadata.width,
      height: metadata.height,
      durationSeconds: metadata.duration,
      frameCount: metadata.frameCount,
      estimatedMs
    })

    if (estimatedMs > 0) {
      entryInitialEstimates.set(entryId, estimatedMs)
      console.debug('[solifloute] estimate: applique a l entree', {
        entryId,
        estimatedRemainingMs: uploadEntries.value.find(entry => entry.id === entryId)?.estimatedRemainingMs
      })
    }
  }

  function getManualFaces(entryId: string) {
    return uploadEntries.value.find(entry => entry.id === entryId)?.faces.filter(isManualFace) ?? []
  }

  function applyDetection(entryId: string, result: DetectResponse) {
    updateEntry(entryId, {
      faces: [...result.faces, ...getManualFaces(entryId)],
      lastDurationMs: result.durationMs
    })
    void persistEntry(entryId)
  }

  function createSettingsSnapshot(entryId: string): EditorSettings {
    const source = getEntrySettings(entryId)

    return {
      confidenceThreshold: source.confidenceThreshold,
      blurIntensity: source.blurIntensity,
      processingMode: source.processingMode,
      excludedFaceIds: [...source.excludedFaceIds],
      detectionModel: source.detectionModel
    }
  }

  async function setProcessedPreview(entryId: string, blob: Blob) {
    const entry = uploadEntries.value.find(candidate => candidate.id === entryId)

    if (!entry) {
      return
    }

    revokeUrl(entry.processedPreviewUrl)
    entryProcessedBlobs.set(entryId, blob)
    updateEntry(entryId, { processedPreviewUrl: URL.createObjectURL(blob) })
    await persistEntry(entryId)
  }

  async function getEntryImageData(entryId: string) {
    let imageData = entryImageData.get(entryId)

    if (!imageData) {
      const entryFile = entryFiles.get(entryId)

      if (!entryFile) {
        return null
      }

      imageData = await fileToImageData(entryFile)
      entryImageData.set(entryId, imageData)
    }

    return imageData
  }

  async function refreshClientPreviewForEntry(entryId: string, nextFaces: Face[] = []) {
    const imageData = await getEntryImageData(entryId)

    if (!imageData) {
      return
    }

    const entry = uploadEntries.value.find(candidate => candidate.id === entryId)
    const facesToBlur = nextFaces.length > 0 ? nextFaces : (entry?.faces ?? [])
    const entrySettings = getEntrySettings(entryId)
    const processedImageData = new ImageData(
      applyBlurEffects(
        imageDataToDetectionInput(imageData),
        facesToBlur,
        entrySettings.excludedFaceIds,
        entrySettings.blurIntensity
      ),
      imageData.width,
      imageData.height
    )

    await setProcessedPreview(entryId, await imageDataToBlob(processedImageData))
  }

  async function refreshClientPreview(nextFaces = faces.value) {
    const entryId = currentEntryId.value

    if (!entryId) {
      return
    }

    await refreshClientPreviewForEntry(entryId, nextFaces)
  }

  async function detectOnClientForEntry(entryId: string) {
    const imageData = await getEntryImageData(entryId)
    const entrySettings = getEntrySettings(entryId)
    const detector = getClientDetector(entrySettings.detectionModel)

    if (!imageData || !detector) {
      throw new Error('La detection des visages dans le navigateur est indisponible.')
    }

    return await detector.detectFaces(
      imageDataToDetectionInput(imageData),
      entrySettings.confidenceThreshold
    )
  }

  async function detectInWorkerForEntry(entryId: string) {
    const imageData = await getEntryImageData(entryId)
    const entrySettings = getEntrySettings(entryId)

    if (!imageData) {
      throw new Error('Aucune image n a ete chargee.')
    }

    try {
      const result = await detectImageData(
        imageData,
        entrySettings.confidenceThreshold,
        getClientModelUrl(entrySettings.detectionModel),
        DETECTION_MODELS[entrySettings.detectionModel].modelType
      )
      return {
        faces: result.faces,
        durationMs: result.durationMs
      }
    } catch {
      return await detectOnClientForEntry(entryId)
    }
  }

  async function detectOnServerForEntry(entryId: string) {
    const entryFile = entryFiles.get(entryId)

    if (!entryFile) {
      throw new Error('Aucune image n a ete chargee.')
    }

    return await $fetch<DetectResponse>('/api/process-image', {
      method: 'POST',
      body: {
        action: 'detect',
        imageBase64: await fileToBase64(entryFile),
        fileName: entryFile.name,
        mimeType: entryFile.type,
        settings: createSettingsSnapshot(entryId)
      }
    })
  }

  async function detectFacesForEntry(entryId: string) {
    const entry = uploadEntries.value.find(candidate => candidate.id === entryId)

    if (!entry || entry.mediaKind !== 'image' || !entryImageData.has(entryId)) {
      return
    }

    const runId = nextRunId(detectRunIds, entryId)
    setEntryStatus(entryId, 'detecting')

    try {
      const result = activeModeForEntry(entryId) === 'server'
        ? await detectOnServerForEntry(entryId)
        : await detectInWorkerForEntry(entryId)

      if (currentRunId(detectRunIds, entryId) !== runId) {
        return
      }

      applyDetection(entryId, result)

      if (activeModeForEntry(entryId) === 'client') {
        await refreshClientPreviewForEntry(entryId)
      }

      setEntryStatus(entryId, 'ready')
    } catch (cause) {
      if (currentRunId(detectRunIds, entryId) !== runId) {
        return
      }

      setEntryStatus(
        entryId,
        'error',
        cause instanceof Error ? cause.message : 'La detection des visages a echoue.'
      )
    }
  }

  function detectFaces() {
    if (currentEntryId.value) {
      return detectFacesForEntry(currentEntryId.value)
    }
  }

  async function processOnClientForEntry(entryId: string) {
    const result = await detectOnClientForEntry(entryId)
    applyDetection(entryId, result)
    await refreshClientPreviewForEntry(entryId)
  }

  async function processInWorkerForEntry(entryId: string) {
    const imageData = await getEntryImageData(entryId)

    if (!imageData) {
      throw new Error('Aucune image n a ete chargee.')
    }

    try {
      const response = await processImageData(
        imageData,
        createSettingsSnapshot(entryId),
        getManualFaces(entryId),
        getClientModelUrl(getEntrySettings(entryId).detectionModel),
        DETECTION_MODELS[getEntrySettings(entryId).detectionModel].modelType
      )

      applyDetection(entryId, {
        faces: response.faces,
        durationMs: response.durationMs
      })
      await setProcessedPreview(entryId, await imageDataToBlob(response.processedImageData))
    } catch {
      await processOnClientForEntry(entryId)
    }
  }

  async function processOnServerForEntry(entryId: string) {
    const entryFile = entryFiles.get(entryId)

    if (!entryFile) {
      throw new Error('Aucune image n a ete chargee.')
    }

    const response = await $fetch.raw('/api/process-image', {
      method: 'POST',
      body: {
        action: 'process',
        imageBase64: await fileToBase64(entryFile),
        fileName: entryFile.name,
        mimeType: entryFile.type,
        settings: createSettingsSnapshot(entryId),
        manualFaces: getManualFaces(entryId)
      },
      responseType: 'blob'
    })

    if (!(response._data instanceof Blob)) {
      throw new Error('Le serveur n a pas renvoye de blob image.')
    }

    await setProcessedPreview(entryId, response._data)
  }

  async function pollServerVideoJob(entryId: string, jobId: string, runId: number) {
    while (runId === currentRunId(videoRunIds, entryId)) {
      const job = await $fetch<VideoJobResponse>(`/api/process-jobs/${jobId}`)

      if (job.status === 'queued') {
        updateEntry(entryId, {
          processingProgress: 0,
          processingMessage: job.queuePosition !== null
            ? `Video en attente dans la file (position ${job.queuePosition}).`
            : 'Video en attente dans la file.',
          estimatedRemainingMs: entryInitialEstimates.get(entryId) ?? null,
          processingQueued: true
        })
      } else if (job.status === 'processing') {
        updateEntry(entryId, { processingQueued: false })
        updateProgress(
          entryId,
          job.progress,
          job.stage || 'Traitement video sur le serveur.',
          job.remainingMs
        )
      }

      if (job.status === 'completed') {
        const downloadUrl = job.downloadUrl || `/api/process-jobs/${jobId}/download`
        const response = await $fetch.raw(downloadUrl, { responseType: 'blob' })

        if (!(response._data instanceof Blob)) {
          throw new Error('Le serveur n a pas renvoye de blob video.')
        }

        return response._data
      }

      if (job.status === 'error') {
        throw new Error(job.error || 'Le traitement de la video a echoue.')
      }

      if (job.status === 'cancelled') {
        throw new Error(job.error || 'Traitement annule.')
      }

      await new Promise(resolve => setTimeout(resolve, 500))
    }

    return null
  }

  async function resumeServerVideoJob(entryId: string, jobId: string) {
    const entry = uploadEntries.value.find(candidate => candidate.id === entryId)

    if (!entry) {
      return
    }

    const runId = nextRunId(videoRunIds, entryId)
    serverVideoJobIds.set(entryId, jobId)
    setEntryStatus(entryId, 'processing')
    updateProgress(entryId, 0, 'Reprise du traitement video sur le serveur.')
    await setInitialVideoEstimate(entryId)

    try {
      const blob = await pollServerVideoJob(entryId, jobId, runId)

      if (runId !== currentRunId(videoRunIds, entryId)) {
        return
      }

      if (!blob) {
        throw new Error('Aucun resultat video n a ete produit.')
      }

      await setProcessedPreview(entryId, blob)
      setEntryStatus(entryId, 'ready')
    } catch (cause) {
      if (runId !== currentRunId(videoRunIds, entryId)) {
        return
      }

      setEntryStatus(
        entryId,
        'error',
        cause instanceof Error ? cause.message : 'Le traitement de la video a echoue.'
      )
    } finally {
      updateEntry(entryId, { serverJobId: null })
      void persistEntry(entryId)

      if (runId === currentRunId(videoRunIds, entryId)) {
        serverVideoJobIds.delete(entryId)
        entryInitialEstimates.delete(entryId)
        entryVideoMetadata.delete(entryId)
        updateEntry(entryId, { processingQueued: false })
        updateProgress(entryId, null)
      }
    }
  }

  async function processVideoForEntry(entryId: string) {
    const entryFile = entryFiles.get(entryId)
    const entry = uploadEntries.value.find(candidate => candidate.id === entryId)

    if (!entryFile || !entry) {
      return
    }

    const runId = nextRunId(videoRunIds, entryId)
    const startedAt = Date.now()
    umTrackEvent('video-job', {
      mode: activeModeForEntry(entryId) === 'client' ? 'browser' : activeModeForEntry(entryId)
    })
    setEntryStatus(entryId, 'processing')
    updateProgress(entryId, 0, 'Preparation du traitement video.')
    await setInitialVideoEstimate(entryId)

    try {
      let blob: Blob | null = null

      if (activeModeForEntry(entryId) === 'server' || activeModeForEntry(entryId) === 'cloud') {
        const body = new FormData()
        body.append('file', entryFile)
        body.append('settings', JSON.stringify(createSettingsSnapshot(entryId)))

        const { jobId } = await $fetch<{ jobId: string }>('/api/process-video', {
          method: 'POST',
          body
        })
        serverVideoJobIds.set(entryId, jobId)
        updateEntry(entryId, { serverJobId: jobId })
        void persistEntry(entryId)
        blob = await pollServerVideoJob(entryId, jobId, runId)
      } else {
        blob = await processVideoInBrowser(entryFile, createSettingsSnapshot(entryId), (progress) => {
          updateProgress(entryId, progress)
        })
      }

      if (runId !== currentRunId(videoRunIds, entryId)) {
        return
      }

      if (!blob) {
        throw new Error('Aucun resultat video n a ete produit.')
      }

      await setProcessedPreview(entryId, blob)
      updateEntry(entryId, {
        lastDurationMs: Date.now() - startedAt
      })
      setEntryStatus(entryId, 'ready')
    } catch (cause) {
      if (runId !== currentRunId(videoRunIds, entryId)) {
        return
      }

      setEntryStatus(
        entryId,
        'error',
        cause instanceof Error ? cause.message : 'Le traitement de la video a echoue.'
      )
    } finally {
      updateEntry(entryId, { serverJobId: null })
      void persistEntry(entryId)

      if (runId === currentRunId(videoRunIds, entryId)) {
        serverVideoJobIds.delete(entryId)
        entryInitialEstimates.delete(entryId)
        entryVideoMetadata.delete(entryId)
        updateEntry(entryId, { processingQueued: false })
        updateProgress(entryId, null)
      }
    }
  }

  async function cancelEntryProcessing(entryId: string) {
    nextRunId(videoRunIds, entryId)
    const jobId = serverVideoJobIds.get(entryId)
    serverVideoJobIds.delete(entryId)
    entryInitialEstimates.delete(entryId)
    entryVideoMetadata.delete(entryId)
    updateEntry(entryId, { processingQueued: false, serverJobId: null })
    updateProgress(entryId, null)
    setEntryStatus(entryId, 'cancelled', 'Traitement annule.')

    if (!jobId) {
      return
    }

    try {
      await $fetch(`/api/process-jobs/${jobId}`, {
        method: 'DELETE'
      })
    } catch {
      // The local cancellation should remain effective even if the job already ended server-side.
    }
  }

  async function processEntry(entryId: string) {
    const entry = uploadEntries.value.find(candidate => candidate.id === entryId)

    if (!entry) {
      return
    }

    if (entry.processedPreviewUrl) {
      revokeUrl(entry.processedPreviewUrl)
      entryProcessedBlobs.delete(entryId)
      updateEntry(entryId, { processedPreviewUrl: '' })
    }

    if (entry.mediaKind === 'video') {
      await processVideoForEntry(entryId)
      return
    }

    umTrackEvent('image-job', {
      mode: activeModeForEntry(entryId) === 'client' ? 'browser' : activeModeForEntry(entryId)
    })
    setEntryStatus(entryId, 'processing')

    try {
      if (activeModeForEntry(entryId) === 'server') {
        await processOnServerForEntry(entryId)
      } else {
        await processInWorkerForEntry(entryId)
      }

      setEntryStatus(entryId, 'ready')
    } catch (cause) {
      setEntryStatus(
        entryId,
        'error',
        cause instanceof Error ? cause.message : 'Le traitement de l image a echoue.'
      )
    }
  }

  async function retryEntry(entryId: string) {
    const entry = uploadEntries.value.find(candidate => candidate.id === entryId)

    if (!entry || entry.status === 'processing') {
      return
    }

    await processEntry(entryId)
  }

  async function loadFile(nextFile: File) {
    file.value = nextFile
    originalImageData.value = null

    const mediaKind = nextFile.type.startsWith('video/') ? 'video' : 'image'
    const entryId = crypto.randomUUID()
    entryFiles.set(entryId, nextFile)
    currentEntryId.value = entryId
    uploadEntries.value.unshift({
      id: entryId,
      createdAt: Date.now(),
      fileName: nextFile.name,
      mediaKind,
      originalPreviewUrl: URL.createObjectURL(nextFile),
      processedPreviewUrl: '',
      faces: [],
      status: mediaKind === 'video' ? 'ready' : 'detecting',
      error: '',
      warning: '',
      lastDurationMs: null,
      processingProgress: null,
      processingMessage: '',
      processingQueued: false,
      estimatedRemainingMs: null,
      serverJobId: null,
      settings: {
        ...settings,
        excludedFaceIds: []
      },
      autoResolvedMode: 'client',
      videoServerOnly: false,
      isUploading: false
    })
    void persistEntry(entryId)

    const inferredMode = await inferProcessingTarget(nextFile.size)
    const resolvedMode = mediaKind === 'video' && (nextFile.size > 12 * 1024 * 1024 || inferredMode !== 'client')
      ? 'server'
      : inferredMode

    if (currentEntry.value) {
      currentEntry.value.autoResolvedMode = resolvedMode
    }

    safariVideoModalOpen.value = mediaKind === 'video' && isSafari

    if (mediaKind === 'video') {
      console.debug('[solifloute] chargement video', { entryId, fileName: nextFile.name, size: nextFile.size, resolvedMode })
      await checkVideoCompatibility(entryId, nextFile)
      void setInitialVideoEstimate(entryId)
      void persistEntry(entryId)
      return entryId
    }

    const imageData = await fileToImageData(nextFile)
    entryImageData.set(entryId, imageData)
    originalImageData.value = imageData
    await detectFacesForEntry(entryId)
    void persistEntry(entryId)

    return entryId
  }

  async function checkVideoCompatibility(entryId: string, nextFile: File) {
    const entry = uploadEntries.value.find(candidate => candidate.id === entryId)

    if (!import.meta.client || !entry) {
      console.debug('[solifloute] compat: entree introuvable', { entryId })
      return
    }

    const metadata = await readVideoMetadata(nextFile).catch((cause) => {
      console.debug('[solifloute] compat: echec lecture metadonnees video', { entryId, cause: cause instanceof Error ? cause.message : String(cause) })
      return null
    })

    if (!metadata) {
      return
    }

    if (metadata.width > 0 && metadata.height > 0 && metadata.duration > 0) {
      entryVideoMetadata.set(entryId, metadata)
    }

    console.debug('[solifloute] compat: metadonnees lues', { entryId, metadata })

    void setInitialVideoEstimate(entryId)

    const browserCannotDecode = metadata.width <= 0 || metadata.height <= 0

    if (browserCannotDecode) {
      console.info('[solifloute] le navigateur ne peut pas decoder la video', {
        fileName: nextFile.name,
        width: metadata.width,
        height: metadata.height,
        duration: metadata.duration
      })
      entry.videoServerOnly = true
      entry.autoResolvedMode = 'server'

      return
    }

    const tooLong = metadata.duration > MAX_BROWSER_VIDEO_DURATION_SECONDS
    const tooBig = metadata.width * metadata.height > MAX_BROWSER_VIDEO_PIXELS

    if (tooLong || tooBig) {
      entry.videoServerOnly = true
      entry.autoResolvedMode = 'server'

      if (entry.settings.processingMode !== 'cloud') {
        entry.settings.processingMode = 'server'
      }

      entry.warning = 'Cette video depasse les limites du traitement navigateur. Les modes serveur et Cloud restent disponibles.'
    }
  }

  function toggleExcludedFace(faceId: string) {
    const entry = currentEntry.value

    if (!entry) {
      return
    }

    entry.settings.excludedFaceIds = entry.settings.excludedFaceIds.includes(faceId)
      ? entry.settings.excludedFaceIds.filter(id => id !== faceId)
      : [...entry.settings.excludedFaceIds, faceId]
  }

  async function addManualFace(bounds: Pick<Face, 'x' | 'y' | 'width' | 'height'>) {
    if (!currentEntry.value || mediaKind.value !== 'image') {
      return
    }

    updateCurrentEntry({
      faces: [
        ...currentEntry.value.faces,
        {
          id: `manual:${crypto.randomUUID()}`,
          confidence: 1,
          ...bounds
        }
      ]
    })

    if (activeModeForEntry(currentEntryId.value ?? '') === 'client' && originalImageData.value) {
      try {
        await refreshClientPreview()
      } catch (cause) {
        setStatus(
          'error',
          cause instanceof Error ? cause.message : 'La mise a jour de l apercu a echoue.'
        )
      }
    }
  }

  function closeSafariVideoModal() {
    safariVideoModalOpen.value = false
  }

  async function restoreEntries() {
    if (!import.meta.client || uploadEntries.value.length > 0) {
      return
    }

    let stored: StoredEntry[]

    try {
      stored = await loadAllEntries()
      console.info('[solifloute] restauration: ' + stored.length + ' entree(s) trouvee(s)')
    } catch (cause) {
      console.error('[solifloute] echec de lecture de la base', cause)
      return
    }

    if (stored.length === 0) {
      return
    }

    stored.sort((left, right) => right.createdAt - left.createdAt)

    const restored: UploadEntry[] = []
    const pendingResumes: Array<{ entryId: string, jobId: string }> = []

    for (const item of stored) {
      const entryFile = new File([item.originalBlob], item.fileName, {
        type: item.originalBlob.type || undefined
      })
      entryFiles.set(item.id, entryFile)

      if (item.processedBlob) {
        entryProcessedBlobs.set(item.id, item.processedBlob)
      }

      const serverJobId = item.serverJobId ?? null
      const inFlight = (
        item.status === 'processing'
        || item.status === 'queued'
        || item.status === 'detecting'
      )
      const status: EditorStatus = inFlight && !item.processedBlob
        ? (serverJobId ? 'processing' : 'error')
        : (item.status as EditorStatus)

      if (serverJobId && !item.processedBlob && (item.status === 'processing' || item.status === 'queued')) {
        pendingResumes.push({ entryId: item.id, jobId: serverJobId })
      }

      const inferredMode = await inferProcessingTarget(item.originalBlob.size)
      const resolvedMode = item.mediaKind === 'video' && (item.originalBlob.size > 12 * 1024 * 1024 || inferredMode !== 'client')
        ? 'server'
        : inferredMode

      restored.push({
        id: item.id,
        createdAt: item.createdAt,
        fileName: item.fileName,
        mediaKind: item.mediaKind,
        originalPreviewUrl: URL.createObjectURL(entryFile),
        processedPreviewUrl: item.processedBlob ? URL.createObjectURL(item.processedBlob) : '',
        faces: item.faces,
        status,
        error: inFlight && status === 'error' && !serverJobId
          ? 'Le traitement a ete interrompu. Relancez-le.'
          : item.error,
        warning: item.warning,
        lastDurationMs: item.lastDurationMs,
        processingProgress: null,
        processingMessage: '',
        processingQueued: false,
        estimatedRemainingMs: null,
        serverJobId,
        settings: sanitizeEntrySettings(item.settings),
        autoResolvedMode: resolvedMode,
        videoServerOnly: false,
        isUploading: false
      })
    }

    uploadEntries.value = restored
    const mostRecent = restored[0]

    if (mostRecent) {
      currentEntryId.value = mostRecent.id
      file.value = entryFiles.get(mostRecent.id) ?? null

      if (mostRecent.mediaKind === 'image') {
        originalImageData.value = await getEntryImageData(mostRecent.id)
      }
    }

    for (const pending of pendingResumes) {
      void resumeServerVideoJob(pending.entryId, pending.jobId)
    }

    for (const item of restored) {
      if (item.mediaKind === 'video') {
        const entryFile = entryFiles.get(item.id)
        void checkVideoCompatibility(item.id, entryFile ?? new File([], item.fileName))
      }
    }
  }

  if (import.meta.client) {
    onMounted(() => {
      void restoreEntries()
    })

    const debugApi = {
      restore: () => restoreEntries(),
      entries: () => uploadEntries.value,
      db: () => loadAllEntries()
    }

    Object.assign(window, { __solifloute: debugApi })
  }

  let watchedSettingsEntryId: string | null = null
  let watchedSettingsSnapshot: string | null = null

  watch(
    () => currentEntry.value?.settings,
    async (nextSettings) => {
      if (!import.meta.client || !nextSettings || !currentEntry.value) {
        return
      }

      const entryId = currentEntry.value.id
      const snapshot = JSON.stringify({
        confidenceThreshold: nextSettings.confidenceThreshold,
        blurIntensity: nextSettings.blurIntensity,
        processingMode: nextSettings.processingMode,
        excludedFaceIds: nextSettings.excludedFaceIds,
        detectionModel: nextSettings.detectionModel
      })

      // First observation of an entry's settings (entry switch or load) is
      // handled by loadFile/restoreEntries: only react to subsequent edits.
      if (watchedSettingsEntryId !== entryId || watchedSettingsSnapshot === snapshot) {
        watchedSettingsEntryId = entryId
        watchedSettingsSnapshot = snapshot
        return
      }

      const previous = JSON.parse(watchedSettingsSnapshot!) as EditorSettings
      watchedSettingsEntryId = entryId
      watchedSettingsSnapshot = snapshot

      const modelChanged = nextSettings.detectionModel !== previous.detectionModel
      const sensitivityChanged = nextSettings.confidenceThreshold !== previous.confidenceThreshold
      const previewChanged = modelChanged
        || sensitivityChanged
        || nextSettings.blurIntensity !== previous.blurIntensity
        || nextSettings.excludedFaceIds.join('|') !== previous.excludedFaceIds.join('|')
        || nextSettings.processingMode !== previous.processingMode

      if (modelChanged) {
        clientDetector.value = useFaceDetector(
          getClientModelUrl(nextSettings.detectionModel),
          DETECTION_MODELS[nextSettings.detectionModel].modelType
        )
      }

      const entry = currentEntry.value

      if (entry.mediaKind === 'video') {
        void setInitialVideoEstimate(entryId)

        if (isSafari && nextSettings.processingMode !== 'server') {
          safariVideoModalOpen.value = true
        }
      } else if (file.value && originalImageData.value && status.value !== 'detecting') {
        if (sensitivityChanged || modelChanged) {
          await detectFaces()
        } else if (activeModeForEntry(entryId) === 'client' && previewChanged) {
          await refreshClientPreview()
        }
      }

      persistSettings(nextSettings)
      void persistEntry(entryId)
    },
    { deep: true }
  )

  return {
    file,
    uploadEntries,
    currentEntryId,
    mediaKind,
    faces,
    status,
    error,
    settings,
    originalPreviewUrl,
    processedPreviewUrl,
    processingProgress,
    estimatedRemainingMs,
    processingMessage,
    lastDurationMs,
    safariVideoModalOpen,
    activeModeForEntry,
    isEntryServerOnly,
    loadFile,
    detectFaces,
    processEntry,
    retryEntry,
    cancelEntryProcessing,
    toggleExcludedFace,
    addManualFace,
    closeSafariVideoModal,
    entryInitialEstimates
  }
}
