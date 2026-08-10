import type { EditorSettings, Face, ProcessingMode } from '~~/shared/types/faces'
import { applyBlurEffects } from '~~/shared/utils/imageProcessing'
import { DETECTION_MODELS, getClientModelUrl } from '~~/shared/utils/detectionModels'
import { useFaceDetector } from '~~/shared/utils/useFaceDetector'
import { fileToBase64, fileToImageData, imageDataToBlob, imageDataToDetectionInput } from '~/utils/image-io'
import { detectImageData, processImageData } from '~/utils/detect-worker'
import { inferProcessingTarget } from '~/utils/machine-profile'
import { loadAllEntries, saveEntry, type StoredEntry } from '~/utils/entry-store'
import {
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
      processingMode: parsed.processingMode === 'client' || parsed.processingMode === 'server'
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

export function useImageEditor() {
  const entryFiles = new Map<string, File>()
  const entryImageData = new Map<string, ImageData>()
  const entryProcessedBlobs = new Map<string, Blob>()
  const videoRunIds = new Map<string, number>()
  const detectRunIds = new Map<string, number>()
  const serverVideoJobIds = new Map<string, string>()

  const file = shallowRef<File | null>(null)
  const originalImageData = shallowRef<ImageData | null>(null)
  const uploadEntries = ref<UploadEntry[]>([])
  const currentEntryId = ref<string | null>(null)
  const settings = reactive<EditorSettings>({ ...DEFAULT_SETTINGS, ...loadPersistedSettings() })
  const autoResolvedMode = ref<Exclude<ProcessingMode, 'auto'>>('client')
  const videoServerOnly = ref(false)
  const safariVideoModalOpen = ref(false)
  const isSafari = isSafariBrowser()
  const clientDetector = shallowRef<ReturnType<typeof useFaceDetector> | null>(null)

  function getClientDetector() {
    if (!import.meta.client) {
      return null
    }

    if (!clientDetector.value) {
      clientDetector.value = useFaceDetector(
        getClientModelUrl(settings.detectionModel),
        DETECTION_MODELS[settings.detectionModel].modelType
      )
    }

    return clientDetector.value
  }

  const currentEntry = computed(() => (
    currentEntryId.value
      ? uploadEntries.value.find(entry => entry.id === currentEntryId.value) || null
      : null
  ))

  const isSafariVideoForcedToServer = computed(() => (
    isSafari
    && mediaKind.value === 'video'
  ))

  const serverOnly = computed(() => (
    isSafariVideoForcedToServer.value || videoServerOnly.value
  ))

  const activeMode = computed(() => {
    if (serverOnly.value) {
      return 'server'
    }

    return settings.processingMode === 'auto' ? autoResolvedMode.value : settings.processingMode
  })

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

  function createSettingsSnapshot(): EditorSettings {
    return {
      confidenceThreshold: settings.confidenceThreshold,
      blurIntensity: settings.blurIntensity,
      processingMode: settings.processingMode,
      excludedFaceIds: [...settings.excludedFaceIds],
      detectionModel: settings.detectionModel
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
    const processedImageData = new ImageData(
      applyBlurEffects(
        imageDataToDetectionInput(imageData),
        facesToBlur,
        settings.excludedFaceIds,
        settings.blurIntensity
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
    const detector = getClientDetector()

    if (!imageData || !detector) {
      throw new Error('La detection des visages dans le navigateur est indisponible.')
    }

    return await detector.detectFaces(
      imageDataToDetectionInput(imageData),
      settings.confidenceThreshold
    )
  }

  async function detectInWorkerForEntry(entryId: string) {
    const imageData = await getEntryImageData(entryId)

    if (!imageData) {
      throw new Error('Aucune image n a ete chargee.')
    }

    try {
      const result = await detectImageData(
        imageData,
        settings.confidenceThreshold,
        getClientModelUrl(settings.detectionModel),
        DETECTION_MODELS[settings.detectionModel].modelType
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
        settings: createSettingsSnapshot()
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
      const result = activeMode.value === 'server'
        ? await detectOnServerForEntry(entryId)
        : await detectInWorkerForEntry(entryId)

      if (currentRunId(detectRunIds, entryId) !== runId) {
        return
      }

      applyDetection(entryId, result)

      if (activeMode.value === 'client') {
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
        createSettingsSnapshot(),
        getManualFaces(entryId),
        getClientModelUrl(settings.detectionModel),
        DETECTION_MODELS[settings.detectionModel].modelType
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
        settings: createSettingsSnapshot(),
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
          estimatedRemainingMs: null,
          processingQueued: true
        })
      } else if (job.status === 'processing') {
        updateEntry(entryId, { processingQueued: false })
        updateProgress(entryId, job.progress, 'Traitement video sur le serveur.', job.remainingMs)
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
    setEntryStatus(entryId, 'processing')
    updateProgress(entryId, 0, 'Preparation du traitement video.')

    try {
      let blob: Blob | null = null

      if (activeMode.value === 'server') {
        const body = new FormData()
        body.append('file', entryFile)
        body.append('settings', JSON.stringify(createSettingsSnapshot()))

        const { jobId } = await $fetch<{ jobId: string }>('/api/process-video', {
          method: 'POST',
          body
        })
        serverVideoJobIds.set(entryId, jobId)
        updateEntry(entryId, { serverJobId: jobId })
        void persistEntry(entryId)
        blob = await pollServerVideoJob(entryId, jobId, runId)
      } else {
        blob = await processVideoInBrowser(entryFile, createSettingsSnapshot(), (progress) => {
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
        updateEntry(entryId, { processingQueued: false })
        updateProgress(entryId, null)
      }
    }
  }

  async function cancelEntryProcessing(entryId: string) {
    nextRunId(videoRunIds, entryId)
    const jobId = serverVideoJobIds.get(entryId)
    serverVideoJobIds.delete(entryId)
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

    setEntryStatus(entryId, 'processing')

    try {
      if (activeMode.value === 'server') {
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
    settings.excludedFaceIds = []
    videoServerOnly.value = false

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
      serverJobId: null
    })
    void persistEntry(entryId)

    const inferredMode = await inferProcessingTarget(nextFile.size)
    autoResolvedMode.value = mediaKind === 'video' && (nextFile.size > 12 * 1024 * 1024 || inferredMode !== 'client')
      ? 'server'
      : inferredMode
    safariVideoModalOpen.value = mediaKind === 'video' && isSafari

    if (mediaKind === 'video') {
      await checkVideoCompatibility(nextFile)
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

  async function checkVideoCompatibility(nextFile: File) {
    if (!import.meta.client || !currentEntry.value) {
      return
    }

    const metadata = await readVideoMetadata(nextFile).catch(() => null)

    if (!metadata) {
      return
    }

    const browserCannotDecode = metadata.width <= 0 || metadata.height <= 0

    if (browserCannotDecode) {
      console.info('[solifloute] le navigateur ne peut pas decoder la video', {
        fileName: nextFile.name,
        width: metadata.width,
        height: metadata.height,
        duration: metadata.duration
      })
      videoServerOnly.value = true
      autoResolvedMode.value = 'server'

      return
    }

    const tooLong = metadata.duration > MAX_BROWSER_VIDEO_DURATION_SECONDS
    const tooBig = metadata.width * metadata.height > MAX_BROWSER_VIDEO_PIXELS

    if (tooLong || tooBig) {
      videoServerOnly.value = true
      autoResolvedMode.value = 'server'

      updateCurrentEntry({
        warning: 'Cette video depasse les limites du traitement navigateur. Le mode serveur est force.'
      })
    }
  }

  function toggleExcludedFace(faceId: string) {
    settings.excludedFaceIds = settings.excludedFaceIds.includes(faceId)
      ? settings.excludedFaceIds.filter(id => id !== faceId)
      : [...settings.excludedFaceIds, faceId]
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

    if (activeMode.value === 'client' && originalImageData.value) {
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
        serverJobId
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

  watch(() => settings.confidenceThreshold, async () => {
    if (mediaKind.value === 'image' && file.value && originalImageData.value) {
      await detectFaces()
    }
  })

  watch(() => settings.detectionModel, async () => {
    if (!import.meta.client) {
      return
    }

    clientDetector.value = useFaceDetector(
      getClientModelUrl(settings.detectionModel),
      DETECTION_MODELS[settings.detectionModel].modelType
    )

    if (mediaKind.value === 'image' && file.value && originalImageData.value) {
      await detectFaces()
    }
  })

  watch(
    () => [settings.processingMode, settings.blurIntensity, settings.excludedFaceIds.join('|')],
    async () => {
      if (
        mediaKind.value !== 'image'
        || !file.value
        || !originalImageData.value
        || activeMode.value !== 'client'
        || status.value === 'detecting'
      ) {
        return
      }

      try {
        await refreshClientPreview()
      } catch (cause) {
        setStatus(
          'error',
          cause instanceof Error ? cause.message : 'La mise a jour de l apercu a echoue.'
        )
      }
    }
  )

  watch(
    () => settings.processingMode,
    () => {
      if (isSafariVideoForcedToServer.value && settings.processingMode !== 'server') {
        safariVideoModalOpen.value = true
      }
    }
  )

  watch(serverOnly, (forced) => {
    if (forced) {
      settings.processingMode = 'server'
    }
  })

  watch(
    () => ({
      confidenceThreshold: settings.confidenceThreshold,
      blurIntensity: settings.blurIntensity,
      processingMode: settings.processingMode
    }),
    (value) => {
      persistSettings(value)
    }
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
    activeMode,
    serverOnly,
    originalPreviewUrl,
    processedPreviewUrl,
    processingProgress,
    estimatedRemainingMs,
    processingMessage,
    lastDurationMs,
    safariVideoModalOpen,
    isSafariVideoForcedToServer,
    loadFile,
    detectFaces,
    processEntry,
    retryEntry,
    cancelEntryProcessing,
    toggleExcludedFace,
    addManualFace,
    closeSafariVideoModal
  }
}
