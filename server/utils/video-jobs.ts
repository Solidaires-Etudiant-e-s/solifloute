import { randomUUID } from 'node:crypto'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { rm } from 'node:fs/promises'
import {
  allJobRows,
  deleteJobRow,
  upsertJobRow,
  type JobRow
} from './job-db'

export type ProcessingJobStatus = 'queued' | 'processing' | 'completed' | 'error' | 'cancelled'

export interface ProcessingJob {
  id: string
  ownerId: string
  kind: 'video-process'
  fileName: string
  mimeType: string
  status: ProcessingJobStatus
  progress: number
  remainingMs: number | null
  error: string
  createdAt: number
  updatedAt: number
  enqueuedAt: number
  durationMs: number | null
  outputPath: string
  tempRoot: string
  inputPath: string
  settingsJson: string
  resumeCount: number
}

const jobs = new Map<string, ProcessingJob>()
const queuedTasks = new Map<string, (signal: AbortSignal) => Promise<void>>()
const activeControllers = new Map<string, AbortController>()
const queue: string[] = []
const JOB_TTL_MS = 24 * 60 * 60 * 1000
export const JOB_RESUME_MAX = Math.max(1, Number(process.env.PROCESS_JOB_RESUME_MAX || 2))
const JOB_TIMEOUT_MS = Number(process.env.PROCESS_JOB_TIMEOUT_MS || 2 * 60 * 60 * 1000)
const JOB_QUEUE_CONCURRENCY = Math.max(1, Number(process.env.PROCESS_JOB_CONCURRENCY || 1))
const JOB_QUEUE_LIMIT_PER_OWNER = Math.max(1, Number(process.env.PROCESS_JOB_LIMIT_PER_OWNER || 5))
const PROGRESS_PERSIST_DELTA = 0.01
const PROGRESS_PERSIST_INTERVAL_MS = 2000
let activeTasks = 0
let lastProgressPersistAt = 0
let lastProgressPersistValue = -1

function isTerminalStatus(status: ProcessingJobStatus) {
  return status === 'completed' || status === 'error' || status === 'cancelled'
}

function toJobRow(job: ProcessingJob): JobRow {
  return {
    id: job.id,
    owner_id: job.ownerId,
    kind: job.kind,
    file_name: job.fileName,
    mime_type: job.mimeType,
    status: job.status,
    progress: job.progress,
    error: job.error,
    created_at: job.createdAt,
    updated_at: job.updatedAt,
    duration_ms: job.durationMs,
    output_path: job.outputPath,
    temp_root: job.tempRoot,
    input_path: job.inputPath,
    settings_json: job.settingsJson,
    resume_count: job.resumeCount
  }
}

function fromJobRow(row: JobRow): ProcessingJob {
  return {
    id: row.id,
    ownerId: row.owner_id,
    kind: 'video-process',
    fileName: row.file_name,
    mimeType: row.mime_type,
    status: row.status as ProcessingJobStatus,
    progress: row.progress,
    remainingMs: null,
    error: row.error,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    enqueuedAt: row.created_at,
    durationMs: row.duration_ms,
    outputPath: row.output_path,
    tempRoot: row.temp_root,
    inputPath: row.input_path,
    settingsJson: row.settings_json,
    resumeCount: row.resume_count
  }
}

function persistJob(job: ProcessingJob) {
  job.updatedAt = Date.now()
  upsertJobRow(toJobRow(job))
}

function scheduleCleanup(jobId: string, delayMs = JOB_TTL_MS) {
  const timer = setTimeout(() => {
    void cleanupJob(jobId)
  }, delayMs)

  timer.unref?.()
}

function loadJobsFromStore() {
  try {
    for (const row of allJobRows()) {
      const job = fromJobRow(row)

      if (job.status === 'processing') {
        job.status = 'queued'
        job.progress = 0
        job.error = ''
        job.resumeCount += 1
        persistJob(job)
      }

      jobs.set(job.id, job)
      scheduleCleanup(job.id, Math.max(0, (job.createdAt + JOB_TTL_MS) - Date.now()))
    }
  } catch {
    // Ignore a corrupt store and allow new jobs to proceed.
  }
}

async function runQueue() {
  while (activeTasks < JOB_QUEUE_CONCURRENCY && queue.length > 0) {
    const jobId = queue[0]
    const job = jobId ? jobs.get(jobId) : undefined
    const task = jobId ? queuedTasks.get(jobId) : undefined

    if (!jobId || !job || !task) {
      queue.shift()
      continue
    }

    queue.shift()
    queuedTasks.delete(jobId)
    const controller = new AbortController()
    activeControllers.set(jobId, controller)
    activeTasks += 1
    job.status = 'processing'
    persistJob(job)
    const jobTimeout = JOB_TIMEOUT_MS > 0
      ? setTimeout(() => controller.abort(), JOB_TIMEOUT_MS)
      : null

    void task(controller.signal)
      .catch(async (cause) => {
        if (controller.signal.aborted) {
          await cancelJob(jobId)
          return
        }

        await failJob(jobId, cause instanceof Error ? cause.message : 'Le traitement a echoue.')
      })
      .finally(() => {
        if (jobTimeout) {
          clearTimeout(jobTimeout)
        }
        activeControllers.delete(jobId)
        activeTasks = Math.max(0, activeTasks - 1)
        void runQueue()
      })
  }
}

export function createJob(input: {
  ownerId: string
  fileName: string
  mimeType?: string
  inputPath: string
  settingsJson: string
  tempRoot: string
}) {
  const activeOwnerJobs = [...jobs.values()].filter(job => (
    job.ownerId === input.ownerId
    && (job.status === 'queued' || job.status === 'processing')
  ))

  if (activeOwnerJobs.length >= JOB_QUEUE_LIMIT_PER_OWNER) {
    throw new Error('Trop de traitements video sont deja en attente pour ce client.')
  }

  const id = randomUUID()
  const now = Date.now()
  const job: ProcessingJob = {
    id,
    ownerId: input.ownerId,
    kind: 'video-process',
    fileName: input.fileName,
    mimeType: input.mimeType || 'video/mp4',
    status: 'queued',
    progress: 0,
    remainingMs: null,
    error: '',
    createdAt: now,
    updatedAt: now,
    enqueuedAt: now,
    durationMs: null,
    outputPath: '',
    tempRoot: input.tempRoot,
    inputPath: input.inputPath,
    settingsJson: input.settingsJson,
    resumeCount: 0
  }

  jobs.set(id, job)
  upsertJobRow(toJobRow(job))
  scheduleCleanup(id)
  return id
}

export function enqueueJob(jobId: string, task: (signal: AbortSignal) => Promise<void>) {
  const job = jobs.get(jobId)

  if (!job) {
    throw new Error('Tache introuvable.')
  }

  if (job.status === 'processing' || isTerminalStatus(job.status)) {
    return
  }

  const alreadyQueued = queue.includes(jobId)

  queuedTasks.set(jobId, task)

  if (!alreadyQueued) {
    job.status = 'queued'
    job.enqueuedAt = job.enqueuedAt || Date.now()

    const insertAt = queue.findIndex(otherId => (
      (jobs.get(otherId)?.enqueuedAt ?? Infinity) > job.enqueuedAt
    ))

    if (insertAt < 0) {
      queue.push(jobId)
    } else {
      queue.splice(insertAt, 0, jobId)
    }
  }

  void runQueue()
}

export function getJob(jobId: string) {
  return jobs.get(jobId) || null
}

export function getJobForOwner(jobId: string, ownerId: string) {
  const job = jobs.get(jobId)

  return job && job.ownerId === ownerId ? job : null
}

export function listJobsForOwner(ownerId: string) {
  return [...jobs.values()]
    .filter(job => job.ownerId === ownerId)
    .sort((left, right) => right.createdAt - left.createdAt)
}

export function listNonTerminalJobs() {
  return [...jobs.values()].filter(job => job.status === 'queued' || job.status === 'processing')
}

export function updateJobProgress(jobId: string, progress: number, remainingMs: number | null = null) {
  const job = jobs.get(jobId)

  if (!job) {
    return
  }

  const nextProgress = Math.max(0, Math.min(1, progress))
  job.progress = nextProgress
  job.remainingMs = remainingMs
  const now = Date.now()

  if (
    nextProgress >= 1
    || nextProgress - lastProgressPersistValue >= PROGRESS_PERSIST_DELTA
    || now - lastProgressPersistAt >= PROGRESS_PERSIST_INTERVAL_MS
  ) {
    lastProgressPersistValue = nextProgress
    lastProgressPersistAt = now
    persistJob(job)
  }
}

export function completeJob(
  jobId: string,
  payload: {
    outputPath: string
    tempRoot: string
    durationMs: number
    mimeType?: string
  }
) {
  const job = jobs.get(jobId)

  if (!job) {
    return
  }

  if (job.status === 'cancelled') {
    void rm(payload.tempRoot, { recursive: true, force: true })
    return
  }

  job.status = 'completed'
  job.progress = 1
  job.outputPath = payload.outputPath
  job.tempRoot = payload.tempRoot
  job.durationMs = payload.durationMs
  job.mimeType = payload.mimeType || job.mimeType
  persistJob(job)
}

export async function failJob(jobId: string, message: string) {
  const job = jobs.get(jobId)

  if (!job) {
    return
  }

  if (job.status === 'cancelled') {
    return
  }

  job.status = 'error'
  job.progress = 0
  job.error = message
  job.outputPath = ''
  persistJob(job)
}

export async function cancelJob(jobId: string) {
  const job = jobs.get(jobId)

  if (!job || isTerminalStatus(job.status)) {
    return job || null
  }

  const controller = activeControllers.get(jobId)
  controller?.abort()
  queuedTasks.delete(jobId)

  const queueIndex = queue.indexOf(jobId)

  if (queueIndex >= 0) {
    queue.splice(queueIndex, 1)
  }

  job.status = 'cancelled'
  job.progress = 0
  job.error = 'Traitement annule.'
  job.outputPath = ''
  persistJob(job)

  return job
}

export async function removeJob(jobId: string) {
  const job = jobs.get(jobId)

  if (!job) {
    return null
  }

  activeControllers.get(jobId)?.abort()
  queuedTasks.delete(jobId)

  const queueIndex = queue.indexOf(jobId)

  if (queueIndex >= 0) {
    queue.splice(queueIndex, 1)
  }

  activeControllers.delete(jobId)
  jobs.delete(jobId)
  deleteJobRow(jobId)

  if (job.tempRoot) {
    await rm(job.tempRoot, { recursive: true, force: true })
  }

  return job
}

export async function removeJobForOwner(jobId: string, ownerId: string) {
  const job = getJobForOwner(jobId, ownerId)

  if (!job) {
    return null
  }

  return await removeJob(jobId)
}

export function getJobOutputStream(jobId: string) {
  const job = jobs.get(jobId)

  if (!job || job.status !== 'completed' || !job.outputPath || !existsSync(job.outputPath)) {
    return null
  }

  const stats = statSync(job.outputPath)

  return {
    stream: createReadStream(job.outputPath),
    size: stats.size,
    mimeType: job.mimeType || 'video/mp4',
    fileName: job.fileName || 'visages-floutes.mp4'
  }
}

export function getQueuePosition(jobId: string, ownerId: string | null = null) {
  const job = jobs.get(jobId)

  if (!job || job.status !== 'queued') {
    return null
  }

  const index = queue.indexOf(jobId)

  if (index < 0) {
    return null
  }

  if (!ownerId) {
    return index + 1
  }

  let position = 1

  for (let i = 0; i < index; i += 1) {
    const otherJobId = queue[i]

    if (!otherJobId) {
      continue
    }

    const queuedJob = jobs.get(otherJobId)

    if (queuedJob && queuedJob.ownerId === ownerId) {
      position += 1
    }
  }

  return position
}

export async function cleanupJob(jobId: string) {
  const job = jobs.get(jobId)

  if (!job) {
    return
  }

  jobs.delete(jobId)
  queuedTasks.delete(jobId)
  activeControllers.get(jobId)?.abort()
  activeControllers.delete(jobId)
  deleteJobRow(jobId)

  if (job.tempRoot) {
    await rm(job.tempRoot, { recursive: true, force: true })
  }
}

loadJobsFromStore()
