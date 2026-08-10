import { join } from 'node:path'
import type { EditorSettings } from '~~/shared/types/faces'
import { processVideoFromPath } from './process-video'
import {
  cancelJob,
  completeJob,
  enqueueJob,
  failJob,
  getJob,
  listNonTerminalJobs,
  updateJobProgress,
  type ProcessingJob
} from './video-jobs'

function defaultStoredSettings(): EditorSettings {
  return {
    confidenceThreshold: 0.2,
    blurIntensity: 0.5,
    processingMode: 'auto',
    excludedFaceIds: [],
    detectionModel: 'fast'
  }
}

function parseStoredSettings(raw: string): EditorSettings {
  try {
    const parsed = JSON.parse(raw) as EditorSettings
    const fallback = defaultStoredSettings()

    return {
      confidenceThreshold: typeof parsed.confidenceThreshold === 'number'
        ? parsed.confidenceThreshold
        : fallback.confidenceThreshold,
      blurIntensity: typeof parsed.blurIntensity === 'number'
        ? parsed.blurIntensity
        : fallback.blurIntensity,
      processingMode: parsed.processingMode === 'client' || parsed.processingMode === 'server'
        ? parsed.processingMode
        : fallback.processingMode,
      excludedFaceIds: Array.isArray(parsed.excludedFaceIds)
        ? parsed.excludedFaceIds.filter(id => typeof id === 'string')
        : [],
      detectionModel: parsed.detectionModel === 'fast' || parsed.detectionModel === 'advanced'
        ? parsed.detectionModel
        : fallback.detectionModel
    }
  } catch {
    return defaultStoredSettings()
  }
}

function buildVideoTask(job: ProcessingJob) {
  const settings = parseStoredSettings(job.settingsJson)
  const workspace = {
    tempRoot: job.tempRoot,
    inputPath: job.inputPath,
    outputPath: join(job.tempRoot, 'output.mp4'),
    framesDir: join(job.tempRoot, 'frames')
  }

  return async (signal: AbortSignal) => {
    const startedAt = Date.now()

    try {
      const { outputPath, tempRoot } = await processVideoFromPath(
        workspace,
        settings,
        (progress, remainingMs) => {
          updateJobProgress(job.id, progress, remainingMs)
        },
        signal
      )

      completeJob(job.id, {
        outputPath,
        tempRoot,
        durationMs: Date.now() - startedAt,
        mimeType: 'video/mp4'
      })
    } catch (error) {
      if (signal.aborted) {
        await cancelJob(job.id)
        return
      }

      await failJob(job.id, error instanceof Error ? error.message : 'Le traitement de la video a echoue.')
    }
  }
}

export function enqueueVideoJob(jobId: string) {
  const job = getJob(jobId)

  if (job) {
    enqueueJob(job.id, buildVideoTask(job))
  }
}

export function resumePendingJobs() {
  for (const job of listNonTerminalJobs()) {
    enqueueJob(job.id, buildVideoTask(job))
  }
}
