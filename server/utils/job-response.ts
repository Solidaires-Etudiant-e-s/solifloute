import type { ProcessingJob } from './video-jobs'
import { getQueuePosition } from './video-jobs'

export function serializeJob(job: ProcessingJob) {
  return {
    id: job.id,
    kind: job.kind,
    status: job.status,
    fileName: job.fileName,
    progress: job.progress,
    error: job.error,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
    durationMs: job.durationMs,
    remainingMs: job.remainingMs,
    queuePosition: getQueuePosition(job.id, job.ownerId),
    downloadUrl: job.outputPath ? `/api/process-jobs/${job.id}/download` : null
  }
}
