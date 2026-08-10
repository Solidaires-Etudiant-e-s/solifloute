import { getRouterParam } from 'h3'
import { getOrCreateClientId } from '../../utils/job-client'
import { serializeJob } from '../../utils/job-response'
import { removeJobForOwner } from '../../utils/video-jobs'

export default defineEventHandler(async (event) => {
  const ownerId = getOrCreateClientId(event)
  const jobId = getRouterParam(event, 'jobId')

  if (!jobId) {
    throw createError({
      statusCode: 400,
      statusMessage: 'Identifiant de tache manquant.'
    })
  }

  const job = await removeJobForOwner(jobId, ownerId)

  if (!job) {
    throw createError({
      statusCode: 404,
      statusMessage: 'Tache introuvable.'
    })
  }

  return serializeJob(job)
})
