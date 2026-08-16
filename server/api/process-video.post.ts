import { rm, writeFile } from 'node:fs/promises'
import type { EditorSettings } from '~~/shared/types/faces'
import { getOrCreateClientId } from '../utils/job-client'
import { createVideoJobWorkspace, getVideoFrameCount } from '../utils/process-video'
import { createJob } from '../utils/video-jobs'
import { enqueueVideoJob } from '../utils/video-job-runner'

const MAX_VIDEO_UPLOAD_BYTES = Number(process.env.PROCESS_MAX_VIDEO_UPLOAD_BYTES || 1024 * 1024 * 1024)

interface MultipartField {
  name?: string
  data?: Buffer
  filename?: string
}

function readSettingsField(value?: Buffer) {
  if (!value) {
    throw createError({
      statusCode: 400,
      statusMessage: 'Les reglages video sont requis.'
    })
  }

  try {
    return JSON.parse(value.toString('utf8')) as EditorSettings
  } catch {
    throw createError({
      statusCode: 400,
      statusMessage: 'Les reglages video sont invalides.'
    })
  }
}

export default defineEventHandler(async (event) => {
  const ownerId = getOrCreateClientId(event)

  const contentLength = Number(getRequestHeader(event, 'content-length') || 0)

  if (contentLength <= 0) {
    throw createError({
      statusCode: 411,
      statusMessage: 'La longueur du contenu est requise.'
    })
  }

  if (contentLength > MAX_VIDEO_UPLOAD_BYTES) {
    throw createError({
      statusCode: 413,
      statusMessage: 'Le fichier video depasse la taille maximale autorisee.'
    })
  }

  const parts = await readMultipartFormData(event)

  if (!parts?.length) {
    throw createError({
      statusCode: 400,
      statusMessage: 'Un fichier video et des reglages sont requis.'
    })
  }

  const filePart = parts.find(part => part.name === 'file') as MultipartField | undefined
  const settingsPart = parts.find(part => part.name === 'settings') as MultipartField | undefined

  if (!filePart?.data || !filePart.filename) {
    throw createError({
      statusCode: 400,
      statusMessage: 'Le fichier video est requis.'
    })
  }

  if (filePart.data.length > MAX_VIDEO_UPLOAD_BYTES) {
    throw createError({
      statusCode: 413,
      statusMessage: 'Le fichier video depasse la taille maximale autorisee.'
    })
  }

  const settings = readSettingsField(settingsPart?.data)
  const workspace = await createVideoJobWorkspace(filePart.filename)
  let jobId = ''

  try {
    await writeFile(workspace.inputPath, filePart.data)
  } catch {
    await rm(workspace.tempRoot, { recursive: true, force: true })
    throw createError({
      statusCode: 500,
      statusMessage: 'Impossible de stocker le fichier video.'
    })
  }

  const frameCount = await getVideoFrameCount(workspace.inputPath)

  try {
    jobId = createJob({
      ownerId,
      fileName: filePart.filename,
      mimeType: 'video/mp4',
      inputPath: workspace.inputPath,
      settingsJson: JSON.stringify(settings),
      tempRoot: workspace.tempRoot,
      frameCount
    })
  } catch (error) {
    await rm(workspace.tempRoot, { recursive: true, force: true })
    throw createError({
      statusCode: 429,
      statusMessage: error instanceof Error ? error.message : 'Trop de traitements video sont deja en attente.'
    })
  }

  enqueueVideoJob(jobId)

  return { jobId }
})
