import { readVideoMetadataFromBuffer } from '../utils/process-video'
import type { H3Event } from 'h3'

const MAX_METADATA_UPLOAD_BYTES = Number(process.env.PROCESS_MAX_METADATA_UPLOAD_BYTES || 256 * 1024 * 1024)
const MAX_METADATA_CONCURRENCY = Number(process.env.PROCESS_METADATA_MAX_CONCURRENCY || 2)

let activeMetadataReads = 0
const metadataWaiters: (() => void)[] = []

async function acquireMetadataSlot(): Promise<() => void> {
  if (activeMetadataReads < MAX_METADATA_CONCURRENCY) {
    activeMetadataReads += 1
    return () => {
      activeMetadataReads -= 1
      metadataWaiters.shift()?.()
    }
  }

  await new Promise<void>(resolve => metadataWaiters.push(resolve))
  return acquireMetadataSlot()
}

export default defineEventHandler(async (event) => {
  const contentLength = Number(getRequestHeader(event, 'content-length') || 0)

  if (contentLength <= 0) {
    throw createError({
      statusCode: 411,
      statusMessage: 'La longueur du contenu est requise.'
    })
  }

  if (contentLength > MAX_METADATA_UPLOAD_BYTES) {
    throw createError({
      statusCode: 413,
      statusMessage: 'Le fichier video depasse la taille maximale autorisee.'
    })
  }

  const release = await acquireMetadataSlot()

  try {
    return await handleMetadataRequest(event)
  } finally {
    release()
  }
})

async function handleMetadataRequest(event: H3Event) {
  const parts = await readMultipartFormData(event)

  if (!parts?.length) {
    throw createError({
      statusCode: 400,
      statusMessage: 'Un fichier video est requis.'
    })
  }

  const filePart = parts.find(part => part.name === 'file')

  if (!filePart?.filename || !Buffer.isBuffer(filePart.data)) {
    throw createError({
      statusCode: 400,
      statusMessage: 'Le fichier video est requis.'
    })
  }

  if (filePart.data.length > MAX_METADATA_UPLOAD_BYTES) {
    throw createError({
      statusCode: 413,
      statusMessage: 'Le fichier video depasse la taille maximale autorisee.'
    })
  }

  try {
    const metadata = await readVideoMetadataFromBuffer(filePart.data)

    return {
      width: metadata.width,
      height: metadata.height,
      fps: metadata.fps,
      frameCount: metadata.frameCount,
      duration: metadata.videoDuration ?? (metadata.frameCount / Math.max(0.001, metadata.fps))
    }
  } catch (cause) {
    throw createError({
      statusCode: 422,
      statusMessage: cause instanceof Error ? cause.message : 'Impossible de lire les metadonnees video.'
    })
  }
}
