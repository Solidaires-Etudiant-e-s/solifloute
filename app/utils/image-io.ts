import type { DetectionInput } from '~~/shared/types/faces'

export async function fileToBase64(file: File) {
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader()

    reader.onload = () => {
      const result = reader.result

      if (typeof result !== 'string') {
        reject(new Error('Impossible d encoder le fichier selectionne.'))
        return
      }

      resolve(result.split(',')[1] || '')
    }

    reader.onerror = () => {
      reject(reader.error || new Error('Impossible d encoder le fichier selectionne.'))
    }

    reader.readAsDataURL(file)
  })
}

export async function fileToImageData(file: File): Promise<ImageData> {
  const bitmap = await createImageBitmap(file)
  const canvas = document.createElement('canvas')
  canvas.width = bitmap.width
  canvas.height = bitmap.height
  const context = canvas.getContext('2d', { willReadFrequently: true })

  if (!context) {
    throw new Error('Le contexte canvas 2D est indisponible.')
  }

  context.drawImage(bitmap, 0, 0)
  const imageData = context.getImageData(0, 0, bitmap.width, bitmap.height)
  bitmap.close()

  return imageData
}

export function imageDataToDetectionInput(imageData: ImageData): DetectionInput {
  return {
    data: imageData.data,
    width: imageData.width,
    height: imageData.height
  }
}

export function imageDataToBlob(imageData: ImageData) {
  const canvas = document.createElement('canvas')
  canvas.width = imageData.width
  canvas.height = imageData.height
  const context = canvas.getContext('2d')

  if (!context) {
    throw new Error('Le contexte canvas 2D est indisponible.')
  }

  context.putImageData(imageData, 0, 0)

  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('Impossible de creer le blob d apercu.'))
        return
      }

      resolve(blob)
    }, 'image/png')
  })
}
