import type { Face } from '../types/faces'

const MASK_SCALE = 1.3
const MIN_BLUR_RADIUS = 2

interface RasterImage {
  data: Uint8ClampedArray
  width: number
  height: number
}

interface BlurScratch {
  temporary: Uint8ClampedArray
  target: Uint8ClampedArray
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function getPixelIndex(x: number, y: number, width: number) {
  return (y * width + x) * 4
}

function readByte(data: Uint8ClampedArray, index: number) {
  return data[index] ?? 0
}

function ensureBuffer(buffer: Uint8ClampedArray, length: number) {
  return buffer.length >= length ? buffer : new Uint8ClampedArray(length)
}

function ensureScratch(scratch: BlurScratch, length: number) {
  scratch.temporary = ensureBuffer(scratch.temporary, length)
  scratch.target = ensureBuffer(scratch.target, length)
}

function scaleFaceAroundCenter(face: Face, scale: number): Face {
  const width = Math.max(1, Math.round(face.width * scale))
  const height = Math.max(1, Math.round(face.height * scale))

  return {
    ...face,
    x: Math.round(face.x - ((width - face.width) / 2)),
    y: Math.round(face.y - ((height - face.height) / 2)),
    width,
    height
  }
}

function createEllipseAlpha(face: Face) {
  const centerX = face.x + (face.width / 2)
  const centerY = face.y + (face.height / 2)
  const radiusX = Math.max(1, face.width / 2)
  const radiusY = Math.max(1, face.height / 2)

  return (x: number, y: number) => {
    const dx = (x + 0.5 - centerX) / radiusX
    const dy = (y + 0.5 - centerY) / radiusY

    return Math.hypot(dx, dy) <= 1 ? 1 : 0
  }
}

function extractRegion(image: RasterImage, face: Face, padding: number) {
  const left = clamp(Math.floor(face.x - padding), 0, image.width - 1)
  const top = clamp(Math.floor(face.y - padding), 0, image.height - 1)
  const right = clamp(Math.ceil(face.x + face.width + padding), 0, image.width)
  const bottom = clamp(Math.ceil(face.y + face.height + padding), 0, image.height)
  const width = Math.max(1, right - left)
  const height = Math.max(1, bottom - top)
  const data = new Uint8ClampedArray(width * height * 4)

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const sourceIndex = getPixelIndex(left + x, top + y, image.width)
      const targetIndex = getPixelIndex(x, y, width)

      data[targetIndex] = readByte(image.data, sourceIndex)
      data[targetIndex + 1] = readByte(image.data, sourceIndex + 1)
      data[targetIndex + 2] = readByte(image.data, sourceIndex + 2)
      data[targetIndex + 3] = readByte(image.data, sourceIndex + 3)
    }
  }

  return { left, top, width, height, data }
}

function boxBlurHorizontal(input: Uint8ClampedArray, output: Uint8ClampedArray, width: number, height: number, radius: number) {
  const windowSize = (radius * 2) + 1

  for (let y = 0; y < height; y += 1) {
    let red = 0
    let green = 0
    let blue = 0
    let alpha = 0

    for (let offset = -radius; offset <= radius; offset += 1) {
      const sampleX = clamp(offset, 0, width - 1)
      const sampleIndex = getPixelIndex(sampleX, y, width)
      red += readByte(input, sampleIndex)
      green += readByte(input, sampleIndex + 1)
      blue += readByte(input, sampleIndex + 2)
      alpha += readByte(input, sampleIndex + 3)
    }

    for (let x = 0; x < width; x += 1) {
      const index = getPixelIndex(x, y, width)
      output[index] = Math.round(red / windowSize)
      output[index + 1] = Math.round(green / windowSize)
      output[index + 2] = Math.round(blue / windowSize)
      output[index + 3] = Math.round(alpha / windowSize)

      const removeX = clamp(x - radius, 0, width - 1)
      const addX = clamp(x + radius + 1, 0, width - 1)
      const removeIndex = getPixelIndex(removeX, y, width)
      const addIndex = getPixelIndex(addX, y, width)

      red += readByte(input, addIndex) - readByte(input, removeIndex)
      green += readByte(input, addIndex + 1) - readByte(input, removeIndex + 1)
      blue += readByte(input, addIndex + 2) - readByte(input, removeIndex + 2)
      alpha += readByte(input, addIndex + 3) - readByte(input, removeIndex + 3)
    }
  }
}

function boxBlurVertical(input: Uint8ClampedArray, output: Uint8ClampedArray, width: number, height: number, radius: number) {
  const windowSize = (radius * 2) + 1

  for (let x = 0; x < width; x += 1) {
    let red = 0
    let green = 0
    let blue = 0
    let alpha = 0

    for (let offset = -radius; offset <= radius; offset += 1) {
      const sampleY = clamp(offset, 0, height - 1)
      const sampleIndex = getPixelIndex(x, sampleY, width)
      red += readByte(input, sampleIndex)
      green += readByte(input, sampleIndex + 1)
      blue += readByte(input, sampleIndex + 2)
      alpha += readByte(input, sampleIndex + 3)
    }

    for (let y = 0; y < height; y += 1) {
      const index = getPixelIndex(x, y, width)
      output[index] = Math.round(red / windowSize)
      output[index + 1] = Math.round(green / windowSize)
      output[index + 2] = Math.round(blue / windowSize)
      output[index + 3] = Math.round(alpha / windowSize)

      const removeY = clamp(y - radius, 0, height - 1)
      const addY = clamp(y + radius + 1, 0, height - 1)
      const removeIndex = getPixelIndex(x, removeY, width)
      const addIndex = getPixelIndex(x, addY, width)

      red += readByte(input, addIndex) - readByte(input, removeIndex)
      green += readByte(input, addIndex + 1) - readByte(input, removeIndex + 1)
      blue += readByte(input, addIndex + 2) - readByte(input, removeIndex + 2)
      alpha += readByte(input, addIndex + 3) - readByte(input, removeIndex + 3)
    }
  }
}

function boxBlurRegion(region: RasterImage, radiusX: number, radiusY: number, scratch: BlurScratch) {
  if (radiusX <= 1 && radiusY <= 1) {
    return region.data
  }

  ensureScratch(scratch, region.data.length)

  if (radiusX <= 1) {
    boxBlurVertical(region.data, scratch.target, region.width, region.height, radiusY)
    return scratch.target
  }

  boxBlurHorizontal(region.data, scratch.temporary, region.width, region.height, radiusX)

  if (radiusY <= 1) {
    scratch.target.set(scratch.temporary)
    return scratch.target
  }

  boxBlurVertical(scratch.temporary, scratch.target, region.width, region.height, radiusY)
  return scratch.target
}

export function applyBlurEffects(
  image: RasterImage,
  faces: Face[],
  excludedFaceIds: string[] = [],
  blurIntensity = 0.5
) {
  const output = new Uint8ClampedArray(image.data)
  const excludedFaces = new Set(excludedFaceIds)
  const blurFaces = faces.filter(face => !excludedFaces.has(face.id))

  if (blurFaces.length === 0) {
    return output
  }

  const normalizedIntensity = Number.isFinite(blurIntensity) ? clamp(blurIntensity, 0, 1) : 0.5
  const scratch: BlurScratch = {
    temporary: new Uint8ClampedArray(0),
    target: new Uint8ClampedArray(0)
  }
  const pixelCount = image.width * image.height
  const alphaAccum = new Float32Array(pixelCount)
  const blurredAccum = new Uint8ClampedArray(pixelCount * 4)

  for (const face of blurFaces) {
    const scaled = scaleFaceAroundCenter(face, MASK_SCALE)
    const region = extractRegion(image, scaled, 0)
    const radiusX = Math.max(MIN_BLUR_RADIUS, Math.round((scaled.width * normalizedIntensity) / 2))
    const radiusY = Math.max(MIN_BLUR_RADIUS, Math.round((scaled.height * normalizedIntensity) / 2))
    const blurred = boxBlurRegion(region, radiusX, radiusY, scratch)
    const getMaskAlpha = createEllipseAlpha(scaled)

    for (let y = 0; y < region.height; y += 1) {
      for (let x = 0; x < region.width; x += 1) {
        const globalX = region.left + x
        const globalY = region.top + y
        const alpha = getMaskAlpha(globalX, globalY)

        if (alpha <= 0) {
          continue
        }

        const pixelIndex = (globalY * image.width) + globalX
        const sourceIndex = getPixelIndex(x, y, region.width)

        if (alpha > (alphaAccum[pixelIndex] ?? 0)) {
          alphaAccum[pixelIndex] = alpha
          const targetIndex = pixelIndex * 4
          blurredAccum[targetIndex] = readByte(blurred, sourceIndex)
          blurredAccum[targetIndex + 1] = readByte(blurred, sourceIndex + 1)
          blurredAccum[targetIndex + 2] = readByte(blurred, sourceIndex + 2)
          blurredAccum[targetIndex + 3] = readByte(blurred, sourceIndex + 3)
        }
      }
    }
  }

  for (let index = 0; index < pixelCount; index += 1) {
    const alpha = alphaAccum[index] ?? 0

    if (alpha <= 0) {
      continue
    }

    const targetIndex = index * 4
    output[targetIndex] = Math.round((readByte(blurredAccum, targetIndex) * alpha) + (readByte(image.data, targetIndex) * (1 - alpha)))
    output[targetIndex + 1] = Math.round((readByte(blurredAccum, targetIndex + 1) * alpha) + (readByte(image.data, targetIndex + 1) * (1 - alpha)))
    output[targetIndex + 2] = Math.round((readByte(blurredAccum, targetIndex + 2) * alpha) + (readByte(image.data, targetIndex + 2) * (1 - alpha)))
    output[targetIndex + 3] = Math.round((readByte(blurredAccum, targetIndex + 3) * alpha) + (readByte(image.data, targetIndex + 3) * (1 - alpha)))
  }

  return output
}
