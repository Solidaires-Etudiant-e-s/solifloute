import type { Face } from '../types/faces'

export const FACE_MATCH_MAX_SCORE = 1.5
const HIGH_CONFIDENCE_FACE = 0.9
const FACE_MATCH_DISTANCE_RATIO = 1.5
const SMALL_FACE_MAX_SIZE = 36
const FAST_MOTION_RATIO = 0.2

export interface FaceSample {
  frameIndex: number
  faces: Face[]
}

export type ResolvedFaceSource = 'detection' | 'interpolated' | 'appearance-ext' | 'disappearance-ext'

export interface ResolvedFace extends Face {
  source: ResolvedFaceSource
}

interface Detection {
  frameIndex: number
  face: Face
}

interface FaceTrack {
  detections: Detection[]
}

function centerDistance(faceA: Face, faceB: Face) {
  const ax = faceA.x + (faceA.width / 2)
  const ay = faceA.y + (faceA.height / 2)
  const bx = faceB.x + (faceB.width / 2)
  const by = faceB.y + (faceB.height / 2)

  return Math.hypot(ax - bx, ay - by)
}

function overlapRatio(faceA: Face, faceB: Face) {
  const left = Math.max(faceA.x, faceB.x)
  const top = Math.max(faceA.y, faceB.y)
  const right = Math.min(faceA.x + faceA.width, faceB.x + faceB.width)
  const bottom = Math.min(faceA.y + faceA.height, faceB.y + faceB.height)
  const intersectionArea = Math.max(0, right - left) * Math.max(0, bottom - top)
  const smallestArea = Math.max(1, Math.min(faceA.width * faceA.height, faceB.width * faceB.height))

  return intersectionArea / smallestArea
}

function sizeDifference(faceA: Face, faceB: Face) {
  return (
    Math.abs(faceA.width - faceB.width) / Math.max(1, Math.max(faceA.width, faceB.width))
    + Math.abs(faceA.height - faceB.height) / Math.max(1, Math.max(faceA.height, faceB.height))
  )
}

function matchScore(faceA: Face, faceB: Face) {
  const averageSize = Math.max(1, (faceA.width + faceA.height + faceB.width + faceB.height) / 4)

  return centerDistance(faceA, faceB) / averageSize + sizeDifference(faceA, faceB)
}

function isHighConfidenceMatch(faceA: Face, faceB: Face) {
  const smallerMaxSize = Math.min(
    Math.max(faceA.width, faceA.height),
    Math.max(faceB.width, faceB.height)
  )
  const closeEnough = centerDistance(faceA, faceB) <= smallerMaxSize * FACE_MATCH_DISTANCE_RATIO

  return (
    Math.max(faceA.confidence, faceB.confidence) >= HIGH_CONFIDENCE_FACE
    && (closeEnough || overlapRatio(faceA, faceB) >= 0.1)
  )
}

function canMatch(faceA: Face, faceB: Face, score: number) {
  return score <= FACE_MATCH_MAX_SCORE || isHighConfidenceMatch(faceA, faceB)
}

function matchFaces(previousFaces: Face[], nextFaces: Face[]) {
  const remainingPrevious = new Set(previousFaces.map((_, index) => index))
  const remainingNext = new Set(nextFaces.map((_, index) => index))
  const pairs: Array<{ fromIndex: number, toIndex: number }> = []

  while (remainingPrevious.size > 0 && remainingNext.size > 0) {
    let bestPreviousIndex = -1
    let bestNextIndex = -1
    let bestScore = Number.POSITIVE_INFINITY

    for (const previousIndex of remainingPrevious) {
      for (const nextIndex of remainingNext) {
        const score = matchScore(previousFaces[previousIndex]!, nextFaces[nextIndex]!)

        if (score < bestScore) {
          bestScore = score
          bestPreviousIndex = previousIndex
          bestNextIndex = nextIndex
        }
      }
    }

    if (
      bestPreviousIndex < 0
      || bestNextIndex < 0
      || !canMatch(previousFaces[bestPreviousIndex]!, nextFaces[bestNextIndex]!, bestScore)
    ) {
      break
    }

    pairs.push({ fromIndex: bestPreviousIndex, toIndex: bestNextIndex })
    remainingPrevious.delete(bestPreviousIndex)
    remainingNext.delete(bestNextIndex)
  }

  return pairs
}

function buildTracks(samples: FaceSample[]): FaceTrack[] {
  const trackIdByFaceKey = new Map<string, number>()
  const tracks: FaceTrack[] = []
  let nextTrackId = 0

  function ensureTrack(keys: string[]) {
    for (const key of keys) {
      const existing = trackIdByFaceKey.get(key)
      if (existing !== undefined) {
        return existing
      }
    }

    const trackId = nextTrackId
    nextTrackId += 1
    tracks.push({ detections: [] })
    for (const key of keys) {
      trackIdByFaceKey.set(key, trackId)
    }

    return trackId
  }

  for (let sampleIndex = 0; sampleIndex < samples.length - 1; sampleIndex += 1) {
    const previousFaces = samples[sampleIndex]!.faces
    const nextFaces = samples[sampleIndex + 1]!.faces

    for (const { fromIndex, toIndex } of matchFaces(previousFaces, nextFaces)) {
      ensureTrack([`${sampleIndex}:${fromIndex}`, `${sampleIndex + 1}:${toIndex}`])
    }
  }

  for (let sampleIndex = 0; sampleIndex < samples.length; sampleIndex += 1) {
    const sample = samples[sampleIndex]!

    for (let faceIndex = 0; faceIndex < sample.faces.length; faceIndex += 1) {
      const trackId = trackIdByFaceKey.get(`${sampleIndex}:${faceIndex}`) ?? ensureTrack([`${sampleIndex}:${faceIndex}`])
      tracks[trackId]!.detections.push({
        frameIndex: sample.frameIndex,
        face: sample.faces[faceIndex]!
      })
    }
  }

  return tracks.filter(track => track.detections.length > 0)
}

function removeIsolatedTracks(tracks: FaceTrack[], samples: FaceSample[], windowFrames: number): FaceTrack[] {
  return tracks.filter((track) => {
    if (track.detections.length > 1) {
      return true
    }

    const detection = track.detections[0]!
    for (const sample of samples) {
      const frameDistance = Math.abs(sample.frameIndex - detection.frameIndex)
      if (frameDistance === 0 || frameDistance > windowFrames) {
        continue
      }
      for (const face of sample.faces) {
        if (canMatch(detection.face, face, matchScore(detection.face, face))) {
          return true
        }
      }
    }

    return false
  })
}

function tracksCompatible(previousFace: Face, nextFace: Face) {
  const maxSize = Math.max(
    previousFace.width,
    previousFace.height,
    nextFace.width,
    nextFace.height
  )

  return centerDistance(previousFace, nextFace) <= maxSize * FACE_MATCH_DISTANCE_RATIO || overlapRatio(previousFace, nextFace) >= 0.1
}

function mergeTracks(tracks: FaceTrack[], gapFrames: number): FaceTrack[] {
  if (gapFrames <= 0) {
    return tracks
  }

  const orderedTracks = [...tracks].sort((left, right) => {
    return left.detections[0]!.frameIndex - right.detections[0]!.frameIndex
  })
  const mergedTracks: FaceTrack[] = []

  for (const track of orderedTracks) {
    const currentStartFrame = track.detections[0]!.frameIndex
    const currentFirstFace = track.detections[0]!.face
    let bestPreviousTrack: FaceTrack | null = null
    let bestDistance = Number.POSITIVE_INFINITY

    for (let index = mergedTracks.length - 1; index >= 0; index -= 1) {
      const previousTrack = mergedTracks[index]!
      const previousEndFrame = previousTrack.detections[previousTrack.detections.length - 1]!.frameIndex
      const gap = currentStartFrame - previousEndFrame

      if (gap <= 0 || gap > gapFrames) {
        continue
      }

      const previousLastFace = previousTrack.detections[previousTrack.detections.length - 1]!.face

      if (!tracksCompatible(previousLastFace, currentFirstFace)) {
        continue
      }

      const distance = centerDistance(previousLastFace, currentFirstFace)

      if (distance < bestDistance) {
        bestDistance = distance
        bestPreviousTrack = previousTrack
      }
    }

    if (bestPreviousTrack) {
      bestPreviousTrack.detections.push(...track.detections)
      bestPreviousTrack.detections.sort((left, right) => left.frameIndex - right.frameIndex)
      continue
    }

    mergedTracks.push(track)
  }

  return mergedTracks
}

function trackIsUnstable(track: FaceTrack) {
  const detections = track.detections

  if (detections.length === 0) {
    return false
  }

  let sizeSum = 0
  let motionSum = 0
  let motionCount = 0
  let hasFastSegment = false

  for (let index = 0; index < detections.length; index += 1) {
    const face = detections[index]!.face
    sizeSum += Math.max(face.width, face.height)

    if (index > 0) {
      const previousFace = detections[index - 1]!.face
      const gap = Math.max(1, detections[index]!.frameIndex - detections[index - 1]!.frameIndex)
      const motionPerFrame = centerDistance(previousFace, face) / gap
      motionSum += motionPerFrame
      motionCount += 1

      const segmentSize = Math.max(
        Math.max(previousFace.width, previousFace.height),
        Math.max(face.width, face.height)
      )
      if (motionPerFrame / Math.max(1, segmentSize) > FAST_MOTION_RATIO) {
        hasFastSegment = true
      }
    }
  }

  const averageSize = sizeSum / detections.length

  if (averageSize < SMALL_FACE_MAX_SIZE) {
    return true
  }

  const averageMotion = motionCount > 0 ? motionSum / motionCount : 0
  return averageMotion / Math.max(1, averageSize) > FAST_MOTION_RATIO || hasFastSegment
}

function smoothTrack(track: FaceTrack, windowSize: number, smoothPosition = true) {
  const half = Math.floor(windowSize / 2)

  for (let index = 0; index < track.detections.length; index += 1) {
    const start = Math.max(0, index - half)
    const end = Math.min(track.detections.length - 1, index + half)
    let weightSum = 0
    let x = 0
    let y = 0
    let width = 0
    let height = 0

    for (let otherIndex = start; otherIndex <= end; otherIndex += 1) {
      const other = track.detections[otherIndex]!
      const weight = 1 / (Math.abs(otherIndex - index) + 1)
      x += other.face.x * weight
      y += other.face.y * weight
      width += other.face.width * weight
      height += other.face.height * weight
      weightSum += weight
    }

    const face = track.detections[index]!.face

    if (smoothPosition) {
      face.x = Math.round(x / weightSum)
      face.y = Math.round(y / weightSum)
    }

    face.width = Math.round(width / weightSum)
    face.height = Math.round(height / weightSum)
  }
}

function lerp(start: number, end: number, t: number) {
  return start + ((end - start) * t)
}

function clampNumber(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function cloneFace(face: Face): Face {
  return { ...face }
}

interface GlobalMotion {
  dx: number
  dy: number
}

function faceCenter(face: Face) {
  return { x: face.x + (face.width / 2), y: face.y + (face.height / 2) }
}

function median(values: number[]) {
  if (values.length === 0) {
    return 0
  }
  const sorted = [...values].sort((left, right) => left - right)
  return sorted[Math.floor(sorted.length / 2)]!
}

function estimateGlobalMotion(tracks: FaceTrack[]): Map<number, GlobalMotion> {
  const samplesByFrame = new Map<number, Array<{ dx: number, dy: number }>>()

  for (const track of tracks) {
    for (let index = 1; index < track.detections.length; index += 1) {
      const current = track.detections[index]!
      const previous = track.detections[index - 1]!

      if (current.frameIndex !== previous.frameIndex + 1) {
        continue
      }

      const previousCenter = faceCenter(previous.face)
      const currentCenter = faceCenter(current.face)
      const samples = samplesByFrame.get(previous.frameIndex) ?? []
      samples.push({
        dx: currentCenter.x - previousCenter.x,
        dy: currentCenter.y - previousCenter.y
      })
      samplesByFrame.set(previous.frameIndex, samples)
    }
  }

  const motion = new Map<number, GlobalMotion>()

  for (const [frameIndex, samples] of samplesByFrame) {
    motion.set(frameIndex, {
      dx: median(samples.map(sample => sample.dx)),
      dy: median(samples.map(sample => sample.dy))
    })
  }

  return motion
}

function cameraMotionAt(globalMotion: Map<number, GlobalMotion>, frameIndex: number): GlobalMotion {
  const maxDistance = 60

  for (let distance = 0; distance <= maxDistance; distance += 1) {
    const after = globalMotion.get(frameIndex + distance)

    if (after) {
      return after
    }

    const before = globalMotion.get(frameIndex - distance)

    if (before) {
      return before
    }
  }

  return { dx: 0, dy: 0 }
}

function sumCameraMotion(globalMotion: Map<number, GlobalMotion>, startFrame: number, endFrame: number): GlobalMotion {
  let dx = 0
  let dy = 0

  for (let frameIndex = startFrame; frameIndex <= endFrame; frameIndex += 1) {
    const motion = cameraMotionAt(globalMotion, frameIndex)
    dx += motion.dx
    dy += motion.dy
  }

  return { dx, dy }
}

function projectBeforeTrack(track: FaceTrack, frameIndex: number, globalMotion: Map<number, GlobalMotion>): Face {
  const firstDetection = track.detections[0]!
  const framesBefore = firstDetection.frameIndex - frameIndex

  if (framesBefore <= 0) {
    return cloneFace(firstDetection.face)
  }

  let residualVelocityX = 0
  let residualVelocityY = 0
  const maxShift = Math.max(4, Math.min(48, firstDetection.face.width * 0.6))

  if (track.detections.length >= 2) {
    const secondDetection = track.detections[1]!
    const span = Math.max(1, secondDetection.frameIndex - firstDetection.frameIndex)
    const faceVelocityX = (secondDetection.face.x - firstDetection.face.x) / span
    const faceVelocityY = (secondDetection.face.y - firstDetection.face.y) / span
    const cameraMotion = sumCameraMotion(globalMotion, firstDetection.frameIndex, secondDetection.frameIndex - 1)
    const cameraVelocityX = cameraMotion.dx / span
    const cameraVelocityY = cameraMotion.dy / span
    residualVelocityX = clampNumber(faceVelocityX - cameraVelocityX, -maxShift, maxShift)
    residualVelocityY = clampNumber(faceVelocityY - cameraVelocityY, -maxShift, maxShift)
  }

  const cameraShift = sumCameraMotion(globalMotion, frameIndex, firstDetection.frameIndex - 1)
  const velocityWidth = track.detections.length >= 2
    ? (track.detections[1]!.face.width - firstDetection.face.width) / Math.max(1, track.detections[1]!.frameIndex - firstDetection.frameIndex)
    : 0
  const velocityHeight = track.detections.length >= 2
    ? (track.detections[1]!.face.height - firstDetection.face.height) / Math.max(1, track.detections[1]!.frameIndex - firstDetection.frameIndex)
    : 0

  const residualShiftX = clampNumber(residualVelocityX * framesBefore, -maxShift, maxShift)
  const residualShiftY = clampNumber(residualVelocityY * framesBefore, -maxShift, maxShift)
  const totalShiftX = clampNumber(cameraShift.dx + residualShiftX, -maxShift, maxShift)
  const totalShiftY = clampNumber(cameraShift.dy + residualShiftY, -maxShift, maxShift)

  return {
    id: firstDetection.face.id,
    x: Math.round(firstDetection.face.x - totalShiftX),
    y: Math.round(firstDetection.face.y - totalShiftY),
    width: Math.round(clampNumber(
      firstDetection.face.width - (velocityWidth * framesBefore),
      firstDetection.face.width * 0.75,
      firstDetection.face.width * 1.25
    )),
    height: Math.round(clampNumber(
      firstDetection.face.height - (velocityHeight * framesBefore),
      firstDetection.face.height * 0.75,
      firstDetection.face.height * 1.25
    )),
    confidence: firstDetection.face.confidence
  }
}

function projectAfterTrack(track: FaceTrack, frameIndex: number, globalMotion: Map<number, GlobalMotion>): Face {
  const lastDetection = track.detections[track.detections.length - 1]!
  const framesAfter = frameIndex - lastDetection.frameIndex

  if (framesAfter <= 0) {
    return cloneFace(lastDetection.face)
  }

  let residualVelocityX = 0
  let residualVelocityY = 0
  const maxShift = Math.max(4, Math.min(48, lastDetection.face.width * 0.6))

  if (track.detections.length >= 2) {
    const previousDetection = track.detections[track.detections.length - 2]!
    const span = Math.max(1, lastDetection.frameIndex - previousDetection.frameIndex)
    const faceVelocityX = (lastDetection.face.x - previousDetection.face.x) / span
    const faceVelocityY = (lastDetection.face.y - previousDetection.face.y) / span
    const cameraMotion = sumCameraMotion(globalMotion, previousDetection.frameIndex, lastDetection.frameIndex - 1)
    const cameraVelocityX = cameraMotion.dx / span
    const cameraVelocityY = cameraMotion.dy / span
    residualVelocityX = clampNumber(faceVelocityX - cameraVelocityX, -maxShift, maxShift)
    residualVelocityY = clampNumber(faceVelocityY - cameraVelocityY, -maxShift, maxShift)
  }

  const cameraShift = sumCameraMotion(globalMotion, lastDetection.frameIndex + 1, frameIndex)
  const velocityWidth = track.detections.length >= 2
    ? (lastDetection.face.width - track.detections[track.detections.length - 2]!.face.width) / Math.max(1, lastDetection.frameIndex - track.detections[track.detections.length - 2]!.frameIndex)
    : 0
  const velocityHeight = track.detections.length >= 2
    ? (lastDetection.face.height - track.detections[track.detections.length - 2]!.face.height) / Math.max(1, lastDetection.frameIndex - track.detections[track.detections.length - 2]!.frameIndex)
    : 0

  const residualShiftX = clampNumber(residualVelocityX * framesAfter, -maxShift, maxShift)
  const residualShiftY = clampNumber(residualVelocityY * framesAfter, -maxShift, maxShift)
  const totalShiftX = clampNumber(cameraShift.dx + residualShiftX, -maxShift, maxShift)
  const totalShiftY = clampNumber(cameraShift.dy + residualShiftY, -maxShift, maxShift)

  return {
    id: lastDetection.face.id,
    x: Math.round(lastDetection.face.x + totalShiftX),
    y: Math.round(lastDetection.face.y + totalShiftY),
    width: Math.round(clampNumber(
      lastDetection.face.width + (velocityWidth * framesAfter),
      lastDetection.face.width * 0.75,
      lastDetection.face.width * 1.25
    )),
    height: Math.round(clampNumber(
      lastDetection.face.height + (velocityHeight * framesAfter),
      lastDetection.face.height * 0.75,
      lastDetection.face.height * 1.25
    )),
    confidence: lastDetection.face.confidence
  }
}

function interpolateFace(faceA: Face, faceB: Face, t: number): Face {
  return {
    id: faceA.id,
    x: Math.round(lerp(faceA.x, faceB.x, t)),
    y: Math.round(lerp(faceA.y, faceB.y, t)),
    width: Math.round(lerp(faceA.width, faceB.width, t)),
    height: Math.round(lerp(faceA.height, faceB.height, t)),
    confidence: faceA.confidence
  }
}

function pushFace(resolvedFrames: Map<number, ResolvedFace[]>, frameIndex: number, face: Face, source: ResolvedFaceSource) {
  const faces = resolvedFrames.get(frameIndex) ?? []
  faces.push({ ...face, source })
  resolvedFrames.set(frameIndex, faces)
}

function clampToFrame(face: Face, frameWidth: number, frameHeight: number): Face {
  return {
    ...face,
    x: clampNumber(face.x, 0, Math.max(0, frameWidth - face.width)),
    y: clampNumber(face.y, 0, Math.max(0, frameHeight - face.height))
  }
}

function fillTrackGaps(track: FaceTrack, gapFrames: number, resolvedFrames: Map<number, ResolvedFace[]>) {
  for (let index = 0; index < track.detections.length - 1; index += 1) {
    const current = track.detections[index]!
    const next = track.detections[index + 1]!
    const gap = next.frameIndex - current.frameIndex

    if (gap <= 1 || gap > gapFrames) {
      continue
    }

    for (let frameIndex = current.frameIndex + 1; frameIndex < next.frameIndex; frameIndex += 1) {
      const t = (frameIndex - current.frameIndex) / gap
      pushFace(resolvedFrames, frameIndex, interpolateFace(current.face, next.face, t), 'interpolated')
    }
  }
}

function fillTrackExtension(
  track: FaceTrack,
  options: { appearanceFrames: number, disappearanceFrames: number, frameCount: number },
  resolvedFrames: Map<number, ResolvedFace[]>,
  globalMotion: Map<number, GlobalMotion>,
  frameWidth: number,
  frameHeight: number
) {
  const { appearanceFrames, disappearanceFrames, frameCount } = options
  const firstDetection = track.detections[0]!
  const lastDetection = track.detections[track.detections.length - 1]!

  if (track.detections.length < 2) {
    return
  }

  if (appearanceFrames > 0) {
    const firstFrame = Math.max(0, firstDetection.frameIndex - appearanceFrames)
    for (let frameIndex = firstFrame; frameIndex < firstDetection.frameIndex; frameIndex += 1) {
      pushFace(resolvedFrames, frameIndex, clampToFrame(projectBeforeTrack(track, frameIndex, globalMotion), frameWidth, frameHeight), 'appearance-ext')
    }
  }

  if (disappearanceFrames > 0) {
    const lastFrame = Math.min(frameCount - 1, lastDetection.frameIndex + disappearanceFrames)
    for (let frameIndex = lastDetection.frameIndex + 1; frameIndex <= lastFrame; frameIndex += 1) {
      pushFace(resolvedFrames, frameIndex, clampToFrame(projectAfterTrack(track, frameIndex, globalMotion), frameWidth, frameHeight), 'disappearance-ext')
    }
  }
}

function deduplicateFrameFaces(faces: ResolvedFace[]): ResolvedFace[] {
  const sorted = [...faces].sort((left, right) => {
    const priority = (source: ResolvedFaceSource) => (
      source === 'detection'
        ? 2
        : source === 'interpolated'
          ? 1
          : 0
    )
    return priority(right.source) - priority(left.source) || right.confidence - left.confidence
  })
  const kept: ResolvedFace[] = []

  for (const face of sorted) {
    const isDuplicate = kept.some((other) => {
      const minSize = Math.min(face.width, face.height, other.width, other.height)
      const closeCenters = centerDistance(face, other) <= minSize * 0.3
      const sizesSimilar = Math.max(face.width, face.height, other.width, other.height) <= minSize * 1.5
      const bothDetections = face.source === 'detection' && other.source === 'detection'

      if (bothDetections) {
        return sizesSimilar && (overlapRatio(face, other) >= 0.5 || closeCenters)
      }

      return overlapRatio(face, other) >= 0.2 || closeCenters
    })

    if (!isDuplicate) {
      kept.push(face)
    }
  }

  kept.sort((left, right) => (right.width * right.height) - (left.width * left.height))

  return kept
}

export interface FaceResolverOptions {
  gapFrames?: number
  appearanceFrames?: number
  disappearanceFrames?: number
}

export function createFaceResolver(samples: FaceSample[], options: FaceResolverOptions = {}) {
  if (samples.length === 0) {
    return () => [] as ResolvedFace[]
  }

  const normalized = [...samples].sort((left, right) => left.frameIndex - right.frameIndex)
  const frameCount = normalized[normalized.length - 1]!.frameIndex + 1
  let frameWidth = 0
  let frameHeight = 0
  for (const sample of normalized) {
    for (const face of sample.faces) {
      frameWidth = Math.max(frameWidth, face.x + face.width)
      frameHeight = Math.max(frameHeight, face.y + face.height)
    }
  }
  const gapFrames = Math.max(0, options.gapFrames ?? 4)
  const appearanceFrames = Math.max(0, options.appearanceFrames ?? 0)
  const disappearanceFrames = Math.max(0, options.disappearanceFrames ?? 0)
  const tracks = mergeTracks(
    removeIsolatedTracks(buildTracks(normalized), normalized, 2),
    gapFrames
  )
  const globalMotion = estimateGlobalMotion(tracks)
  const resolvedFrames = new Map<number, ResolvedFace[]>()

  for (const track of tracks) {
    const unstable = trackIsUnstable(track)
    smoothTrack(track, 5, !unstable)
    for (const detection of track.detections) {
      pushFace(resolvedFrames, detection.frameIndex, detection.face, 'detection')
    }
    fillTrackGaps(track, gapFrames, resolvedFrames)
    fillTrackExtension(track, {
      appearanceFrames,
      disappearanceFrames,
      frameCount
    }, resolvedFrames, globalMotion, frameWidth, frameHeight)
  }

  for (const [frameIndex, faces] of resolvedFrames) {
    resolvedFrames.set(frameIndex, deduplicateFrameFaces(faces))
  }

  return (frameIndex: number) => {
    if (frameIndex < 0 || frameIndex >= frameCount) {
      return []
    }

    return resolvedFrames.get(frameIndex) ?? []
  }
}
