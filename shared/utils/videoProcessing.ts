import type { EditorSettings, Face } from '../types/faces'
import { applyBlurEffects } from './imageProcessing'
import { createFaceResolver, type FaceSample } from './videoFaceTracking'

export async function collectFaceSamples(
  frameCount: number,
  detectFacesAtFrame: (frameIndex: number) => Promise<Face[]>,
  onProgress?: (framesDone: number) => void
) {
  const samples: FaceSample[] = []

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    samples.push({
      frameIndex,
      faces: await detectFacesAtFrame(frameIndex)
    })
    onProgress?.(frameIndex + 1)
  }

  return samples
}

export function createVideoFaceResolver(samples: FaceSample[], fps: number) {
  return createFaceResolver(samples, {
    gapFrames: 2,
    appearanceFrames: Math.max(2, Math.round(fps * 0.1)),
    disappearanceFrames: Math.max(2, Math.round(fps * 0.1))
  })
}

export function blurVideoFrame(
  frame: { data: Uint8ClampedArray, width: number, height: number },
  settings: EditorSettings,
  resolveFaces: (frameIndex: number) => Face[],
  frameIndex: number
) {
  return applyBlurEffects(
    frame,
    resolveFaces(frameIndex),
    settings.excludedFaceIds,
    settings.blurIntensity
  )
}
