export interface AdaptiveProgressPhase {
  frames: number
  weight: number
}

export interface AdaptiveProgressReport {
  progress: number
  remainingMs: number | null
}

export function createAdaptiveProgress(phases: AdaptiveProgressPhase[]) {
  const weights = phases.map(phase => phase.weight)
  const frames = phases.map(phase => phase.frames)
  const measuredMs = new Array<number>(phases.length).fill(0)

  let phaseIndex = 0
  let phaseStartedAt = performance.now()
  let framesDone = 0

  function completedMs() {
    let total = 0

    for (let index = 0; index < phaseIndex; index += 1) {
      total += measuredMs[index] ?? 0
    }

    return total
  }

  function learnedScale() {
    let weightSum = 0
    let durationSum = 0

    for (let index = 0; index < phaseIndex; index += 1) {
      weightSum += weights[index] ?? 0
      durationSum += measuredMs[index] ?? 0
    }

    return weightSum > 0 && durationSum > 0 ? durationSum / weightSum : null
  }

  function estimateTotalMs() {
    const elapsedMs = performance.now() - phaseStartedAt
    const completed = completedMs()

    if (phaseIndex >= phases.length) {
      return Math.max(1, completed + elapsedMs)
    }

    const currentWeight = weights[phaseIndex] ?? 0
    const currentFrames = frames[phaseIndex] ?? 0
    const scale = learnedScale()
    const throughputMs = framesDone > 0 ? elapsedMs / framesDone : 0
    let currentRemainingMs = 0

    if (throughputMs > 0) {
      currentRemainingMs = throughputMs * Math.max(0, currentFrames - framesDone)
    } else if (scale !== null) {
      currentRemainingMs = Math.max(0, (currentWeight * scale) - elapsedMs)
    }

    let futureMs = 0

    if (scale !== null) {
      for (let index = phaseIndex + 1; index < phases.length; index += 1) {
        futureMs += (weights[index] ?? 0) * scale
      }
    } else if (currentWeight > 0) {
      const currentTotalMs = elapsedMs + currentRemainingMs

      for (let index = phaseIndex + 1; index < phases.length; index += 1) {
        futureMs += ((weights[index] ?? 0) * currentTotalMs) / currentWeight
      }
    }

    return Math.max(1, completed + elapsedMs + currentRemainingMs + futureMs)
  }

  function report(done: number): AdaptiveProgressReport {
    if (phaseIndex < phases.length) {
      framesDone = Math.max(0, Math.min(frames[phaseIndex] ?? 0, done))
    }

    const totalMs = estimateTotalMs()
    const elapsedMs = completedMs() + (performance.now() - phaseStartedAt)

    return {
      progress: Math.max(0, Math.min(1, elapsedMs / totalMs)),
      remainingMs: Math.max(0, totalMs - elapsedMs)
    }
  }

  return {
    setFrames(nextFrames: number) {
      if (phaseIndex < phases.length) {
        frames[phaseIndex] = Math.max(0, nextFrames)
      }
    },
    report,
    nextPhase(): AdaptiveProgressReport {
      measuredMs[phaseIndex] = Math.max(0, performance.now() - phaseStartedAt)
      phaseIndex = Math.min(phases.length, phaseIndex + 1)
      phaseStartedAt = performance.now()
      framesDone = 0

      if (phaseIndex >= phases.length) {
        return { progress: 1, remainingMs: 0 }
      }

      return report(0)
    }
  }
}
