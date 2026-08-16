export interface AdaptiveProgressPhase {
  frames: number
  weight: number
  label?: string
  labelBuilder?: (done: number, total: number) => string
}

export interface AdaptiveProgressReport {
  progress: number
  remainingMs: number | null
  message: string
}

export function createAdaptiveProgress(phases: AdaptiveProgressPhase[]) {
  const weights = phases.map(phase => phase.weight)
  const frames = phases.map(phase => phase.frames)
  const labels = phases.map(phase => phase.label ?? '')
  const labelBuilders = phases.map(phase => phase.labelBuilder)
  const measuredMs = new Array<number>(phases.length).fill(0)

  let phaseIndex = 0
  let phaseStartedAt = performance.now()
  let framesDone = 0
  let lastProgress = 0

  function totalWeight() {
    let total = 0

    for (const weight of weights) {
      total += weight
    }

    return total > 0 ? total : 1
  }

  function completedWeight() {
    let total = 0

    for (let index = 0; index < phaseIndex; index += 1) {
      total += weights[index] ?? 0
    }

    return total
  }

  function phaseFraction() {
    if (phaseIndex >= phases.length) {
      return 1
    }

    const total = frames[phaseIndex] ?? 0
    return total > 0 ? Math.max(0, Math.min(1, framesDone / total)) : 0
  }

  function completedMs() {
    let total = 0

    for (let index = 0; index < phaseIndex; index += 1) {
      total += measuredMs[index] ?? 0
    }

    return total
  }

  function learnedScale() {
    const achieved = completedWeight() + ((weights[phaseIndex] ?? 0) * phaseFraction())
    const elapsedMs = completedMs() + (performance.now() - phaseStartedAt)

    return achieved > 0 && elapsedMs > 0 ? elapsedMs / achieved : null
  }

  function estimateTotalMs() {
    const elapsedMs = performance.now() - phaseStartedAt

    if (phaseIndex >= phases.length) {
      return Math.max(1, completedMs() + elapsedMs)
    }

    const fraction = phaseFraction()
    const currentWeight = weights[phaseIndex] ?? 0
    const currentTotalMs = fraction > 0
      ? elapsedMs / fraction
      : currentWeight > 0
        ? elapsedMs * 2
        : 0
    let futureMs = 0
    const scale = learnedScale()

    if (scale !== null) {
      for (let index = phaseIndex + 1; index < phases.length; index += 1) {
        futureMs += (weights[index] ?? 0) * scale
      }
    } else if (currentWeight > 0) {
      for (let index = phaseIndex + 1; index < phases.length; index += 1) {
        futureMs += ((weights[index] ?? 0) * currentTotalMs) / currentWeight
      }
    }

    return Math.max(1, completedMs() + currentTotalMs + futureMs)
  }

  function buildMessage() {
    if (phaseIndex >= phases.length) {
      return 'Video traitee.'
    }

    const builder = labelBuilders[phaseIndex]
    const total = frames[phaseIndex] ?? 0

    if (builder) {
      return builder(framesDone, total)
    }

    return labels[phaseIndex] ?? ''
  }

  function report(done: number): AdaptiveProgressReport {
    if (phaseIndex < phases.length) {
      framesDone = Math.max(0, Math.min(frames[phaseIndex] ?? 0, done))
    }

    const achieved = completedWeight() + ((weights[phaseIndex] ?? 0) * phaseFraction())
    const nextProgress = Math.min(1, achieved / totalWeight())
    const progress = Math.max(lastProgress, nextProgress)
    lastProgress = progress
    const totalMs = estimateTotalMs()
    const elapsedMs = completedMs() + (performance.now() - phaseStartedAt)

    return {
      progress,
      remainingMs: Math.max(0, totalMs - elapsedMs),
      message: buildMessage()
    }
  }

  return {
    setFrames(nextFrames: number) {
      if (phaseIndex < phases.length) {
        frames[phaseIndex] = Math.max(0, nextFrames)
      }
    },
    report,
    nextPhase(nextFrames?: number): AdaptiveProgressReport {
      measuredMs[phaseIndex] = Math.max(0, performance.now() - phaseStartedAt)
      phaseIndex = Math.min(phases.length, phaseIndex + 1)
      phaseStartedAt = performance.now()
      framesDone = 0

      if (phaseIndex >= phases.length) {
        return { progress: 1, remainingMs: 0, message: 'Video traitee.' }
      }

      if (nextFrames !== undefined) {
        frames[phaseIndex] = Math.max(0, nextFrames)
      }

      return report(0)
    }
  }
}
