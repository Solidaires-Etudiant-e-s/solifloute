interface BoxCandidate {
  x1: number
  y1: number
  x2: number
  y2: number
  score: number
}

function intersectionOverUnion(a: BoxCandidate, b: BoxCandidate) {
  const x1 = Math.max(a.x1, b.x1)
  const y1 = Math.max(a.y1, b.y1)
  const x2 = Math.min(a.x2, b.x2)
  const y2 = Math.min(a.y2, b.y2)
  const width = Math.max(0, x2 - x1)
  const height = Math.max(0, y2 - y1)
  const intersection = width * height
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1)
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1)
  const union = areaA + areaB - intersection

  return union <= 0 ? 0 : intersection / union
}

export function hardNonMaxSuppression(boxes: BoxCandidate[], iouThreshold = 0.3, topK = -1) {
  const order = boxes
    .map((box, index) => ({ box, index }))
    .sort((a, b) => b.box.score - a.box.score)
    .map(entry => entry.index)
  const selected: number[] = []
  const suppressed = new Uint8Array(boxes.length)

  for (let orderIndex = 0; orderIndex < order.length; orderIndex += 1) {
    const currentIndex = order[orderIndex]!

    if (suppressed[currentIndex]) {
      continue
    }

    selected.push(currentIndex)

    if (topK > 0 && selected.length >= topK) {
      break
    }

    const current = boxes[currentIndex]!

    for (let otherOrderIndex = orderIndex + 1; otherOrderIndex < order.length; otherOrderIndex += 1) {
      const otherIndex = order[otherOrderIndex]!

      if (!suppressed[otherIndex] && intersectionOverUnion(current, boxes[otherIndex]!) > iouThreshold) {
        suppressed[otherIndex] = 1
      }
    }
  }

  return selected.map(index => boxes[index]!)
}
