import type { DetectionInput, DetectionResult } from '../types/faces'
import {
  createModelInputData,
  DEFAULT_PROBABILITY_THRESHOLD,
  extractFacesFromOutputs,
  extractFacesFromYunetOutputs,
  getPaddedInputSize
} from './faceDetectionCore'

const isNodeRuntime = typeof window === 'undefined' && typeof self === 'undefined' && typeof process !== 'undefined'

const DEFAULT_MODEL_PATH = isNodeRuntime
  ? `${process.cwd()}/public/models/centerface.onnx`
  : '/models/centerface.onnx'

export type FaceDetectionModelType = 'centerface' | 'yunet'

type OrtWebModule = typeof import('onnxruntime-web')
type OrtNodeModule = typeof import('onnxruntime-node')
type OrtModule = OrtWebModule | OrtNodeModule
type InferenceSession = Awaited<ReturnType<OrtModule['InferenceSession']['create']>>

const sessionCache = new Map<string, Promise<InferenceSession>>()
let ortPromise: Promise<OrtModule> | null = null

function getExecutionProviders() {
  return isNodeRuntime ? ['cpu'] : ['wasm']
}

async function getOrt() {
  if (!ortPromise) {
    ortPromise = isNodeRuntime
      ? import('onnxruntime-node')
      : import('onnxruntime-web').then((ort) => {
          ort.env.wasm.wasmPaths = {
            mjs: '/ort/ort-wasm-simd-threaded.mjs',
            wasm: '/ort/ort-wasm-simd-threaded.wasm'
          }
          ort.env.wasm.numThreads = 1
          return ort
        })
  }

  return await ortPromise
}

async function loadSession(modelPath: string) {
  const cacheKey = `${isNodeRuntime ? 'server' : 'client'}:${modelPath}`

  if (!sessionCache.has(cacheKey)) {
    sessionCache.set(cacheKey, (async () => {
      const ort = await getOrt()
      return await ort.InferenceSession.create(modelPath, {
        executionProviders: getExecutionProviders()
      })
    })())
  }

  return await sessionCache.get(cacheKey)!
}

export function useFaceDetector(
  modelPath = DEFAULT_MODEL_PATH,
  modelType: FaceDetectionModelType = 'centerface'
) {
  return {
    async warmup() {
      await Promise.all([
        getOrt(),
        loadSession(modelPath)
      ])
    },

    async detectFaces(
      input: DetectionInput,
      probabilityThreshold = DEFAULT_PROBABILITY_THRESHOLD
    ): Promise<DetectionResult> {
      const startedAt = performance.now()
      const [ort, session] = await Promise.all([
        getOrt(),
        loadSession(modelPath)
      ])
      const inputName = session.inputNames[0]

      if (!inputName) {
        throw new Error('Le modele de detection de visages n a pas de nom d entree.')
      }

      const { width, height } = getPaddedInputSize(input.width, input.height)
      const tensor = new ort.Tensor(
        'float32',
        createModelInputData(input, width, height),
        [1, 3, height, width]
      )
      const outputs = await session.run({ [inputName]: tensor })

      return {
        faces: modelType === 'yunet'
          ? extractFacesFromYunetOutputs(outputs, input, probabilityThreshold)
          : extractFacesFromOutputs(outputs, input, probabilityThreshold),
        durationMs: performance.now() - startedAt
      }
    }
  }
}
