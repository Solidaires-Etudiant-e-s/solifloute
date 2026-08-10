/* SoliFloute service worker — caches the heavy static assets used for
   browser-side face detection and video processing (ONNX runtime, the face
   detection model, FFmpeg core, fonts) so repeated sessions are fast and
   resilient to network issues. */
const VERSION = 'solifloute-v3'
const HEAVY_STATIC_PREFIXES = ['/models/', '/ort/', '/ffmpeg/']

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(VERSION)
    await Promise.allSettled([
      cache.add('/Scotchlidaires.ttf'),
      cache.add('/models/centerface.onnx'),
      cache.add('/ort/ort-wasm-simd-threaded.mjs'),
      cache.add('/ort/ort-wasm-simd-threaded.wasm'),
      cache.add('/ffmpeg/ffmpeg-core.js'),
      cache.add('/ffmpeg/ffmpeg-core.wasm')
    ])
    await self.skipWaiting()
  })())
})

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys()

    await Promise.all(
      keys.filter(key => key !== VERSION).map(key => caches.delete(key))
    )
    await self.clients.claim()
  })())
})

self.addEventListener('fetch', (event) => {
  const request = event.request

  if (request.method !== 'GET') {
    return
  }

  const url = new URL(request.url)

  if (url.origin !== self.location.origin || url.protocol !== 'http:' && url.protocol !== 'https:' || url.pathname.startsWith('/api/')) {
    return
  }

  const isHeavyStatic = HEAVY_STATIC_PREFIXES.some(prefix => url.pathname.startsWith(prefix))

  event.respondWith((async () => {
    const cache = await caches.open(VERSION)

    if (isHeavyStatic) {
      const cached = await cache.match(request)

      if (cached) {
        return cached
      }

      const response = await fetch(request)

      if (response.ok) {
        cache.put(request, response.clone())
      }

      return response
    }

    try {
      const response = await fetch(request)

      if (response.ok && url.pathname.startsWith('/')) {
        cache.put(request, response.clone())
      }

      return response
    } catch (error) {
      const cached = await cache.match(request)

      if (cached) {
        return cached
      }

      throw error
    }
  })())
})
