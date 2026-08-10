export default defineNuxtConfig({
  modules: [
    '@nuxt/eslint',
    '@nuxt/ui'
  ],

  devtools: {
    enabled: true
  },

  css: ['~/assets/css/main.css'],

  compatibilityDate: '2025-01-15',

  nitro: {
    experimental: {
      wasm: true
    },
    externals: {
      external: ['bun:sqlite']
    }
  },

  vite: {
    optimizeDeps: {
      exclude: ['onnxruntime-web', 'onnxruntime-node']
    },
    worker: {
      format: 'es'
    },
    server: {
      allowedHosts: ['localhost', '8b83-147-210-85-232.ngrok-free.app']
    }
  },

  eslint: {
    config: {
      stylistic: {
        commaDangle: 'never',
        braceStyle: '1tbs'
      }
    }
  }
})
