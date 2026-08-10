<script setup lang="ts">
useHead({
  meta: [
    { name: 'viewport', content: 'width=device-width, initial-scale=1' }
  ],
  htmlAttrs: {
    lang: 'fr'
  }
})

const title = 'Solifloute'
const description = 'Détectez, sélectionnez et floutez des visages avec un pipeline ONNX partagé qui fonctionne dans le navigateur ou sur le serveur.'

useSeoMeta({
  title,
  description,
  ogTitle: title,
  ogDescription: description,
  twitterCard: 'summary_large_image'
})

onMounted(() => {
  if (import.meta.env.PROD && 'serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch(() => {
      // Service worker availability is best-effort.
    })
  }
})
</script>

<template>
  <UApp>
    <NuxtPage />
  </UApp>
</template>
