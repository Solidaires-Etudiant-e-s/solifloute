<script setup lang="ts">
useHead({
  meta: [
    { name: 'viewport', content: 'width=device-width, initial-scale=1' },
    { name: 'theme-color', content: '#ffffff' }
  ],
  link: [
    { rel: 'icon', type: 'image/png', href: '/favicon.png' },
    { rel: 'apple-touch-icon', sizes: '180x180', href: '/apple-touch-icon.png' }
  ],
  htmlAttrs: {
    lang: 'fr'
  }
})

const title = 'Solifloute — Floutez les visages de vos photos et vidéos'
const description = 'Floutez automatiquement les visages de vos photos et vidéos en quelques clics. Simple, gratuit et confidentiel : vos fichiers restent privés, sans collecte ni partage de vos données.'

useSeoMeta({
  title,
  description,
  ogTitle: title,
  ogDescription: description,
  ogImage: '/icon-512.png',
  ogType: 'website',
  ogLocale: 'fr_FR',
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
