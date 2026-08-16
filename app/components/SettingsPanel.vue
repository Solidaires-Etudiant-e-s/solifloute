<script setup lang="ts">
import { computed } from 'vue'
import type { EditorSettings, ProcessingMode } from '~~/shared/types/faces'

const settings = defineModel<EditorSettings>({ required: true })

const props = defineProps<{
  activeMode: Exclude<ProcessingMode, 'auto'>
  serverOnly: boolean
  showCloud: boolean
  initialEstimatedMs: number | null
}>()

function formatEstimatedTime(estimatedMs: number) {
  const totalSeconds = Math.max(1, Math.round(estimatedMs / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60

  if (minutes === 0) {
    return `~${seconds}s`
  }

  return `~${minutes} min ${String(seconds).padStart(2, '0')}s`
}

const modelItems = [
  { label: 'Rapide', icon: 'mingcute:flash-fill', value: 'fast' },
  { label: 'Avancé', icon: 'mingcute:sparkles-2-fill', value: 'advanced' }
]

const modelHint = 'Rapide : environ 4x plus rapide, mais ~20% moins precis.'

const detectionModel = computed({
  get: () => settings.value.detectionModel,
  set: (value: string | number) => {
    settings.value.detectionModel = value === 'fast' ? 'fast' : 'advanced'
  }
})

const processingItems = computed(() => {
  const items: Array<{ label: string, icon: string, value: 'client' | 'server' | 'cloud' }> = []

  if (!props.serverOnly) {
    items.push({ label: 'Navigateur', icon: 'mingcute:earth-line', value: 'client' })
  }

  items.push({ label: 'Serveur', icon: 'mingcute:server-line', value: 'server' })

  if (props.showCloud) {
    items.push({ label: 'Cloud', icon: 'mingcute:cloud-line', value: 'cloud' })
  }

  return items
})

const isCloudMode = computed(() => props.activeMode === 'cloud')

const processingMode = computed({
  get: () => props.activeMode,
  set: (value: string | number) => {
    if (value === 'server' || value === 'cloud') {
      settings.value.processingMode = value
    } else {
      settings.value.processingMode = 'client'
    }
  }
})

function normalizeSliderValue(value: number | number[]) {
  const nextValue = Array.isArray(value) ? value[0] : value

  return typeof nextValue === 'number' && Number.isFinite(nextValue) ? nextValue : 0
}

const detectionSensitivity = computed({
  get: () => Number(Math.min(0.9, 1 - settings.value.confidenceThreshold).toFixed(2)),
  set: (value: number | number[]) => {
    const nextValue = normalizeSliderValue(value)
    settings.value.confidenceThreshold = Number((1 - nextValue).toFixed(2))
  }
})
</script>

<template>
  <UCard class="border-(--ui-border) bg-(--ui-bg-muted)">
    <template #header>
      <div>
        <h2 class="text-2xl">
          Parametres de floutage
        </h2>
      </div>
    </template>

    <div class="space-y-6">
      <UFormField>
        <template #label>
          <div class="flex items-center gap-1.5">
            <span>Modele de detection</span>
            <UTooltip :text="modelHint">
              <UIcon
                name="mingcute:information-line"
                class="size-4 cursor-help text-(--ui-text-dimmed)"
              />
            </UTooltip>
          </div>
        </template>

        <UTabs
          v-model="detectionModel"
          :items="modelItems"
          :content="false"
          variant="pill"
        />
      </UFormField>

      <UFormField label="Cible de traitement">
        <UTabs
          v-model="processingMode"
          :items="processingItems"
          :content="false"
          variant="pill"
        />
      </UFormField>

      <p
        v-if="initialEstimatedMs !== null"
        class="text-sm text-(--ui-text-dimmed)"
      >
        Temps de traitement estimé : {{ formatEstimatedTime(initialEstimatedMs) }}
      </p>

      <UAlert
        v-if="isCloudMode"
        color="primary"
        variant="soft"
        title="Traitement Cloud via Modal"
        icon="mingcute:cloud-line"
        description="Votre vidéo est transmise et traitée auprès d'un service externe. Bien que celui-ci assure une politique de non-rétention des données, le mode cloud est à éviter pour les vidéos extrêmement sensibles."
      />

      <div class="space-y-2">
        <div class="flex items-center justify-between text-sm">
          <span>Sensibilité de détection</span>
          <span class="font-semibold">{{ detectionSensitivity.toFixed(2) }}</span>
        </div>
        <USlider
          v-model="detectionSensitivity"
          :min="0"
          :max="0.9"
          :step="0.01"
        />
      </div>
    </div>
  </UCard>
</template>
