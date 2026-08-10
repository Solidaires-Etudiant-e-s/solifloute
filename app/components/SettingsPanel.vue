<script setup lang="ts">
import { computed } from 'vue'
import type { EditorSettings, ProcessingMode } from '~~/shared/types/faces'

const settings = defineModel<EditorSettings>({ required: true })

const props = defineProps<{
  activeMode: Exclude<ProcessingMode, 'auto'>
  serverOnly: boolean
}>()

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
  if (props.serverOnly) {
    return [{ label: 'Serveur', icon: 'mingcute:server-line', value: 'server' }]
  }

  return [
    { label: 'Navigateur', icon: 'mingcute:earth-line', value: 'client' },
    { label: 'Serveur', icon: 'mingcute:server-line', value: 'server' }
  ]
})

const processingMode = computed({
  get: () => props.activeMode,
  set: (value: string | number) => {
    settings.value.processingMode = value === 'server' ? 'server' : 'client'
  }
})

function normalizeSliderValue(value: number | number[]) {
  const nextValue = Array.isArray(value) ? value[0] : value

  return typeof nextValue === 'number' && Number.isFinite(nextValue) ? nextValue : 0
}

const detectionSensitivity = computed({
  get: () => Number((1 - settings.value.confidenceThreshold).toFixed(2)),
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

      <div class="space-y-2">
        <div class="flex items-center justify-between text-sm">
          <span>Sensibilité de détection</span>
          <span class="font-semibold">{{ detectionSensitivity.toFixed(2) }}</span>
        </div>
        <USlider
          v-model="detectionSensitivity"
          :min="0"
          :max="1"
          :step="0.01"
        />
      </div>
    </div>
  </UCard>
</template>
