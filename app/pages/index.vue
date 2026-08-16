<script setup lang="ts">
const inputRef = ref<HTMLInputElement | null>(null)
const editor = useImageEditor()
const entries = computed(() => editor.uploadEntries.value)

function formatRemainingTime(remainingMs: number | null) {
  if (remainingMs === null) {
    return ''
  }

  const totalSeconds = Math.max(1, Math.round(remainingMs / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60

  if (minutes === 0) {
    return `${seconds}s restantes`
  }

  return `${minutes} min ${String(seconds).padStart(2, '0')}s restantes`
}

function formatDuration(durationMs: number | null) {
  if (durationMs === null) {
    return ''
  }

  const totalSeconds = durationMs / 1000

  if (totalSeconds < 10) {
    return `${totalSeconds.toFixed(1)}s`
  }

  if (totalSeconds < 60) {
    return `${Math.round(totalSeconds)}s`
  }

  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.round(totalSeconds % 60)

  if (seconds === 60) {
    return `${minutes + 1} min`
  }

  if (seconds === 0) {
    return `${minutes} min`
  }

  return `${minutes} min ${String(seconds).padStart(2, '0')}s`
}

function downloadName(entry: { mediaKind: 'image' | 'video', fileName: string }) {
  const baseName = entry.fileName.replace(/\.[^/.]+$/, '')
  const extension = entry.mediaKind === 'video' ? 'mp4' : 'png'

  return `visages-floutes-${baseName}.${extension}`
}

async function onFileChange(event: Event) {
  const target = event.target as HTMLInputElement
  const nextFiles = Array.from(target.files || [])

  if (nextFiles.length === 0) {
    return
  }

  for (const nextFile of nextFiles) {
    try {
      await editor.loadFile(nextFile)
    } catch {
      // Best-effort: keep importing the remaining media.
    }
  }

  target.value = ''
}

async function onDrop(event: DragEvent) {
  const nextFiles = Array.from(event.dataTransfer?.files || [])

  for (const nextFile of nextFiles) {
    try {
      await editor.loadFile(nextFile)
    } catch {
      // Best-effort: keep importing the remaining media.
    }
  }
}

function openPicker() {
  inputRef.value?.click()
}

function isCurrentEntry(entryId: string) {
  return editor.currentEntryId.value === entryId
}
</script>

<template>
  <main
    class="mx-auto flex min-h-screen max-w-7xl flex-col gap-8 px-4 py-8 sm:px-6 lg:px-8"
    @dragenter.prevent
    @dragover.prevent
    @drop.prevent="onDrop"
  >
    <UModal
      v-model:open="editor.safariVideoModalOpen.value"
      title="Safari, c'est guez"
      :dismissible="true"
      :ui="{ footer: 'justify-end' }"
    >
      <template #body>
        <p class="text-sm text-muted">
          Safari est nul pour l encodage video dans le navigateur. Pour les videos, Solifloute force donc le mode serveur sur Safari.
        </p>
        <p class="mt-3 text-sm text-muted">
          Si vous voulez vraiment le mode navigateur, utilisez Chrome ou Firefox. Sinon, restez en mode serveur ici.
        </p>
      </template>

      <template #footer>
        <UButton
          color="neutral"
          variant="outline"
          @click="editor.closeSafariVideoModal"
        >
          Rester en mode serveur
        </UButton>
        <UButton
          href="https://www.google.com/chrome/"
          target="_blank"
          rel="noreferrer"
          color="primary"
        >
          Utiliser Chrome
        </UButton>
      </template>
    </UModal>

    <section class="border border-default bg-muted pt-8 p-6">
      <div class="grid gap-6 lg:grid-cols-2 lg:items-center">
        <div class="flex flex-col items-start gap-6">
          <h1 class="text-5xl leading-none sm:text-6xl">
            SoliFloute
          </h1>

          <div class="flex flex-wrap items-center gap-3">
            <UButton
              size="xl"
              color="primary"
              @click="openPicker"
            >
              Importer un media
            </UButton>
          </div>
        </div>

        <p class="max-w-2xl text-md text-muted">
          Les vidéos sont conservées un maximum de 24 heures. Le traitement est effectué dans votre navigateur, sur notre serveur ou, si vous choisissez le Cloud, chez un fournisseur externe (Modal). Le navigateur et le serveur ne collectent et ne partagent aucune donnée.
        </p>
      </div>

      <input
        ref="inputRef"
        type="file"
        accept="image/*,video/*"
        multiple
        class="hidden"
        @change="onFileChange"
      >
    </section>

    <section
      v-if="entries.length === 0"
      class="border border-dashed border-default bg-muted p-8 text-center text-muted"
    >
      Veuillez importer un media.
    </section>

    <section
      v-for="entry in entries"
      :key="entry.id"
      class="grid gap-6 xl:grid-cols-2"
    >
      <UCard>
        <template #header>
          <div class="flex items-center justify-between gap-3">
            <div>
              <h2 class="text-2xl">
                {{ entry.mediaKind === 'video' ? 'Apercu source' : 'Overlay de detection' }}
              </h2>
              <p class="text-xs text-(--ui-text-dimmed)">
                {{ entry.fileName }}
              </p>
            </div>

            <UBadge
              v-if="entry.mediaKind === 'image' && entry.lastDurationMs !== null"
              color="neutral"
              variant="subtle"
            >
              {{ formatDuration(entry.lastDurationMs) }}
            </UBadge>
          </div>
        </template>

        <div class="space-y-4">
          <FaceOverlay
            v-if="entry.mediaKind === 'image'"
            :src="entry.originalPreviewUrl"
            :faces="entry.faces"
            :excluded-face-ids="entry.settings.excludedFaceIds"
            @toggle="(faceId) => isCurrentEntry(entry.id) && editor.toggleExcludedFace(faceId)"
            @create="(face) => isCurrentEntry(entry.id) && editor.addManualFace(face)"
          />

          <video
            v-else
            :src="entry.originalPreviewUrl"
            controls
            class="block w-full border border-(--ui-border)"
          />

          <p
            v-if="entry.warning"
            class="text-sm text-(--ui-text-toned)"
          >
            {{ entry.warning }}
          </p>

          <p class="text-sm text-(--ui-text-muted)">
            {{ entry.mediaKind === 'image' ? 'Cliquez sur un cadre pour l exclure du floutage, ou tracez une zone manuelle directement sur l image.' : '' }}
          </p>
        </div>
      </UCard>

      <UCard>
        <template #header>
          <div class="flex items-center justify-between gap-3">
            <h2 class="text-2xl">
              Apercu traite
            </h2>

            <div class="flex items-center gap-3">
              <UBadge
                v-if="entry.mediaKind === 'video' && entry.lastDurationMs !== null"
                color="neutral"
                variant="subtle"
              >
                {{ formatDuration(entry.lastDurationMs) }}
              </UBadge>

              <UButton
                v-if="entry.processedPreviewUrl"
                color="neutral"
                variant="outline"
                @click="editor.retryEntry(entry.id)"
              >
                Recalculer
              </UButton>

              <UButton
                v-if="entry.processedPreviewUrl"
                :href="entry.processedPreviewUrl"
                :download="downloadName(entry)"
                color="primary"
                variant="soft"
              >
                Telecharger
              </UButton>

              <UButton
                v-if="entry.status === 'error'"
                color="neutral"
                variant="outline"
                @click="editor.retryEntry(entry.id)"
              >
                Reessayer
              </UButton>
            </div>
          </div>
        </template>

        <SettingsPanel
          v-if="entry.status !== 'processing'"
          v-model="entry.settings"
          :active-mode="editor.activeModeForEntry(entry.id)"
          :server-only="editor.isEntryServerOnly(entry.id)"
          :show-cloud="entry.mediaKind === 'video'"
          :initial-estimated-ms="editor.entryInitialEstimates.get(entry.id) ?? null"
        />

        <div
          v-if="entry.processedPreviewUrl"
          class="mt-4 space-y-4"
        >
          <img
            v-if="entry.mediaKind === 'image'"
            :src="entry.processedPreviewUrl"
            alt="Apercu de l image traitee"
            class="block w-full border border-(--ui-border)"
          >

          <video
            v-else
            :src="entry.processedPreviewUrl"
            controls
            class="block w-full border border-(--ui-border)"
          />
        </div>

        <div
          v-else
          class="space-y-3 border border-dashed border-(--ui-border) bg-(--ui-bg-muted) p-6"
        >
          <div
            v-if="entry.processingProgress !== null"
            class="space-y-2"
          >
            <p
              v-if="entry.processingMessage"
              class="text-sm font-medium text-(--ui-text)"
            >
              {{ entry.processingMessage }}
            </p>
            <div class="h-2 w-full overflow-hidden bg-(--ui-bg-elevated)">
              <div
                v-if="entry.processingQueued"
                class="h-full w-full animate-pulse bg-(--color-solired-500)"
              />
              <div
                v-else
                class="h-full bg-(--color-solired-500) transition-all"
                :style="{ width: `${Math.round(entry.processingProgress * 100)}%` }"
              />
            </div>
            <p
              v-if="!entry.processingQueued"
              class="text-sm text-(--ui-text-muted)"
            >
              Progression : {{ Math.round(entry.processingProgress * 100) }}%
            </p>
            <p
              v-if="entry.estimatedRemainingMs !== null"
              class="text-sm text-(--ui-text-dimmed)"
            >
              Temps restant estimé : {{ formatRemainingTime(entry.estimatedRemainingMs) }}
            </p>
          </div>

          <p
            v-if="entry.error"
            class="text-sm text-(--ui-text-toned)"
          >
            {{ entry.error }}
          </p>

          <div class="flex justify-center gap-3 pt-4">
            <UButton
              v-if="entry.isUploading"
              disabled
              color="primary"
              variant="soft"
              :loading="true"
            >
              Envoi en cours...
            </UButton>
            <UButton
              v-else-if="entry.status !== 'detecting' && entry.status !== 'processing'"
              color="primary"
              @click="editor.processEntry(entry.id)"
            >
              Traiter le media
            </UButton>
            <UButton
              v-if="entry.status === 'processing' && entry.mediaKind === 'video'"
              color="neutral"
              variant="outline"
              @click="editor.cancelEntryProcessing(entry.id)"
            >
              Annuler
            </UButton>
          </div>
        </div>
      </UCard>
    </section>
  </main>
</template>
