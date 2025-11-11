<template>
  <div class="candle-builds-page">
    <section class="build-form-section">
      <Card class="form-card">
        <template #title>Candle Build from Source</template>
        <template #subtitle>
          Configure a build of <span class="brand">candle-vllm</span> and launch it directly from the UI.
        </template>

        <template #content>
          <form class="build-form surface-section p-4 border-round-lg" @submit.prevent="handleBuild">
            <div class="form-grid">
            <div class="form-field">
              <label for="gitRef">Git Reference</label>
              <InputText
                id="gitRef"
                v-model="form.gitRef"
                placeholder="main"
              />
            </div>

            <div class="form-field">
              <label for="profile">Build Profile</label>
              <Dropdown
                id="profile"
                v-model="form.buildProfile"
                :options="profileOptions"
                optionLabel="label"
                optionValue="value"
              />
            </div>

            <div class="form-field">
              <label>CUDA Backend</label>
              <div class="toggle-row">
                <Checkbox
                  inputId="enableCuda"
                  v-model="form.enableCuda"
                  :binary="true"
                  :disabled="!cudaAvailable"
                  @change="handleCudaToggle"
                />
                <label for="enableCuda">Enable CUDA kernels</label>
              </div>
              <small class="hint">
                {{ capabilityLabel('cuda') }}
              </small>
            </div>

            <div class="form-field" v-if="form.enableCuda">
              <label>CUDA Extensions</label>
              <div class="toggle-vertical">
                <div class="toggle-row">
                  <Checkbox
                    inputId="enableFlash"
                    v-model="form.enableFlashAttention"
                    :binary="true"
                    :disabled="!cudaAvailable"
                  />
                  <label for="enableFlash">Flash Attention kernels</label>
                </div>
                <div class="toggle-row">
                  <Checkbox
                    inputId="enableGraph"
                    v-model="form.enableGraph"
                    :binary="true"
                    :disabled="!cudaAvailable"
                  />
                  <label for="enableGraph">CUDA Graph launch optimizations</label>
                </div>
                <div class="toggle-row">
                  <Checkbox
                    inputId="enableNccl"
                    v-model="form.enableNccl"
                    :binary="true"
                    :disabled="!cudaAvailable"
                  />
                  <label for="enableNccl">NCCL (multi-GPU)</label>
                </div>
                <div class="toggle-row">
                  <Checkbox
                    inputId="enableMarlin"
                    v-model="form.enableMarlin"
                    :binary="true"
                    :disabled="!cudaAvailable"
                  />
                  <label for="enableMarlin">Marlin quant kernels</label>
                </div>
              </div>
            </div>

            <div class="form-field" v-if="metalAvailable">
              <label>Metal Backend</label>
              <div class="toggle-row">
                <Checkbox
                  inputId="enableMetal"
                  v-model="form.enableMetal"
                  :binary="true"
                  :disabled="form.enableCuda"
                  @change="handleMetalToggle"
                />
                <label for="enableMetal">Enable Metal (macOS)</label>
              </div>
              <small class="hint">
                {{ capabilityLabel('metal') }}
              </small>
            </div>

            <div class="form-field">
              <label for="cudaArch">CUDA Architectures</label>
              <InputText
                id="cudaArch"
                v-model="form.cudaArchitectures"
                :disabled="!form.enableCuda"
                placeholder="e.g. 80;86 (optional)"
              />
            </div>

            <div class="form-field span-2">
              <label for="features">Additional Cargo Features</label>
              <InputText
                id="features"
                v-model="form.customFeatures"
                placeholder="Comma separated (optional)"
              />
            </div>

            <div class="form-field span-2">
              <label for="rustflags">Custom RUSTFLAGS</label>
              <Textarea
                id="rustflags"
                v-model="form.customRustflags"
                autoResize
                rows="2"
                placeholder="e.g. -C target-cpu=native"
              />
            </div>

            <div class="form-field span-2">
              <div class="toggle-row">
                <Checkbox
                  inputId="markActive"
                  v-model="form.markActive"
                  :binary="true"
                />
                <label for="markActive">Mark build as active once complete</label>
              </div>
            </div>
          </div>

          <div class="form-actions">
            <Button
              type="submit"
              label="Start Build"
              icon="pi pi-cog"
              :loading="buildStore.building"
            />
          </div>
          </form>
        </template>
      </Card>
    </section>

    <section class="build-progress-section" v-if="progressEntries.length">
      <Card v-for="entry in progressEntries" :key="entry.task_id" class="progress-card">
        <template #title>
          Build Task <code>{{ entry.task_id }}</code>
        </template>
        <template #content>
          <div class="progress-header">
            <span class="stage">{{ entry.stage }}</span>
            <span class="percent">{{ entry.progress }}%</span>
          </div>
          <ProgressBar :value="entry.progress" />
          <pre class="progress-log">{{ entry.message }}</pre>
        </template>
      </Card>
    </section>

    <section class="build-list-section">
      <Card>
        <template #title>Existing Builds</template>
        <template #subtitle>
          Manage compiled binaries and activate your preferred runtime.
        </template>
        <template #content>
          <div v-if="buildStore.loading" class="empty-state">
            <i class="pi pi-spin pi-spinner"></i>
            <span>Loading builds...</span>
          </div>

          <div v-else-if="!buildStore.hasBuilds" class="empty-state">
            <i class="pi pi-box"></i>
            <span>No builds yet. Start one above to create a candle-vllm binary.</span>
          </div>

          <div v-else class="build-table-wrapper">
            <table class="build-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Git Ref</th>
                  <th>Features</th>
                  <th>Binary</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="build in buildStore.builds" :key="build.id">
                  <td>
                    <div class="cell-title">
                      <strong>{{ build.name }}</strong>
                      <small>{{ formatDate(build.created_at) }}</small>
                    </div>
                  </td>
                  <td>{{ build.git_ref || '—' }}</td>
                  <td>
                    <div class="feature-tags" v-if="build.features.length">
                      <Tag
                        v-for="feature in build.features"
                        :key="feature"
                        :value="feature"
                        class="feature-tag"
                      />
                    </div>
                    <span v-else>default</span>
                  </td>
                  <td>
                    <div class="binary-status">
                      <i
                        :class="build.has_binary ? 'pi pi-check-circle text-green-500' : 'pi pi-times-circle text-red-500'"
                      ></i>
                      <span>{{ build.has_binary ? 'Available' : 'Missing' }}</span>
                    </div>
                    <small v-if="build.binary_path">{{ build.binary_path }}</small>
                  </td>
                  <td>
                    <Tag
                      :severity="build.is_active ? 'success' : 'secondary'"
                      :value="build.is_active ? 'Active' : 'Inactive'"
                    />
                  </td>
                  <td class="actions">
                    <Button
                      icon="pi pi-check"
                      label="Activate"
                      size="small"
                      severity="success"
                      class="mr-2"
                      @click="activate(build)"
                      :disabled="build.is_active"
                    />
                    <Button
                      icon="pi pi-trash"
                      label="Delete"
                      size="small"
                      severity="danger"
                      outlined
                      @click="remove(build)"
                    />
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </template>
      </Card>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, reactive } from 'vue'
import { toast } from 'vue3-toastify'
import Card from 'primevue/card'
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'
import Checkbox from 'primevue/checkbox'
import Dropdown from 'primevue/dropdown'
import Textarea from 'primevue/textarea'
import Tag from 'primevue/tag'
import ProgressBar from 'primevue/progressbar'

import { useBuildStore } from '@/stores/builds'
import { useWebSocketStore } from '@/stores/websocket'

const buildStore = useBuildStore()
const wsStore = useWebSocketStore()

const form = reactive({
  gitRef: 'master',
  buildProfile: 'release',
  enableCuda: false,
  enableMetal: false,
  enableNccl: false,
  enableFlashAttention: false,
  enableGraph: false,
  enableMarlin: false,
  cudaArchitectures: '',
  customFeatures: '',
  customRustflags: '',
  markActive: false
})

const profileOptions = [
  { label: 'Release (recommended)', value: 'release' },
  { label: 'Debug', value: 'debug' }
]

const capabilityLabel = (key) => {
  return buildStore.capabilities?.[key]?.reason || 'Capability unavailable'
}

const cudaAvailable = computed(() => !!buildStore.capabilities?.cuda?.available)
const metalAvailable = computed(() => !!buildStore.capabilities?.metal?.available)

const progressEntries = computed(() => Object.values(buildStore.progress))

const handleCudaToggle = () => {
  if (!form.enableCuda) {
    form.enableFlashAttention = false
    form.enableNccl = false
    form.enableGraph = false
    form.enableMarlin = false
  } else {
    form.enableMetal = false
  }
}

const handleMetalToggle = () => {
  if (form.enableMetal) {
    form.enableCuda = false
    form.enableFlashAttention = false
    form.enableNccl = false
    form.enableGraph = false
    form.enableMarlin = false
  }
}

const parseCustomFeatures = () => {
  if (!form.customFeatures) return []
  return form.customFeatures
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

const handleBuild = async () => {
  if (!form.enableCuda && !form.enableMetal) {
    toast.error('Select CUDA or Metal backend before building.')
    return
  }
  try {
    const payload = {
      git_ref: form.gitRef || 'master',
      build_profile: form.buildProfile,
      enable_cuda: !!form.enableCuda,
      enable_metal: !!form.enableMetal,
      enable_nccl: !!form.enableNccl,
      enable_flash_attention: !!form.enableFlashAttention,
      enable_graph: !!form.enableGraph,
      enable_marlin: !!form.enableMarlin,
      cuda_architectures: form.enableCuda ? form.cudaArchitectures || '' : '',
      custom_features: parseCustomFeatures(),
      custom_rustflags: form.customRustflags || '',
      mark_active: !!form.markActive
    }

    const response = await buildStore.startBuild(payload)
    if (response?.task_id) {
      buildStore.updateProgress({
        task_id: response.task_id,
        stage: 'queued',
        progress: 0,
        message: 'Build enqueued...'
      })
      toast.info(`Build queued (task ${response.task_id})`)
    }
  } catch (error) {
    console.error('Failed to start build', error)
    toast.error(error?.response?.data?.detail || 'Failed to start build')
  }
}

const remove = async (build) => {
  const confirmDelete = window.confirm(`Delete build "${build.name}"?`)
  if (!confirmDelete) return
  await buildStore.deleteBuild(build.id)
  toast.success(`Deleted build "${build.name}"`)
}

const activate = async (build) => {
  await buildStore.activateBuild(build.id)
  toast.success(`Activated build "${build.name}"`)
}

const formatDate = (value) => {
  if (!value) return '—'
  return new Date(value).toLocaleString()
}

let unsubscribe = null

onMounted(async () => {
  await Promise.all([buildStore.fetchBuilds(), buildStore.fetchCapabilities()])
  unsubscribe = wsStore.subscribeToBuildProgress(async (message) => {
    buildStore.updateProgress(message)
    if (message.stage === 'complete') {
      await buildStore.fetchBuilds()
      setTimeout(() => buildStore.removeProgress(message.task_id), 5000)
    }
  })
})

onUnmounted(() => {
  if (unsubscribe) {
    unsubscribe()
  }
})
</script>

<style scoped>
.candle-builds-page {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  padding-bottom: 4rem;
}

.build-form {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  margin-top: 1rem;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1.5rem;
}

.form-field {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-field label {
  font-weight: 600;
  color: var(--text-primary);
}

.toggle-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.toggle-vertical {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.hint {
  font-size: 0.75rem;
  color: var(--text-secondary);
}

.span-2 {
  grid-column: span 2;
}

.form-actions {
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
}

.build-progress-section {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1rem;
  background: var(--surface-ground);
  border: 1px solid var(--surface-border);
  padding: 1rem;
  border-radius: var(--radius-lg);
}

.progress-card {
  background: var(--surface-card);
  border: 1px solid var(--surface-border);
}

.progress-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
  font-weight: 600;
  color: var(--text-primary);
}

.progress-log {
  margin-top: 0.75rem;
  padding: 0.75rem;
  background: var(--surface-ground);
  border-radius: var(--radius-md);
  font-family: var(--font-mono);
  font-size: 0.8rem;
  white-space: pre-wrap;
  max-height: 120px;
  overflow: auto;
  border: 1px solid var(--surface-border);
}

.build-table-wrapper {
  overflow-x: auto;
}

.build-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 1rem;
}

.build-table th,
.build-table td {
  padding: 0.75rem;
  text-align: left;
  border-bottom: 1px solid var(--surface-border);
}

.build-table thead {
  background: var(--surface-ground);
}

.cell-title {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.feature-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.feature-tag {
  font-size: 0.75rem;
}

.binary-status {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  color: var(--text-secondary);
}

.actions {
  display: flex;
  gap: 0.5rem;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 3rem 1rem;
  color: var(--text-secondary);
}

.brand {
  font-weight: 600;
  color: var(--accent-primary);
}

@media (max-width: 768px) {
  .span-2 {
    grid-column: span 1;
  }
  .form-actions {
    justify-content: stretch;
  }
  .actions {
    flex-direction: column;
  }
}
</style>

