<template>
  <Dialog 
    :visible="visible" 
    @update:visible="$emit('update:visible', $event)"
    :modal="true"
    :style="{ width: '600px' }"
    :header="`Preview ${type} Changes`"
    :dismissableMask="true"
    class="config-preview-dialog"
    aria-label="Configuration change preview"
    @touchstart="handleTouchStart"
    @touchmove="handleTouchMove"
    @touchend="handleTouchEnd"
  >
    <div class="preview-content">
      <div class="preview-header">
        <i class="pi pi-info-circle" aria-hidden="true"></i>
        <p>{{ type === 'preset' ? `${presetName} preset will change:` : 'Smart Auto will change:' }}</p>
      </div>

      <div class="changes-list">
        <div 
          v-for="change in changes" 
          :key="change.field"
          class="change-item"
          :class="{ 'has-impact': change.impact }"
        >
          <div class="change-field">
            <strong>{{ change.field }}</strong>
          </div>
          <div class="change-values">
            <span class="value-before">{{ formatValue(change.before) }}</span>
            <i class="pi pi-arrow-right" aria-hidden="true"></i>
            <span class="value-after">{{ formatValue(change.after) }}</span>
          </div>
          <div v-if="change.description" class="change-description">
            {{ change.description }}
          </div>
        </div>
      </div>

      <div v-if="impact" class="impact-preview">
        <h4>Expected Impact:</h4>
        <div class="impact-items">
          <div v-if="impact.performance" class="impact-item performance">
            <i class="pi pi-chart-line" aria-hidden="true"></i>
            <span>{{ impact.performance }}</span>
          </div>
          <div v-if="impact.vram" class="impact-item vram">
            <i class="pi pi-memory" aria-hidden="true"></i>
            <span>{{ impact.vram }}</span>
          </div>
          <div v-if="impact.ram" class="impact-item ram">
            <i class="pi pi-server" aria-hidden="true"></i>
            <span>{{ impact.ram }}</span>
          </div>
        </div>
      </div>

      <div class="preview-warning" v-if="hasWarnings">
        <i class="pi pi-exclamation-triangle" aria-hidden="true"></i>
        <p>Some changes may affect memory usage. Review the impact above.</p>
      </div>

      <div v-if="planSummary" class="plan-section">
        <h4>Topology Plan</h4>
        <div class="plan-summary-grid">
          <div>
            <span class="metric-label">Strategy</span>
            <span class="metric-value">{{ planSummary.tp_strategy === 'tensor_parallel' ? 'Tensor Parallel' : 'Replica' }}</span>
          </div>
          <div>
            <span class="metric-label">KV Pool</span>
            <span class="metric-value">{{ formatMb(planSummary.total_kv_pool_mb) }}</span>
          </div>
          <div>
            <span class="metric-label">Recommended Concurrency</span>
            <span class="metric-value">{{ planSummary.recommended_max_num_seqs }}</span>
          </div>
          <div>
            <span class="metric-label">Prefill Chunk</span>
            <span class="metric-value">{{ planSummary.prefill_chunk_size }}</span>
          </div>
          <div>
            <span class="metric-label">NVLink Only</span>
            <span class="metric-value">{{ planSummary.restrict_to_nvlink ? 'Yes' : 'No' }}</span>
          </div>
        </div>
        <div v-if="planSummary.warnings?.length" class="plan-warnings">
          <i class="pi pi-exclamation-circle"></i>
          <ul>
            <li v-for="(warning, idx) in planSummary.warnings" :key="idx">{{ warning }}</li>
          </ul>
        </div>
        <div class="plan-groups">
          <div v-for="(group, idx) in planGroups" :key="group.id || idx" class="plan-group-card">
            <div class="plan-group-header">
              <div>
                <h5>Group {{ idx + 1 }} · {{ group.link_type === 'nvlink' ? 'NVLink' : 'PCIe' }}</h5>
                <p>TP Degree: {{ group.tp_degree }} · KV Pool: {{ formatMb(group.kv_pool_mb) }}</p>
              </div>
              <div v-if="group.expected_penalty_pct" class="plan-group-penalty">
                Penalty ~{{ group.expected_penalty_pct }}%
              </div>
            </div>
            <table class="plan-device-table">
              <thead>
                <tr>
                  <th>GPU</th>
                  <th>Weights</th>
                  <th>Activations</th>
                  <th>KV Cache</th>
                  <th>Safety</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="device in group.devices" :key="device.index">
                  <td>GPU{{ device.index }}</td>
                  <td>{{ formatMb(device.weights_mb) }}</td>
                  <td>{{ formatMb(device.activations_mb) }}</td>
                  <td>{{ formatMb(device.kv_budget_mb) }}</td>
                  <td>{{ formatMb(device.safety_margin_mb) }}</td>
                </tr>
              </tbody>
            </table>
            <div v-if="group.warnings?.length" class="plan-warnings group-plan-warnings">
              <i class="pi pi-exclamation-triangle"></i>
              <ul>
                <li v-for="(warning, wIdx) in group.warnings" :key="wIdx">{{ warning }}</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div v-if="debug" class="debug-section">
        <h4>Smart Auto Rationale</h4>
        <div class="debug-summary">
          <div class="debug-card">
            <h5>Inputs</h5>
            <ul class="debug-list">
              <li>
                <strong>Slider</strong>
                <span>{{ debug.slider_value }}</span>
                <span v-if="debug.adjusted_slider !== undefined && debug.adjusted_slider !== debug.slider_value" class="muted">
                  (adjusted {{ debug.adjusted_slider }})
                </span>
              </li>
              <li v-for="([key, value]) in overrideEntries" :key="`override-${key}`">
                <strong>{{ formatLabel(key) }}</strong>
                <span>{{ formatValue(value) }}</span>
              </li>
            </ul>
          </div>
          <div class="debug-card" v-if="hardwareEntries.length">
            <h5>Hardware Snapshot</h5>
            <ul class="debug-list">
              <li v-for="([key, value]) in hardwareEntries" :key="`hw-${key}`">
                <strong>{{ formatLabel(key) }}</strong>
                <span>{{ formatValue(value) }}</span>
              </li>
            </ul>
          </div>
          <div class="debug-card" v-if="modelEntries.length">
            <h5>Model Profile</h5>
            <ul class="debug-list">
              <li v-for="([key, value]) in modelEntries" :key="`model-${key}`">
                <strong>{{ formatLabel(key) }}</strong>
                <span>{{ formatValue(value) }}</span>
              </li>
            </ul>
          </div>
          <div class="debug-card" v-if="precisionEntries.length">
            <h5>Precision</h5>
            <ul class="debug-list">
              <li v-for="([key, value]) in precisionEntries" :key="`precision-${key}`">
                <strong>{{ formatLabel(key) }}</strong>
                <span>{{ formatValue(value) }}</span>
              </li>
            </ul>
          </div>
        </div>

        <div class="decision-list" v-if="decisionEntries.length">
          <h5>Key Decisions</h5>
          <div
            class="decision-item"
            v-for="decision in decisionEntries"
            :key="decision.key"
          >
            <div class="decision-header">
              <strong>{{ formatLabel(decision.key) }}</strong>
              <span>{{ formatValue(decision.value) }}</span>
            </div>
            <p class="decision-reason">{{ decision.reason }}</p>
            <ul v-if="decision.notes.length" class="decision-notes">
              <li v-for="(note, index) in decision.notes" :key="index">{{ note }}</li>
            </ul>
            <div v-if="hasContext(decision.context)" class="decision-context">
              <div
                v-for="(contextValue, contextKey) in decision.context"
                :key="`${decision.key}-${contextKey}`"
              >
                <small>{{ formatLabel(contextKey) }}</small>
                <span>{{ formatValue(contextValue) }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <template #footer>
      <Button 
        label="Cancel" 
        icon="pi pi-times" 
        @click="$emit('cancel')"
        severity="secondary"
        outlined
        aria-label="Cancel configuration changes"
      />
      <Button 
        label="Apply Changes" 
        icon="pi pi-check" 
        @click="$emit('apply')"
        :loading="applying"
        aria-label="Apply configuration changes"
      />
    </template>
  </Dialog>
</template>

<script setup>
import { computed, ref } from 'vue'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  type: {
    type: String,
    default: 'smart-auto' // 'smart-auto' or 'preset'
  },
  presetName: {
    type: String,
    default: ''
  },
  changes: {
    type: Array,
    default: () => []
  },
  impact: {
    type: Object,
    default: null
  },
  debug: {
    type: Object,
    default: null
  },
  plan: {
    type: Object,
    default: null
  },
  applying: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update:visible', 'apply', 'cancel'])

// Touch gesture handling for swipe to dismiss
const touchStartX = ref(0)
const touchStartY = ref(0)
const touchThreshold = 50

const handleTouchStart = (e) => {
  if (e.touches && e.touches.length > 0) {
    touchStartX.value = e.touches[0].clientX
    touchStartY.value = e.touches[0].clientY
  }
}

const handleTouchMove = (e) => {
  if (e.touches && e.touches.length > 0) {
    const deltaX = e.touches[0].clientX - touchStartX.value
    const deltaY = e.touches[0].clientY - touchStartY.value
    
    if (deltaY > touchThreshold && Math.abs(deltaX) < Math.abs(deltaY)) {
      e.preventDefault()
    }
  }
}

const handleTouchEnd = (e) => {
  if (e.changedTouches && e.changedTouches.length > 0) {
    const deltaX = e.changedTouches[0].clientX - touchStartX.value
    const deltaY = e.changedTouches[0].clientY - touchStartY.value
    
    if (deltaY > touchThreshold && Math.abs(deltaX) < Math.abs(deltaY)) {
      emit('cancel')
    }
  }
  
  touchStartX.value = 0
  touchStartY.value = 0
}

const formatValue = (value) => {
  if (value === null || value === undefined) return 'Not set'
  if (typeof value === 'boolean') return value ? 'Enabled' : 'Disabled'
  if (typeof value === 'number') {
    if (value >= 1000 && value < 1000000) return `${(value / 1000).toFixed(1)}K`
    if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`
    return value.toString()
  }
  return value.toString()
}

const formatLabel = (value) => {
  if (!value) return ''
  return value
    .toString()
    .replace(/[_\-]/g, ' ')
    .replace(/\b\w/g, char => char.toUpperCase())
}

const hardwareEntries = computed(() => {
  if (!props.debug?.hardware) return []
  return Object.entries(props.debug.hardware)
})

const modelEntries = computed(() => {
  if (!props.debug?.model) return []
  return Object.entries(props.debug.model)
})

const precisionEntries = computed(() => {
  if (!props.debug?.precision) return []
  return Object.entries(props.debug.precision)
})

const overrideEntries = computed(() => {
  const overrides = props.debug?.requested_overrides
  if (!overrides) return []
  return Object.entries(overrides).filter(([, value]) => value !== null && value !== undefined)
})

const decisionEntries = computed(() => {
  if (!props.debug?.decisions) return []
  return Object.entries(props.debug.decisions).map(([key, entry]) => ({
    key,
    value: entry?.value,
    reason: entry?.reason,
    notes: entry?.notes || [],
    context: entry?.context || {}
  }))
})

const hasContext = (entry) => entry && Object.keys(entry).length > 0

const hasWarnings = computed(() => {
  return props.impact && (props.impact.vram || props.impact.ram)
})

const planSummary = computed(() => props.plan?.global ?? null)
const planGroups = computed(() => props.plan?.groups ?? [])

const formatMb = (mb) => {
  if (mb === null || mb === undefined) return '—'
  const value = Number(mb)
  if (Number.isNaN(value)) return '—'
  if (value >= 1024) return `${(value / 1024).toFixed(1)} GB`
  return `${value} MB`
}
</script>

<style scoped>
.preview-content {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
}

.preview-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  background: rgba(34, 211, 238, 0.1);
  border: 1px solid rgba(34, 211, 238, 0.2);
  border-radius: var(--radius-md);
}

.preview-header i {
  font-size: 1.5rem;
  color: var(--accent-cyan);
}

.preview-header p {
  margin: 0;
  font-size: 1rem;
  color: var(--text-primary);
  font-weight: 500;
}

.changes-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  max-height: 400px;
  overflow-y: auto;
}

.change-item {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  padding: var(--spacing-md);
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  transition: all var(--transition-normal);
}

.change-item.has-impact {
  border-left: 3px solid var(--accent-cyan);
}

.change-field {
  font-weight: 600;
  color: var(--text-primary);
  font-size: 0.95rem;
}

.change-values {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-size: 0.9rem;
}

.value-before {
  color: var(--text-secondary);
  text-decoration: line-through;
}

.value-after {
  color: var(--accent-cyan);
  font-weight: 600;
}

.change-values i {
  color: var(--text-secondary);
  font-size: 0.8rem;
}

.change-description {
  font-size: 0.85rem;
  color: var(--text-secondary);
  font-style: italic;
  margin-top: var(--spacing-xs);
}

.impact-preview {
  padding: var(--spacing-md);
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
}

.impact-preview h4 {
  margin: 0 0 var(--spacing-md) 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.impact-items {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.impact-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  border-radius: var(--radius-sm);
}

.impact-item.performance {
  background: rgba(34, 197, 94, 0.1);
  color: #22c55e;
}

.impact-item.vram {
  background: rgba(245, 158, 11, 0.1);
  color: #f59e0b;
}

.impact-item.ram {
  background: rgba(59, 130, 246, 0.1);
  color: var(--accent-blue);
}

.impact-item i {
  font-size: 1.1rem;
}

.preview-warning {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.3);
  border-radius: var(--radius-md);
}

.preview-warning i {
  color: #f59e0b;
  font-size: 1.2rem;
  flex-shrink: 0;
}

.preview-warning p {
  margin: 0;
  color: var(--text-primary);
  font-size: 0.9rem;
}

.plan-section {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
  background: var(--bg-surface);
}

.plan-summary-grid {
  display: grid;
  gap: var(--spacing-md);
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
}

.plan-warnings {
  display: flex;
  gap: var(--spacing-sm);
  align-items: flex-start;
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.2);
  color: var(--text-warning);
  font-size: 0.8rem;
}

.plan-warnings ul {
  margin: 0;
  padding-left: var(--spacing-md);
}

.plan-groups {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.plan-group-card {
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  padding: var(--spacing-sm);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.plan-group-header {
  display: flex;
  justify-content: space-between;
  gap: var(--spacing-md);
  align-items: baseline;
}

.plan-group-header h5 {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-primary);
}

.plan-group-header p {
  margin: var(--spacing-xs) 0 0 0;
  color: var(--text-secondary);
  font-size: 0.8rem;
}

.plan-group-penalty {
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  background: rgba(239, 68, 68, 0.12);
  color: var(--status-error);
  font-size: 0.75rem;
  font-weight: 600;
}

.plan-device-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.75rem;
}

.plan-device-table th,
.plan-device-table td {
  padding: var(--spacing-2xs) var(--spacing-xs);
  border-bottom: 1px solid var(--border-primary);
  text-align: left;
}

.plan-device-table th {
  font-weight: 600;
  color: var(--text-secondary);
}

.plan-device-table td {
  color: var(--text-primary);
}

.group-plan-warnings {
  background: rgba(239, 68, 68, 0.12);
  border-color: rgba(239, 68, 68, 0.3);
  color: var(--status-error);
}

.metric-label {
  font-size: 0.75rem;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.metric-value {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-primary);
}

.debug-section {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
  border: 1px solid rgba(59, 130, 246, 0.2);
}

.debug-section h4 {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.debug-summary {
  display: grid;
  gap: var(--spacing-md);
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}

.debug-card {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
  background: var(--bg-contrast);
}

.debug-card h5 {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-primary);
}

.debug-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.debug-list li {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: var(--spacing-sm);
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.debug-list strong {
  color: var(--text-primary);
}

.debug-list span {
  color: var(--text-secondary);
}

.debug-list .muted {
  color: var(--text-tertiary);
  font-size: 0.75rem;
}

.decision-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.decision-list h5 {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-primary);
}

.decision-item {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
  background: var(--bg-surface);
}

.decision-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: var(--spacing-sm);
}

.decision-header strong {
  font-weight: 600;
  color: var(--text-primary);
}

.decision-header span {
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.decision-reason {
  margin: 0;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.decision-notes {
  list-style: disc;
  margin: 0;
  padding-left: var(--spacing-lg);
  font-size: 0.8rem;
  color: var(--text-tertiary);
}

.decision-context {
  display: grid;
  gap: var(--spacing-sm);
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  font-size: 0.8rem;
}

.decision-context small {
  display: block;
  color: var(--text-tertiary);
}

.decision-context span {
  color: var(--text-secondary);
}
</style>

