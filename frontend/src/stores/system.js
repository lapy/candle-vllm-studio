import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'

const toPercent = (used, total) => {
  if (!total || total <= 0) return 0
  return Math.min(100, Math.max(0, (used / total) * 100))
}

const normalizeGpuEntry = (entry, indexHint = 0) => {
  if (!entry) return null

  const index = entry.index ?? entry.device_id ?? indexHint
  const total = entry.total ?? entry.memory?.total ?? 0
  const used = entry.used ?? entry.memory?.used ?? 0
  const free = entry.free ?? entry.memory?.free ?? Math.max(total - used, 0)
  const percent =
    entry.memory?.percent ??
    toPercent(used, total)

  return {
    index,
    name: entry.name ?? entry.model ?? `GPU ${index}`,
    memory: {
      total,
      used,
      free,
      percent,
    },
    utilization: {
      gpu: entry.utilization?.gpu ?? entry.utilization ?? 0,
      memory: entry.utilization?.memory ?? entry.memory_utilization ?? 0,
    },
    temperature: entry.temperature ?? null,
    raw: entry,
  }
}

const normalizeGpuInfo = (info = {}) => {
  const normalized = { ...info }
  const vramData = info.vram_data

  let gpuEntries = []
  if (Array.isArray(info.gpus) && info.gpus.length > 0) {
    gpuEntries = info.gpus.map((gpu, idx) => normalizeGpuEntry(gpu, idx)).filter(Boolean)
  } else if (vramData?.gpus && vramData.gpus.length > 0) {
    gpuEntries = vramData.gpus.map((gpu, idx) => normalizeGpuEntry(gpu, idx)).filter(Boolean)
  }

  normalized.gpus = gpuEntries

  const totalFromEntries = gpuEntries.reduce((sum, gpu) => sum + (gpu?.memory?.total ?? 0), 0)
  const usedFromEntries = gpuEntries.reduce((sum, gpu) => sum + (gpu?.memory?.used ?? 0), 0)

  const total =
    info.total_vram ??
    vramData?.total ??
    vramData?.total_vram ??
    totalFromEntries

  const used =
    info.used_vram ??
    vramData?.used ??
    vramData?.used_vram ??
    (total > 0 ? usedFromEntries : 0)

  const available =
    info.available_vram ??
    vramData?.free ??
    vramData?.free_vram ??
    (total > 0 ? Math.max(0, total - used) : 0)

  normalized.total_vram = total
  normalized.available_vram = available
  normalized.used_vram = used
  normalized.vram_percent =
    info.vram_percent ??
    vramData?.percent ??
    toPercent(used, total)

  normalized.device_count = info.device_count ?? vramData?.device_count ?? gpuEntries.length
  normalized.cpu_only_mode = info.cpu_only_mode ?? (normalized.device_count === 0)

  if (!normalized.cuda_version && vramData?.cuda_version) {
    normalized.cuda_version = vramData.cuda_version
  }

  return normalized
}

export const useSystemStore = defineStore('system', () => {
  const systemStatus = ref({})
  const gpuInfo = ref({})
  const loading = ref(false)

  const fetchSystemStatus = async () => {
    loading.value = true
    try {
      const [statusResponse, gpuResponse] = await Promise.all([
        axios.get('/api/status'),
        axios.get('/api/gpu-info')
      ])

      systemStatus.value = statusResponse.data
      gpuInfo.value = normalizeGpuInfo(gpuResponse.data)
    } catch (error) {
      console.error('Failed to fetch system status:', error)
      throw error
    } finally {
      loading.value = false
    }
  }

  const updateSystemStatus = (status) => {
    systemStatus.value = { ...systemStatus.value, ...status }
  }

  const updateGpuInfo = (gpuData) => {
    const merged = { ...gpuInfo.value, ...gpuData }
    gpuInfo.value = normalizeGpuInfo(merged)
  }

  return {
    systemStatus,
    gpuInfo,
    loading,
    fetchSystemStatus,
    updateSystemStatus,
    updateGpuInfo
  }
})
