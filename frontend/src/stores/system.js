import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'

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
      gpuInfo.value = gpuResponse.data
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
    gpuInfo.value = { ...gpuInfo.value, ...gpuData }
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
