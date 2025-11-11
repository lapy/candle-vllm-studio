import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

export const useBuildStore = defineStore('builds', () => {
  const builds = ref([])
  const capabilities = ref({})
  const loading = ref(false)
  const building = ref(false)
  const progress = ref({})

  const hasBuilds = computed(() => builds.value.length > 0)

  const fetchBuilds = async () => {
    loading.value = true
    try {
      const response = await axios.get('/api/candle-builds')
      builds.value = response.data
    } finally {
      loading.value = false
    }
  }

  const fetchCapabilities = async () => {
    try {
      const response = await axios.get('/api/candle-builds/capabilities')
      capabilities.value = response.data || {}
    } catch (error) {
      console.warn('Failed to fetch build capabilities', error)
      capabilities.value = {}
    }
  }

  const startBuild = async (payload) => {
    building.value = true
    try {
      const response = await axios.post('/api/candle-builds/build', payload)
      return response.data
    } finally {
      building.value = false
    }
  }

  const deleteBuild = async (id) => {
    await axios.delete(`/api/candle-builds/${id}`)
    await fetchBuilds()
  }

  const activateBuild = async (id) => {
    await axios.post(`/api/candle-builds/${id}/activate`, { mark_active: true })
    await fetchBuilds()
  }

  const updateProgress = (message) => {
    if (!message?.task_id) return
    progress.value[message.task_id] = message
  }

  const removeProgress = (taskId) => {
    delete progress.value[taskId]
  }

  return {
    builds,
    capabilities,
    loading,
    building,
    progress,
    hasBuilds,
    fetchBuilds,
    fetchCapabilities,
    startBuild,
    deleteBuild,
    activateBuild,
    updateProgress,
    removeProgress
  }
})

