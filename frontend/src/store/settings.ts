import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface Settings {
  // Embedding Models
  defaultEmbeddingModel: string
  maxSequenceLength: number

  // LLM Models
  defaultLLMModel: string
  llmDevice: string
  llmMaxMemoryGB: number

  // Analysis Parameters
  chunkSize: number
  chunkOverlap: number
  defaultSegments: number
  combinedStrategies: {
    semantic: boolean
    style: boolean
    tfidf: boolean
    emotion: boolean
    llm: boolean
  }

  // Performance
  maxWorkers: number
  cacheTTL: number

  // UI
  theme: string
  language: string
  historyLimit: number
  autoRemoveCompletedTasks: boolean
  showNotifications: boolean
}

export const useSettingsStore = defineStore('settings', () => {
  // Default settings
  const defaultSettings: Settings = {
    defaultEmbeddingModel: 'multilingual-e5-small',
    maxSequenceLength: 512,
    defaultLLMModel: 'qwen2.5-3b',
    llmDevice: 'cuda',
    llmMaxMemoryGB: 24,
    chunkSize: 1000,
    chunkOverlap: 200,
    defaultSegments: 10,
    combinedStrategies: {
      semantic: true,
      style: true,
      tfidf: true,
      emotion: true,
      llm: true
    },
    maxWorkers: 3,
    cacheTTL: 3600,
    theme: 'dark',
    language: 'ru',
    historyLimit: 5,
    autoRemoveCompletedTasks: false,
    showNotifications: true
  }

  // State
  const settings = ref<Settings>({ ...defaultSettings })

  // Getters
  const getEmbeddingModel = computed(() => settings.value.defaultEmbeddingModel)
  const getLLMModel = computed(() => settings.value.defaultLLMModel)
  const getTheme = computed(() => settings.value.theme)
  const getHistoryLimit = computed(() => settings.value.historyLimit)
  const getShouldAutoRemoveTasks = computed(() => settings.value.autoRemoveCompletedTasks)
  const getShouldShowNotifications = computed(() => settings.value.showNotifications)
  const getCombinedStrategies = computed(() => {
    const strategies = settings.value.combinedStrategies
    return Object.keys(strategies).filter(key => strategies[key as keyof typeof strategies])
  })

  // Actions
  function loadSettings() {
    const saved = localStorage.getItem('app-settings')
    if (saved) {
      try {
        const parsed = JSON.parse(saved)
        settings.value = { ...defaultSettings, ...parsed }
      } catch (e) {
        console.error('Failed to load settings:', e)
      }
    }
  }

  function saveSettings() {
    try {
      localStorage.setItem('app-settings', JSON.stringify(settings.value))
      return true
    } catch (e) {
      console.error('Failed to save settings:', e)
      return false
    }
  }

  function updateSetting<K extends keyof Settings>(key: K, value: Settings[K]) {
    settings.value[key] = value
    saveSettings()
  }

  function resetToDefaults() {
    settings.value = { ...defaultSettings }
    saveSettings()
  }

  function exportSettings(): string {
    return JSON.stringify(settings.value, null, 2)
  }

  function importSettings(json: string): boolean {
    try {
      const imported = JSON.parse(json)
      settings.value = { ...defaultSettings, ...imported }
      saveSettings()
      return true
    } catch (e) {
      console.error('Failed to import settings:', e)
      return false
    }
  }

  // Initialize on store creation
  loadSettings()

  return {
    settings,
    getEmbeddingModel,
    getLLMModel,
    getTheme,
    getHistoryLimit,
    getShouldAutoRemoveTasks,
    getShouldShowNotifications,
    getCombinedStrategies,
    loadSettings,
    saveSettings,
    updateSetting,
    resetToDefaults,
    exportSettings,
    importSettings
  }
})
