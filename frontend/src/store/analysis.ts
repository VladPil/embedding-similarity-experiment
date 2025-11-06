import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import { analysisAPI } from '../services/api'

interface HistoryItem {
  id: string
  type: string
  text1_title: string
  text2_title: string
  similarity: number
  timestamp: number
  interpretation?: string
}

const HISTORY_KEY = 'emb_analysis_history'
const MAX_HISTORY_ITEMS = 50

export const useAnalysisStore = defineStore('analysis', () => {
  const results = ref<any[]>([])
  const loading = ref(false)
  const history = ref<HistoryItem[]>([])
  const selectedHistoryId = ref<string | null>(null)

  // Load history from localStorage
  function loadHistory() {
    try {
      const stored = localStorage.getItem(HISTORY_KEY)
      if (stored) {
        history.value = JSON.parse(stored)
      }
    } catch (e) {
      console.error('Failed to load history:', e)
    }
  }

  // Save history to localStorage
  function saveHistory() {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history.value))
    } catch (e) {
      console.error('Failed to save history:', e)
    }
  }

  // Watch history changes and save to localStorage
  watch(history, saveHistory, { deep: true })

  // Add item to history
  function addToHistory(item: HistoryItem) {
    history.value.unshift(item)
    if (history.value.length > MAX_HISTORY_ITEMS) {
      history.value = history.value.slice(0, MAX_HISTORY_ITEMS)
    }
  }

  // Clear history
  function clearHistory() {
    history.value = []
    localStorage.removeItem(HISTORY_KEY)
  }

  // Delete history item
  function deleteHistoryItem(id: string) {
    const idx = history.value.findIndex(item => item.id === id)
    if (idx >= 0) {
      history.value.splice(idx, 1)
    }
  }

  async function analyze(type: string, params: any, text1Title?: string, text2Title?: string) {
    loading.value = true
    try {
      const { data } = await (analysisAPI as any)[type](params)
      if (data.task_id) {
        return { isTask: true, taskId: data.task_id }
      }

      const result = { type, ...data, timestamp: Date.now() }
      results.value.unshift(result)

      // Add to history
      if (text1Title && text2Title) {
        addToHistory({
          id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          type,
          text1_title: text1Title,
          text2_title: text2Title,
          similarity: data.similarity || 0,
          timestamp: Date.now(),
          interpretation: data.interpretation
        })
      }

      return { isTask: false, data }
    } finally {
      loading.value = false
    }
  }

  // Initialize
  loadHistory()

  return {
    results,
    loading,
    history,
    selectedHistoryId,
    analyze,
    clearHistory,
    deleteHistoryItem
  }
})
