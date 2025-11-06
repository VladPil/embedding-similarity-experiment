import { defineStore } from 'pinia'
import { ref } from 'vue'
import { textsAPI } from '../services/api'

export const useTextsStore = defineStore('texts', () => {
  const texts = ref<any[]>([])
  const loading = ref(false)

  async function fetchTexts() {
    loading.value = true
    try {
      const { data } = await textsAPI.list()
      texts.value = data.texts
    } finally {
      loading.value = false
    }
  }

  async function uploadText(title: string, text: string) {
    const { data } = await textsAPI.upload({ title, text })
    await fetchTexts()
    return data
  }

  async function uploadFB2(file: File) {
    const { data } = await textsAPI.uploadFB2(file)
    await fetchTexts()
    return data
  }

  async function deleteText(id: string) {
    await textsAPI.delete(id)
    // Remove from local array instead of fetching all again
    const index = texts.value.findIndex(t => t.id === id)
    if (index !== -1) {
      texts.value.splice(index, 1)
    }
  }

  async function deleteMultipleTexts(ids: string[]) {
    // Delete all texts in parallel
    const results = await Promise.allSettled(
      ids.map(id => textsAPI.delete(id))
    )

    // Remove successfully deleted texts from local array
    const successfulIds = ids.filter((id, index) =>
      results[index].status === 'fulfilled'
    )

    texts.value = texts.value.filter(t => !successfulIds.includes(t.id))

    // Return info about failures
    const failedCount = results.filter(r => r.status === 'rejected').length
    return {
      deleted: successfulIds.length,
      failed: failedCount
    }
  }

  return { texts, loading, fetchTexts, uploadText, uploadFB2, deleteText, deleteMultipleTexts }
})
