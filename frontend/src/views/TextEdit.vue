<template>
  <div class="page-wrapper">
    <HistorySidebar />
    <div class="container">
      <div v-if="loading" class="loading">⏳ Загрузка...</div>
      <div v-else-if="error" class="error-state">
        <h2>Ошибка</h2>
        <p>{{ error }}</p>
        <button @click="$router.push('/')" class="btn-primary">← Вернуться</button>
      </div>
      <div v-else-if="text" class="text-edit-container">
        <header class="header">
          <div class="header-left">
            <button @click="$router.push('/')" class="btn-back">← Назад</button>
            <h1>Редактирование текста</h1>
          </div>
          <div class="header-actions">
            <button @click="saveText" :disabled="saving || !hasChanges" class="btn-primary">
              {{ saving ? '⏳ Сохранение...' : '💾 Сохранить' }}
            </button>
            <button @click="deleteText" :disabled="deleting" class="btn-danger">
              {{ deleting ? '⏳' : '🗑️' }} Удалить
            </button>
          </div>
        </header>

        <div class="card edit-card">
          <!-- Statistics moved to top -->
          <div class="text-stats-top">
            <div class="stat">
              <span class="stat-icon">📝</span>
              <span class="stat-label">Символов:</span>
              <span class="stat-value">{{ formatNumber(editedContent.length) }}</span>
            </div>
            <div class="stat">
              <span class="stat-icon">📄</span>
              <span class="stat-label">Строк:</span>
              <span class="stat-value">{{ formatNumber(editedContent.split('\n').length) }}</span>
            </div>
            <div class="stat">
              <span class="stat-icon">📖</span>
              <span class="stat-label">Слов:</span>
              <span class="stat-value">{{ formatNumber(countWords(editedContent)) }}</span>
            </div>
          </div>

          <div class="form-group">
            <label>Название</label>
            <input v-model="editedTitle" class="input" placeholder="Введите название текста" />
          </div>

          <div class="form-group">
            <label>Текст</label>
            <textarea v-model="editedContent" class="textarea" rows="20" placeholder="Введите текст..."></textarea>
          </div>
        </div>
      </div>
    </div>
    <Toast ref="toast" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api } from '../services/api'
import HistorySidebar from '../components/HistorySidebar.vue'
import Toast from '../components/Toast.vue'
import { formatNumber } from '../utils/format'

const route = useRoute()
const router = useRouter()
const toast = ref<any>(null)

const text = ref<any>(null)
const editedTitle = ref('')
const editedContent = ref('')
const loading = ref(true)
const error = ref('')
const saving = ref(false)
const deleting = ref(false)

const hasChanges = computed(() => {
  if (!text.value) return false
  return editedTitle.value !== text.value.title || editedContent.value !== (text.value.text || '')
})

onMounted(async () => {
  await loadText()
})

async function loadText() {
  loading.value = true
  error.value = ''
  try {
    const textId = route.params.id as string
    const response = await api.get(`/api/v1/texts/${textId}`)
    text.value = response.data
    editedTitle.value = text.value.title || ''

    // Get text content from 'text' field (API response structure)
    let textContent = text.value.text || ''

    // If text is large (>10k chars), truncate it
    const MAX_CHARS = 10000
    if (textContent.length > MAX_CHARS) {
      textContent = textContent.substring(0, MAX_CHARS)
      toast.value?.show(
        `Текст слишком большой (${formatNumber(text.value.length)} символов). Показаны первые ${formatNumber(MAX_CHARS)} символов.`,
        'info'
      )
    }

    editedContent.value = textContent
  } catch (err: any) {
    const status = err.response?.status

    // Redirect to error pages for 404 and 500
    if (status === 404) {
      router.push('/404')
    } else if (status >= 500) {
      router.push('/500')
    } else {
      error.value = err.response?.data?.error || 'Не удалось загрузить текст'
    }
  } finally {
    loading.value = false
  }
}

async function saveText() {
  if (!hasChanges.value) return

  saving.value = true
  try {
    const textId = route.params.id as string
    await api.put(`/api/v1/texts/${textId}`, {
      title: editedTitle.value,
      text: editedContent.value
    })

    text.value.title = editedTitle.value
    text.value.content = editedContent.value

    toast.value?.show('Текст успешно сохранен', 'success')
  } catch (err: any) {
    toast.value?.show(err.response?.data?.error || 'Ошибка сохранения текста', 'error')
  } finally {
    saving.value = false
  }
}

async function deleteText() {
  if (!confirm('Вы уверены, что хотите удалить этот текст?')) {
    return
  }

  deleting.value = true
  try {
    const textId = route.params.id as string
    await api.delete(`/api/v1/texts/${textId}`)
    toast.value?.show('Текст успешно удален', 'success')
    setTimeout(() => router.push('/'), 500)
  } catch (err: any) {
    toast.value?.show(err.response?.data?.error || 'Ошибка удаления текста', 'error')
  } finally {
    deleting.value = false
  }
}

function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(word => word.length > 0).length
}
</script>

<style scoped>
.page-wrapper {
  display: flex;
  min-height: 100vh;
}

.container {
  flex: 1;
  padding: 2rem;
  max-width: 1200px;
}

.loading, .error-state {
  text-align: center;
  padding: 4rem 2rem;
  color: var(--text-muted);
}

.error-state h2 {
  color: var(--error, #ef4444);
  margin-bottom: 1rem;
}

.error-state p {
  margin-bottom: 2rem;
}

.text-edit-container {
  width: 100%;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  padding-bottom: 1.5rem;
  border-bottom: 2px solid var(--border);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.header-left h1 {
  margin: 0;
  font-size: 1.75rem;
  font-weight: 700;
}

.btn-back {
  padding: 0.5rem 1rem;
  background: var(--bg-hover);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.95rem;
  transition: all 0.2s;
}

.btn-back:hover {
  background: var(--bg-active);
}

.header-actions {
  display: flex;
  gap: 0.75rem;
}

.btn-primary, .btn-danger {
  padding: 0.65rem 1.25rem;
  border-radius: 8px;
  font-size: 0.95rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  border: none;
}

.btn-primary {
  background: var(--primary, #3b82f6);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #2563eb;
  transform: translateY(-1px);
}

.btn-danger {
  background: var(--error, #ef4444);
  color: white;
}

.btn-danger:hover:not(:disabled) {
  background: #dc2626;
  transform: translateY(-1px);
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none !important;
}

.edit-card {
  padding: 2rem;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--text);
}

.input, .textarea {
  width: 100%;
  padding: 0.75rem;
  background: var(--bg-hover);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  font-size: 1rem;
  font-family: inherit;
  resize: vertical;
}

.input:focus, .textarea:focus {
  outline: none;
  border-color: var(--primary);
}

.textarea {
  font-family: 'Courier New', monospace;
  line-height: 1.6;
  min-height: 400px;
}

.text-stats-top {
  display: flex;
  gap: 2rem;
  padding: 1rem 1.5rem;
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark, #2563eb) 100%);
  border-radius: 12px;
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
}

.stat {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.stat-icon {
  font-size: 1.25rem;
}

.stat-label {
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.8);
  margin-right: 0.25rem;
}

.stat-value {
  font-size: 1.25rem;
  font-weight: 700;
  color: white;
}

.card {
  background: var(--bg-secondary, #1a1a1a);
  border: 1px solid var(--border, #333);
  border-radius: 12px;
}
</style>
