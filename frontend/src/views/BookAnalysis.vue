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
      <div v-else-if="text" class="book-analysis-container">
        <header class="header">
          <div class="header-left">
            <button @click="$router.push('/')" class="btn-back">← Назад</button>
            <h1>📚 Анализ книги</h1>
          </div>
        </header>

        <div class="card text-info-card">
          <h2>{{ text.title }}</h2>
          <div class="text-stats">
            <div class="stat">
              <span class="stat-icon">📝</span>
              <span class="stat-label">Символов:</span>
              <span class="stat-value">{{ formatNumber(text.length) }}</span>
            </div>
            <div class="stat">
              <span class="stat-icon">📄</span>
              <span class="stat-label">Строк:</span>
              <span class="stat-value">{{ formatNumber(text.lines) }}</span>
            </div>
          </div>
        </div>

        <!-- Analysis Selection -->
        <div class="card analysis-selection-card">
          <h3>Выберите типы анализа</h3>
          <div v-if="loadingAnalyses" class="loading-mini">Загрузка доступных анализов...</div>
          <div v-else class="analysis-options">
            <label
              v-for="analysis in availableAnalyses"
              :key="analysis.name"
              class="analysis-option"
            >
              <input
                type="checkbox"
                :value="analysis.name"
                v-model="selectedAnalyses"
                class="analysis-checkbox"
              />
              <div class="analysis-info">
                <span class="analysis-icon">{{ analysis.icon }}</span>
                <div class="analysis-text">
                  <span class="analysis-label">{{ analysis.label }}</span>
                  <span class="analysis-description">{{ analysis.description }}</span>
                </div>
              </div>
            </label>
          </div>

          <div class="analysis-actions">
            <button
              @click="runAnalysis"
              :disabled="analyzing || selectedAnalyses.length === 0"
              class="btn-primary btn-large"
            >
              {{ analyzing ? '⏳ Анализируем...' : `🔬 Запустить анализ (${selectedAnalyses.length})` }}
            </button>
            <button
              @click="selectAllAnalyses"
              class="btn-secondary"
            >
              {{ allSelected ? '☐ Снять все' : '☑ Выбрать все' }}
            </button>
          </div>
        </div>

        <!-- Results -->
        <div v-if="analysisResults" class="card results-card">
          <h3>Результаты анализа</h3>
          <div v-if="analysisResults.summary" class="summary-section">
            <h4>📋 Общая информация</h4>
            <p>{{ analysisResults.summary }}</p>
          </div>

          <div class="results-grid">
            <div
              v-for="(result, analysisType) in analysisResults.results"
              :key="analysisType"
              class="result-item"
            >
              <div class="result-header">
                <span class="result-icon">{{ getAnalysisIcon(analysisType) }}</span>
                <h4>{{ getAnalysisLabel(analysisType) }}</h4>
              </div>
              <div class="result-content">
                <pre>{{ formatResult(result) }}</pre>
              </div>
            </div>
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
const loading = ref(true)
const error = ref('')
const analyzing = ref(false)
const loadingAnalyses = ref(true)

const availableAnalyses = ref<any[]>([])
const selectedAnalyses = ref<string[]>([])
const analysisResults = ref<any>(null)

const allSelected = computed(() => {
  return selectedAnalyses.value.length === availableAnalyses.value.length
})

onMounted(async () => {
  await loadText()
  await loadAvailableAnalyses()
})

async function loadText() {
  loading.value = true
  error.value = ''
  try {
    const textId = route.params.id as string
    const response = await api.get(`/api/v1/texts/${textId}`)
    text.value = response.data
  } catch (err: any) {
    const status = err.response?.status

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

async function loadAvailableAnalyses() {
  loadingAnalyses.value = true
  try {
    const response = await api.get('/api/v1/book/available-analyses')
    availableAnalyses.value = response.data.analyses
    // Select all by default
    selectedAnalyses.value = availableAnalyses.value.map(a => a.name)
  } catch (err: any) {
    toast.value?.show('Не удалось загрузить список анализов', 'error')
  } finally {
    loadingAnalyses.value = false
  }
}

async function runAnalysis() {
  if (selectedAnalyses.value.length === 0) return

  analyzing.value = true
  analysisResults.value = null
  try {
    const textId = route.params.id as string
    const response = await api.post('/api/v1/book/analyze-book', {
      text_id: textId,
      analyses: selectedAnalyses.value,
      chunk_size: 2000
    })

    analysisResults.value = response.data
    toast.value?.show('Анализ успешно завершен', 'success')
  } catch (err: any) {
    toast.value?.show(err.response?.data?.detail || 'Ошибка анализа', 'error')
  } finally {
    analyzing.value = false
  }
}

function selectAllAnalyses() {
  if (allSelected.value) {
    selectedAnalyses.value = []
  } else {
    selectedAnalyses.value = availableAnalyses.value.map(a => a.name)
  }
}

function getAnalysisIcon(type: string): string {
  const analysis = availableAnalyses.value.find(a => a.name === type)
  return analysis?.icon || '📊'
}

function getAnalysisLabel(type: string): string {
  const analysis = availableAnalyses.value.find(a => a.name === type)
  return analysis?.label || type
}

function formatResult(result: any): string {
  if (typeof result === 'string') return result
  return JSON.stringify(result, null, 2)
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
  max-width: 1400px;
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

.book-analysis-container {
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

.card {
  background: var(--bg-secondary, #1a1a1a);
  border: 1px solid var(--border, #333);
  border-radius: 12px;
  padding: 2rem;
  margin-bottom: 1.5rem;
}

.text-info-card h2 {
  margin: 0 0 1rem 0;
  font-size: 1.5rem;
}

.text-stats {
  display: flex;
  gap: 2rem;
  padding: 1rem;
  background: var(--bg-hover);
  border-radius: 8px;
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
  color: var(--text-muted);
  margin-right: 0.25rem;
}

.stat-value {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text);
}

.analysis-selection-card h3 {
  margin: 0 0 1.5rem 0;
  font-size: 1.25rem;
}

.loading-mini {
  text-align: center;
  padding: 2rem;
  color: var(--text-muted);
}

.analysis-options {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.analysis-option {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 1rem;
  background: var(--bg-hover);
  border: 2px solid transparent;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.analysis-option:hover {
  background: var(--bg-active);
  border-color: var(--primary);
}

.analysis-option:has(.analysis-checkbox:checked) {
  background: rgba(59, 130, 246, 0.1);
  border-color: var(--primary);
}

.analysis-checkbox {
  margin-top: 0.25rem;
  width: 18px;
  height: 18px;
  cursor: pointer;
}

.analysis-info {
  display: flex;
  gap: 0.75rem;
  flex: 1;
}

.analysis-icon {
  font-size: 1.5rem;
}

.analysis-text {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.analysis-label {
  font-weight: 600;
  font-size: 1rem;
  color: var(--text);
}

.analysis-description {
  font-size: 0.85rem;
  color: var(--text-muted);
}

.analysis-actions {
  display: flex;
  gap: 1rem;
  justify-content: flex-start;
}

.btn-primary, .btn-secondary {
  padding: 0.65rem 1.25rem;
  border-radius: 8px;
  font-size: 0.95rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  border: none;
}

.btn-large {
  padding: 0.85rem 1.5rem;
  font-size: 1rem;
}

.btn-primary {
  background: var(--primary, #3b82f6);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #2563eb;
  transform: translateY(-1px);
}

.btn-secondary {
  background: var(--bg-hover);
  color: var(--text);
  border: 1px solid var(--border);
}

.btn-secondary:hover:not(:disabled) {
  background: var(--bg-active);
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none !important;
}

.results-card h3 {
  margin: 0 0 1.5rem 0;
  font-size: 1.25rem;
}

.summary-section {
  padding: 1rem;
  background: var(--bg-hover);
  border-radius: 8px;
  margin-bottom: 1.5rem;
}

.summary-section h4 {
  margin: 0 0 0.75rem 0;
  font-size: 1rem;
}

.summary-section p {
  margin: 0;
  line-height: 1.6;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 1.5rem;
}

.result-item {
  background: var(--bg-hover);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}

.result-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background: rgba(59, 130, 246, 0.1);
  border-bottom: 1px solid var(--border);
}

.result-icon {
  font-size: 1.5rem;
}

.result-header h4 {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
}

.result-content {
  padding: 1rem;
}

.result-content pre {
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
  line-height: 1.5;
  color: var(--text);
}
</style>
