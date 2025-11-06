<template>
  <div class="page-layout">
    <aside class="sidebar-wrapper">
      <HistorySidebar />
    </aside>
    <main class="main-content">
      <AppHeader title="📚 Embedding Similarity" subtitle="Анализ семантической схожести текстов">
        <template #actions>
          <button @click="clearAll" class="btn-danger" :disabled="clearing">
            {{ clearing ? '⏳' : '🗑️' }} Очистить все
          </button>
          <button @click="showUpload = !showUpload" class="btn-primary">
            {{ showUpload ? '✕ Закрыть' : '+ Добавить текст' }}
          </button>
          <button @click="textsStore.fetchTexts()" class="btn-secondary" :disabled="textsStore.loading">
            🔄 Обновить
          </button>
        </template>
      </AppHeader>

      <!-- Upload Section -->
      <div v-if="showUpload" class="card upload-section">
        <h3>Загрузить текст</h3>
        <div class="upload-tabs">
          <button
            @click="uploadMode = 'text'"
            :class="{ active: uploadMode === 'text' }"
            class="tab-btn"
          >
            Текст
          </button>
          <button
            @click="uploadMode = 'file'"
            :class="{ active: uploadMode === 'file' }"
            class="tab-btn"
          >
            Файл FB2
          </button>
        </div>

        <div v-if="uploadMode === 'text'" class="upload-form">
          <input v-model="uploadTitle" placeholder="Название" class="input" />
          <textarea v-model="uploadText" placeholder="Введите текст..." rows="8" class="textarea"></textarea>
          <div class="upload-actions">
            <button @click="handleUpload" :disabled="!uploadTitle || !uploadText || uploading" class="btn-primary">
              {{ uploading ? '⏳ Загрузка...' : '✓ Загрузить' }}
            </button>
            <button @click="showUpload = false" class="btn-secondary">Отмена</button>
          </div>
        </div>

        <div v-else class="upload-form">
          <div class="file-upload">
            <input
              type="file"
              @change="handleFileSelect"
              accept=".fb2"
              ref="fileInput"
              class="file-input"
              multiple
            />
            <button @click="$refs.fileInput.click()" class="btn-file">
              📁 Выбрать файлы FB2 (до 10)
            </button>
            <div v-if="selectedFiles.length > 0" class="selected-files">
              <div class="files-header">
                Выбрано файлов: {{ selectedFiles.length }} / 10
              </div>
              <div v-for="(file, index) in selectedFiles" :key="index" class="file-item">
                <span class="file-name">{{ file.name }}</span>
                <span class="file-size">({{ formatFileSize(file.size) }})</span>
                <button @click="removeFile(index)" class="btn-remove-file">✕</button>
              </div>
            </div>
          </div>
          <div class="upload-actions">
            <button @click="handleMultipleFileUpload" :disabled="selectedFiles.length === 0 || uploading" class="btn-primary">
              {{ uploading ? `⏳ Загрузка... (${uploadProgress.current}/${uploadProgress.total})` : `✓ Загрузить (${selectedFiles.length})` }}
            </button>
            <button @click="showUpload = false; selectedFiles = []" class="btn-secondary">Отмена</button>
          </div>
          <div v-if="uploadProgress.total > 0" class="upload-progress">
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: `${(uploadProgress.current / uploadProgress.total) * 100}%` }"></div>
            </div>
            <span class="progress-text">{{ uploadProgress.current }} из {{ uploadProgress.total }}</span>
          </div>
        </div>
      </div>

      <!-- Main Tabs -->
      <div class="main-tabs">
        <button
          @click="activeTab = 'texts'"
          :class="{ active: activeTab === 'texts' }"
          class="tab-btn-main"
        >
          📚 Тексты ({{ textsStore.texts.length }})
        </button>
        <button
          @click="activeTab = 'analysis'"
          :class="{ active: activeTab === 'analysis' }"
          class="tab-btn-main"
        >
          🔬 Анализ {{ selectedTexts.length >= 2 ? `(${selectedTexts.length} выбрано)` : '' }}
        </button>
        <button
          @click="activeTab = 'results'"
          :class="{ active: activeTab === 'results' }"
          class="tab-btn-main"
        >
          📈 Результаты ({{ analysisStore.results.length }})
        </button>
      </div>

      <!-- Texts Tab -->
      <div v-if="activeTab === 'texts'" class="tab-content">
        <!-- Filters -->
        <div class="filters-section">
          <div class="filter-group">
            <label class="filter-label">🔍 Поиск:</label>
            <input
              v-model="searchQuery"
              type="text"
              placeholder="Название текста..."
              class="filter-input"
            />
          </div>

          <div class="filter-group">
            <label class="filter-label">📅 Сортировка:</label>
            <select v-model="sortBy" class="filter-select">
              <option value="date-desc">Новые первые</option>
              <option value="date-asc">Старые первые</option>
              <option value="size-desc">Большие первые</option>
              <option value="size-asc">Маленькие первые</option>
              <option value="name">По названию</option>
            </select>
          </div>

          <div class="filter-group">
            <label class="filter-label">📏 Размер:</label>
            <select v-model="sizeFilter" class="filter-select">
              <option value="all">Все размеры</option>
              <option value="small">Маленькие (&lt; 1000)</option>
              <option value="medium">Средние (1000-10000)</option>
              <option value="large">Большие (&gt; 10000)</option>
            </select>
          </div>

          <button
            @click="selectAllOnPage"
            class="btn-secondary"
          >
            {{ allOnPageSelected ? '☐ Снять выделение' : '☑ Выбрать все' }}
          </button>

          <button
            v-if="selectedForDelete.length > 0"
            @click="deleteSelected"
            class="btn-danger"
            :disabled="deleting"
          >
            {{ deleting ? '⏳' : '🗑️' }} Удалить ({{ selectedForDelete.length }})
          </button>
        </div>

        <!-- Texts List -->
        <div v-if="textsStore.loading" class="loading">⏳ Загрузка текстов...</div>
        <div v-else-if="!filteredTexts.length" class="empty-state">
          {{ searchQuery || sizeFilter !== 'all' ? 'Нет текстов по заданным фильтрам' : 'Нет загруженных текстов. Добавьте свой первый текст!' }}
        </div>
        <div v-else class="texts-list">
          <div v-for="text in paginatedTexts" :key="text.id" class="text-row">
            <input
              type="checkbox"
              :checked="selectedForDelete.includes(text.id)"
              @change="toggleDeleteSelect(text.id)"
              class="row-checkbox"
            />
            <div class="text-info">
              <span class="text-title" @click="navigateToEdit(text.id)">{{ text.title }}</span>
              <span class="text-stats">
                <span class="stat">📝 {{ formatNumber(text.length) }} симв.</span>
                <span class="stat">📄 {{ formatNumber(text.lines) }} строк</span>
                <span class="stat">📅 {{ formatDate(text.created_at) }}</span>
              </span>
            </div>
            <div class="text-actions">
              <button
                @click="navigateToAnalyze(text.id)"
                class="btn-analyze"
                title="Анализ книги"
              >
                📚 Анализ
              </button>
              <button
                @click="selectText(text)"
                :class="{ selected: isSelected(text.id) }"
                class="btn-select-row"
              >
                {{ isSelected(text.id) ? '✓ Выбрано' : 'Выбрать' }}
              </button>
            </div>
          </div>
        </div>

        <!-- Pagination -->
        <div v-if="totalPages > 1" class="pagination">
          <button
            @click="currentPage = 1"
            :disabled="currentPage === 1"
            class="page-btn"
          >
            ⏮️
          </button>
          <button
            @click="currentPage--"
            :disabled="currentPage === 1"
            class="page-btn"
          >
            ◀️
          </button>
          <span class="page-info">
            Страница {{ currentPage }} из {{ totalPages }}
          </span>
          <button
            @click="currentPage++"
            :disabled="currentPage === totalPages"
            class="page-btn"
          >
            ▶️
          </button>
          <button
            @click="currentPage = totalPages"
            :disabled="currentPage === totalPages"
            class="page-btn"
          >
            ⏭️
          </button>
        </div>
      </div>

      <!-- Analysis Tab -->
      <div v-if="activeTab === 'analysis'" class="tab-content">
        <div v-if="selectedTexts.length < 2" class="empty-state">
          Выберите минимум 2 текста для анализа
        </div>
        <div v-else class="analysis-panel">
          <!-- Pairwise analysis for 2 texts -->
          <div v-if="selectedTexts.length === 2">
            <div class="selected-texts">
              <div class="selected-text">
                <span class="label">Текст 1:</span> {{ selectedTexts[0].title }}
              </div>
              <div class="vs-divider">VS</div>
              <div class="selected-text">
                <span class="label">Текст 2:</span> {{ selectedTexts[1].title }}
              </div>
            </div>

            <!-- Model Selector -->
            <div class="model-selector">
              <label class="model-label">
                <span class="label-icon">🧠</span> Модель эмбеддингов:
              </label>
              <select v-model="selectedModel" class="model-select">
                <option v-for="model in availableModels" :key="model.value" :value="model.value">
                  {{ model.label }}
                </option>
              </select>
              <div class="model-hint">
                Используется для семантического анализа, комбо и чанков
              </div>
            </div>

            <div class="analysis-methods">
              <button @click="runAnalysis('semantic')" :disabled="analyzing" class="analysis-btn" :title="analysisDescriptions.semantic">
                {{ analyzing === 'semantic' ? '⏳' : '🧠' }} Семантика
              </button>
              <button @click="runAnalysis('style')" :disabled="analyzing" class="analysis-btn" :title="analysisDescriptions.style">
                {{ analyzing === 'style' ? '⏳' : '✍️' }} Стиль
              </button>
              <button @click="runAnalysis('tfidf')" :disabled="analyzing" class="analysis-btn" :title="analysisDescriptions.tfidf">
                {{ analyzing === 'tfidf' ? '⏳' : '📊' }} TF-IDF
              </button>
              <button @click="runAnalysis('emotion')" :disabled="analyzing" class="analysis-btn" :title="analysisDescriptions.emotion">
                {{ analyzing === 'emotion' ? '⏳' : '😊' }} Эмоции
              </button>
              <button @click="runAnalysis('llm')" :disabled="analyzing" class="analysis-btn" :title="analysisDescriptions.llm">
                {{ analyzing === 'llm' ? '⏳' : '🤖' }} LLM
              </button>
              <button @click="runAnalysis('combined')" :disabled="analyzing" class="analysis-btn btn-special" :title="analysisDescriptions.combined">
                {{ analyzing === 'combined' ? '⏳' : '🎯' }} Комбо
              </button>
              <button @click="runChunkedAnalysis()" :disabled="analyzing" class="analysis-btn" :title="analysisDescriptions.chunked">
                {{ analyzing === 'chunked' ? '⏳' : '🔀' }} По чанкам
              </button>
            </div>
          </div>

          <!-- Matrix analysis for 3+ texts -->
          <div v-else>
            <div class="selected-texts-list">
              <div class="selected-text-item" v-for="(text, i) in selectedTexts" :key="text.id">
                <span class="text-number">{{ i + 1 }}.</span> {{ text.title }}
                <button @click="unselectText(text)" class="btn-remove">✕</button>
              </div>
            </div>
            <p class="matrix-hint">Выбрано {{ selectedTexts.length }} текстов для матричного анализа</p>
            <div class="analysis-methods">
              <button @click="runMatrixAnalysis('semantic')" :disabled="analyzing" class="analysis-btn">
                {{ analyzing === 'matrix-semantic' ? '⏳' : '🧠' }} Матрица (Семантика)
              </button>
              <button @click="runMatrixAnalysis('style')" :disabled="analyzing" class="analysis-btn">
                {{ analyzing === 'matrix-style' ? '⏳' : '✍️' }} Матрица (Стиль)
              </button>
              <button @click="runMatrixAnalysis('tfidf')" :disabled="analyzing" class="analysis-btn">
                {{ analyzing === 'matrix-tfidf' ? '⏳' : '📊' }} Матрица (TF-IDF)
              </button>
              <button @click="runMatrixAnalysis('emotion')" :disabled="analyzing" class="analysis-btn">
                {{ analyzing === 'matrix-emotion' ? '⏳' : '😊' }} Матрица (Эмоции)
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Results Tab -->
      <div v-if="activeTab === 'results'" class="tab-content">
        <div v-if="!analysisStore.results.length" class="empty-state">
          Пока нет результатов анализа
        </div>
        <div v-else class="results-list">
          <div v-for="(result, i) in analysisStore.results" :key="i" class="result-card card">
            <div class="result-header">
              <h4>{{ getAnalysisTitle(result.type) }}</h4>
              <div class="result-actions">
                <div class="similarity-score" :style="{ background: getScoreColor(result.similarity) }">
                  {{ (result.similarity * 100).toFixed(1) }}%
                </div>
                <button @click="removeResult(i)" class="btn-remove-result">🗑️</button>
              </div>
            </div>

            <!-- For LLM analysis - show detailed metrics -->
            <div v-if="result.type === 'llm' && hasLLMMetrics(result)" class="llm-metrics">
              <div class="metric-grid">
                <div v-for="metric in getLLMMetrics(result)" :key="metric.key" class="metric-item">
                  <div class="metric-label">{{ metric.label }}</div>
                  <div class="metric-bar">
                    <div class="metric-bar-fill" :style="{ width: `${metric.value * 100}%`, background: getScoreColor(metric.value) }"></div>
                  </div>
                  <div class="metric-value">{{ (metric.value * 100).toFixed(1) }}%</div>
                </div>
              </div>
            </div>

            <!-- For Combined analysis - show breakdown -->
            <div v-if="result.type === 'combined' && result.results" class="combined-breakdown">
              <h5>Вклад каждого метода:</h5>
              <div class="breakdown-grid">
                <div v-for="(strategyResult, strategyName) in result.results" :key="strategyName" class="breakdown-item">
                  <div class="breakdown-header">
                    <span class="breakdown-name">{{ formatStrategyName(strategyName) }}</span>
                    <span class="breakdown-weight" v-if="strategyResult.weight">вес: {{ (strategyResult.weight * 100).toFixed(0) }}%</span>
                  </div>
                  <div class="breakdown-score">
                    <div class="breakdown-bar">
                      <div class="breakdown-bar-fill" :style="{ width: `${(strategyResult.similarity || 0) * 100}%`, background: getScoreColor(strategyResult.similarity || 0) }"></div>
                    </div>
                    <span class="breakdown-value">{{ ((strategyResult.similarity || 0) * 100).toFixed(1) }}%</span>
                  </div>
                  <div v-if="strategyResult.weighted_score !== undefined" class="breakdown-contribution">
                    Вклад: {{ (strategyResult.weighted_score * 100).toFixed(1) }}%
                  </div>
                </div>
              </div>
            </div>

            <!-- Collapsible Interpretation -->
            <div v-if="result.interpretation" class="interpretation-section">
              <div
                class="interpretation-header"
                @click="toggleResultInterpretation(i)"
              >
                <span class="interpretation-toggle">
                  {{ expandedResultInterpretations[i] ? '▼' : '▶' }}
                </span>
                <span class="interpretation-label">Интерпретация</span>
              </div>
              <pre
                v-if="expandedResultInterpretations[i]"
                class="result-interpretation"
              >{{ result.interpretation }}</pre>
            </div>
          </div>
        </div>
      </div>

      <Toast ref="toast" />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useTextsStore } from '../store/texts'
import { useAnalysisStore } from '../store/analysis'
import { useSettingsStore } from '../store/settings'
import AppHeader from '../components/AppHeader.vue'
import HistorySidebar from '../components/HistorySidebar.vue'
import Toast from '../components/Toast.vue'
import { formatNumber } from '../utils/format'

const textsStore = useTextsStore()
const analysisStore = useAnalysisStore()
const settingsStore = useSettingsStore()
const router = useRouter()
const toast = ref<any>(null)

// Tab state
const activeTab = ref<'texts' | 'analysis' | 'results'>('texts')

// Upload state
const showUpload = ref(false)
const uploadMode = ref<'text' | 'file'>('text')
const uploadTitle = ref('')
const uploadText = ref('')
const uploading = ref(false)
const selectedFile = ref<File | null>(null)
const selectedFiles = ref<File[]>([])
const fileInput = ref<HTMLInputElement>()
const uploadProgress = ref({ current: 0, total: 0 })

// Selection state
const selectedTexts = ref<any[]>([])
const selectedForDelete = ref<string[]>([])
const deleting = ref(false)
const analyzing = ref<string | null>(null)
const clearing = ref(false)

// Filters and pagination
const searchQuery = ref('')
const sortBy = ref('date-desc')
const sizeFilter = ref('all')
const currentPage = ref(1)
const itemsPerPage = 50

// Collapsible interpretation state
const expandedResultInterpretations = ref<Record<number, boolean>>({})

// Embedding model selection
const selectedModel = ref(settingsStore.getEmbeddingModel)
const availableModels = [
  { value: 'multilingual-e5-large', label: '🌍 E5-Large (лучшее качество, медленно)' },
  { value: 'multilingual-e5-base', label: '🌍 E5-Base (баланс)' },
  { value: 'multilingual-e5-small', label: '🌍 E5-Small (быстро, рекомендуется)' },
  { value: 'labse', label: '🔤 LaBSE (многоязычный)' },
  { value: 'rubert-base', label: '🇷🇺 RuBERT-Base (русский)' },
  { value: 'rubert-tiny', label: '🇷🇺 RuBERT-Tiny (русский, быстро)' },
  { value: 'minilm-l12', label: '⚡ MiniLM-L12 (универсальный)' },
  { value: 'minilm-l6', label: '⚡ MiniLM-L6 (очень быстро)' },
  { value: 'mpnet-base', label: '🎯 MPNet-Base (точность)' },
  { value: 'distilbert', label: '📦 DistilBERT (компактный)' }
]

// Analysis method descriptions
const analysisDescriptions = {
  semantic: 'Семантический анализ использует нейросетевые эмбеддинги (векторные представления) текстов для определения смысловой близости. Модель преобразует тексты в многомерные векторы и вычисляет косинусное сходство между ними. Учитывает контекст, синонимы и семантические связи между словами.',
  style: 'Стилистический анализ оценивает сходство по формальным признакам текста: средняя длина предложений, разнообразие лексики (type-token ratio), использование знаков препинания, частота союзов и предлогов. Помогает определить авторский стиль независимо от содержания.',
  tfidf: 'TF-IDF (Term Frequency-Inverse Document Frequency) анализирует частоту встречаемости слов с учётом их уникальности. Часто встречающиеся в обоих текстах редкие слова дают больший вклад в схожесть. Эффективен для поиска текстов на одну тему.',
  emotion: 'Эмоциональный анализ определяет тональность и эмоциональную окраску текстов. Использует словари эмоционально окрашенных слов для оценки позитивных, негативных и нейтральных настроений. Сравнивает эмоциональные профили двух текстов.',
  llm: 'LLM-анализ использует большую языковую модель (Qwen 2.5) для глубокой оценки схожести текстов. Модель анализирует сюжет, стиль повествования, жанр, персонажей и язык. Даёт детальное объяснение сходств и различий на естественном языке.',
  combined: 'Комбинированный анализ объединяет результаты нескольких методов (семантика, стиль, TF-IDF, эмоции) с весовыми коэффициентами. Даёт наиболее полную оценку схожести текстов, учитывая как содержание, так и форму.',
  chunked: 'Чанкованный анализ разбивает тексты на фрагменты (чанки) и строит детальную карту схожести между ними. Показывает какие части текстов наиболее похожи. Полезен для поиска заимствований и анализа структуры произведений.'
}

// Computed properties
const filteredTexts = computed(() => {
  let texts = [...textsStore.texts]

  // Search filter
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    texts = texts.filter(t => t.title.toLowerCase().includes(query))
  }

  // Size filter
  if (sizeFilter.value !== 'all') {
    texts = texts.filter(t => {
      const len = t.length || 0
      switch (sizeFilter.value) {
        case 'small': return len < 1000
        case 'medium': return len >= 1000 && len <= 10000
        case 'large': return len > 10000
        default: return true
      }
    })
  }

  // Sorting
  texts.sort((a, b) => {
    switch (sortBy.value) {
      case 'date-desc':
        return (b.created_at || '').localeCompare(a.created_at || '')
      case 'date-asc':
        return (a.created_at || '').localeCompare(b.created_at || '')
      case 'size-desc':
        return (b.length || 0) - (a.length || 0)
      case 'size-asc':
        return (a.length || 0) - (b.length || 0)
      case 'name':
        return a.title.localeCompare(b.title)
      default:
        return 0
    }
  })

  return texts
})

const totalPages = computed(() => Math.ceil(filteredTexts.value.length / itemsPerPage))

const paginatedTexts = computed(() => {
  const start = (currentPage.value - 1) * itemsPerPage
  const end = start + itemsPerPage
  return filteredTexts.value.slice(start, end)
})

const allOnPageSelected = computed(() => {
  if (paginatedTexts.value.length === 0) return false
  return paginatedTexts.value.every(text => selectedForDelete.value.includes(text.id))
})

// Watch for filter changes to reset page
watch([searchQuery, sizeFilter, sortBy], () => {
  currentPage.value = 1
})

// Switch to analysis tab when texts are selected
watch(selectedTexts, (newVal) => {
  if (newVal.length >= 2 && activeTab.value === 'texts') {
    activeTab.value = 'analysis'
  }
}, { deep: true })

// Save selected embedding model to settings
watch(selectedModel, (newModel) => {
  settingsStore.setEmbeddingModel(newModel)
})

onMounted(() => {
  textsStore.fetchTexts()
})

function formatDate(date: string | undefined): string {
  if (!date) return ''
  const d = new Date(date)
  return d.toLocaleDateString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    year: '2-digit'
  })
}

function handleFileSelect(event: Event) {
  const target = event.target as HTMLInputElement
  if (target.files) {
    // For single file mode (legacy support)
    if (uploadMode.value === 'file' && target.files[0]) {
      selectedFile.value = target.files[0]
    }

    // For multiple files
    const files = Array.from(target.files)
    const fb2Files = files.filter(f => f.name.endsWith('.fb2'))

    // Limit to 10 files
    const maxFiles = 10 - selectedFiles.value.length
    const filesToAdd = fb2Files.slice(0, maxFiles)

    selectedFiles.value = [...selectedFiles.value, ...filesToAdd]

    if (selectedFiles.value.length > 10) {
      selectedFiles.value = selectedFiles.value.slice(0, 10)
      toast.value?.show('Максимум 10 файлов за раз', 'warning')
    }

    // Clear input to allow re-selection of same files
    target.value = ''
  }
}

function removeFile(index: number) {
  selectedFiles.value.splice(index, 1)
}

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
}

async function handleUpload() {
  if (!uploadTitle.value || !uploadText.value) return

  uploading.value = true
  try {
    await textsStore.uploadText(uploadTitle.value, uploadText.value)
    toast.value?.show('Текст успешно загружен!', 'success')
    uploadTitle.value = ''
    uploadText.value = ''
    showUpload.value = false
  } catch (error: any) {
    toast.value?.show(error.message || 'Ошибка загрузки текста', 'error')
  } finally {
    uploading.value = false
  }
}

async function handleFileUpload() {
  if (!selectedFile.value) return

  uploading.value = true
  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    if (uploadTitle.value) {
      formData.append('title', uploadTitle.value)
    }

    const response = await fetch('/api/v1/texts/upload-fb2', {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.error || 'Ошибка загрузки файла')
    }

    await textsStore.fetchTexts()
    toast.value?.show('Файл успешно загружен!', 'success')
    uploadTitle.value = ''
    selectedFile.value = null
    showUpload.value = false
  } catch (error: any) {
    toast.value?.show(error.message || 'Ошибка загрузки файла', 'error')
  } finally {
    uploading.value = false
  }
}

async function handleMultipleFileUpload() {
  if (selectedFiles.value.length === 0) return

  uploading.value = true
  uploadProgress.value = { current: 0, total: selectedFiles.value.length }

  let successCount = 0
  const failedFiles: string[] = []

  try {
    // Upload files sequentially to show progress
    for (let i = 0; i < selectedFiles.value.length; i++) {
      const file = selectedFiles.value[i]
      uploadProgress.value.current = i + 1

      try {
        const formData = new FormData()
        formData.append('file', file)

        // Use filename without extension as title
        const titleFromFilename = file.name.replace('.fb2', '')
        formData.append('title', titleFromFilename)

        const response = await fetch('/api/v1/texts/upload-fb2', {
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          const error = await response.json()
          throw new Error(error.error || 'Ошибка загрузки')
        }

        successCount++
      } catch (err: any) {
        console.error(`Failed to upload ${file.name}:`, err)
        failedFiles.push(file.name)
      }
    }

    // Refresh texts list
    await textsStore.fetchTexts()

    // Show results
    if (failedFiles.length === 0) {
      toast.value?.show(`Успешно загружено ${successCount} файлов`, 'success')
    } else {
      toast.value?.show(
        `Загружено ${successCount} из ${selectedFiles.value.length} файлов. Ошибки: ${failedFiles.join(', ')}`,
        'warning'
      )
    }

    // Reset form
    showUpload.value = false
    selectedFiles.value = []
    uploadProgress.value = { current: 0, total: 0 }
  } catch (err: any) {
    toast.value?.show(err.message || 'Ошибка загрузки файлов', 'error')
  } finally {
    uploading.value = false
    uploadProgress.value = { current: 0, total: 0 }
  }
}

function selectText(text: any) {
  const idx = selectedTexts.value.findIndex(t => t.id === text.id)
  if (idx >= 0) {
    selectedTexts.value.splice(idx, 1)
  } else {
    selectedTexts.value.push(text)
  }
}

function unselectText(text: any) {
  const idx = selectedTexts.value.findIndex(t => t.id === text.id)
  if (idx >= 0) {
    selectedTexts.value.splice(idx, 1)
  }
}

function isSelected(id: string) {
  return selectedTexts.value.some(t => t.id === id)
}

function toggleDeleteSelect(id: string) {
  const idx = selectedForDelete.value.indexOf(id)
  if (idx >= 0) {
    selectedForDelete.value.splice(idx, 1)
  } else {
    selectedForDelete.value.push(id)
  }
}

async function deleteSelected() {
  if (!selectedForDelete.value.length) return

  if (!confirm(`Вы уверены, что хотите удалить ${selectedForDelete.value.length} текстов?`)) {
    return
  }

  deleting.value = true

  try {
    const result = await textsStore.deleteMultipleTexts(selectedForDelete.value)

    // Clear selections for deleted texts
    selectedTexts.value = selectedTexts.value.filter(t => !selectedForDelete.value.includes(t.id))
    selectedForDelete.value = []

    if (result.failed > 0) {
      toast.value?.show(`Удалено ${result.deleted} текстов, ошибок: ${result.failed}`, 'warning')
    } else {
      toast.value?.show(`Успешно удалено ${result.deleted} текстов`, 'success')
    }
  } catch (error: any) {
    toast.value?.show(error.message || 'Ошибка удаления', 'error')
  } finally {
    deleting.value = false
  }
}

function selectAllOnPage() {
  const pageTextIds = paginatedTexts.value.map(t => t.id)

  if (allOnPageSelected.value) {
    // Deselect all on page
    selectedForDelete.value = selectedForDelete.value.filter(id => !pageTextIds.includes(id))
  } else {
    // Select all on page
    const newSelections = pageTextIds.filter(id => !selectedForDelete.value.includes(id))
    selectedForDelete.value.push(...newSelections)
  }
}

async function runAnalysis(type: string) {
  analyzing.value = type
  try {
    const params: any = {
      text_id1: selectedTexts.value[0].id,
      text_id2: selectedTexts.value[1].id
    }

    // Add model parameter for semantic and chunked analysis
    if (type === 'semantic' || type === 'chunked' || type === 'combined') {
      params.model = selectedModel.value
    }

    const result = await analysisStore.analyze(
      type,
      params,
      selectedTexts.value[0].title,
      selectedTexts.value[1].title
    )

    // Check if it's a background task
    if (result && result.isTask) {
      toast.value?.show('⏳ Анализ запущен в фоне. Результат появится в разделе "Задачи"', 'info')
    } else {
      toast.value?.show('✓ Анализ завершен!', 'success')
      activeTab.value = 'results'
    }
  } catch (error: any) {
    toast.value?.show(error.message || 'Ошибка анализа', 'error')
  } finally {
    analyzing.value = null
  }
}

async function runMatrixAnalysis(analysisType: string) {
  analyzing.value = `matrix-${analysisType}`
  try {
    const textIds = selectedTexts.value.map(t => t.id)
    const result = await analysisStore.analyze(
      'matrix',
      {
        text_ids: textIds,
        analysis_type: analysisType
      }
    )

    // Matrix analysis is always a background task
    if (result && result.isTask) {
      toast.value?.show(`⏳ Матричный анализ (${selectedTexts.value.length} текстов) запущен в фоне`, 'info')
      router.push('/tasks')
    } else {
      toast.value?.show('✓ Матричный анализ завершен!', 'success')
    }
  } catch (error: any) {
    toast.value?.show(error.message || 'Ошибка матричного анализа', 'error')
  } finally {
    analyzing.value = null
  }
}

async function runChunkedAnalysis() {
  analyzing.value = 'chunked'
  try {
    const result = await analysisStore.analyze(
      'chunked',
      {
        text_id1: selectedTexts.value[0].id,
        text_id2: selectedTexts.value[1].id,
        chunk_size: 1000,
        overlap: 100,
        split_by: 'sentences',
        top_n: 10
      }
    )

    // Chunked analysis is a background task
    if (result && result.isTask) {
      toast.value?.show('⏳ Чанковый анализ запущен в фоне', 'info')
      router.push('/tasks')
    } else {
      toast.value?.show('✓ Чанковый анализ завершен!', 'success')
    }
  } catch (error: any) {
    toast.value?.show(error.message || 'Ошибка чанкового анализа', 'error')
  } finally {
    analyzing.value = null
  }
}

function removeResult(index: number) {
  analysisStore.results.splice(index, 1)
  toast.value?.show('Результат удален', 'info')
}

function toggleResultInterpretation(index: number) {
  expandedResultInterpretations.value[index] = !expandedResultInterpretations.value[index]
}

function getAnalysisTitle(type: string): string {
  const titles: Record<string, string> = {
    semantic: '🧠 Семантический анализ',
    style: '✍️ Анализ стиля',
    tfidf: '📊 TF-IDF анализ',
    emotion: '😊 Эмоциональный анализ',
    llm: '🤖 LLM анализ',
    combined: '🎯 Комбинированный анализ'
  }
  return titles[type] || type
}

function getScoreColor(score: number) {
  const hue = score * 120
  return `hsl(${hue}, 70%, 50%)`
}

function hasLLMMetrics(result: any): boolean {
  return result.plot_similarity !== undefined ||
         result.style_similarity !== undefined ||
         result.genre_similarity !== undefined ||
         result.characters_similarity !== undefined ||
         result.language_similarity !== undefined
}

function getLLMMetrics(result: any) {
  const metrics = [
    { key: 'plot_similarity', label: '📖 Сюжет', value: result.plot_similarity },
    { key: 'style_similarity', label: '✍️ Стиль', value: result.style_similarity },
    { key: 'genre_similarity', label: '🎭 Жанр', value: result.genre_similarity },
    { key: 'characters_similarity', label: '👥 Персонажи', value: result.characters_similarity },
    { key: 'language_similarity', label: '🗣️ Язык', value: result.language_similarity }
  ]
  return metrics.filter(m => m.value !== undefined)
}

function formatStrategyName(name: string): string {
  const names: Record<string, string> = {
    semantic: '🧠 Семантика',
    style: '✍️ Стиль',
    tfidf: '📊 TF-IDF',
    emotion: '😊 Эмоции',
    llm: '🤖 LLM'
  }
  return names[name] || name
}

function navigateToEdit(textId: string) {
  router.push(`/text/${textId}`)
}

function navigateToAnalyze(textId: string) {
  router.push(`/text/${textId}/analyze`)
}

async function clearAll() {
  if (!confirm('Вы уверены, что хотите удалить все тексты и очистить историю?')) {
    return
  }

  clearing.value = true
  try {
    // Clear history
    analysisStore.clearHistory()

    // Get all text IDs
    const allTextIds = textsStore.texts.map(t => t.id)

    if (allTextIds.length > 0) {
      // Delete all texts using the batch function
      const result = await textsStore.deleteMultipleTexts(allTextIds)

      if (result.failed > 0) {
        toast.value?.show(`Удалено ${result.deleted} текстов, ошибок: ${result.failed}`, 'warning')
      } else if (result.deleted > 0) {
        toast.value?.show(`Успешно удалено ${result.deleted} текстов`, 'success')
      }
    }

    // Clear results
    analysisStore.results = []

    // Clear selections
    selectedTexts.value = []
    selectedForDelete.value = []

    // Clear localStorage cache
    localStorage.clear()

    toast.value?.show('Все данные успешно удалены', 'success')
    selectedTexts.value = []
    selectedForDelete.value = []
  } catch (error: any) {
    toast.value?.show(error.message || 'Ошибка очистки данных', 'error')
  } finally {
    clearing.value = false
  }
}
</script>

<style scoped>
.page-layout {
  display: grid;
  grid-template-columns: 320px 1fr;
  min-height: 100vh;
  gap: 20px;
}

.sidebar-wrapper {
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
}

.main-content {
  padding: 2rem;
  max-width: 1400px;
  overflow-x: hidden;
}

/* Main Tabs */
.main-tabs {
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
  border-bottom: 2px solid var(--border);
  padding-bottom: 0;
}

.tab-btn-main {
  padding: 1rem 2rem;
  background: transparent;
  border: none;
  color: var(--text-muted);
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  border-bottom: 3px solid transparent;
  margin-bottom: -2px;
}

.tab-btn-main:hover {
  color: var(--text);
  background: var(--bg-hover);
}

.tab-btn-main.active {
  color: var(--primary, #3b82f6);
  border-bottom-color: var(--primary, #3b82f6);
  background: transparent;
}

.tab-content {
  animation: fadeIn 0.3s ease-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Filters Section */
.filters-section {
  display: flex;
  gap: 1rem;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 1.5rem;
  padding: 1rem;
  background: var(--bg-secondary);
  border-radius: 8px;
}

.filter-group {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.filter-label {
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--text-muted);
}

.filter-input, .filter-select {
  padding: 0.5rem 0.75rem;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  font-size: 0.9rem;
}

.filter-input:focus, .filter-select:focus {
  outline: none;
  border-color: var(--primary);
}

.filter-input {
  width: 200px;
}

/* Texts List */
.texts-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.text-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  transition: all 0.2s;
}

.text-row:hover {
  background: var(--bg-hover);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.row-checkbox {
  width: 20px;
  height: 20px;
  cursor: pointer;
  flex-shrink: 0;
}

.text-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  min-width: 0;
}

.text-title {
  font-size: 1rem;
  font-weight: 500;
  color: var(--text);
  cursor: pointer;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  transition: color 0.2s;
}

.text-title:hover {
  color: var(--primary);
}

.text-stats {
  display: flex;
  gap: 1rem;
  font-size: 0.85rem;
  color: var(--text-muted);
}

.stat {
  white-space: nowrap;
}

.text-actions {
  display: flex;
  gap: 0.5rem;
  flex-shrink: 0;
}

.btn-analyze {
  padding: 0.5rem 1rem;
  background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
  border: none;
  color: white;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.2s;
  flex-shrink: 0;
}

.btn-analyze:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
}

.btn-select-row {
  padding: 0.5rem 1rem;
  background: var(--bg);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.2s;
  flex-shrink: 0;
}

.btn-select-row.selected {
  background: var(--success, #10b981);
  color: white;
  border-color: var(--success);
}

.btn-select-row:hover:not(.selected) {
  background: var(--primary);
  color: white;
  border-color: var(--primary);
}

/* Pagination */
.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
  margin-top: 2rem;
  padding: 1rem;
}

.page-btn {
  padding: 0.5rem 0.75rem;
  background: var(--bg-hover);
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.2s;
  color: var(--text);
}

.page-btn:hover:not(:disabled) {
  background: var(--primary);
  color: white;
  border-color: var(--primary);
}

.page-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-info {
  font-size: 0.95rem;
  color: var(--text-muted);
  font-weight: 500;
}

/* Analysis Panel */
.analysis-panel {
  padding: 1.5rem;
  background: var(--bg-secondary);
  border-radius: 12px;
}

.selected-texts {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.5rem;
  padding: 1rem;
  background: var(--bg);
  border-radius: 8px;
}

.selected-text {
  flex: 1;
  font-size: 0.95rem;
}

.label {
  font-weight: 600;
  color: var(--primary);
}

.vs-divider {
  font-weight: 700;
  color: var(--text-muted);
  font-size: 1.2rem;
}

.selected-texts-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding: 1rem;
  background: var(--bg);
  border-radius: 8px;
  margin-bottom: 1rem;
}

.selected-text-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.95rem;
  color: var(--text);
}

.text-number {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  background: var(--primary, #3b82f6);
  color: white;
  border-radius: 50%;
  font-weight: 600;
  font-size: 0.85rem;
}

.btn-remove {
  margin-left: auto;
  padding: 0.25rem 0.5rem;
  background: var(--error, #ef4444);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s;
}

.btn-remove:hover {
  background: #dc2626;
}

/* Results */
.results-list {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.result-actions {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.btn-remove-result {
  padding: 0.5rem;
  background: var(--bg-hover);
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
  color: var(--text-muted);
}

.btn-remove-result:hover {
  background: var(--error);
  color: white;
  border-color: var(--error);
}

/* Keep existing button styles */
.btn-primary, .btn-secondary, .btn-danger, .btn-file {
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

.btn-secondary {
  background: var(--bg-hover);
  color: var(--text);
  border: 1px solid var(--border);
}

.btn-secondary:hover:not(:disabled) {
  background: var(--bg-active);
}

.btn-danger {
  background: var(--error, #ef4444);
  color: white;
}

.btn-danger:hover:not(:disabled) {
  background: #dc2626;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none !important;
}

/* Upload section remains the same */
.upload-section {
  margin-bottom: 2rem;
  animation: slideDown 0.3s ease-out;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.upload-tabs {
  display: flex;
  gap: 0.5rem;
  margin: 1rem 0;
}

.tab-btn {
  padding: 0.5rem 1rem;
  background: var(--bg-hover);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.tab-btn.active {
  background: var(--primary);
  color: white;
  border-color: var(--primary);
}

.upload-form {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-top: 1rem;
}

.input, .textarea {
  padding: 0.75rem;
  background: var(--bg-hover);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  font-size: 1rem;
}

.input:focus, .textarea:focus {
  outline: none;
  border-color: var(--primary);
}

.file-upload {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  width: 100%;
}

.file-input {
  display: none;
}

.btn-file {
  background: var(--bg-hover);
  border: 1px solid var(--border);
  color: var(--text);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  transition: all 0.2s;
}

.btn-file:hover:not(:disabled) {
  border-color: var(--primary);
  transform: translateY(-1px);
}

.file-name {
  font-size: 0.9rem;
  color: var(--text-muted);
}

.selected-files {
  background: var(--bg-hover);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
}

.selected-files h4 {
  margin: 0 0 0.75rem 0;
  font-size: 0.95rem;
  color: var(--text);
}

.file-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  max-height: 200px;
  overflow-y: auto;
}

.file-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem;
  background: var(--bg-secondary);
  border-radius: 6px;
  font-size: 0.9rem;
  color: var(--text);
}

.file-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-size {
  color: var(--text-muted);
  font-size: 0.85rem;
  margin-left: 1rem;
}

.upload-progress {
  margin-top: 1rem;
  padding: 0.75rem;
  background: var(--bg-hover);
  border-radius: 6px;
  border: 1px solid var(--primary);
}

.progress-text {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.9rem;
  color: var(--text);
  margin-bottom: 0.5rem;
}

.progress-bar {
  height: 8px;
  background: var(--bg-secondary);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--primary);
  transition: width 0.3s ease;
}

.upload-actions {
  display: flex;
  gap: 0.75rem;
}

/* Model Selector */
.model-selector {
  margin: 1.5rem 0;
  padding: 1rem;
  background: var(--bg);
  border-radius: 8px;
  border: 1px solid var(--border);
}

.model-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 0.75rem;
  font-size: 0.95rem;
}

.label-icon {
  font-size: 1.1rem;
}

.model-select {
  width: 100%;
  padding: 0.75rem;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  font-size: 0.95rem;
  cursor: pointer;
  transition: all 0.2s;
}

.model-select:hover {
  border-color: var(--primary);
}

.model-select:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.model-hint {
  margin-top: 0.5rem;
  font-size: 0.85rem;
  color: var(--text-muted);
  font-style: italic;
}

.analysis-methods {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.analysis-btn {
  padding: 0.75rem 1.5rem;
  background: var(--bg);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.2s;
}

.analysis-btn:hover:not(:disabled) {
  background: var(--primary);
  color: white;
  border-color: var(--primary);
  transform: translateY(-2px);
}

.analysis-btn.btn-special {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
}

.matrix-hint {
  margin: 0.5rem 0 1rem 0;
  padding: 0.75rem;
  background: var(--bg-active, #252525);
  border-left: 3px solid var(--primary, #3b82f6);
  border-radius: 4px;
  color: var(--text-muted);
  font-size: 0.9rem;
}

.empty-state, .loading {
  text-align: center;
  padding: 3rem 2rem;
  color: var(--text-muted);
  font-size: 1.1rem;
}

.result-card {
  padding: 1.5rem;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 12px;
  transition: box-shadow 0.2s;
}

.result-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.result-header h4 {
  margin: 0;
  font-size: 1.2rem;
}

.similarity-score {
  display: inline-block;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-weight: bold;
  font-size: 1.1rem;
  color: white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.interpretation-section {
  margin-top: 1rem;
  background: var(--bg-hover, rgba(255, 255, 255, 0.03));
  border-radius: 8px;
  overflow: hidden;
  border-left: 3px solid var(--primary, #3b82f6);
}

.interpretation-header {
  padding: 0.75rem 1rem;
  cursor: pointer;
  user-select: none;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: background 0.2s;
}

.interpretation-header:hover {
  background: rgba(59, 130, 246, 0.1);
}

.interpretation-toggle {
  font-size: 0.75rem;
  color: var(--primary);
}

.interpretation-label {
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--primary);
}

.result-interpretation {
  margin: 0;
  padding: 1rem;
  line-height: 1.6;
  color: var(--text-muted);
  white-space: pre-wrap;
  font-family: inherit;
  font-size: 0.9rem;
  background: transparent;
  border: none;
  animation: slideDown 0.2s ease;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Keep existing LLM and breakdown styles as they were */
.llm-metrics {
  margin: 1.5rem 0;
  padding: 1rem;
  background: var(--bg);
  border-radius: 8px;
}

.metric-grid {
  display: grid;
  gap: 1rem;
}

.metric-item {
  display: grid;
  grid-template-columns: 120px 1fr 60px;
  align-items: center;
  gap: 1rem;
}

.metric-label {
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--text);
}

.metric-bar {
  height: 20px;
  background: var(--bg-active);
  border-radius: 10px;
  overflow: hidden;
}

.metric-bar-fill {
  height: 100%;
  transition: width 0.3s ease;
  border-radius: 10px;
}

.metric-value {
  font-size: 0.9rem;
  font-weight: 600;
  text-align: right;
  color: var(--text);
}

.combined-breakdown {
  margin: 1.5rem 0;
  padding: 1rem;
  background: var(--bg);
  border-radius: 8px;
}

.combined-breakdown h5 {
  margin: 0 0 1rem 0;
  font-size: 1rem;
  color: var(--text);
}

.breakdown-grid {
  display: grid;
  gap: 1rem;
}

.breakdown-item {
  padding: 0.75rem;
  background: var(--bg-active);
  border-radius: 6px;
}

.breakdown-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.breakdown-name {
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--text);
}

.breakdown-weight {
  font-size: 0.85rem;
  color: var(--text-muted);
  font-style: italic;
}

.breakdown-score {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
}

.breakdown-bar {
  flex: 1;
  height: 16px;
  background: var(--bg-secondary);
  border-radius: 8px;
  overflow: hidden;
}

.breakdown-bar-fill {
  height: 100%;
  transition: width 0.3s ease;
  border-radius: 8px;
}

.breakdown-value {
  font-size: 0.9rem;
  font-weight: 600;
  min-width: 50px;
  text-align: right;
  color: var(--text);
}

.breakdown-contribution {
  font-size: 0.85rem;
  color: var(--primary);
  font-weight: 500;
}
</style>