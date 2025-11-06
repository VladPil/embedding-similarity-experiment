<template>
  <div class="page-layout">
    <aside class="sidebar-wrapper">
      <HistorySidebar />
    </aside>
    <main class="main-content">
      <AppHeader title="🔧 Настройки" subtitle="Кастомизация параметров анализа и интерфейса">
        <template #actions>
          <button @click="resetToDefaults" class="btn-secondary">
            🔄 Сбросить
          </button>
          <button @click="saveSettings" class="btn-primary">
            💾 Сохранить
          </button>
        </template>
      </AppHeader>

      <div class="settings-content">
        <!-- Embedding Models Section -->
        <section class="settings-section">
          <div class="section-header" @click="toggleSection('embeddings')">
            <div class="section-title">
              <h2>🧠 Модели Embeddings</h2>
              <span class="toggle-icon">{{ expandedSections.embeddings ? '▼' : '▶' }}</span>
            </div>
            <p>Выбор модели для семантического анализа</p>
          </div>
          <div v-show="expandedSections.embeddings" class="section-body">
            <div class="settings-grid">
              <div class="setting-item">
                <label class="setting-label">
                  <span class="label-text">
                    Модель по умолчанию
                    <button @click="showHelp('embeddingModel')" class="help-btn" title="Подробнее">?</button>
                  </span>
                </label>
                <select v-model="settings.defaultEmbeddingModel" class="setting-select">
                  <option value="multilingual-e5-large">🌍 E5-Large (1024 dim)</option>
                  <option value="multilingual-e5-base">🌍 E5-Base (768 dim)</option>
                  <option value="multilingual-e5-small">🌍 E5-Small (384 dim)</option>
                  <option value="labse">🔤 LaBSE (768 dim)</option>
                  <option value="rubert-base">🇷🇺 RuBERT-Base</option>
                  <option value="rubert-tiny">🇷🇺 RuBERT-Tiny</option>
                  <option value="minilm-l12">⚡ MiniLM-L12</option>
                  <option value="minilm-l6">⚡ MiniLM-L6</option>
                  <option value="mpnet-base">🎯 MPNet-Base</option>
                  <option value="distilbert">📦 DistilBERT</option>
                </select>
                <div v-if="activeHelp === 'embeddingModel'" class="help-text">
                  <strong>Что это:</strong> Модель для преобразования текста в векторное представление (embedding).<br>
                  <strong>На что влияет:</strong> Качество семантического анализа, скорость обработки, потребление памяти.<br>
                  <strong>Рекомендации:</strong>
                  <ul>
                    <li><strong>E5-Large:</strong> Лучшее качество, но медленнее (1024 измерения)</li>
                    <li><strong>E5-Small:</strong> Быстро, подходит для большинства задач (384 измерения)</li>
                    <li><strong>RuBERT:</strong> Оптимизирован для русского языка</li>
                    <li><strong>MiniLM:</strong> Очень быстрый, компактный</li>
                  </ul>
                </div>
              </div>

              <div class="setting-item">
                <label class="setting-label">
                  <span class="label-text">
                    Макс. длина последовательности
                    <button @click="showHelp('maxSeq')" class="help-btn" title="Подробнее">?</button>
                  </span>
                </label>
                <input v-model.number="settings.maxSequenceLength" type="number" min="128" max="2048" step="128" class="setting-input" />
                <div v-if="activeHelp === 'maxSeq'" class="help-text">
                  <strong>Что это:</strong> Максимальное количество токенов (слов) для обработки за один раз.<br>
                  <strong>На что влияет:</strong> Размер обрабатываемых фрагментов текста.<br>
                  <strong>Рекомендации:</strong> 512 оптимально для большинства моделей. Больше = медленнее, но длиннее контекст.
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- LLM Models Section -->
        <section class="settings-section">
          <div class="section-header" @click="toggleSection('llm')">
            <div class="section-title">
              <h2>🤖 LLM Модели</h2>
              <span class="toggle-icon">{{ expandedSections.llm ? '▼' : '▶' }}</span>
            </div>
            <p>Параметры для анализа с большими языковыми моделями</p>
          </div>
          <div v-show="expandedSections.llm" class="section-body">
            <div class="settings-grid">
              <div class="setting-item">
                <label class="setting-label">
                  <span class="label-text">
                    LLM модель
                    <button @click="showHelp('llmModel')" class="help-btn" title="Подробнее">?</button>
                  </span>
                </label>
                <select v-model="settings.defaultLLMModel" class="setting-select">
                  <option value="qwen2.5-0.5b">Qwen2.5-0.5B (быстро)</option>
                  <option value="qwen2.5-1.5b">Qwen2.5-1.5B (баланс)</option>
                  <option value="qwen2.5-3b">Qwen2.5-3B (качество)</option>
                  <option value="qwen2.5-7b">Qwen2.5-7B (лучшее)</option>
                </select>
                <div v-if="activeHelp === 'llmModel'" class="help-text">
                  <strong>Что это:</strong> Большая языковая модель для глубокого анализа текстов.<br>
                  <strong>На что влияет:</strong> Качество анализа, скорость, потребление VRAM.<br>
                  <strong>Рекомендации:</strong>
                  <ul>
                    <li><strong>0.5B:</strong> ~500 MB VRAM, быстро, базовое качество</li>
                    <li><strong>3B:</strong> ~3 GB VRAM, хорошее качество (рекомендуется)</li>
                    <li><strong>7B:</strong> ~7 GB VRAM, лучшее качество, медленнее</li>
                  </ul>
                </div>
              </div>

              <div class="setting-item">
                <label class="setting-label">
                  <span class="label-text">
                    Устройство
                    <button @click="showHelp('llmDevice')" class="help-btn" title="Подробнее">?</button>
                  </span>
                </label>
                <select v-model="settings.llmDevice" class="setting-select">
                  <option value="cuda">🎮 GPU (CUDA)</option>
                  <option value="cpu">💻 CPU</option>
                </select>
                <div v-if="activeHelp === 'llmDevice'" class="help-text">
                  <strong>Что это:</strong> Устройство для запуска LLM модели.<br>
                  <strong>На что влияет:</strong> Скорость генерации (GPU в 10-100 раз быстрее).<br>
                  <strong>Рекомендации:</strong> Используйте GPU если доступен.
                </div>
              </div>

              <div class="setting-item">
                <label class="setting-label">
                  <span class="label-text">
                    Макс. памяти GPU (GB)
                    <button @click="showHelp('llmMemory')" class="help-btn" title="Подробнее">?</button>
                  </span>
                </label>
                <input v-model.number="settings.llmMaxMemoryGB" type="number" min="1" max="48" step="1" class="setting-input" />
                <div v-if="activeHelp === 'llmMemory'" class="help-text">
                  <strong>Что это:</strong> Лимит использования видеопамяти для LLM.<br>
                  <strong>На что влияет:</strong> Предотвращает переполнение VRAM.<br>
                  <strong>Рекомендации:</strong> Установите на 1-2 GB меньше доступной VRAM.
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- Analysis Parameters Section -->
        <section class="settings-section">
          <div class="section-header" @click="toggleSection('analysis')">
            <div class="section-title">
              <h2>📊 Параметры Анализа</h2>
              <span class="toggle-icon">{{ expandedSections.analysis ? '▼' : '▶' }}</span>
            </div>
            <p>Настройка алгоритмов сравнения текстов</p>
          </div>
          <div v-show="expandedSections.analysis" class="section-body">
            <div class="settings-grid">
              <div class="setting-item">
                <label class="setting-label">
                  <span class="label-text">
                    Размер чанка (символов)
                    <button @click="showHelp('chunkSize')" class="help-btn" title="Подробнее">?</button>
                  </span>
                </label>
                <input v-model.number="settings.chunkSize" type="number" min="500" max="5000" step="100" class="setting-input" />
                <div v-if="activeHelp === 'chunkSize'" class="help-text">
                  <strong>Что это:</strong> Размер фрагмента текста для chunk-анализа.<br>
                  <strong>На что влияет:</strong> Гранулярность сравнения больших текстов.<br>
                  <strong>Рекомендации:</strong> 1000-2000 для книг, 500-1000 для статей.
                </div>
              </div>

              <div class="setting-item">
                <label class="setting-label">
                  <span class="label-text">
                    Перекрытие (символов)
                    <button @click="showHelp('chunkOverlap')" class="help-btn" title="Подробнее">?</button>
                  </span>
                </label>
                <input v-model.number="settings.chunkOverlap" type="number" min="0" max="1000" step="50" class="setting-input" />
                <div v-if="activeHelp === 'chunkOverlap'" class="help-text">
                  <strong>Что это:</strong> Количество символов пересечения между соседними чанками.<br>
                  <strong>На что влияет:</strong> Непрерывность анализа на границах фрагментов.<br>
                  <strong>Рекомендации:</strong> 10-20% от размера чанка (например, 200 при чанке 1000).
                </div>
              </div>

              <div class="setting-item">
                <label class="setting-label">
                  <span class="label-text">
                    Количество сегментов
                    <button @click="showHelp('segments')" class="help-btn" title="Подробнее">?</button>
                  </span>
                </label>
                <input v-model.number="settings.defaultSegments" type="number" min="5" max="50" step="5" class="setting-input" />
                <div v-if="activeHelp === 'segments'" class="help-text">
                  <strong>Что это:</strong> На сколько частей разбивать текст при адаптивном chunking.<br>
                  <strong>На что влияет:</strong> Детальность анализа структуры текста.<br>
                  <strong>Рекомендации:</strong> 10-20 для большинства задач.
                </div>
              </div>

              <div class="setting-item full-width">
                <label class="setting-label">
                  <span class="label-text">
                    Стратегии для комбо-анализа
                    <button @click="showHelp('strategies')" class="help-btn" title="Подробнее">?</button>
                  </span>
                </label>
                <div class="checkbox-group">
                  <label class="checkbox-label">
                    <input type="checkbox" v-model="settings.combinedStrategies.semantic" />
                    <span>Семантический</span>
                  </label>
                  <label class="checkbox-label">
                    <input type="checkbox" v-model="settings.combinedStrategies.style" />
                    <span>Стилистический</span>
                  </label>
                  <label class="checkbox-label">
                    <input type="checkbox" v-model="settings.combinedStrategies.tfidf" />
                    <span>TF-IDF</span>
                  </label>
                  <label class="checkbox-label">
                    <input type="checkbox" v-model="settings.combinedStrategies.emotion" />
                    <span>Эмоциональный</span>
                  </label>
                  <label class="checkbox-label">
                    <input type="checkbox" v-model="settings.combinedStrategies.llm" />
                    <span>LLM</span>
                  </label>
                </div>
                <div v-if="activeHelp === 'strategies'" class="help-text">
                  <strong>Что это:</strong> Методы анализа для комбинированного подхода.<br>
                  <strong>На что влияет:</strong> Полнота и скорость комплексного анализа.<br>
                  <strong>Рекомендации:</strong>
                  <ul>
                    <li><strong>Семантический:</strong> Смысловая близость (обязательно)</li>
                    <li><strong>Стилистический:</strong> Авторский стиль, формальность</li>
                    <li><strong>TF-IDF:</strong> Уникальные ключевые слова</li>
                    <li><strong>Эмоциональный:</strong> Тональность текста</li>
                    <li><strong>LLM:</strong> Глубокий анализ (медленно)</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- Performance Section -->
        <section class="settings-section">
          <div class="section-header" @click="toggleSection('performance')">
            <div class="section-title">
              <h2>⚡ Производительность</h2>
              <span class="toggle-icon">{{ expandedSections.performance ? '▼' : '▶' }}</span>
            </div>
            <p>Настройка кэширования и параллелизма</p>
          </div>
          <div v-show="expandedSections.performance" class="section-body">
            <div class="settings-grid">
              <div class="setting-item">
                <label class="setting-label">
                  <span class="label-text">
                    Макс. параллельных задач
                    <button @click="showHelp('workers')" class="help-btn" title="Подробнее">?</button>
                  </span>
                </label>
                <input v-model.number="settings.maxWorkers" type="number" min="1" max="10" step="1" class="setting-input" />
                <div v-if="activeHelp === 'workers'" class="help-text">
                  <strong>Что это:</strong> Количество одновременно выполняемых анализов.<br>
                  <strong>На что влияет:</strong> Скорость обработки нескольких задач, нагрузка на систему.<br>
                  <strong>Рекомендации:</strong> 3-5 для мощных систем, 1-2 для слабых.
                </div>
              </div>

              <div class="setting-item">
                <label class="setting-label">
                  <span class="label-text">
                    TTL кэша (секунды)
                    <button @click="showHelp('cacheTTL')" class="help-btn" title="Подробнее">?</button>
                  </span>
                </label>
                <input v-model.number="settings.cacheTTL" type="number" min="600" max="86400" step="600" class="setting-input" />
                <div v-if="activeHelp === 'cacheTTL'" class="help-text">
                  <strong>Что это:</strong> Время хранения embeddings в Redis (кэш L1).<br>
                  <strong>На что влияет:</strong> Баланс между скоростью и использованием памяти.<br>
                  <strong>Рекомендации:</strong> 3600 (1 час) для активной работы, 86400 (1 день) для долгосрочных проектов.
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- UI Section -->
        <section class="settings-section">
          <div class="section-header" @click="toggleSection('ui')">
            <div class="section-title">
              <h2>🎨 Интерфейс</h2>
              <span class="toggle-icon">{{ expandedSections.ui ? '▼' : '▶' }}</span>
            </div>
            <p>Персонализация внешнего вида</p>
          </div>
          <div v-show="expandedSections.ui" class="section-body">
            <div class="settings-grid">
              <div class="setting-item">
                <label class="setting-label">
                  <span class="label-text">Тема оформления</span>
                </label>
                <select v-model="settings.theme" class="setting-select">
                  <option value="light">☀️ Светлая</option>
                  <option value="dark">🌙 Тёмная</option>
                  <option value="auto">🔄 Автоматически</option>
                </select>
              </div>

              <div class="setting-item">
                <label class="setting-label">
                  <span class="label-text">Элементов в истории</span>
                </label>
                <input v-model.number="settings.historyLimit" type="number" min="3" max="20" step="1" class="setting-input" />
              </div>

              <div class="setting-item">
                <label class="setting-label">
                  <span class="label-text">Автоудаление задач</span>
                </label>
                <div class="toggle-switch">
                  <input type="checkbox" id="autoRemove" v-model="settings.autoRemoveCompletedTasks" />
                  <label for="autoRemove" class="toggle-label"></label>
                </div>
              </div>

              <div class="setting-item">
                <label class="setting-label">
                  <span class="label-text">Уведомления</span>
                </label>
                <div class="toggle-switch">
                  <input type="checkbox" id="showNotifications" v-model="settings.showNotifications" />
                  <label for="showNotifications" class="toggle-label"></label>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- Cache Management Section -->
        <section class="settings-section">
          <div class="section-header" @click="toggleSection('cache')">
            <div class="section-title">
              <h2>🗄️ Управление Кэшем</h2>
              <span class="toggle-icon">{{ expandedSections.cache ? '▼' : '▶' }}</span>
            </div>
            <p>Просмотр и очистка кэшированных данных</p>
          </div>
          <div v-show="expandedSections.cache" class="section-body">
            <div class="cache-stats" v-if="cacheStats">
              <div class="stat-card">
                <div class="stat-icon">📦</div>
                <div class="stat-content">
                  <div class="stat-value">{{ cacheStats.total_embeddings }}</div>
                  <div class="stat-label">Всего embeddings</div>
                </div>
              </div>
              <div class="stat-card">
                <div class="stat-icon">🔑</div>
                <div class="stat-content">
                  <div class="stat-value">{{ cacheStats.redis?.keys || 0 }}</div>
                  <div class="stat-label">Redis ключей</div>
                </div>
              </div>
              <div class="stat-card">
                <div class="stat-icon">💾</div>
                <div class="stat-content">
                  <div class="stat-value">{{ cacheStats.redis?.memory_used || 'N/A' }}</div>
                  <div class="stat-label">Памяти использовано</div>
                </div>
              </div>
            </div>

            <div class="cache-actions">
              <button @click="loadCacheStats" class="btn-secondary" :disabled="loadingCache">
                {{ loadingCache ? '⏳ Загрузка...' : '🔄 Обновить' }}
              </button>
              <button @click="clearCache" class="btn-danger" :disabled="loadingCache">
                🗑️ Очистить кэш
              </button>
            </div>
          </div>
        </section>

        <!-- Action Buttons -->
        <div class="settings-actions">
          <button @click="exportSettings" class="btn-secondary">
            📥 Экспорт
          </button>
          <button @click="importSettings" class="btn-secondary">
            📤 Импорт
          </button>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import axios from 'axios'
import AppHeader from '../components/AppHeader.vue'
import HistorySidebar from '../components/HistorySidebar.vue'

interface Settings {
  defaultEmbeddingModel: string
  maxSequenceLength: number
  defaultLLMModel: string
  llmDevice: string
  llmMaxMemoryGB: number
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
  maxWorkers: number
  cacheTTL: number
  theme: string
  language: string
  historyLimit: number
  autoRemoveCompletedTasks: boolean
  showNotifications: boolean
}

interface CacheStats {
  total_embeddings: number
  by_model: Record<string, number>
  redis: {
    keys: number
    memory_used: string
    hits: number
    misses: number
  }
}

const API_BASE_URL = 'http://localhost:8000/api/v1'

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

const settings = reactive<Settings>({ ...defaultSettings })
const cacheStats = ref<CacheStats | null>(null)
const loadingCache = ref(false)
const activeHelp = ref<string | null>(null)
const expandedSections = reactive({
  embeddings: true,
  llm: false,
  analysis: false,
  performance: false,
  ui: false,
  cache: false
})

onMounted(() => {
  loadSettings()
  loadCacheStats()
})

// Section toggle
function toggleSection(section: keyof typeof expandedSections) {
  expandedSections[section] = !expandedSections[section]
}

// Help toggle
function showHelp(helpId: string) {
  activeHelp.value = activeHelp.value === helpId ? null : helpId
}

// Settings management
function loadSettings() {
  const saved = localStorage.getItem('app-settings')
  if (saved) {
    try {
      const parsed = JSON.parse(saved)
      Object.assign(settings, parsed)
    } catch (e) {
      console.error('Failed to load settings:', e)
    }
  }
}

function saveSettings() {
  try {
    localStorage.setItem('app-settings', JSON.stringify(settings))
    alert('✅ Настройки сохранены!')
  } catch (e) {
    console.error('Failed to save settings:', e)
    alert('❌ Ошибка сохранения настроек')
  }
}

function resetToDefaults() {
  if (confirm('Вы уверены? Все настройки будут сброшены к значениям по умолчанию.')) {
    Object.assign(settings, defaultSettings)
    saveSettings()
  }
}

function exportSettings() {
  const dataStr = JSON.stringify(settings, null, 2)
  const dataBlob = new Blob([dataStr], { type: 'application/json' })
  const url = URL.createObjectURL(dataBlob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'settings.json'
  link.click()
  URL.revokeObjectURL(url)
}

function importSettings() {
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = '.json'
  input.onchange = (e: any) => {
    const file = e.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (event: any) => {
        try {
          const imported = JSON.parse(event.target.result)
          Object.assign(settings, imported)
          saveSettings()
          alert('✅ Настройки импортированы!')
        } catch (e) {
          alert('❌ Ошибка чтения файла')
        }
      }
      reader.readAsText(file)
    }
  }
  input.click()
}

// Cache management
async function loadCacheStats() {
  loadingCache.value = true
  try {
    const response = await axios.get(`${API_BASE_URL}/cache/stats`)
    if (response.data.success) {
      cacheStats.value = response.data.data
    }
  } catch (error: any) {
    console.error('Failed to load cache stats:', error)
    // Don't show alert on initial load
    if (cacheStats.value !== null) {
      alert('❌ Не удалось загрузить статистику кэша')
    }
  } finally {
    loadingCache.value = false
  }
}

async function clearCache() {
  if (!confirm('Вы уверены? Весь кэш embeddings будет удалён.')) {
    return
  }

  loadingCache.value = true
  try {
    await axios.post(`${API_BASE_URL}/cache/clear`)
    alert('✅ Кэш очищен!')
    await loadCacheStats()
  } catch (error) {
    console.error('Failed to clear cache:', error)
    alert('❌ Ошибка очистки кэша')
  } finally {
    loadingCache.value = false
  }
}
</script>

<style scoped>
/* Layout */
.page-layout {
  display: grid;
  grid-template-columns: 320px 1fr;
  min-height: 100vh;
  gap: 20px;
  background: var(--bg);
}

.sidebar-wrapper {
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
}

.main-content {
  max-width: 1400px;
  padding: 2rem;
  overflow-x: hidden;
}

/* Settings Content */
.settings-content {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.settings-section {
  background: var(--bg-secondary);
  border-radius: 12px;
  border: 1px solid var(--border);
  overflow: hidden;
}

.section-header {
  padding: 1.25rem 1.5rem;
  cursor: pointer;
  border-bottom: 1px solid var(--border);
  transition: background 0.2s;
}

.section-header:hover {
  background: rgba(255, 255, 255, 0.02);
}

.section-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.25rem;
}

.section-title h2 {
  font-size: 1.25rem;
  margin: 0;
  color: var(--text);
}

.toggle-icon {
  color: var(--text-muted);
  font-size: 0.875rem;
}

.section-header p {
  color: var(--text-muted);
  font-size: 0.875rem;
  margin: 0;
}

.section-body {
  padding: 1.5rem;
}

/* Settings Grid - 2 columns */
.settings-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
}

.setting-item {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.setting-item.full-width {
  grid-column: 1 / -1;
}

.setting-label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.label-text {
  font-weight: 600;
  color: var(--text);
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.help-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--primary);
  color: white;
  border: none;
  cursor: pointer;
  font-size: 0.75rem;
  font-weight: bold;
  transition: transform 0.2s;
}

.help-btn:hover {
  transform: scale(1.1);
}

.help-text {
  margin-top: 0.5rem;
  padding: 1rem;
  background: rgba(59, 130, 246, 0.1);
  border-left: 3px solid var(--primary);
  border-radius: 4px;
  font-size: 0.875rem;
  line-height: 1.6;
  color: var(--text);
}

.help-text strong {
  color: var(--primary);
}

.help-text ul {
  margin: 0.5rem 0 0 0;
  padding-left: 1.5rem;
}

.help-text li {
  margin: 0.25rem 0;
}

.setting-select,
.setting-input {
  padding: 0.625rem;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  font-size: 0.875rem;
  transition: all 0.2s;
}

.setting-select:hover,
.setting-input:hover {
  border-color: var(--primary);
}

.setting-select:focus,
.setting-input:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.checkbox-group {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  padding: 0.5rem 1rem;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  transition: all 0.2s;
  font-size: 0.875rem;
}

.checkbox-label:hover {
  border-color: var(--primary);
  background: rgba(59, 130, 246, 0.05);
}

.checkbox-label input[type="checkbox"] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

/* Toggle Switch */
.toggle-switch {
  position: relative;
  display: inline-block;
  width: 50px;
  height: 28px;
}

.toggle-switch input[type="checkbox"] {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-label {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #ccc;
  transition: 0.4s;
  border-radius: 28px;
}

.toggle-label:before {
  position: absolute;
  content: "";
  height: 20px;
  width: 20px;
  left: 4px;
  bottom: 4px;
  background-color: white;
  transition: 0.4s;
  border-radius: 50%;
}

input[type="checkbox"]:checked + .toggle-label {
  background-color: var(--primary);
}

input[type="checkbox"]:checked + .toggle-label:before {
  transform: translateX(22px);
}

/* Cache Stats */
.cache-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
}

.stat-card {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
}

.stat-icon {
  font-size: 1.75rem;
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--text);
}

.stat-label {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.cache-actions {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

/* Action Buttons */
.settings-actions {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  justify-content: center;
  padding: 1rem 0;
}

/* Buttons */
.btn-primary,
.btn-secondary,
.btn-danger {
  padding: 0.625rem 1.25rem;
  border: none;
  border-radius: 6px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 0.875rem;
}

.btn-primary {
  background: var(--primary);
  color: white;
}

.btn-primary:hover {
  background: #2563eb;
  transform: translateY(-1px);
}

.btn-secondary {
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--border);
}

.btn-secondary:hover {
  background: var(--bg-secondary);
  transform: translateY(-1px);
}

.btn-danger {
  background: #ef4444;
  color: white;
}

.btn-danger:hover {
  background: #dc2626;
  transform: translateY(-1px);
}

.btn-primary:disabled,
.btn-secondary:disabled,
.btn-danger:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}

/* Responsive */
@media (max-width: 768px) {
  .container {
    padding: 1rem;
  }

  .header-top {
    flex-direction: column;
    gap: 1rem;
  }

  .header-actions {
    width: 100%;
    flex-direction: column;
  }

  .header-nav {
    flex-wrap: wrap;
  }

  .nav-tab {
    flex: 1;
    min-width: 120px;
  }

  .settings-grid {
    grid-template-columns: 1fr;
  }

  .cache-stats {
    grid-template-columns: 1fr;
  }
}
</style>
