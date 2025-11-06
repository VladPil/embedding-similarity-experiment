<template>
  <div class="page-layout">
    <aside class="sidebar-wrapper">
      <HistorySidebar />
    </aside>
    <main class="main-content">
      <AppHeader title="⚙️ Мониторинг задач" subtitle="Отслеживание статуса фоновых задач" />

      <div class="page-actions">
        <button @click="fetchTasks" :disabled="loading" class="btn-secondary">
          {{ loading ? '⏳' : '🔄' }} Обновить
        </button>
      </div>

      <div class="stats-cards">
        <div class="stat-card">
          <div class="stat-icon">⏳</div>
          <div class="stat-info">
            <div class="stat-value">{{ stats.running }}</div>
            <div class="stat-label">Выполняются</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">📋</div>
          <div class="stat-info">
            <div class="stat-value">{{ stats.pending }}</div>
            <div class="stat-label">В очереди</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">✅</div>
          <div class="stat-info">
            <div class="stat-value">{{ stats.completed }}</div>
            <div class="stat-label">Завершено</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">❌</div>
          <div class="stat-info">
            <div class="stat-value">{{ stats.failed }}</div>
            <div class="stat-label">Ошибок</div>
          </div>
        </div>
      </div>

      <div v-if="loading && !tasks.length && !hasLoadedOnce" class="loading">⏳ Загрузка задач...</div>
      <div v-else-if="!loading && !tasks.length" class="empty-state">
        Нет активных задач. Все задачи завершены!
      </div>
      <div v-else class="tasks-list">
        <div v-for="task in tasks" :key="task.id" class="task-card card">
          <div class="task-header">
            <div class="task-status" :class="task.status">
              {{ getStatusIcon(task.status) }}
            </div>
            <div class="task-info">
              <h4>{{ getTaskTitle(task) }}</h4>
              <div class="task-meta">
                <span class="task-id">ID: {{ task.id.substring(0, 8) }}...</span>
                <span class="task-time">{{ formatTime(task.created_at) }}</span>
              </div>
            </div>
            <div class="task-progress">
              <div v-if="task.status === 'running'" class="spinner"></div>
              <div v-else-if="task.status === 'completed'" class="check-icon">✓</div>
              <div v-else-if="task.status === 'failed'" class="error-icon">✗</div>
              <div v-else class="pending-icon">⋯</div>
            </div>
          </div>

          <div v-if="task.result" class="task-result">
            <div class="result-label">Результат:</div>
            <div class="result-content">
              <div v-if="task.result.similarity !== undefined && !isNaN(task.result.similarity)" class="similarity-badge" :style="{ background: getScoreColor(task.result.similarity) }">
                {{ (task.result.similarity * 100).toFixed(1) }}%
              </div>
              <div v-else-if="task.result.combined_similarity !== undefined && !isNaN(task.result.combined_similarity)" class="similarity-badge" :style="{ background: getScoreColor(task.result.combined_similarity) }">
                {{ (task.result.combined_similarity * 100).toFixed(1) }}%
              </div>
              <div v-if="task.result.interpretation" class="result-interpretation">
                {{ task.result.interpretation }}
              </div>
            </div>
          </div>

          <div v-if="task.error" class="task-error">
            <div class="error-label">Ошибка:</div>
            <div class="error-content">{{ task.error }}</div>
          </div>

          <div v-if="task.status === 'running'" class="task-progress-bar">
            <div class="progress-bar-fill"></div>
          </div>
        </div>
      </div>
      <Toast ref="toast" />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { api } from '../services/api'
import { useAnalysisStore } from '../store/analysis'
import AppHeader from '../components/AppHeader.vue'
import HistorySidebar from '../components/HistorySidebar.vue'
import Toast from '../components/Toast.vue'

const analysisStore = useAnalysisStore()
const toast = ref<any>(null)

const tasks = ref<any[]>([])
const stats = ref({
  running: 0,
  pending: 0,
  completed: 0,
  failed: 0
})
const loading = ref(false)
const hasLoadedOnce = ref(false)
const autoRefresh = ref(false)
let refreshInterval: NodeJS.Timeout | null = null

onMounted(() => {
  fetchTasks()
  // Автообновление отключено по запросу
  // startAutoRefresh()
})

onUnmounted(() => {
  stopAutoRefresh()
})

async function fetchTasks() {
  // Only show loading indicator on initial load (when there are no tasks yet)
  if (tasks.value.length === 0) {
    loading.value = true
  }

  try {
    const response = await api.get('/api/v1/tasks')
    tasks.value = response.data.tasks || []

    // Calculate stats
    stats.value = {
      running: tasks.value.filter(t => t.status === 'running').length,
      pending: tasks.value.filter(t => t.status === 'pending').length,
      completed: tasks.value.filter(t => t.status === 'completed').length,
      failed: tasks.value.filter(t => t.status === 'failed').length
    }

    // Move completed tasks to history
    tasks.value.forEach(task => {
      if (task.status === 'completed' && task.result) {
        moveTaskToHistory(task)
      }
    })

    // Keep completed tasks visible - don't remove them
    // setTimeout(() => {
    //   tasks.value = tasks.value.filter(t => t.status !== 'completed')
    // }, 2000)
  } catch (error: any) {
    console.error('Failed to fetch tasks:', error)
  } finally {
    loading.value = false
    hasLoadedOnce.value = true
  }
}

function moveTaskToHistory(task: any) {
  if (!task.result) return

  // Check if already in history
  const existingIndex = analysisStore.history.findIndex(h => h.task_id === task.id)
  if (existingIndex >= 0) return

  // Add to history
  const historyItem = {
    id: task.id,
    task_id: task.id,
    type: task.type || task.name || 'unknown',
    text1_title: task.metadata?.text1_title || task.params?.text1_title || 'Text 1',
    text2_title: task.metadata?.text2_title || task.params?.text2_title || 'Text 2',
    similarity: task.result.similarity || task.result.combined_similarity,
    interpretation: task.result.interpretation || '',
    timestamp: new Date(task.created_at).getTime()
  }

  analysisStore.history.unshift(historyItem)
}

function startAutoRefresh() {
  if (refreshInterval) return
  autoRefresh.value = true
  refreshInterval = setInterval(() => {
    fetchTasks()
  }, 2000)
}

function stopAutoRefresh() {
  if (refreshInterval) {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
}

function getTaskTitle(task: any): string {
  const typeNames: Record<string, string> = {
    semantic: 'Семантический анализ',
    style: 'Анализ стиля',
    tfidf: 'TF-IDF анализ',
    emotion: 'Эмоциональный анализ',
    llm: 'LLM анализ',
    combined: 'Комбинированный анализ',
    chunked_analysis: 'Чанковый анализ',
    matrix: 'Матричный анализ'
  }

  // Проверяем metadata для более точного типа
  if (task.metadata?.type) {
    const metaType = task.metadata.type
    if (typeNames[metaType]) {
      return typeNames[metaType]
    }
  }

  // Если есть имя задачи
  if (task.name) {
    return task.name
  }

  // Если есть тип в основном поле
  if (task.type && typeNames[task.type]) {
    return typeNames[task.type]
  }

  // Генерируем название по параметрам
  if (task.params) {
    if (task.params.text_id1 && task.params.text_id2) {
      return 'Анализ текстов'
    }
    if (task.params.text_ids) {
      return `Матричный анализ (${task.params.text_ids.length} текстов)`
    }
  }

  // Fallback
  return task.type ? `Задача: ${task.type}` : 'Задача анализа'
}

function getStatusIcon(status: string): string {
  const icons: Record<string, string> = {
    pending: '📋',
    running: '⏳',
    completed: '✅',
    failed: '❌'
  }
  return icons[status] || '❓'
}

function formatTime(timestamp: string): string {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now.getTime() - date.getTime()

  const seconds = Math.floor(diff / 1000)
  const minutes = Math.floor(diff / 60000)
  const hours = Math.floor(diff / 3600000)

  if (seconds < 60) return `${seconds} сек назад`
  if (minutes < 60) return `${minutes} мин назад`
  if (hours < 24) return `${hours} ч назад`

  return date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
}

function getScoreColor(score: number): string {
  const hue = score * 120
  return `hsl(${hue}, 70%, 50%)`
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

.page-actions {
  display: flex;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
}

.btn-secondary {
  padding: 0.65rem 1.25rem;
  background: var(--bg-hover);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.95rem;
  transition: all 0.2s;
}

.btn-secondary:hover:not(:disabled) {
  background: var(--bg-active);
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.auto-refresh-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: var(--bg-hover);
  border-radius: 6px;
  font-size: 0.85rem;
  color: var(--text-muted);
}

.auto-refresh-indicator.active {
  background: #10b98120;
  color: #10b981;
}

.pulse {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #10b981;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.3;
  }
}

.stats-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.stat-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  display: flex;
  align-items: center;
  gap: 1rem;
  transition: transform 0.2s;
}

.stat-card:hover {
  transform: translateY(-2px);
}

.stat-icon {
  font-size: 2.5rem;
}

.stat-info {
  flex: 1;
}

.stat-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--text);
  min-width: 2ch;
}

.stat-label {
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-top: 0.25rem;
}

.loading, .empty-state {
  text-align: center;
  padding: 4rem 2rem;
  color: var(--text-muted);
  font-size: 1.1rem;
}

.tasks-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.task-card {
  padding: 1.5rem;
  transition: all 0.2s;
}

.task-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.task-header {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
}

.task-status {
  font-size: 2rem;
  line-height: 1;
}

.task-status.running {
  animation: bounce 1s ease-in-out infinite;
}

@keyframes bounce {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-4px);
  }
}

.task-info {
  flex: 1;
}

.task-info h4 {
  margin: 0 0 0.5rem 0;
  font-size: 1.1rem;
}

.task-meta {
  display: flex;
  gap: 1rem;
  font-size: 0.85rem;
  color: var(--text-muted);
}

.task-progress {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
}

.spinner {
  width: 32px;
  height: 32px;
  border: 3px solid var(--bg-hover);
  border-top-color: var(--primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.check-icon {
  width: 32px;
  height: 32px;
  background: #10b981;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.25rem;
  font-weight: bold;
}

.error-icon {
  width: 32px;
  height: 32px;
  background: #ef4444;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.25rem;
  font-weight: bold;
}

.pending-icon {
  font-size: 1.5rem;
  color: var(--text-muted);
}

.task-result {
  margin-top: 1rem;
  padding: 1rem;
  background: var(--bg-hover);
  border-radius: 8px;
}

.result-label {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-muted);
  margin-bottom: 0.5rem;
}

.result-content {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.similarity-badge {
  display: inline-block;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  font-weight: bold;
  font-size: 1.1rem;
  color: white;
  align-self: flex-start;
}

.result-interpretation {
  color: var(--text);
  line-height: 1.5;
}

.task-error {
  margin-top: 1rem;
  padding: 1rem;
  background: #ef444420;
  border-left: 4px solid #ef4444;
  border-radius: 8px;
}

.error-label {
  font-size: 0.85rem;
  font-weight: 600;
  color: #ef4444;
  margin-bottom: 0.5rem;
}

.error-content {
  color: var(--text);
  font-size: 0.9rem;
}

.task-progress-bar {
  margin-top: 1rem;
  height: 4px;
  background: var(--bg-hover);
  border-radius: 2px;
  overflow: hidden;
}

.progress-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--primary) 0%, var(--primary) 50%, transparent 50%, transparent 100%);
  background-size: 200% 100%;
  animation: progress 1.5s linear infinite;
}

@keyframes progress {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: 0 0;
  }
}

.card {
  background: var(--bg-secondary, #1a1a1a);
  border: 1px solid var(--border, #333);
  border-radius: 12px;
}
</style>
