<template>
  <div class="page-layout">
    <aside class="sidebar-wrapper">
      <HistorySidebar />
    </aside>
    <main class="main-content">
      <AppHeader title="📋 Задачи" subtitle="Все выполненные анализы текстов">
        <template #actions>
          <button @click="clearHistory" :disabled="!analysisStore.history.length" class="btn-danger">
            🗑️ Очистить историю
          </button>
        </template>
      </AppHeader>

      <div v-if="!analysisStore.history.length" class="empty-state">
        <h2>История пуста</h2>
        <p>Здесь будет отображаться история всех ваших анализов</p>
        <button @click="$router.push('/')" class="btn-primary">← Вернуться на главную</button>
      </div>

      <div v-else class="history-grid">
        <div
          v-for="item in analysisStore.history"
          :key="item.id"
          :data-history-id="item.id"
          :class="['history-card', 'card', { 'highlighted': analysisStore.selectedHistoryId === item.id }]"
        >
          <div class="card-header">
            <div class="card-title">
              <span class="history-type" :style="{ color: getTypeColor(item.type) }">
                {{ formatType(item.type) }}
              </span>
              <span class="history-time">{{ formatTime(item.timestamp) }}</span>
            </div>
            <button @click="deleteItem(item.id)" class="delete-btn" title="Удалить">
              ×
            </button>
          </div>

          <div class="history-texts">
            <div class="text-block">
              <div class="text-label">Текст 1:</div>
              <div class="text-title">{{ item.text1_title }}</div>
            </div>
            <div class="vs-divider">VS</div>
            <div class="text-block">
              <div class="text-label">Текст 2:</div>
              <div class="text-title">{{ item.text2_title }}</div>
            </div>
          </div>

          <div class="similarity-container">
            <div class="similarity-label">Схожесть:</div>
            <div v-if="item.similarity !== undefined && !isNaN(item.similarity)" class="similarity-score" :style="{ background: getScoreColor(item.similarity) }">
              {{ (item.similarity * 100).toFixed(1) }}%
            </div>
            <div v-else class="similarity-score" style="background: var(--text-muted)">
              N/A
            </div>
          </div>

          <div v-if="item.interpretation" class="interpretation">
            <div
              class="interpretation-header"
              @click="toggleInterpretation(item.id)"
            >
              <span class="interpretation-label">
                {{ expandedInterpretations[item.id] ? '▼' : '▶' }} Интерпретация
              </span>
            </div>
            <div
              v-if="expandedInterpretations[item.id]"
              class="interpretation-content"
            >
              <pre class="interpretation-text">{{ item.interpretation }}</pre>
            </div>
          </div>
        </div>
      </div>
      <Toast ref="toast" />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAnalysisStore } from '../store/analysis'
import HistorySidebar from '../components/HistorySidebar.vue'
import AppHeader from '../components/AppHeader.vue'
import Toast from '../components/Toast.vue'

const analysisStore = useAnalysisStore()
const router = useRouter()
const toast = ref<any>(null)
const expandedInterpretations = ref<Record<string, boolean>>({})

// Clear selected history after animation
onMounted(() => {
  if (analysisStore.selectedHistoryId) {
    setTimeout(() => {
      analysisStore.selectedHistoryId = null
    }, 3000)
  }
})

const typeNames: Record<string, string> = {
  semantic: 'Семантика',
  style: 'Стиль',
  tfidf: 'TF-IDF',
  emotion: 'Эмоции',
  llm: 'LLM',
  combined: 'Комбо',
  matrix: 'Матрица'
}

const typeColors: Record<string, string> = {
  semantic: '#3b82f6',
  style: '#8b5cf6',
  tfidf: '#10b981',
  emotion: '#f59e0b',
  llm: '#ef4444',
  combined: '#ec4899',
  matrix: '#06b6d4'
}

function formatType(type: string): string {
  return typeNames[type] || type
}

function getTypeColor(type: string): string {
  return typeColors[type] || '#6b7280'
}

function formatTime(timestamp: number): string {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now.getTime() - date.getTime()

  const minutes = Math.floor(diff / 60000)
  const hours = Math.floor(diff / 3600000)
  const days = Math.floor(diff / 86400000)

  if (minutes < 1) return 'только что'
  if (minutes < 60) return `${minutes} мин назад`
  if (hours < 24) return `${hours} ч назад`
  if (days < 7) return `${days} д назад`

  return date.toLocaleDateString('ru-RU', {
    day: 'numeric',
    month: 'long',
    year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
  })
}

function getScoreColor(score: number): string {
  const hue = score * 120
  return `hsl(${hue}, 70%, 50%)`
}

function deleteItem(id: string) {
  if (!confirm('Удалить этот анализ из истории?')) {
    return
  }
  analysisStore.deleteHistoryItem(id)
  toast.value?.show('Анализ удален из истории', 'success')
}

function clearHistory() {
  if (!confirm('Вы уверены, что хотите очистить всю историю?')) {
    return
  }
  analysisStore.clearHistory()
  toast.value?.show('История успешно очищена', 'success')
}

function toggleInterpretation(id: string) {
  expandedInterpretations.value[id] = !expandedInterpretations.value[id]
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

.empty-state {
  text-align: center;
  padding: 4rem 2rem;
  color: var(--text-muted);
}

.empty-state h2 {
  font-size: 1.5rem;
  margin-bottom: 1rem;
  color: var(--text);
}

.empty-state p {
  margin-bottom: 2rem;
  font-size: 1.1rem;
}

.history-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 1.5rem;
}

.card {
  background: var(--bg-secondary, #1a1a1a);
  border: 1px solid var(--border, #333);
  border-radius: 12px;
}

.history-card {
  padding: 1.5rem;
  transition: transform 0.2s, box-shadow 0.2s;
}

.history-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1rem;
}

.card-title {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.history-type {
  font-weight: 700;
  font-size: 1.1rem;
}

.history-time {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.delete-btn {
  background: transparent;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 1.75rem;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  transition: all 0.2s;
}

.delete-btn:hover {
  background: var(--error, #ef4444);
  color: white;
}

.history-texts {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 1rem;
  align-items: center;
  margin-bottom: 1rem;
  padding: 1rem;
  background: var(--bg-hover, #2a2a2a);
  border-radius: 8px;
}

.text-block {
  min-width: 0;
}

.text-label {
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-bottom: 0.25rem;
}

.text-title {
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.vs-divider {
  font-weight: 700;
  color: var(--text-muted);
  font-size: 0.85rem;
}

.similarity-container {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
}

.similarity-label {
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--text);
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

.interpretation {
  background: var(--bg-hover, #2a2a2a);
  border-radius: 8px;
  border-left: 3px solid var(--primary, #3b82f6);
  overflow: hidden;
}

.interpretation-header {
  padding: 1rem;
  cursor: pointer;
  user-select: none;
  transition: background 0.2s;
}

.interpretation-header:hover {
  background: rgba(59, 130, 246, 0.1);
}

.interpretation-label {
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--primary);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.interpretation-content {
  padding: 0 1rem 1rem;
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

.interpretation-text {
  margin: 0;
  font-family: inherit;
  font-size: 0.9rem;
  line-height: 1.6;
  color: var(--text-muted);
  white-space: pre-wrap;
  word-wrap: break-word;
  background: transparent;
  border: none;
  padding: 0;
}

/* Highlight animation for selected item */
.history-card.highlighted {
  border-color: var(--primary, #3b82f6);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
}

@keyframes highlight-pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4);
  }
  50% {
    box-shadow: 0 0 0 10px rgba(59, 130, 246, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0);
  }
}

.highlight-animation {
  animation: highlight-pulse 2s ease-out;
  border-color: var(--primary, #3b82f6) !important;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3) !important;
}
</style>
