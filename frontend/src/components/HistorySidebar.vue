<template>
  <div class="sidebar">
    <div class="sidebar-header">
      <h3>📋 Задачи</h3>
    </div>

    <div class="sidebar-content">
      <div v-if="!analysisStore.history.length" class="empty-state">
        История пуста
      </div>

      <div v-else class="history-list">
        <div
          v-for="item in recentHistory"
          :key="item.id"
          class="history-item"
          @click="loadHistoryItem(item)"
        >
          <div class="history-item-main">
            <div class="history-item-header">
              <div class="history-item-type-time">
                <span class="history-type">{{ formatType(item.type) }}</span>
                <span class="history-time">{{ formatTime(item.timestamp) }}</span>
              </div>
              <div v-if="item.similarity !== undefined && !isNaN(item.similarity)"
                   class="history-similarity-badge"
                   :style="{ background: getScoreColor(item.similarity) }">
                {{ (item.similarity * 100).toFixed(0) }}%
              </div>
              <div v-else class="history-similarity-badge" style="background: var(--text-muted)">
                N/A
              </div>
            </div>

            <div class="history-texts">
              <div class="history-text">📄 {{ item.text1_title }}</div>
              <div class="history-text">📄 {{ item.text2_title }}</div>
            </div>
          </div>

          <button @click.stop="analysisStore.deleteHistoryItem(item.id)" class="delete-btn">
            🗑️
          </button>
        </div>
      </div>

      <div v-if="analysisStore.history.length > 3" class="sidebar-footer">
        <button @click="navigateToHistory" class="btn-all-history">
          Вся история
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useAnalysisStore } from '../store/analysis'

const analysisStore = useAnalysisStore()
const router = useRouter()

const recentHistory = computed(() => analysisStore.history.slice(0, 3))

function navigateToHistory() {
  router.push('/history')
}

const typeNames: Record<string, string> = {
  semantic: 'Семантика',
  style: 'Стиль',
  tfidf: 'TF-IDF',
  emotion: 'Эмоции',
  llm: 'LLM',
  combined: 'Комбо',
  matrix: 'Матрица'
}

function formatType(type: string): string {
  return typeNames[type] || type
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

  return date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' })
}

function getScoreColor(score: number): string {
  const hue = score * 120
  return `hsl(${hue}, 70%, 50%)`
}

function loadHistoryItem(item: any) {
  // Set the selected history item for highlighting
  analysisStore.selectedHistoryId = item.id

  // Navigate to history page
  router.push('/history')

  // Optionally scroll to the item after navigation
  setTimeout(() => {
    const element = document.querySelector(`[data-history-id="${item.id}"]`)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' })
      // Add highlight animation
      element.classList.add('highlight-animation')
      setTimeout(() => element.classList.remove('highlight-animation'), 2000)
    }
  }, 100)
}
</script>

<style scoped>
.sidebar {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 1rem;
  background: transparent;
}

.sidebar-header {
  padding: 1rem 0.5rem;
  margin-bottom: 0.5rem;
}

.sidebar-header h3 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text);
}

.sidebar-content {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.sidebar-footer {
  padding: 1rem 0.5rem 0;
  margin-top: 0.5rem;
}

.btn-all-history {
  width: 100%;
  padding: 0.75rem;
  background: var(--primary, #3b82f6);
  border: none;
  color: white;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.95rem;
  font-weight: 500;
  transition: all 0.2s;
}

.btn-all-history:hover {
  background: #2563eb;
  transform: translateY(-1px);
}

.empty-state {
  padding: 2rem 1rem;
  text-align: center;
  color: var(--text-muted, #888);
  font-size: 0.875rem;
}

.history-list {
  flex: 1;
  overflow-y: auto;
}

.history-item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem;
  margin-bottom: 0.75rem;
  cursor: pointer;
  transition: all 0.2s;
  border-radius: 10px;
  background: var(--bg-secondary, rgba(30, 30, 30, 0.4));
  border: 1px solid var(--border, rgba(255, 255, 255, 0.1));
}

.history-item:hover {
  background: var(--bg-hover, rgba(59, 130, 246, 0.1));
  border-color: var(--primary, #3b82f6);
  transform: translateX(2px);
}

.history-item-main {
  flex: 1;
  min-width: 0;
}

.history-item-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  gap: 0.5rem;
}

.history-item-type-time {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
  min-width: 0;
}

.history-type {
  font-weight: 600;
  font-size: 0.8rem;
  color: var(--primary, #3b82f6);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.history-time {
  font-size: 0.65rem;
  color: var(--text-muted, #888);
  white-space: nowrap;
}

.history-similarity-badge {
  padding: 0.15rem 0.4rem;
  border-radius: 12px;
  font-weight: 700;
  font-size: 0.75rem;
  color: white;
  white-space: nowrap;
}

.delete-btn {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  background: transparent;
  border: none;
  color: var(--text-muted, #666);
  cursor: pointer;
  font-size: 0.85rem;
  padding: 0.25rem;
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  transition: all 0.2s;
  opacity: 0;
}

.history-item:hover .delete-btn {
  opacity: 1;
}

.delete-btn:hover {
  background: var(--error, #ef4444);
  color: white;
  transform: scale(1.1);
}

.history-texts {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.history-text {
  font-size: 0.8rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--text, #e5e5e5);
  display: flex;
  align-items: center;
  gap: 0.3rem;
}
</style>
