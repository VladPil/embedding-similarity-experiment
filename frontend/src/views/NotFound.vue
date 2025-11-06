<template>
  <div class="page-wrapper">
    <HistorySidebar />
    <div class="container">
      <header class="header">
        <div class="header-top">
          <div class="header-left">
            <h1>📚 Embedding Similarity</h1>
            <p class="subtitle">Анализ семантической схожести текстов</p>
          </div>
          <div class="header-actions">
            <button @click="$router.push('/')" class="btn-primary">
              🏠 На главную
            </button>
          </div>
        </div>
        <nav class="header-nav">
          <button @click="$router.push('/')" class="nav-tab">
            🏠 Главная
          </button>
          <button @click="$router.push('/history')" class="nav-tab">
            📜 История
          </button>
          <button @click="$router.push('/tasks')" class="nav-tab">
            ⚙️ Задачи
          </button>
        </nav>
      </header>

      <div class="error-container">
        <div class="error-icon">404</div>
        <h2 class="error-title">Страница не найдена</h2>
        <p class="error-message">{{ message }}</p>
        <div class="error-actions">
          <button @click="$router.push('/')" class="btn-primary">
            🏠 Вернуться на главную
          </button>
          <button @click="$router.back()" class="btn-secondary">
            ← Назад
          </button>
        </div>

        <div class="error-suggestions">
          <h3>Возможно, вы искали:</h3>
          <ul>
            <li><router-link to="/">Главная страница</router-link></li>
            <li><router-link to="/history">История анализов</router-link></li>
            <li><router-link to="/tasks">Мониторинг задач</router-link></li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import HistorySidebar from '../components/HistorySidebar.vue'

const route = useRoute()

const message = computed(() => {
  const path = route.path

  if (path.includes('/text/')) {
    return 'Текст с указанным ID не найден. Возможно, он был удален или ID устарел после обновления.'
  }

  return `Страница "${path}" не существует. Проверьте правильность URL или вернитесь на главную страницу.`
})
</script>

<style scoped>
.page-wrapper {
  display: flex;
  min-height: 100vh;
}

.container {
  flex: 1;
  padding: 2rem;
  margin-left: 320px;
  max-width: 1400px;
}

.header {
  display: flex;
  flex-direction: column;
  margin-bottom: 2rem;
  padding-bottom: 0;
  border-bottom: 2px solid var(--border);
}

.header-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding-bottom: 1.5rem;
}

.header-left h1 {
  margin: 0 0 0.5rem 0;
  font-size: 2rem;
  font-weight: 700;
}

.subtitle {
  margin: 0;
  color: var(--text-muted);
  font-size: 0.95rem;
}

.header-actions {
  display: flex;
  gap: 0.75rem;
}

.header-nav {
  display: flex;
  gap: 0.5rem;
  padding: 0.5rem 0 0 0;
  border-top: 1px solid var(--border);
}

.nav-tab {
  padding: 0.75rem 1.5rem;
  background: transparent;
  border: none;
  color: var(--text-muted);
  font-size: 0.95rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  border-bottom: 3px solid transparent;
  margin-bottom: -2px;
}

.nav-tab:hover {
  color: var(--text);
  background: var(--bg-hover);
  border-radius: 8px 8px 0 0;
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

.btn-primary {
  background: var(--primary, #3b82f6);
  color: white;
}

.btn-primary:hover {
  background: #2563eb;
}

.btn-secondary {
  background: var(--bg-hover);
  border: 1px solid var(--border);
  color: var(--text);
}

.btn-secondary:hover {
  background: var(--bg-active);
}

.error-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem 2rem;
  text-align: center;
  min-height: 60vh;
}

.error-icon {
  font-size: 8rem;
  font-weight: 900;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 1rem;
  line-height: 1;
}

.error-title {
  font-size: 2rem;
  font-weight: 700;
  margin: 0 0 1rem 0;
  color: var(--text);
}

.error-message {
  font-size: 1.1rem;
  color: var(--text-muted);
  max-width: 600px;
  margin: 0 0 2rem 0;
  line-height: 1.6;
}

.error-actions {
  display: flex;
  gap: 1rem;
  margin-bottom: 3rem;
}

.error-suggestions {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 2rem;
  max-width: 500px;
  width: 100%;
  text-align: left;
}

.error-suggestions h3 {
  margin: 0 0 1rem 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text);
}

.error-suggestions ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.error-suggestions li {
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--border);
}

.error-suggestions li:last-child {
  border-bottom: none;
}

.error-suggestions a {
  color: var(--primary);
  text-decoration: none;
  font-weight: 500;
  transition: color 0.2s;
}

.error-suggestions a:hover {
  color: #2563eb;
  text-decoration: underline;
}
</style>
