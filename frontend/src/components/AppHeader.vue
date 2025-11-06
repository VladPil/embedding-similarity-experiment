<template>
  <header class="header">
    <div class="header-top">
      <div class="header-left">
        <h1>{{ title }}</h1>
        <p v-if="subtitle" class="subtitle">{{ subtitle }}</p>
      </div>
      <div v-if="$slots.actions" class="header-actions">
        <slot name="actions"></slot>
      </div>
    </div>
    <nav class="header-nav">
      <button
        @click="navigateTo('/')"
        :class="{ active: currentPath === '/' }"
        class="nav-tab"
      >
        🏠 Главная
      </button>
      <button
        @click="navigateTo('/history')"
        :class="{ active: currentPath === '/history' }"
        class="nav-tab"
      >
        📜 История
      </button>
      <button
        @click="navigateTo('/tasks')"
        :class="{ active: currentPath === '/tasks' }"
        class="nav-tab"
      >
        ⚙️ История запусков
      </button>
      <button
        @click="navigateTo('/settings')"
        :class="{ active: currentPath === '/settings' }"
        class="nav-tab"
      >
        🔧 Настройки
      </button>
    </nav>
  </header>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'

interface Props {
  title: string
  subtitle?: string
}

defineProps<Props>()

const router = useRouter()
const route = useRoute()

const currentPath = computed(() => route.path)

function navigateTo(path: string) {
  router.push(path)
}
</script>

<style scoped>
.header {
  background: var(--bg-secondary);
  border-radius: 12px;
  padding: 1.5rem 1.5rem 0.5rem 1.5rem;
  margin-bottom: 2rem;
  border: 1px solid var(--border);
}

.header-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1.75rem;
}

.header-left h1 {
  font-size: 2rem;
  margin: 0 0 0.5rem 0;
  color: var(--text);
}

.subtitle {
  color: var(--text-muted);
  margin: 0;
}

.header-actions {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.header-nav {
  display: flex;
  gap: 0.5rem;
  border-top: 1px solid var(--border);
  padding-top: 1rem;
  padding-bottom: 1rem;
}

.nav-tab {
  padding: 0.75rem 1.5rem;
  background: transparent;
  border: none;
  border-radius: 6px;
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.2s;
  font-weight: 500;
}

.nav-tab:hover {
  background: var(--bg);
  color: var(--text);
}

.nav-tab.active {
  background: var(--primary);
  color: white;
}

@media (max-width: 768px) {
  .header-top {
    flex-direction: column;
    gap: 1rem;
  }

  .header-actions {
    width: 100%;
  }

  .header-nav {
    flex-wrap: wrap;
  }

  .nav-tab {
    flex: 1;
    min-width: 120px;
  }
}
</style>
