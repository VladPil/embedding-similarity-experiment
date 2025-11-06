<template>
  <div class="toast-container">
    <div
      v-for="toast in toasts"
      :key="toast.id"
      class="toast"
      :class="toast.type"
    >
      <span>{{ toast.message }}</span>
      <button @click="remove(toast.id)" class="toast-close">×</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

interface Toast {
  id: string
  message: string
  type: 'success' | 'error' | 'info'
}

const toasts = ref<Toast[]>([])

function show(message: string, type: 'success' | 'error' | 'info' = 'info') {
  const id = Date.now().toString()
  toasts.value.push({ id, message, type })

  setTimeout(() => {
    remove(id)
  }, 5000)
}

function remove(id: string) {
  const index = toasts.value.findIndex(t => t.id === id)
  if (index > -1) {
    toasts.value.splice(index, 1)
  }
}

defineExpose({ show })
</script>

<style scoped>
.toast-container {
  position: fixed;
  top: 1rem;
  right: 1rem;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.toast {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem 1.5rem;
  min-width: 300px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  animation: slideIn 0.3s ease-out;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  color: white;
  opacity: 1;
}

.toast.success {
  border-color: #10b981;
  background: #10b981;
}

.toast.error {
  border-color: #ef4444;
  background: #ef4444;
}

.toast.info {
  border-color: #3b82f6;
  background: #3b82f6;
}

.toast-close {
  background: transparent;
  border: none;
  color: var(--text);
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
}

.toast-close:hover {
  background: var(--bg-hover);
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}
</style>
