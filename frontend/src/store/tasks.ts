import { defineStore } from 'pinia'
import { ref } from 'vue'
import { tasksAPI } from '../services/api'

export const useTasksStore = defineStore('tasks', () => {
  const tasks = ref<any[]>([])
  const loading = ref(false)

  async function fetchTasks() {
    loading.value = true
    try {
      const { data } = await tasksAPI.list()
      tasks.value = data.tasks
    } finally {
      loading.value = false
    }
  }

  async function getTask(id: string) {
    const { data } = await tasksAPI.get(id)
    return data.task
  }

  async function deleteTask(id: string) {
    await tasksAPI.delete(id)
    await fetchTasks()
  }

  async function clearCompleted() {
    await tasksAPI.clear()
    await fetchTasks()
  }

  return { tasks, loading, fetchTasks, getTask, deleteTask, clearCompleted }
})
