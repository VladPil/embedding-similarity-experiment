import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' }
})

export const textsAPI = {
  list: () => api.get('/api/v1/texts'),
  upload: (data: { title: string; text: string }) => api.post('/api/v1/texts/upload', data),
  uploadFB2: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/api/v1/texts/upload_fb2', formData)
  },
  delete: (id: string) => api.delete(`/api/v1/texts/${id}`)
}

export const analysisAPI = {
  semantic: (data: any) => api.post('/api/v1/analysis/semantic', data),
  style: (data: any) => api.post('/api/v1/analysis/style', data),
  tfidf: (data: any) => api.post('/api/v1/analysis/tfidf', data),
  emotion: (data: any) => api.post('/api/v1/analysis/emotion', data),
  llm: (data: any) => api.post('/api/v1/analysis/llm', data),
  combined: (data: any) => api.post('/api/v1/analysis/combined', data),
  matrix: (data: any) => api.post('/api/v1/analysis/matrix', data),
  chunked: (data: any) => api.post('/api/v1/analysis/chunked', data)
}

export const tasksAPI = {
  list: (status?: string) => api.get('/api/v1/tasks', { params: { status } }),
  get: (id: string) => api.get(`/api/v1/tasks/${id}`),
  delete: (id: string) => api.delete(`/api/v1/tasks/${id}`),
  clear: () => api.post('/api/v1/tasks/clear')
}

export const embeddingsAPI = {
  models: () => api.get('/api/v1/embeddings/models')
}
