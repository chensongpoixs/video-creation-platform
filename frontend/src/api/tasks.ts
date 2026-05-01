import api from './index'

export interface TaskInfo {
  task_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  prompt: string
  result: string | null
  created_at: string
  error: string | null
  progress?: number
  script?: any
}

export interface TaskListResponse {
  tasks: TaskInfo[]
  total: number
}

export const tasksApi = {
  create(prompt: string): Promise<TaskInfo> {
    return api.post('/api/tasks', { prompt }).then((r) => r.data)
  },

  get(taskId: string): Promise<TaskInfo> {
    return api.get(`/api/tasks/${taskId}`).then((r) => r.data)
  },

  list(skip = 0, limit = 10, status?: string): Promise<TaskListResponse> {
    const params: any = { skip, limit }
    if (status) params.status = status
    return api.get('/api/tasks', { params }).then((r) => r.data)
  },

  delete(taskId: string) {
    return api.delete(`/api/tasks/${taskId}`).then((r) => r.data)
  },
}
