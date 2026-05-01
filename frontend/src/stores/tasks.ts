import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { TaskInfo, TaskListResponse } from '@/api/tasks'
import { tasksApi } from '@/api/tasks'

export const useTasksStore = defineStore('tasks', () => {
  const currentTask = ref<TaskInfo | null>(null)
  const tasks = ref<TaskInfo[]>([])
  const total = ref(0)
  const loading = ref(false)

  async function createTask(prompt: string): Promise<TaskInfo> {
    const task = await tasksApi.create(prompt)
    return task
  }

  async function fetchTask(taskId: string): Promise<TaskInfo> {
    const task = await tasksApi.get(taskId)
    if (task.task_id === currentTask.value?.task_id) {
      currentTask.value = task
    }
    return task
  }

  async function fetchTasks(page = 1, pageSize = 10, status?: string) {
    loading.value = true
    try {
      const skip = (page - 1) * pageSize
      const res: TaskListResponse = await tasksApi.list(skip, pageSize, status)
      tasks.value = res.tasks
      total.value = res.total
    } finally {
      loading.value = false
    }
  }

  async function deleteTask(taskId: string) {
    await tasksApi.delete(taskId)
    tasks.value = tasks.value.filter((t) => t.task_id !== taskId)
    total.value--
  }

  return {
    currentTask,
    tasks,
    total,
    loading,
    createTask,
    fetchTask,
    fetchTasks,
    deleteTask,
  }
})
