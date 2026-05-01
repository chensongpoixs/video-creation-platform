<template>
  <div class="task-detail-view">
    <div class="detail-header">
      <el-button text @click="$router.back()">
        <el-icon><ArrowLeft /></el-icon>
        返回
      </el-button>
    </div>

    <div v-loading="loading" class="detail-content">
      <template v-if="task">
        <div class="detail-card">
          <div class="detail-top">
            <div class="detail-info">
              <h1 class="detail-title">任务详情</h1>
              <p class="detail-id">ID: {{ task.task_id }}</p>
            </div>
            <el-tag :type="statusTagType(task.status)" size="large" effect="plain">
              {{ statusLabel(task.status) }}
            </el-tag>
          </div>

          <div class="detail-body">
            <div class="detail-section">
              <label>创作指令</label>
              <p class="detail-prompt">{{ task.prompt }}</p>
            </div>

            <div v-if="task.error" class="detail-section">
              <label>错误信息</label>
              <el-alert :title="task.error" type="error" :closable="false" show-icon />
            </div>

            <div class="detail-section">
              <label>创建时间</label>
              <p>{{ formatTime(task.created_at) }}</p>
            </div>
          </div>
        </div>

        <!-- Video Player -->
        <div v-if="task.status === 'completed' && task.result" class="detail-card">
          <h2 class="section-title">生成结果</h2>
          <VideoPlayer :src="'/' + task.result" :task-id="task.task_id" />
        </div>

        <!-- Processing Status -->
        <div v-if="task.status === 'processing'" class="detail-card status-card">
          <h2 class="section-title">生成进度</h2>
          <div class="processing-status">
            <el-icon class="processing-icon is-loading" :size="48"><Loading /></el-icon>
            <p class="processing-text">视频正在生成中，请稍候...</p>
            <p class="processing-hint">页面会自动刷新状态</p>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useTasksStore } from '@/stores/tasks'
import VideoPlayer from '@/components/video/VideoPlayer.vue'
import type { TaskInfo } from '@/api/tasks'

const route = useRoute()
const store = useTasksStore()
const loading = ref(true)
const task = ref<TaskInfo | null>(null)
let pollTimer: ReturnType<typeof setInterval> | null = null

onMounted(async () => {
  await loadTask()
  startPolling()
})

onUnmounted(() => {
  stopPolling()
})

watch(() => route.params.id, () => {
  stopPolling()
  loadTask()
  startPolling()
})

async function loadTask() {
  loading.value = true
  try {
    task.value = await store.fetchTask(route.params.id as string)
  } finally {
    loading.value = false
  }
}

function startPolling() {
  if (task.value?.status === 'pending' || task.value?.status === 'processing') {
    pollTimer = setInterval(async () => {
      try {
        await loadTask()
        if (task.value?.status === 'completed' || task.value?.status === 'failed') {
          stopPolling()
          store.fetchTasks(1, 10)
        }
      } catch {
        stopPolling()
      }
    }, 3000)
  }
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

function statusTagType(status: string) {
  const map: Record<string, string> = {
    pending: 'info',
    processing: 'warning',
    completed: 'success',
    failed: 'danger',
  }
  return map[status] || 'info'
}

function statusLabel(status: string) {
  const map: Record<string, string> = {
    pending: '待处理',
    processing: '处理中',
    completed: '已完成',
    failed: '失败',
  }
  return map[status] || status
}

function formatTime(iso: string) {
  if (!iso) return '-'
  return new Date(iso).toLocaleString('zh-CN')
}
</script>

<style scoped>
.task-detail-view {
  max-width: 900px;
  margin: 0 auto;
}

.detail-header {
  margin-bottom: 16px;
}

.detail-content {
  min-height: 300px;
}

.detail-card {
  background: #ffffff;
  border: 1px solid var(--color-border-light);
  border-radius: var(--radius-lg);
  padding: 24px;
  margin-bottom: 20px;
}

.detail-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--color-border-light);
}

.detail-title {
  font-size: var(--font-size-xl);
  font-weight: 600;
  margin-bottom: 4px;
}

.detail-id {
  font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
  font-size: var(--font-size-sm);
  color: var(--color-text-tertiary);
}

.detail-body {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.detail-section label {
  display: block;
  font-size: var(--font-size-sm);
  font-weight: 600;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
}

.detail-prompt {
  font-size: var(--font-size-lg);
  color: var(--color-text-primary);
  line-height: 1.7;
}

.section-title {
  font-size: var(--font-size-lg);
  font-weight: 600;
  margin-bottom: 16px;
}

.status-card {
  text-align: center;
}

.processing-status {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 32px;
}

.processing-icon {
  color: var(--color-primary);
}

.processing-text {
  font-size: var(--font-size-lg);
  color: var(--color-text-primary);
  font-weight: 500;
}

.processing-hint {
  font-size: var(--font-size-sm);
  color: var(--color-text-tertiary);
}
</style>
