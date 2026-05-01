<template>
  <div class="task-card" @click="$emit('click')">
    <div class="task-card-left">
      <span class="task-card-id">#{{ task.task_id.slice(0, 8) }}</span>
      <span class="task-card-prompt">{{ task.prompt }}</span>
    </div>
    <div class="task-card-right">
      <el-tag :type="statusType" size="small" effect="plain">{{ statusLabel }}</el-tag>
      <span class="task-card-time">{{ formatTime(task.created_at) }}</span>
      <el-icon v-if="task.status === 'processing'" class="is-loading" color="var(--color-primary)">
        <Loading />
      </el-icon>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { TaskInfo } from '@/api/tasks'
import { Loading } from '@element-plus/icons-vue'

const props = defineProps<{
  task: TaskInfo
}>()

defineEmits<{
  (e: 'click'): void
}>()

const statusType = computed(() => {
  const map: Record<string, string> = {
    pending: 'info',
    processing: 'warning',
    completed: 'success',
    failed: 'danger',
  }
  return map[props.task.status] || 'info'
})

const statusLabel = computed(() => {
  const map: Record<string, string> = {
    pending: '待处理',
    processing: '处理中',
    completed: '已完成',
    failed: '失败',
  }
  return map[props.task.status] || props.task.status
})

function formatTime(iso: string) {
  if (!iso) return '-'
  return new Date(iso).toLocaleString('zh-CN')
}
</script>

<style scoped>
.task-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background: #ffffff;
  border: 1px solid var(--color-border-light);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: box-shadow 0.2s, border-color 0.2s;
}

.task-card:hover {
  box-shadow: var(--shadow-sm);
  border-color: var(--color-border);
}

.task-card-left {
  display: flex;
  align-items: center;
  gap: 16px;
  min-width: 0;
  flex: 1;
}

.task-card-id {
  font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
  flex-shrink: 0;
}

.task-card-prompt {
  font-size: var(--font-size-base);
  color: var(--color-text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-card-right {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
  margin-left: 16px;
}

.task-card-time {
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
}
</style>
