<template>
  <div class="tasks-view">
    <div class="tasks-header">
      <h1 class="tasks-title">任务管理</h1>
      <el-select
        v-model="statusFilter"
        placeholder="筛选状态"
        clearable
        style="width: 160px"
        @change="handleFilterChange"
      >
        <el-option label="全部" value="" />
        <el-option label="待处理" value="pending" />
        <el-option label="处理中" value="processing" />
        <el-option label="已完成" value="completed" />
        <el-option label="失败" value="failed" />
      </el-select>
    </div>

    <div v-loading="store.loading" class="tasks-content">
      <template v-if="store.tasks.length > 0">
        <div class="tasks-list">
          <div
            v-for="task in store.tasks"
            :key="task.task_id"
            class="task-row"
            @click="$router.push(`/tasks/${task.task_id}`)"
          >
            <div class="task-main">
              <span class="task-id">#{{ task.task_id.slice(0, 8) }}</span>
              <span class="task-prompt">{{ task.prompt }}</span>
            </div>
            <div class="task-meta">
              <el-tag :type="statusTagType(task.status)" size="small" effect="plain">
                {{ statusLabel(task.status) }}
              </el-tag>
              <span class="task-time">{{ formatTime(task.created_at) }}</span>
              <el-button text type="danger" size="small" @click.stop="handleDelete(task)">
                <el-icon><Delete /></el-icon>
              </el-button>
            </div>
          </div>
        </div>
        <div class="pagination-wrap">
          <el-pagination
            v-model:current-page="currentPage"
            :page-size="pageSize"
            :total="store.total"
            layout="prev, pager, next"
            @current-change="handlePageChange"
          />
        </div>
      </template>
      <el-empty v-else description="暂无任务记录">
        <el-button type="primary" @click="$router.push('/')">去创作</el-button>
      </el-empty>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useTasksStore } from '@/stores/tasks'
import { ElMessageBox, ElMessage } from 'element-plus'
import type { TaskInfo } from '@/api/tasks'

const store = useTasksStore()
const currentPage = ref(1)
const pageSize = ref(10)
const statusFilter = ref('')

onMounted(() => {
  store.fetchTasks(currentPage.value, pageSize.value, statusFilter.value || undefined)
})

function handlePageChange(page: number) {
  currentPage.value = page
  store.fetchTasks(page, pageSize.value, statusFilter.value || undefined)
}

function handleFilterChange() {
  currentPage.value = 1
  store.fetchTasks(1, pageSize.value, statusFilter.value || undefined)
}

async function handleDelete(task: TaskInfo) {
  try {
    await ElMessageBox.confirm('确定要删除该任务吗？', '确认删除', {
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      type: 'warning',
    })
    await store.deleteTask(task.task_id)
    ElMessage.success('已删除')
  } catch {
    // cancelled
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
.tasks-view {
  max-width: 900px;
  margin: 0 auto;
}

.tasks-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.tasks-title {
  font-size: var(--font-size-2xl);
  font-weight: 600;
}

.tasks-content {
  min-height: 300px;
}

.tasks-list {
  border: 1px solid var(--color-border-light);
  border-radius: var(--radius-lg);
  overflow: hidden;
  background: #ffffff;
}

.task-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid var(--color-border-light);
  cursor: pointer;
  transition: background 0.15s;
}

.task-row:last-child {
  border-bottom: none;
}

.task-row:hover {
  background: var(--color-bg-secondary);
}

.task-main {
  display: flex;
  align-items: center;
  gap: 16px;
  flex: 1;
  min-width: 0;
}

.task-id {
  font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
  flex-shrink: 0;
}

.task-prompt {
  font-size: var(--font-size-base);
  color: var(--color-text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-meta {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
  margin-left: 16px;
}

.task-time {
  font-size: var(--font-size-sm);
  color: var(--color-text-tertiary);
}

.pagination-wrap {
  display: flex;
  justify-content: center;
  margin-top: 24px;
}
</style>
