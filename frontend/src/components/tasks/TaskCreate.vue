<template>
  <div class="task-create-card">
    <div class="card-header">
      <h2 class="card-title">创建视频</h2>
      <p class="card-desc">输入您的创作想法，AI 将自动生成脚本和视频</p>
    </div>
    <div class="card-body">
      <el-input
        v-model="prompt"
        type="textarea"
        :rows="4"
        placeholder="例如：制作一段关于森林探险的短视频，包含河流、小动物和阳光穿过树叶的画面..."
        resize="none"
        maxlength="500"
        show-word-limit
        :disabled="submitting"
      />
      <div class="card-actions">
        <span class="char-hint">详细描述场景、氛围和期望效果，可获得更精准的生成结果</span>
        <el-button
          type="primary"
          size="large"
          :loading="submitting"
          :disabled="!prompt.trim()"
          @click="handleSubmit"
        >
          <el-icon v-if="!submitting"><VideoPlay /></el-icon>
          {{ submitting ? '正在创建...' : '开始生成视频' }}
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useTasksStore } from '@/stores/tasks'
import { useAuthStore } from '@/stores/auth'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'

const emit = defineEmits<{
  (e: 'task-created', task: any): void
}>()

const tasksStore = useTasksStore()
const authStore = useAuthStore()
const router = useRouter()
const prompt = ref('')
const submitting = ref(false)

async function handleSubmit() {
  if (!prompt.value.trim()) return

  if (!authStore.isAuthenticated) {
    router.push('/login')
    ElMessage.warning('请先登录')
    return
  }

  submitting.value = true
  try {
    const task = await tasksStore.createTask(prompt.value.trim())
    ElMessage.success('任务已创建，正在生成视频...')
    prompt.value = ''
    emit('task-created', task)
  } catch (e: any) {
    const detail = e.response?.data?.detail || '创建任务失败'
    ElMessage.error(detail)
  } finally {
    submitting.value = false
  }
}
</script>

<style scoped>
.task-create-card {
  background: #ffffff;
  border: 1px solid var(--color-border-light);
  border-radius: var(--radius-xl);
  padding: 28px;
  box-shadow: var(--shadow-xs);
}

.card-header {
  margin-bottom: 20px;
}

.card-title {
  font-size: var(--font-size-lg);
  font-weight: 600;
  margin-bottom: 6px;
}

.card-desc {
  font-size: var(--font-size-base);
  color: var(--color-text-secondary);
}

.card-body {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.card-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.char-hint {
  font-size: var(--font-size-xs);
  color: var(--color-text-tertiary);
}

@media (max-width: 640px) {
  .card-actions {
    flex-direction: column;
    gap: 12px;
    align-items: stretch;
  }

  .char-hint {
    text-align: center;
  }
}
</style>
