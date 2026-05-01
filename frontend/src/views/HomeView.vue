<template>
  <div class="home-view">
    <section class="hero-section">
      <h1 class="hero-title">AI 驱动的视频内容创作</h1>
      <p class="hero-subtitle">输入自然语言描述，自动生成专业视频。从脚本到成品，一站式完成。</p>
    </section>

    <section class="create-section">
      <TaskCreate @task-created="handleTaskCreated" />
    </section>

    <section v-if="activeTasks.length > 0" class="active-tasks-section">
      <h2 class="section-title">进行中的任务</h2>
      <div class="active-tasks-grid">
        <TaskCard
          v-for="task in activeTasks"
          :key="task.task_id"
          :task="task"
          @click="$router.push(`/tasks/${task.task_id}`)"
        />
      </div>
    </section>

    <section class="features-section">
      <h2 class="section-title">平台能力</h2>
      <div class="features-grid">
        <div v-for="feature in features" :key="feature.title" class="feature-card">
          <el-icon :size="32" class="feature-icon">
            <component :is="feature.icon" />
          </el-icon>
          <h3 class="feature-title">{{ feature.title }}</h3>
          <p class="feature-desc">{{ feature.desc }}</p>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import TaskCreate from '@/components/tasks/TaskCreate.vue'
import TaskCard from '@/components/tasks/TaskCard.vue'
import type { TaskInfo } from '@/api/tasks'

const router = useRouter()
const activeTasks = ref<TaskInfo[]>([])

function handleTaskCreated(task: TaskInfo) {
  activeTasks.value.unshift(task)
  router.push(`/tasks/${task.task_id}`)
}

const features = [
  { icon: 'Edit', title: '自然语言输入', desc: '输入创作描述，AI 自动理解并转化为视频创作指令' },
  { icon: 'Document', title: '智能脚本生成', desc: 'LLM 大语言模型生成详细分镜脚本，包含场景、镜头和动作描述' },
  { icon: 'VideoPlay', title: '视频自动生成', desc: '基于扩散模型的视频生成引擎，支持多场景视频片段生成' },
  { icon: 'Setting', title: '专业后处理', desc: '自动视频拼接、滤镜、字幕、背景音乐和画质优化' },
  { icon: 'Monitor', title: '任务管理', desc: '实时任务状态追踪，支持历史记录查询和视频回放下载' },
  { icon: 'Lock', title: '私有化部署', desc: '全部数据和模型运行在本地，无需联网，保证数据安全' },
]
</script>

<style scoped>
.home-view {
  max-width: 900px;
  margin: 0 auto;
}

.hero-section {
  text-align: center;
  padding: 48px 0 40px;
}

.hero-title {
  font-size: var(--font-size-3xl);
  font-weight: 700;
  color: var(--color-text-primary);
  margin-bottom: 12px;
  letter-spacing: -0.5px;
}

.hero-subtitle {
  font-size: var(--font-size-lg);
  color: var(--color-text-secondary);
  max-width: 560px;
  margin: 0 auto;
  line-height: 1.7;
}

.create-section {
  margin: 0 0 48px;
}

.section-title {
  font-size: var(--font-size-xl);
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: 20px;
}

.active-tasks-section {
  margin-bottom: 48px;
}

.active-tasks-grid {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.features-section {
  margin-bottom: 32px;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.feature-card {
  padding: 28px 24px;
  border: 1px solid var(--color-border-light);
  border-radius: var(--radius-lg);
  background: #ffffff;
  transition: box-shadow 0.2s, border-color 0.2s;
}

.feature-card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--color-border);
}

.feature-icon {
  color: var(--color-primary);
  margin-bottom: 16px;
}

.feature-title {
  font-size: var(--font-size-base);
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: 8px;
}

.feature-desc {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
  line-height: 1.6;
}

@media (max-width: 768px) {
  .features-grid {
    grid-template-columns: 1fr;
  }
}
</style>
