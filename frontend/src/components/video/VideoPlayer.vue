<template>
  <div class="video-player">
    <div class="video-container">
      <video
        ref="videoRef"
        :src="src"
        class="video-element"
        controls
        playsinline
        @loadedmetadata="onLoaded"
        @error="onError"
      />
    </div>
    <div class="video-meta" v-if="videoDuration">
      <div class="meta-item">
        <el-icon><Timer /></el-icon>
        <span>{{ formattedDuration }}</span>
      </div>
      <div class="meta-item">
        <el-icon><VideoPlay /></el-icon>
        <span>{{ resolution }}</span>
      </div>
      <div class="meta-item">
        <el-button type="primary" size="small" @click="handleDownload">
          <el-icon><Download /></el-icon>
          下载视频
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { Timer, Download } from '@element-plus/icons-vue'

const props = defineProps<{
  src: string
  taskId: string
}>()

const videoRef = ref<HTMLVideoElement | null>(null)
const videoDuration = ref(0)
const videoWidth = ref(0)
const videoHeight = ref(0)

const formattedDuration = computed(() => {
  const mins = Math.floor(videoDuration.value / 60)
  const secs = Math.floor(videoDuration.value % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
})

const resolution = computed(() => {
  if (videoWidth.value && videoHeight.value) {
    return `${videoWidth.value} x ${videoHeight.value}`
  }
  return '-'
})

function onLoaded() {
  if (videoRef.value) {
    videoDuration.value = videoRef.value.duration
    videoWidth.value = videoRef.value.videoWidth
    videoHeight.value = videoRef.value.videoHeight
  }
}

function onError() {
  // handle gracefully
}

function handleDownload() {
  const a = document.createElement('a')
  a.href = props.src
  a.download = `video_${props.taskId}.mp4`
  a.click()
}
</script>

<style scoped>
.video-player {
  background: #ffffff;
  border-radius: var(--radius-md);
  overflow: hidden;
}

.video-container {
  background: #000;
  border-radius: var(--radius-md);
  overflow: hidden;
}

.video-element {
  width: 100%;
  display: block;
  max-height: 540px;
}

.video-meta {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 16px 0;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
}
</style>
