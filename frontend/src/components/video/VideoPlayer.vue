<template>
  <div class="video-player">
    <div class="video-container" v-if="!loadError">
      <video
        ref="videoRef"
        :src="src"
        class="video-element"
        controls
        playsinline
        preload="metadata"
        @loadedmetadata="onLoaded"
        @error="onError"
      >
        <p>您的浏览器不支持视频播放，请下载后查看</p>
      </video>
    </div>
    <div v-else class="video-error">
      <el-icon :size="48"><VideoPlay /></el-icon>
      <p>视频无法加载</p>
      <p class="error-hint">请尝试下载或检查文件是否存在</p>
      <el-button type="primary" size="small" @click="handleDownload">
        <el-icon><Download /></el-icon>
        下载视频
      </el-button>
    </div>
    <div class="video-meta" v-if="videoDuration">
      <div class="meta-item">
        <el-icon><Timer /></el-icon>
        <span>{{ formattedDuration }}</span>
      </div>
      <div class="meta-item" v-if="videoWidth && videoHeight">
        <span>{{ resolution }}</span>
      </div>
      <div class="meta-item">
        <el-button type="primary" size="small" @click="handleDownload">
          <el-icon><Download /></el-icon>
          下载视频
        </el-button>
      </div>
    </div>
    <!-- 即使没有时长元数据也显示下载按钮 -->
    <div class="video-meta" v-else>
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
import { Timer, Download, VideoPlay } from '@element-plus/icons-vue'

const props = defineProps<{
  src: string | null
  taskId: string
}>()

const videoRef = ref<HTMLVideoElement | null>(null)
const videoDuration = ref(0)
const videoWidth = ref(0)
const videoHeight = ref(0)
const loadError = ref(false)

const formattedDuration = computed(() => {
  const mins = Math.floor(videoDuration.value / 60)
  const secs = Math.floor(videoDuration.value % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
})

const resolution = computed(() => {
  if (videoWidth.value && videoHeight.value) {
    return `${videoWidth.value} x ${videoHeight.value}`
  }
  return ''
})

function onLoaded() {
  if (videoRef.value) {
    videoDuration.value = videoRef.value.duration
    videoWidth.value = videoRef.value.videoWidth
    videoHeight.value = videoRef.value.videoHeight
    loadError.value = false
  }
}

function onError(e: Event) {
  loadError.value = true
}

function handleDownload() {
  if (!props.src) return
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

.video-error {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 48px;
  background: #fafafa;
  border: 2px dashed var(--color-border-light);
  border-radius: var(--radius-md);
  color: var(--color-text-secondary);
}

.error-hint {
  font-size: var(--font-size-sm);
  color: var(--color-text-tertiary);
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
