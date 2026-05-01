<template>
  <div class="skeleton">
    <div class="skeleton-line" v-for="n in lines" :key="n" :style="{ width: getWidth(n) }" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  lines?: number
}>(), {
  lines: 4,
})

function getWidth(index: number): string {
  const widths = ['60%', '90%', '40%', '75%']
  return widths[(index - 1) % widths.length]
}
</script>

<style scoped>
.skeleton {
  padding: 24px;
}

.skeleton-line {
  height: 16px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 4px;
  margin-bottom: 12px;
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
</style>
