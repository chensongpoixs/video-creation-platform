<template>
  <footer class="app-footer">
    <div class="footer-inner">
      <p class="footer-text">多模态视频创作平台 &copy; {{ year }} — 本地私有化部署 · 数据安全保障</p>
    </div>
  </footer>
</template>

<script setup lang="ts">
const year = new Date().getFullYear()
</script>

<style scoped>
.app-footer {
  border-top: 1px solid var(--color-border-light);
  background: var(--color-bg-secondary);
  padding: 24px;
  margin-top: auto;
}

.footer-inner {
  max-width: 1200px;
  margin: 0 auto;
  text-align: center;
}

.footer-text {
  font-size: var(--font-size-sm);
  color: var(--color-text-tertiary);
}
</style>
