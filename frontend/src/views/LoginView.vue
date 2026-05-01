<template>
  <div class="auth-view">
    <div class="auth-card">
      <div class="auth-header">
        <h1 class="auth-title">登录</h1>
        <p class="auth-subtitle">欢迎回来，请登录您的账号</p>
      </div>
      <LoginForm />
      <div class="auth-footer">
        还没有账号？<router-link to="/register">立即注册</router-link>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import LoginForm from '@/components/auth/LoginForm.vue'
</script>

<style scoped>
.auth-view {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 60vh;
  padding: 48px 16px;
}

.auth-card {
  width: 100%;
  max-width: 420px;
  padding: 40px;
  background: #ffffff;
  border: 1px solid var(--color-border-light);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-md);
}

.auth-header {
  text-align: center;
  margin-bottom: 32px;
}

.auth-title {
  font-size: var(--font-size-2xl);
  font-weight: 700;
  color: var(--color-text-primary);
  margin-bottom: 8px;
}

.auth-subtitle {
  font-size: var(--font-size-base);
  color: var(--color-text-secondary);
}

.auth-footer {
  margin-top: 24px;
  text-align: center;
  font-size: var(--font-size-base);
  color: var(--color-text-secondary);
}

.auth-footer a {
  color: var(--color-primary);
  font-weight: 500;
}
</style>
