<template>
  <header class="app-header">
    <div class="header-inner">
      <router-link to="/" class="logo">
        <el-icon :size="24"><VideoCamera /></el-icon>
        <span class="logo-text">多模态视频创作平台</span>
      </router-link>

      <nav class="header-nav">
        <router-link to="/" class="nav-item" active-class="nav-item--active">
          <el-icon><Edit /></el-icon>
          <span>创作</span>
        </router-link>
        <router-link v-if="auth.isAuthenticated" to="/tasks" class="nav-item" active-class="nav-item--active">
          <el-icon><List /></el-icon>
          <span>任务</span>
        </router-link>
      </nav>

      <div class="header-actions">
        <template v-if="auth.isAuthenticated">
          <el-dropdown trigger="click">
            <div class="user-info">
              <el-avatar :size="32" icon="UserFilled" />
              <span class="user-name">{{ auth.user?.username }}</span>
              <el-icon class="dropdown-icon"><ArrowDown /></el-icon>
            </div>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item disabled>
                  <div class="dropdown-user-detail">
                    <div>配额: {{ auth.user?.remaining_quota }} / {{ auth.user?.quota }}</div>
                  </div>
                </el-dropdown-item>
                <el-dropdown-item divided @click="handleLogout">
                  <el-icon><SwitchButton /></el-icon>
                  退出登录
                </el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </template>
        <template v-else>
          <el-button text @click="$router.push('/login')">登录</el-button>
          <el-button type="primary" size="small" @click="$router.push('/register')">注册</el-button>
        </template>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
import { useAuthStore } from '@/stores/auth'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

const auth = useAuthStore()
const router = useRouter()

async function handleLogout() {
  await auth.logout()
  ElMessage.success('已退出登录')
  router.push('/')
}
</script>

<style scoped>
.app-header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: #ffffff;
  border-bottom: 1px solid var(--color-border-light);
  box-shadow: var(--shadow-xs);
  height: 60px;
}

.header-inner {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
  height: 100%;
  display: flex;
  align-items: center;
  gap: 32px;
}

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text-primary);
  font-size: var(--font-size-lg);
  font-weight: 600;
  text-decoration: none;
  flex-shrink: 0;
}

.logo .el-icon {
  color: var(--color-primary);
}

.header-nav {
  display: flex;
  gap: 4px;
  flex: 1;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border-radius: var(--radius-md);
  color: var(--color-text-secondary);
  font-size: var(--font-size-base);
  font-weight: 500;
  transition: all 0.2s;
  text-decoration: none;
}

.nav-item:hover {
  color: var(--color-text-primary);
  background: var(--color-bg-secondary);
}

.nav-item--active {
  color: var(--color-primary);
  background: rgba(26, 115, 232, 0.08);
}

.header-actions {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: var(--radius-md);
  transition: background 0.2s;
}

.user-info:hover {
  background: var(--color-bg-secondary);
}

.user-name {
  font-size: var(--font-size-base);
  font-weight: 500;
  color: var(--color-text-primary);
}

.dropdown-icon {
  font-size: 12px;
  color: var(--color-text-tertiary);
}

.dropdown-user-detail {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
  padding: 4px 0;
}
</style>
