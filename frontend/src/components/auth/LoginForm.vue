<template>
  <el-form ref="formRef" :model="form" :rules="rules" size="large" @submit.prevent="handleSubmit">
    <el-form-item prop="username">
      <el-input
        v-model="form.username"
        placeholder="用户名或邮箱"
        :prefix-icon="User"
        clearable
      />
    </el-form-item>

    <el-form-item prop="password">
      <el-input
        v-model="form.password"
        type="password"
        placeholder="密码"
        :prefix-icon="Lock"
        show-password
        @keyup.enter="handleSubmit"
      />
    </el-form-item>

    <el-form-item>
      <el-button type="primary" native-type="submit" :loading="loading" style="width: 100%">
        登录
      </el-button>
    </el-form-item>
  </el-form>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { ElMessage } from 'element-plus'
import { User, Lock } from '@element-plus/icons-vue'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()
const loading = ref(false)

const form = reactive({
  username: '',
  password: '',
})

const rules = {
  username: [{ required: true, message: '请输入用户名或邮箱', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
}

async function handleSubmit() {
  loading.value = true
  try {
    await authStore.login({
      username: form.username,
      password: form.password,
    })
    ElMessage.success('登录成功')
    const redirect = (route.query.redirect as string) || '/'
    router.push(redirect)
  } catch (e: any) {
    const detail = e.response?.data?.detail || '登录失败'
    ElMessage.error(detail)
  } finally {
    loading.value = false
  }
}
</script>
