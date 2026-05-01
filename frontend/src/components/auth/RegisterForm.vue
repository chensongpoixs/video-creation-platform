<template>
  <el-form ref="formRef" :model="form" :rules="rules" size="large" @submit.prevent="handleSubmit">
    <el-form-item prop="username">
      <el-input
        v-model="form.username"
        placeholder="用户名（3-50个字符）"
        :prefix-icon="User"
        clearable
      />
    </el-form-item>

    <el-form-item prop="email">
      <el-input
        v-model="form.email"
        placeholder="邮箱"
        :prefix-icon="Message"
        clearable
      />
    </el-form-item>

    <el-form-item prop="password">
      <el-input
        v-model="form.password"
        type="password"
        placeholder="密码（至少8位，含大小写字母和数字）"
        :prefix-icon="Lock"
        show-password
      />
    </el-form-item>

    <el-form-item prop="confirmPassword">
      <el-input
        v-model="form.confirmPassword"
        type="password"
        placeholder="确认密码"
        :prefix-icon="Lock"
        show-password
        @keyup.enter="handleSubmit"
      />
    </el-form-item>

    <el-form-item>
      <el-button type="primary" native-type="submit" :loading="loading" style="width: 100%">
        注册
      </el-button>
    </el-form-item>
  </el-form>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { ElMessage } from 'element-plus'
import { User, Lock, Message } from '@element-plus/icons-vue'

const router = useRouter()
const authStore = useAuthStore()
const loading = ref(false)

const form = reactive({
  username: '',
  email: '',
  password: '',
  confirmPassword: '',
})

const validateConfirmPassword = (_rule: any, value: string, callback: any) => {
  if (value !== form.password) {
    callback(new Error('两次输入的密码不一致'))
  } else {
    callback()
  }
}

const rules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, max: 50, message: '用户名长度 3-50 个字符', trigger: 'blur' },
    { pattern: /^[a-zA-Z0-9_]+$/, message: '用户名只能包含字母、数字和下划线', trigger: 'blur' },
  ],
  email: [
    { required: true, message: '请输入邮箱', trigger: 'blur' },
    { type: 'email', message: '邮箱格式不正确', trigger: 'blur' },
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 8, message: '密码长度至少 8 位', trigger: 'blur' },
    { pattern: /[A-Z]/, message: '密码必须包含大写字母', trigger: 'blur' },
    { pattern: /[a-z]/, message: '密码必须包含小写字母', trigger: 'blur' },
    { pattern: /\d/, message: '密码必须包含数字', trigger: 'blur' },
  ],
  confirmPassword: [
    { required: true, message: '请确认密码', trigger: 'blur' },
    { validator: validateConfirmPassword, trigger: 'blur' },
  ],
}

async function handleSubmit() {
  loading.value = true
  try {
    await authStore.register({
      username: form.username,
      email: form.email,
      password: form.password,
    })
    ElMessage.success('注册成功，请登录')
    router.push('/login')
  } catch (e: any) {
    const detail = e.response?.data?.detail || '注册失败'
    ElMessage.error(detail)
  } finally {
    loading.value = false
  }
}
</script>
