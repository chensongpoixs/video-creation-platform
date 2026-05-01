import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { UserInfo, LoginRequest, RegisterRequest } from '@/api/auth'
import { authApi } from '@/api/auth'

export const useAuthStore = defineStore('auth', () => {
  const user = ref<UserInfo | null>(null)
  const accessToken = ref<string>(localStorage.getItem('access_token') || '')
  const refreshToken = ref<string>(localStorage.getItem('refresh_token') || '')

  const isAuthenticated = computed(() => !!accessToken.value)

  function setTokens(access: string, refresh: string) {
    accessToken.value = access
    refreshToken.value = refresh
    localStorage.setItem('access_token', access)
    localStorage.setItem('refresh_token', refresh)
  }

  function clearTokens() {
    accessToken.value = ''
    refreshToken.value = ''
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
  }

  async function login(data: LoginRequest) {
    const res = await authApi.login(data)
    setTokens(res.access_token, res.refresh_token)
    await fetchUser()
  }

  async function register(data: RegisterRequest) {
    await authApi.register(data)
  }

  async function fetchUser() {
    try {
      const res = await authApi.me()
      user.value = res
    } catch {
      clearTokens()
      user.value = null
    }
  }

  async function logout() {
    try {
      await authApi.logout()
    } catch {
      // ignore
    }
    user.value = null
    clearTokens()
  }

  return {
    user,
    accessToken,
    refreshToken,
    isAuthenticated,
    login,
    register,
    fetchUser,
    logout,
    setTokens,
    clearTokens,
  }
})
