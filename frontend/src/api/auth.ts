import api from './index'

export interface LoginRequest {
  username: string
  password: string
}

export interface RegisterRequest {
  username: string
  email: string
  password: string
}

export interface TokenResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

export interface UserInfo {
  id: number
  username: string
  email: string
  quota: number
  used_quota: number
  remaining_quota: number
  is_active: boolean
  created_at: string
  last_login: string | null
}

export const authApi = {
  login(data: LoginRequest): Promise<TokenResponse> {
    return api.post('/api/auth/login', data).then((r) => r.data)
  },

  register(data: RegisterRequest) {
    return api.post('/api/auth/register', data).then((r) => r.data)
  },

  me(): Promise<UserInfo> {
    return api.get('/api/auth/me').then((r) => r.data)
  },

  refresh(refreshToken: string): Promise<TokenResponse> {
    return api.post('/api/auth/refresh', { refresh_token: refreshToken }).then((r) => r.data)
  },

  changePassword(oldPassword: string, newPassword: string) {
    return api.post('/api/auth/change-password', {
      old_password: oldPassword,
      new_password: newPassword,
    }).then((r) => r.data)
  },

  logout() {
    return api.post('/api/auth/logout').then((r) => r.data)
  },
}
