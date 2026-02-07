# 用户认证系统使用指南

## 📋 目录
1. [概述](#概述)
2. [快速开始](#快速开始)
3. [API 文档](#api-文档)
4. [认证流程](#认证流程)
5. [安全最佳实践](#安全最佳实践)
6. [常见问题](#常见问题)

---

## 1. 概述

### 1.1 功能特性

- ✅ **用户注册**: 用户名、邮箱、密码注册
- ✅ **用户登录**: 支持用户名或邮箱登录
- ✅ **JWT 认证**: 基于 JWT 的无状态认证
- ✅ **Token 刷新**: 访问令牌过期后可刷新
- ✅ **密码修改**: 用户可修改密码
- ✅ **密码加密**: 使用 bcrypt 加密存储
- ✅ **输入验证**: 完整的数据验证

### 1.2 技术栈

- **认证方案**: JWT (JSON Web Token)
- **密码加密**: bcrypt
- **框架**: FastAPI
- **数据库**: SQLAlchemy

---

## 2. 快速开始

### 2.1 安装依赖

```bash
pip install PyJWT bcrypt passlib python-jose email-validator
```

### 2.2 配置环境变量

创建 `.env` 文件：

```bash
# JWT 密钥（生产环境必须修改）
JWT_SECRET_KEY=your-secret-key-change-in-production

# JWT 配置
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### 2.3 初始化数据库

```bash
python scripts/init_database.py
```

### 2.4 启动服务

```bash
python backend/main.py
```

服务将在 `http://localhost:8000` 启动。

---

## 3. API 文档

### 3.1 用户注册

**端点**: `POST /api/auth/register`

**请求体**:
```json
{
  "username": "testuser",
  "email": "test@example.com",
  "password": "Test1234"
}
```

**响应**:
```json
{
  "message": "注册成功",
  "user_id": 1,
  "username": "testuser"
}
```

**验证规则**:
- 用户名: 3-50字符，只能包含字母、数字和下划线
- 邮箱: 有效的邮箱格式
- 密码: 至少8位，包含大小写字母和数字

### 3.2 用户登录

**端点**: `POST /api/auth/login`

**请求体**:
```json
{
  "username": "testuser",
  "password": "Test1234"
}
```

**响应**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**说明**:
- 支持使用用户名或邮箱登录
- `access_token`: 访问令牌，用于访问受保护的 API
- `refresh_token`: 刷新令牌，用于获取新的访问令牌
- `expires_in`: 访问令牌过期时间（秒）

### 3.3 获取当前用户信息

**端点**: `GET /api/auth/me`

**请求头**:
```
Authorization: Bearer <access_token>
```

**响应**:
```json
{
  "id": 1,
  "username": "testuser",
  "email": "test@example.com",
  "quota": 100,
  "used_quota": 10,
  "remaining_quota": 90,
  "is_active": true,
  "created_at": "2024-01-01T00:00:00",
  "last_login": "2024-01-01T12:00:00"
}
```

### 3.4 刷新访问令牌

**端点**: `POST /api/auth/refresh`

**请求体**:
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

**响应**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### 3.5 修改密码

**端点**: `POST /api/auth/change-password`

**请求头**:
```
Authorization: Bearer <access_token>
```

**请求体**:
```json
{
  "old_password": "Test1234",
  "new_password": "NewPass1234"
}
```

**响应**:
```json
{
  "message": "密码修改成功"
}
```

### 3.6 用户登出

**端点**: `POST /api/auth/logout`

**请求头**:
```
Authorization: Bearer <access_token>
```

**响应**:
```json
{
  "message": "登出成功"
}
```

**说明**:
- 由于使用 JWT，服务端无法主动撤销令牌
- 客户端应该删除本地存储的令牌

---

## 4. 认证流程

### 4.1 注册流程

```
1. 用户提交注册信息（用户名、邮箱、密码）
2. 服务端验证数据格式
3. 检查用户名和邮箱是否已存在
4. 使用 bcrypt 加密密码
5. 创建用户记录
6. 返回成功响应
```

### 4.2 登录流程

```
1. 用户提交登录凭证（用户名/邮箱、密码）
2. 服务端查找用户
3. 验证密码是否正确
4. 生成访问令牌和刷新令牌
5. 更新最后登录时间
6. 返回令牌
```

### 4.3 认证流程

```
1. 客户端在请求头中携带访问令牌
2. 服务端提取并验证令牌
3. 解析令牌获取用户信息
4. 检查用户是否存在且激活
5. 执行业务逻辑
6. 返回响应
```

### 4.4 Token 刷新流程

```
1. 访问令牌过期
2. 客户端使用刷新令牌请求新的访问令牌
3. 服务端验证刷新令牌
4. 生成新的访问令牌
5. 返回新令牌
```

---

## 5. 安全最佳实践

### 5.1 密码安全

#### 密码强度要求
- ✅ 最小长度: 8 字符
- ✅ 必须包含: 大写字母、小写字母、数字
- ⚠️ 建议包含: 特殊字符

#### 密码存储
- ✅ 使用 bcrypt 加密
- ✅ 自动加盐
- ✅ 不存储明文密码

### 5.2 Token 安全

#### Token 配置
```python
JWT_CONFIG = {
    "SECRET_KEY": "your-secret-key-here",  # 从环境变量读取
    "ALGORITHM": "HS256",
    "ACCESS_TOKEN_EXPIRE_MINUTES": 60,
    "REFRESH_TOKEN_EXPIRE_DAYS": 7
}
```

#### Token 存储建议
- ✅ **推荐**: HTTP-Only Cookie（防 XSS）
- ⚠️ **可选**: localStorage / sessionStorage
- ❌ **禁止**: URL 参数

#### Token 使用
```javascript
// 存储 Token
localStorage.setItem('access_token', token);

// 使用 Token
fetch('/api/auth/me', {
  headers: {
    'Authorization': `Bearer ${localStorage.getItem('access_token')}`
  }
});

// 删除 Token（登出）
localStorage.removeItem('access_token');
localStorage.removeItem('refresh_token');
```

### 5.3 HTTPS

- ✅ 生产环境必须使用 HTTPS
- ✅ 使用有效的 SSL/TLS 证书
- ✅ 强制 HTTPS 重定向

### 5.4 防护措施

#### 防暴力破解
- 登录失败次数限制
- 账户锁定机制
- 验证码（可选）

#### 防 CSRF
- 使用 CSRF Token
- 验证 Referer 头

#### 防 XSS
- 输入验证和转义
- Content Security Policy

---

## 6. 常见问题

### 6.1 如何在前端使用认证？

**注册**:
```javascript
async function register(username, email, password) {
  const response = await fetch('/api/auth/register', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ username, email, password })
  });
  
  const data = await response.json();
  if (response.ok) {
    console.log('注册成功:', data);
  } else {
    console.error('注册失败:', data.detail);
  }
}
```

**登录**:
```javascript
async function login(username, password) {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ username, password })
  });
  
  const data = await response.json();
  if (response.ok) {
    // 保存 Token
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('refresh_token', data.refresh_token);
    console.log('登录成功');
  } else {
    console.error('登录失败:', data.detail);
  }
}
```

**访问受保护的 API**:
```javascript
async function getProfile() {
  const token = localStorage.getItem('access_token');
  
  const response = await fetch('/api/auth/me', {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (response.ok) {
    const user = await response.json();
    console.log('用户信息:', user);
  } else if (response.status === 401) {
    // Token 过期，尝试刷新
    await refreshToken();
  }
}
```

**刷新 Token**:
```javascript
async function refreshToken() {
  const refresh_token = localStorage.getItem('refresh_token');
  
  const response = await fetch('/api/auth/refresh', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ refresh_token })
  });
  
  if (response.ok) {
    const data = await response.json();
    localStorage.setItem('access_token', data.access_token);
    console.log('Token 刷新成功');
  } else {
    // 刷新失败，需要重新登录
    console.log('Token 刷新失败，请重新登录');
    logout();
  }
}
```

**登出**:
```javascript
function logout() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
  window.location.href = '/login';
}
```

### 6.2 如何保护 API 端点？

使用 `get_current_active_user` 依赖：

```python
from fastapi import APIRouter, Depends
from middleware.auth_middleware import get_current_active_user
from models.user import User

router = APIRouter()

@router.get("/protected")
async def protected_route(current_user: User = Depends(get_current_active_user)):
    """受保护的路由，需要认证"""
    return {"message": f"Hello, {current_user.username}!"}
```

### 6.3 Token 过期了怎么办？

1. **自动刷新**: 在请求拦截器中检测 401 错误，自动使用刷新令牌获取新的访问令牌
2. **手动刷新**: 用户点击刷新按钮
3. **重新登录**: 刷新令牌也过期时，需要重新登录

### 6.4 如何修改 Token 过期时间？

修改 `backend/config.py`:

```python
JWT_CONFIG = {
    "access_token_expire_minutes": 120,  # 2 小时
    "refresh_token_expire_days": 30,  # 30 天
}
```

### 6.5 如何在生产环境部署？

1. **修改密钥**: 设置环境变量 `JWT_SECRET_KEY`
2. **启用 HTTPS**: 配置 SSL/TLS 证书
3. **配置 CORS**: 限制允许的域名
4. **启用日志**: 记录认证相关操作
5. **监控**: 监控异常登录行为

---

## 7. 测试

### 7.1 运行测试

```bash
# 运行所有认证测试
pytest tests/test_auth.py -v

# 运行特定测试类
pytest tests/test_auth.py::TestUserRegistration -v

# 运行特定测试
pytest tests/test_auth.py::TestUserLogin::test_login_success -v
```

### 7.2 测试覆盖率

```bash
pytest tests/test_auth.py --cov=backend/services/auth_service --cov-report=html
```

---

## 8. 总结

### 8.1 实现的功能

- ✅ 用户注册和登录
- ✅ JWT Token 认证
- ✅ Token 刷新机制
- ✅ 密码修改
- ✅ 密码加密存储
- ✅ 完整的输入验证
- ✅ 认证中间件
- ✅ 完整的测试用例

### 8.2 安全特性

- ✅ bcrypt 密码加密
- ✅ JWT Token 认证
- ✅ 密码强度验证
- ✅ 输入验证和转义
- ✅ Token 过期机制

### 8.3 下一步

- ⏳ 添加邮箱验证
- ⏳ 添加密码重置功能
- ⏳ 添加第三方登录（OAuth）
- ⏳ 添加双因素认证（2FA）
- ⏳ 添加登录历史记录

---

**认证系统已完成！** 🎉
