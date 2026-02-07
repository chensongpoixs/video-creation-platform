# 用户认证系统实现方案

## 📋 目录
1. [需求分析](#需求分析)
2. [技术选型](#技术选型)
3. [系统设计](#系统设计)
4. [实现方案](#实现方案)
5. [安全考虑](#安全考虑)

---

## 1. 需求分析

### 1.1 功能需求

#### 核心功能
- ✅ 用户注册（用户名、邮箱、密码）
- ✅ 用户登录（用户名/邮箱 + 密码）
- ✅ 密码加密存储
- ✅ JWT Token 认证
- ✅ Token 刷新机制
- ✅ 用户登出
- ✅ 密码修改
- ✅ 密码重置（邮箱验证）

#### 扩展功能
- ⏳ 第三方登录（OAuth）
- ⏳ 双因素认证（2FA）
- ⏳ 会话管理
- ⏳ 登录历史记录

### 1.2 非功能需求

- **安全性**: 密码加密、Token 安全、防暴力破解
- **性能**: 认证响应 < 100ms
- **可用性**: 简单易用的 API
- **可扩展性**: 支持多种认证方式

---

## 2. 技术选型

### 2.1 认证方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **JWT** | 无状态、可扩展、跨域 | Token 无法撤销 | API 服务 |
| Session | 服务端控制、可撤销 | 有状态、难扩展 | 传统 Web |
| OAuth | 第三方登录、标准化 | 复杂、依赖第三方 | 社交登录 |

#### 选择：JWT（推荐）

**理由**:
1. ✅ 无状态，适合 API 服务
2. ✅ 可扩展，支持分布式
3. ✅ 跨域友好
4. ✅ 标准化，生态好

### 2.2 密码加密

| 算法 | 安全性 | 性能 | 推荐 |
|------|--------|------|------|
| MD5 | ❌ 低 | 快 | ❌ |
| SHA256 | ⚠️ 中 | 快 | ⚠️ |
| **bcrypt** | ✅ 高 | 慢 | ✅ |
| argon2 | ✅ 最高 | 慢 | ✅ |

#### 选择：bcrypt

**理由**:
1. ✅ 自动加盐
2. ✅ 可调节复杂度
3. ✅ 防彩虹表攻击
4. ✅ Python 支持好

### 2.3 技术栈

- **JWT**: PyJWT
- **密码加密**: bcrypt / passlib
- **验证**: pydantic
- **框架**: FastAPI

---

## 3. 系统设计

### 3.1 认证流程

#### 注册流程
```
用户 → 提交注册信息 → 验证数据 → 加密密码 → 存储用户 → 返回成功
```

#### 登录流程
```
用户 → 提交凭证 → 验证用户 → 验证密码 → 生成 Token → 返回 Token
```

#### 认证流程
```
请求 → 提取 Token → 验证 Token → 解析用户 → 执行操作
```

### 3.2 数据模型

#### User 模型（已有，需增强）
```python
class User(Base):
    id: int
    username: str
    email: str
    password_hash: str  # 新增
    api_key: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_login: datetime  # 新增
```

#### Token 模型（新增）
```python
class RefreshToken(Base):
    id: int
    user_id: int
    token: str
    expires_at: datetime
    created_at: datetime
```

### 3.3 API 设计

#### 认证 API

| 端点 | 方法 | 说明 | 认证 |
|------|------|------|------|
| `/api/auth/register` | POST | 用户注册 | ❌ |
| `/api/auth/login` | POST | 用户登录 | ❌ |
| `/api/auth/logout` | POST | 用户登出 | ✅ |
| `/api/auth/refresh` | POST | 刷新 Token | ✅ |
| `/api/auth/me` | GET | 获取当前用户 | ✅ |
| `/api/auth/change-password` | POST | 修改密码 | ✅ |

#### 请求/响应格式

**注册请求**:
```json
{
  "username": "user123",
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**登录请求**:
```json
{
  "username": "user123",
  "password": "SecurePass123!"
}
```

**登录响应**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### 3.4 JWT 结构

#### Access Token
```json
{
  "sub": "user_id",
  "username": "user123",
  "email": "user@example.com",
  "exp": 1234567890,
  "iat": 1234567890,
  "type": "access"
}
```

#### Refresh Token
```json
{
  "sub": "user_id",
  "exp": 1234567890,
  "iat": 1234567890,
  "type": "refresh"
}
```

---

## 4. 实现方案

### 4.1 目录结构

```
backend/
├── models/
│   ├── user.py              # 用户模型（增强）
│   └── refresh_token.py     # 刷新令牌模型（新增）
├── schemas/
│   └── auth.py              # 认证相关 Schema（新增）
├── services/
│   ├── auth_service.py      # 认证服务（新增）
│   └── password_service.py  # 密码服务（新增）
├── api/
│   └── auth.py              # 认证 API（新增）
├── middleware/
│   └── auth_middleware.py   # 认证中间件（新增）
└── utils/
    ├── jwt_utils.py         # JWT 工具（新增）
    └── security.py          # 安全工具（新增）
```

### 4.2 核心组件

#### 4.2.1 密码服务
```python
class PasswordService:
    @staticmethod
    def hash_password(password: str) -> str:
        """加密密码"""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """验证密码"""
        return bcrypt.checkpw(password.encode(), hashed.encode())
```

#### 4.2.2 JWT 工具
```python
class JWTUtils:
    @staticmethod
    def create_access_token(user_id: int, username: str) -> str:
        """创建访问令牌"""
        payload = {
            "sub": str(user_id),
            "username": username,
            "exp": datetime.utcnow() + timedelta(hours=1),
            "type": "access"
        }
        return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    
    @staticmethod
    def verify_token(token: str) -> dict:
        """验证令牌"""
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
```

#### 4.2.3 认证服务
```python
class AuthService:
    def register(self, username: str, email: str, password: str) -> User:
        """用户注册"""
        # 1. 验证用户名/邮箱是否存在
        # 2. 加密密码
        # 3. 创建用户
        # 4. 返回用户
    
    def login(self, username: str, password: str) -> dict:
        """用户登录"""
        # 1. 查找用户
        # 2. 验证密码
        # 3. 生成 Token
        # 4. 返回 Token
    
    def refresh_token(self, refresh_token: str) -> dict:
        """刷新令牌"""
        # 1. 验证刷新令牌
        # 2. 生成新的访问令牌
        # 3. 返回新令牌
```

#### 4.2.4 认证依赖
```python
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """获取当前用户（依赖注入）"""
    try:
        payload = JWTUtils.verify_token(token)
        user_id = payload.get("sub")
        user = UserRepository(db).get(user_id)
        if not user:
            raise HTTPException(401, "Invalid token")
        return user
    except JWTError:
        raise HTTPException(401, "Invalid token")
```

### 4.3 API 实现

#### 注册 API
```python
@router.post("/register")
async def register(
    data: RegisterSchema,
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    user = auth_service.register(
        username=data.username,
        email=data.email,
        password=data.password
    )
    return {"message": "User registered successfully", "user_id": user.id}
```

#### 登录 API
```python
@router.post("/login")
async def login(
    data: LoginSchema,
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    tokens = auth_service.login(
        username=data.username,
        password=data.password
    )
    return tokens
```

#### 受保护的 API
```python
@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user.to_dict()
```

---

## 5. 安全考虑

### 5.1 密码安全

#### 密码强度要求
- 最小长度: 8 字符
- 必须包含: 大写字母、小写字母、数字
- 可选: 特殊字符

#### 密码验证
```python
def validate_password(password: str) -> bool:
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    return True
```

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

#### Token 存储
- ✅ 客户端: localStorage / sessionStorage
- ✅ HTTP-Only Cookie（更安全）
- ❌ 不要存储在 URL 中

### 5.3 防护措施

#### 防暴力破解
```python
# 登录失败次数限制
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = 15  # 分钟

# 实现
if user.failed_login_attempts >= MAX_LOGIN_ATTEMPTS:
    if user.locked_until > datetime.utcnow():
        raise HTTPException(429, "Account locked")
```

#### 防 CSRF
```python
# 使用 CSRF Token
from fastapi_csrf_protect import CsrfProtect

@app.post("/api/auth/login")
async def login(csrf_protect: CsrfProtect = Depends()):
    csrf_protect.validate_csrf(request)
```

#### 防 XSS
```python
# 输入验证和转义
from pydantic import validator

class RegisterSchema(BaseModel):
    username: str
    
    @validator('username')
    def validate_username(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v
```

### 5.4 HTTPS

- ✅ 生产环境必须使用 HTTPS
- ✅ 使用 SSL/TLS 证书
- ✅ 强制 HTTPS 重定向

---

## 6. 实施步骤

### 步骤 1: 增强用户模型（30分钟）

**文件**: `backend/models/user.py`

**任务**:
- ✅ 添加 password_hash 字段
- ✅ 添加 last_login 字段
- ✅ 添加密码验证方法

### 步骤 2: 创建认证 Schema（30分钟）

**文件**: `backend/schemas/auth.py`

**任务**:
- ✅ RegisterSchema
- ✅ LoginSchema
- ✅ TokenSchema
- ✅ ChangePasswordSchema

### 步骤 3: 实现密码服务（30分钟）

**文件**: `backend/services/password_service.py`

**任务**:
- ✅ 密码加密
- ✅ 密码验证
- ✅ 密码强度验证

### 步骤 4: 实现 JWT 工具（1小时）

**文件**: `backend/utils/jwt_utils.py`

**任务**:
- ✅ 创建访问令牌
- ✅ 创建刷新令牌
- ✅ 验证令牌
- ✅ 解析令牌

### 步骤 5: 实现认证服务（1小时）

**文件**: `backend/services/auth_service.py`

**任务**:
- ✅ 用户注册
- ✅ 用户登录
- ✅ 刷新令牌
- ✅ 修改密码

### 步骤 6: 实现认证 API（1小时）

**文件**: `backend/api/auth.py`

**任务**:
- ✅ 注册端点
- ✅ 登录端点
- ✅ 登出端点
- ✅ 刷新端点
- ✅ 获取当前用户端点

### 步骤 7: 实现认证中间件（30分钟）

**文件**: `backend/middleware/auth_middleware.py`

**任务**:
- ✅ Token 提取
- ✅ Token 验证
- ✅ 用户注入

### 步骤 8: 集成到主应用（30分钟）

**文件**: `backend/main.py`

**任务**:
- ✅ 注册认证路由
- ✅ 配置认证中间件
- ✅ 更新现有 API

### 步骤 9: 测试（1小时）

**文件**: `tests/test_auth.py`

**任务**:
- ✅ 注册测试
- ✅ 登录测试
- ✅ Token 验证测试
- ✅ 权限测试

### 步骤 10: 文档（30分钟）

**文件**: `docs/AUTH_GUIDE.md`

**任务**:
- ✅ 认证流程说明
- ✅ API 使用文档
- ✅ 安全最佳实践

---

## 7. 配置

### 7.1 环境变量

```bash
# .env
JWT_SECRET_KEY=your-secret-key-here-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### 7.2 依赖安装

```bash
pip install PyJWT
pip install bcrypt
pip install passlib[bcrypt]
pip install python-jose[cryptography]
pip install python-multipart
```

---

## 8. 验收标准

### 功能验收
- ✅ 用户可以注册
- ✅ 用户可以登录
- ✅ Token 认证正常
- ✅ Token 刷新正常
- ✅ 密码修改正常
- ✅ 受保护的 API 需要认证

### 安全验收
- ✅ 密码加密存储
- ✅ Token 安全生成
- ✅ 防暴力破解
- ✅ 输入验证完整

### 性能验收
- ✅ 认证响应 < 100ms
- ✅ Token 验证 < 10ms
- ✅ 密码验证 < 100ms

---

## 9. 总结

### 实施收益

- ✅ **安全性**: 完整的认证系统
- ✅ **可扩展性**: 支持多种认证方式
- ✅ **用户体验**: 简单易用
- ✅ **标准化**: 遵循行业标准

### 技术亮点

1. **JWT 认证**: 无状态、可扩展
2. **bcrypt 加密**: 安全的密码存储
3. **Token 刷新**: 平衡安全和体验
4. **依赖注入**: 优雅的权限控制
5. **完整防护**: 防暴力破解、CSRF、XSS

### 实施计划

- **总时间**: 6-7 小时
- **优先级**: 高
- **风险**: 低
- **收益**: 高

---

**准备开始实施！** 🚀
