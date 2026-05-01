# 多模态视频创作平台

基于本地私有化部署的多模态视频创作系统，支持从自然语言描述自动生成视频内容。

## 项目简介

本系统采用 **大语言模型（LLM）+ 扩散模型（Diffusion Model）** 的协同架构，实现从用户文本输入到视频生成输出的完整自动化流程。所有模型和数据均在本地运行，无需依赖云端服务，保证数据隐私与安全。

### 核心功能

1. **自然语言输入解析** — 用户通过自然语言描述创作需求
2. **智能脚本与分镜生成** — 基于 ChatGLM3-6B 生成结构化分镜脚本
3. **视频片段生成** — 基于 Stable Video Diffusion (SVD-XT) 生成视频帧
4. **视频后处理** — 滤镜、转场、字幕、音频、画质优化
5. **任务队列管理** — 异步任务处理、状态追踪、历史查询
6. **用户认证系统** — JWT Token 认证、用户注册登录、配额管理
7. **企业级前端界面** — Vue 3 + TypeScript + Element Plus 专业后台
8. **本地私有化部署** — Docker 容器化，完全离线运行

---

## 项目架构

### 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **前端** | Vue 3 + TypeScript + Vite + Element Plus | 企业级 SPA，白色主题专业后台 |
| **API 框架** | FastAPI 0.104 + Uvicorn | 高性能异步 Web 框架 |
| **认证** | JWT (HS256) + bcrypt | Token 认证 + 密码加密 |
| **深度学习** | PyTorch 2.1 + CUDA 11.7 | 模型推理核心 |
| **LLM 模型** | Transformers 4.35 + ChatGLM3-6B | 脚本生成与分镜拆解 |
| **视频生成** | Diffusers 0.24 + Stable Video Diffusion XT | 视频帧生成 |
| **显存优化** | FP16 + xFormers + Attention/VAE Slicing | 显存减半 + 推理加速 |
| **视频处理** | OpenCV 4.8 + MoviePy 1.0 | 帧处理、拼接、后处理 |
| **数据库** | SQLAlchemy 2.0 + SQLite | ORM + 轻量级数据库 |
| **容器化** | Docker + Docker Compose | 一键私有化部署 |

### 项目结构

```
video-creation-platform/
├── backend/                    # 后端服务 (Python/FastAPI)
│   ├── main.py                # FastAPI 主入口
│   ├── config.py              # 系统配置
│   ├── models/                # SQLAlchemy 数据模型
│   │   ├── database.py        # 数据库引擎与会话管理
│   │   ├── user.py            # 用户模型
│   │   ├── task.py            # 任务模型 + TaskStatus 枚举
│   │   ├── video.py           # 视频资源模型
│   │   └── script.py          # 分镜脚本模型
│   ├── api/                   # API 路由
│   │   ├── auth.py            # 认证接口 (/api/auth/*)
│   │   └── tasks.py           # 任务接口 (/api/tasks/*)
│   ├── middleware/            # 中间件
│   │   ├── auth_middleware.py # JWT 认证中间件
│   │   └── performance_middleware.py
│   ├── services/              # 业务逻辑
│   │   ├── auth_service.py    # 用户认证服务
│   │   ├── task_processor.py  # 任务协调处理器
│   │   ├── llm_service.py     # LLM 脚本生成
│   │   ├── video_service.py   # 视频生成服务
│   │   ├── model_loader.py    # 模型加载器 (单例)
│   │   ├── video_processor.py # 视频帧处理
│   │   ├── video_filter.py    # 滤镜处理
│   │   ├── video_optimizer.py # 画质优化
│   │   ├── subtitle_system.py # 字幕系统
│   │   └── audio_processor.py # 音频处理
│   ├── repositories/          # 数据访问层
│   │   ├── base.py            # 泛型基础仓储
│   │   ├── user_repository.py
│   │   ├── task_repository.py
│   │   └── video_repository.py
│   ├── schemas/               # Pydantic 请求/响应模型
│   ├── utils/                 # 工具模块 (JWT, 日志, 缓存, 显存监控)
│   └── requirements.txt       # Python 依赖
├── frontend/                   # 前端 (Vue 3 SPA)
│   ├── src/
│   │   ├── router/            # Vue Router 路由配置
│   │   ├── stores/            # Pinia 状态管理 (auth, tasks)
│   │   ├── api/               # Axios API 封装 (含 JWT 拦截器)
│   │   ├── views/             # 页面组件
│   │   │   ├── HomeView.vue   # 首页 / 创作页
│   │   │   ├── LoginView.vue  # 登录页
│   │   │   ├── RegisterView.vue # 注册页
│   │   │   ├── TasksView.vue  # 任务管理页
│   │   │   └── TaskDetailView.vue # 任务详情 + 视频播放
│   │   ├── components/        # 可复用组件
│   │   │   ├── layout/        # AppHeader, AppFooter
│   │   │   ├── auth/          # LoginForm, RegisterForm
│   │   │   ├── tasks/         # TaskCreate, TaskCard
│   │   │   ├── video/         # VideoPlayer
│   │   │   └── common/        # LoadingSkeleton
│   │   └── styles/            # 全局样式 + 设计 Token
│   ├── package.json
│   ├── vite.config.ts         # Vite 构建配置 (含 API 代理)
│   └── tsconfig.json
├── tests/                      # 测试套件 (pytest)
├── scripts/                    # 工具脚本
│   ├── init_database.py       # 数据库初始化
│   ├── setup.sh               # 环境安装
│   └── download_model.py      # 模型下载
├── docs/                       # 文档
├── docker-compose.yml          # Docker Compose 配置
├── backend/Dockerfile          # 后端容器镜像
└── README.md
```

### 系统流程

```
用户输入创作指令
       ↓
  JWT 认证验证
       ↓
  LLM 生成脚本与分镜 JSON
       ↓
  Stable Video Diffusion 生成视频帧
       ↓
  视频拼接 + 滤镜 + 字幕 + 音频
       ↓
  视频压缩输出 → 前端预览下载
```

---

## 快速开始

### 环境要求

| 组件 | 要求 |
|------|------|
| 操作系统 | Ubuntu 20.04+ / Windows WSL2 |
| GPU | NVIDIA GPU (推荐 RTX 3090/4090, 显存 ≥ 16GB) |
| CUDA | 11.7+ |
| Python | 3.10 |
| Node.js | 18+ (前端开发) |
| Docker | 20.10+ (容器化部署) |

### 1. 创建 Conda 虚拟环境

```bash
# 创建 Python 3.10 虚拟环境
conda create -n video_creation_platform python=3.10

# 激活环境
conda activate video_creation_platform
```

### 2. 安装后端依赖

```bash
cd backend

# 国内用户推荐使用清华源加速（国外用户可省略 -i 参数）
# PyTorch CUDA 版本需从官方源安装
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 安装其余依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **国内镜像源**：清华 `https://pypi.tuna.tsinghua.edu.cn/simple` | 阿里云 `https://mirrors.aliyun.com/pypi/simple` | 中科大 `https://mirrors.ustc.edu.cn/pypi/simple`
>
> **CUDA 版本**：`cu124` 兼容 CUDA 12.4+（含 12.8）。如需其他 CUDA 版本，将 `cu124` 替换为 `cu118` / `cu121` 等。

### 3. 初始化数据库

```bash
python scripts/init_database.py
```

### 4. 配置环境变量（可选）

```bash
# Hugging Face 镜像（国内默认使用 hf-mirror.com，国外用户可设为官方地址）
export HF_MIRROR="https://hf-mirror.com"

# JWT 密钥（生产环境务必修改）
export JWT_SECRET_KEY="your-secret-key-change-in-production"
```

### 5. 下载模型文件

```bash
# 首次运行前需下载模型（默认使用 hf-mirror.com 国内镜像加速）
python scripts/download_model.py --source hf

# 如遇网络问题，可尝试 ModelScope 源
python scripts/download_model.py --source ms
```

> 模型文件较大（ChatGLM3-6B ~12GB, SVD-XT ~5GB），请预留足够磁盘空间。
> 也可在首次启动时自动下载（`config.py` 中 `auto_download: True`）。

### 6. 启动后端服务

```bash
# 开发模式（热重载）
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 或直接运行
python main.py
```

服务启动后访问：
- **API 文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/health
- **模型状态**：http://localhost:8000/api/model/status

### 7. 安装并启动前端（开发模式）

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器（热重载，API 自动代理到 8000 端口）
npm run dev
```

前端开发服务器启动后，终端会输出实际访问地址（端口动态分配，默认从 5173 开始）

### 8. 后端地址配置

前端通过 `public/config.js` 配置后端地址，**部署时修改此文件即可，无需重新构建**：

```javascript
// frontend/public/config.js
window.__APP_CONFIG__ = {
  apiBaseURL: '/',          // 开发环境保持 '/'
                            // 生产环境改为 'http://your-server:8000'
  timeout: 30000,           // 请求超时时间（毫秒）
  backendPort: 8000,        // 后端端口（供参考）
}
```

开发环境下也可通过环境变量指定 Vite proxy 目标：

```bash
VITE_BACKEND_URL=http://192.168.1.100:8000 npm run dev
```

### 9. 生产部署

```bash
# 构建前端（public/config.js 会被复制到 dist/）
cd frontend && npm run build

# 修改 dist/config.js 中的后端地址（如需要）
# 启动后端（会自动服务前端静态文件）
cd ../backend && uvicorn main:app --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000 即可使用完整应用。

---

## Docker 部署

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

---

## API 文档

### 认证 API

| 接口 | 方法 | 说明 | 认证 |
|------|------|------|------|
| `/api/auth/register` | POST | 用户注册 | 否 |
| `/api/auth/login` | POST | 用户登录，返回 JWT Token | 否 |
| `/api/auth/refresh` | POST | 刷新 Access Token | 否 |
| `/api/auth/me` | GET | 获取当前用户信息 | 是 |
| `/api/auth/change-password` | POST | 修改密码 | 是 |
| `/api/auth/logout` | POST | 用户登出 | 是 |

### 任务 API

| 接口 | 方法 | 说明 | 认证 |
|------|------|------|------|
| `/api/tasks` | POST | 创建视频生成任务 | 是 |
| `/api/tasks/{task_id}` | GET | 查询任务状态 | 是 |
| `/api/tasks` | GET | 获取任务列表（支持分页和状态过滤） | 是 |
| `/api/tasks/{task_id}` | DELETE | 删除任务 | 是 |

### 系统 API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/model/status` | GET | GPU 与模型加载状态 |

---

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行认证测试
pytest tests/test_auth.py -v

# 运行后处理测试
pytest tests/test_video_post_processing.py -v

# 运行性能测试
pytest tests/test_performance.py -v --benchmark-only

# 运行压力测试
pytest tests/test_stress.py -v -s

# 运行覆盖度报告
pytest tests/ --cov=backend -v

# Locust 压力测试
locust -f tests/locustfile.py --host=http://localhost:8000

# 前端类型检查
cd frontend && npx vue-tsc --noEmit
```

---

## 文档索引

- [认证系统使用指南](docs/AUTH_GUIDE.md)
- [LLM 集成指南](docs/LLM_INTEGRATION_GUIDE.md)
- [视频模型集成指南](docs/VIDEO_MODEL_INTEGRATION_GUIDE.md)
- [数据库使用指南](docs/DATABASE_GUIDE.md)
- [显存优化指南](docs/MEMORY_OPTIMIZATION_GUIDE.md)
- [测试实施指南](docs/TEST_IMPLEMENTATION_GUIDE.md)
- [API 文档](docs/API.md)
- [架构文档](docs/ARCHITECTURE.md)

---

## 注意事项

- 视频生成需要高性能 GPU（推荐 RTX 3090/4090，显存 ≥ 16GB）
- 首次运行会自动下载模型文件（ChatGLM3-6B, Stable Video Diffusion XT），需要较长时间
- 所有数据和模型均在本地运行，保证隐私安全
- 生产环境请务必修改 JWT_SECRET_KEY
- 前端开发需要 Node.js 18+，生产部署只需构建后的静态文件

## 许可证

MIT License
