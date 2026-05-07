# 安装文档

## 环境要求

### 硬件

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA RTX 3060 (12GB) | RTX 5080 (16GB) |
| 内存 | 16GB | 32GB+ |
| 存储 | 50GB SSD | 100GB+ SSD |

### 软件

| 软件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 推荐 3.11 |
| CUDA | 12.4+ | RTX 50 系列需 12.8 |
| Node.js | 18+ | 前端构建 |
| Git | 任意 | 版本管理 |

---

## 快速安装

### 1. 克隆项目

```bash
git clone https://github.com/chensongpoixs/video-creation-platform
cd video-creation-platform
```

### 2. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / WSL
python3 -m venv venv
source venv/bin/activate




conda create -n video python=3.12


export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### 3. 安装 PyTorch（先装，指定 CUDA 版本）

```bash
# CUDA 12.8（RTX 50 系列 / Blackwell）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# pip install torch torchvision torchaudio -f https://mirrors.tuna.tsinghua.edu.cn/pytorch/whl/cu128/

# CUDA 12.4（RTX 30/40 系列）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU only（无 GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. 安装 Python 依赖

```bash
# 国内镜像（清华源，推荐）
pip install -r backend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或直接安装
pip install -r backend/requirements.txt
```

### 5. 安装前端依赖

```bash
cd frontend
npm install
cd ..
```

---

## 模型下载

项目使用 **2 个模型**（详见 `docs/MODELS.md`）：

| 模型 | 大小 | 显存 | 说明 |
|------|------|------|------|
| ChatGLM3-6B | ~12GB | ~12GB | LLM 剧本生成 |
| CogVideoX-2b | ~12GB | ~8GB (FP16) | 文生视频（支持中文） |

### 自动下载（推荐）

首次启动后端时自动下载，配置已开启 `auto_download=True`：

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 6006

# 等待下载完成（约 10-20 分钟，取决于网速）
```

### 手动下载（可选）

```bash
# 下载全部两个模型
python scripts/download_model.py

# 只下载 LLM 剧本生成模型
python scripts/download_model.py --model llm

# 只下载视频生成模型
python scripts/download_model.py --model video

# 从 ModelScope 下载（国内更快）
python scripts/download_model.py --source ms

# 自定义 HuggingFace 镜像
HF_MIRROR="https://hf-mirror.com" python scripts/download_model.py
```

### CPU 模式（不下载模型）

如果不需要 GPU 模型，LLM 和视频生成会自动退回到备用方案：
- LLM：基于句子拆分的简单分镜
- 视频：OpenCV 渐变色 + 文字叠加的演示视频

在 `backend/config.py` 中将 `device` 设为 `"cpu"` 即可，LLM 需额外设置 `allow_cpu_inference=False`。

---

## 启动服务

### 1. 初始化数据库

```bash
cd backend
python scripts/init_database.py
```

### 2. 启动后端

```bash
cd backend

# 开发模式（热重载）
uvicorn main:app --host 0.0.0.0 --port 8010 --reload
# uvicorn main:app --host 0.0.0.0 --port 6006 --reload

# 生产模式
uvicorn main:app --host 0.0.0.0 --port 8010
```

### 3. 启动前端

```bash
cd frontend

# 开发模式（热重载，自动代理 API 到 8010 端口）
npm run dev

# 生产构建
npm run build
```

### 4. 访问

- 前端页面：http://localhost:5173 （开发模式端口可能不同，看终端输出）
- API 文档：http://localhost:8010/docs
- 健康检查：http://localhost:8010/health

### 5. 前端配置（生产部署）

生产环境部署时，修改 `frontend/public/config.js` 中的后端地址：

```javascript
window.__APP_CONFIG__ = {
  apiBaseURL: 'http://192.168.1.100:8010',   // 后端 API 地址
  videoBaseURL: 'http://192.168.1.100:8010', // 视频文件地址（默认同 API）
  // 如视频托管在 CDN，可单独配置:
  // videoBaseURL: 'https://cdn.example.com',
  timeout: 30000,
  backendPort: 8010,
}
```

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `apiBaseURL` | 后端 API 基础地址 | `/` |
| `videoBaseURL` | 视频文件基础地址 | `/`（同 apiBaseURL） |
| `timeout` | 请求超时时间（ms） | `30000` |
| `backendPort` | 后端端口（参考） | `8010` |

- **开发环境**：保持 `/`，Vite 自动代理 `/api`、`/videos` 到后端
- **生产环境**：修改为后端实际地址，修改后**无需重新构建**，刷新页面即生效

---

## GPU 兼容性说明

### RTX 50 系列（Blackwell，Compute Capability 12.0）

需要 PyTorch 2.7+ + CUDA 12.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### RTX 30/40 系列（Ampere / Ada Lovelace）

PyTorch 2.0+ + CUDA 12.4 即可：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 无 GPU / CPU only

LLM 和视频模型均使用备用方案，无需下载模型。
在 `backend/config.py` 中确保：
```python
LLM_CONFIG["device"] = "cpu"
LLM_CONFIG["allow_cpu_inference"] = False
VIDEO_CONFIG["device"] = "cpu"
```

---

## 常见问题

### 1. `CUDA error: no kernel image is available for execution on the device`

**原因**：PyTorch 版本不支持你的 GPU 架构。

**解决**：升级 PyTorch。
```bash
# 查看 GPU 型号
nvidia-smi --query-gpu=name,compute_cap --format=csv

# RTX 50 系列 → CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# pip install torch torchvision torchaudio -f https://mirrors.tuna.tsinghua.edu.cn/pytorch/whl/cu128/

# RTX 30/40 系列 → CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. `cannot import name 'Dinov2WithRegistersConfig' from 'transformers'`

**原因**：transformers 版本过低，diffusers 需要 >= 4.48.0。

**解决**：
```bash
pip install "transformers>=4.48.0"
```

### 3. `tiktoken` is required / SentencePiece 错误

**原因**：缺少 tiktoken 或 protobuf 版本不兼容。

**解决**：
```bash
pip install tiktoken "protobuf>=3.20.0,<5.0.0"
```

或设置环境变量绕过：
```bash
# Windows PowerShell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"

# Linux / Bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### 4. `Error no file named model_index.json found`

**原因**：模型目录存在但为空（上次下载中断）。

**解决**：
```bash
rm -rf backend/models/cogvideox-2b
# 重启后端，会自动重新下载
```

### 5. 显存不足（Out of Memory）

项目已实现显存管理：视频生成前自动卸载 LLM，释放 ~12GB 显存。

如果仍然不足，在 `backend/config.py` 中调整：
```python
VIDEO_CONFIG["use_fp16"] = True          # FP16 半精度（必须）
VIDEO_CONFIG["num_frames"] = 25           # 减少帧数（默认 49）
VIDEO_CONFIG["enable_vae_slicing"] = True # VAE 切片
```

### 6. OpenH264 警告

```
Failed to load OpenH264 library
```

非致命警告。视频编码器自动降级为 avc1，功能正常。

---

## 配置说明

所有配置在 `backend/config.py` 中，可通过环境变量覆盖：

### 环境变量

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `MODELS_DIR` | 模型根目录 | `backend/models/` |
| `LLM_MODEL_PATH` | LLM 模型路径 | `$MODELS_DIR/chatglm3-6b` |
| `VIDEO_MODEL_PATH` | 视频模型路径 | `$MODELS_DIR/cogvideox-2b` |
| `VIDEO_OUTPUT_DIR` | 视频输出目录 | `backend/videos/` |
| `HF_MIRROR` | HuggingFace 镜像 | `https://hf-mirror.com` |
| `JWT_SECRET_KEY` | JWT 密钥 | （内置默认值） |

### config.py 关键配置项

| 配置 | 说明 | 默认值 |
|------|------|--------|
| `LLM_CONFIG["device"]` | LLM 设备 | `"cpu"` |
| `LLM_CONFIG["allow_cpu_inference"]` | CPU 推理（很慢） | `False` |
| `VIDEO_CONFIG["device"]` | 视频模型设备 | `"cuda"` |
| `VIDEO_CONFIG["use_fp16"]` | FP16 半精度 | `True` |
| `VIDEO_CONFIG["num_frames"]` | 生成帧数 | `49` |
| `VIDEO_CONFIG["auto_download"]` | 自动下载模型 | `True` |
| `DATABASE_URL` | 数据库路径 | `sqlite:///./video_platform.db` |

### 自定义模型路径示例

```bash
# 模型统一放在 /data/models 下
export MODELS_DIR=/data/models

# 或分别指定每个模型的路径
export LLM_MODEL_PATH=/data/llm/chatglm3-6b
export VIDEO_MODEL_PATH=/data/video/cogvideox-2b

# 启动后端（自动读取环境变量）
cd backend
uvicorn main:app --host 0.0.0.0 --port 8010

# 下载模型时也使用相同环境变量
MODELS_DIR=/data/models python scripts/download_model.py


# 视频生成失败: CUDA out of memory. Tried to allocate 1.32 GiB. GPU 0 has a total capacity of 31.36 GiB of which 509.06 MiB is free. Including non-PyTorch memory, this process has 30.85 GiB memory in use. Of the allocated memory 27.60 GiB is allocated by PyTorch, and 2.67 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
# 2026-05-07 07:38:04,038 - services.video_service - ERROR - 场景视频生成失败: CUDA out of memory. Tried to allocate 1.32 GiB. GPU 0 has a total capacity of 31.36 GiB of which 509.06 MiB is free. Including non-PyTorch memory, this process has 30.85 GiB memory in use. Of the allocated memory 27.60 GiB is allocated by PyTorch, and 2.67 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for 
# 在启动脚本前设置环境变量 export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，并启用 CPU Offload 功能。
```

---

## 验证安装

```bash
# 健康检查
curl http://localhost:8010/health

# 模型状态
curl http://localhost:8010/api/model/status

# 运行测试
pytest tests/ -v

# 前端类型检查
cd frontend && npx vue-tsc --noEmit
```
