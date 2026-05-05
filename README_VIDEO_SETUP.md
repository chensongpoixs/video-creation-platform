# 视频生成模型集成快速指南

## 🚀 快速开始

### 方法一：自动下载（推荐）

系统会在首次启动时自动下载模型：

```bash
cd backend
python main.py
```

模型会自动下载到 `backend/models/svd-xt/`

### 方法二：手动下载

#### 使用 Python 脚本

```python
from diffusers import StableVideoDiffusionPipeline
import torch

model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
)

# 模型会下载到 ~/.cache/huggingface/hub/
```

#### 使用 Git LFS

```bash
# 安装 Git LFS
git lfs install

# 克隆模型仓库
cd backend/models
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt svd-xt
```

---

## 📋 系统要求

### 硬件要求
- **GPU**: NVIDIA RTX 3090/4090 (推荐)
- **显存**: 至少 16GB (FP16 模式需要 8GB)
- **内存**: 至少 32GB RAM

### 软件要求
```bash
pip install diffusers torch accelerate xformers
```

---

## ⚙️ 配置选项

编辑 `backend/config.py`:

```python
VIDEO_CONFIG = {
    "model_path": "./models/svd-xt",
    "device": "cuda",
    "use_fp16": True,              # 半精度（省显存）
    "num_inference_steps": 25,     # 推理步数
    "guidance_scale": 7.5,         # 引导强度
    "height": 576,                 # 视频高度
    "width": 1024,                 # 视频宽度
    "num_frames": 25,              # 生成帧数
    "fps": 6,                      # 帧率
}
```

### 显存优化选项

| 配置 | 显存需求 | 速度 | 质量 |
|------|----------|------|------|
| FP32 | ~32GB | 慢 | 最好 |
| FP16 | ~16GB | 中 | 好 |
| 低分辨率 | ~8GB | 快 | 一般 |

---

## 🧪 测试

```bash
# 运行测试
cd backend
pytest tests/test_video_service.py -v

# 测试模型加载
python -c "from services.model_loader import video_loader; video_loader.load_model()"
```

---

## 🔍 验证

启动服务后访问：

```bash
# 检查模型状态
curl http://localhost:8000/api/model/status

# 测试视频生成
curl -X POST http://localhost:8000/api/tasks/ \
  -H "Content-Type: application/json" \
  -d '{"prompt": "制作一段关于森林探险的短视频"}'
```

---

## ❓ 常见问题

### Q: 显存不足怎么办？
A: 
1. 设置 `use_fp16=True`
2. 减少 `num_frames` (从25到15)
3. 降低分辨率 (从1024x576到512x512)

### Q: 生成速度太慢？
A:
1. 减少 `num_inference_steps` (从25到15)
2. 安装 xformers: `pip install xformers`
3. 使用更小的分辨率

### Q: 模型下载失败？
A: 使用镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 📚 更多信息

详细文档请查看：
- [视频模型集成指南](docs/VIDEO_MODEL_INTEGRATION_GUIDE.md)
- [实施计划](VIDEO_MODEL_IMPLEMENTATION_PLAN.md)
- [API 文档](docs/API.md)
