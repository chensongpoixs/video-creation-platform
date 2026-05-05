# LLM 模型集成快速指南

## 🚀 快速开始

### 方法一：自动下载（推荐）

系统会在首次启动时自动下载模型：

```bash
cd backend
python main.py
```

模型会自动下载到 `backend/models/chatglm3-6b/`

### 方法二：手动下载

#### 使用下载脚本

```bash
# 从 Hugging Face 下载
python scripts/download_model.py --source hf --model THUDM/chatglm3-6b

# 从 ModelScope 下载（国内推荐）
python scripts/download_model.py --source ms --model ZhipuAI/chatglm3-6b
```

#### 使用 Git LFS

```bash
# 安装 Git LFS
git lfs install

# 克隆模型仓库
cd backend/models
git clone https://huggingface.co/THUDM/chatglm3-6b

# 或使用镜像（国内）
git clone https://hf-mirror.com/THUDM/chatglm3-6b
```

---

## 📋 系统要求

### 硬件要求
- **GPU**: NVIDIA RTX 3090/4090 (推荐)
- **显存**: 至少 12GB
- **内存**: 至少 16GB RAM

### 软件要求
```bash
pip install transformers torch accelerate
```

---

## ⚙️ 配置选项

编辑 `backend/config.py`:

```python
LLM_CONFIG = {
    "model_path": "./models/chatglm3-6b",  # 模型路径
    "device": "cuda",                       # cuda 或 cpu
    "use_fp16": True,                       # 半精度（省显存）
    "use_int8": False,                      # INT8量化（更省显存）
    "auto_download": True,                  # 自动下载
}
```

### 显存优化选项

| 配置 | 显存需求 | 速度 | 质量 |
|------|----------|------|------|
| FP32 | ~24GB | 慢 | 最好 |
| FP16 | ~12GB | 中 | 好 |
| INT8 | ~6GB | 快 | 较好 |

---

## 🧪 测试

```bash
# 运行测试
cd backend
pytest tests/test_llm_service.py -v

# 测试模型加载
python -c "from services.model_loader import llm_loader; llm_loader.load_model()"
```

---

## 🔍 验证

启动服务后访问：

```bash
# 检查模型状态
curl http://localhost:8000/api/model/status

# 测试脚本生成
curl -X POST http://localhost:8000/api/tasks/ \
  -H "Content-Type: application/json" \
  -d '{"prompt": "制作一段关于森林探险的短视频"}'
```

---

## ❓ 常见问题

### Q: 显存不足怎么办？
A: 设置 `use_int8=True` 或使用更小的模型

### Q: 下载速度慢？
A: 使用 ModelScope 或设置镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 模型加载失败？
A: 检查：
1. 模型文件是否完整
2. CUDA 是否可用
3. 依赖是否安装完整

---

## 📚 更多信息

详细文档请查看：
- [LLM 集成指南](docs/LLM_INTEGRATION_GUIDE.md)
- [API 文档](docs/API.md)
- [开发指南](docs/DEVELOPMENT.md)
