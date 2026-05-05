# 项目使用模型说明

## 模型清单

项目使用 **2 个模型**，分别负责剧本生成和视频生成：

| 模型 | 用途 | 参数量 | 显存 | 加载框架 |
|------|------|--------|------|----------|
| ChatGLM3-6B | LLM 剧本生成 | 6B | ~12GB | transformers |
| CogVideoX-2b | 文生视频（中文）| 2B | ~8GB (FP16) | diffusers |

## 模型详情

### 1. ChatGLM3-6B（LLM 剧本生成）

```
仓库:     THUDM/chatglm3-6b
本地路径: backend/models/chatglm3-6b/
加载器:   transformers.AutoModel + AutoTokenizer
配置:     backend/config.py → LLM_CONFIG
```

**作用**：接收用户输入的自然语言创作指令，生成分镜头剧本（场景描述、时长、运镜、灯光等结构化数据）。

**加载代码**（`backend/services/model_loader.py:LLMModelLoader.load_model`）：
```python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
```

**备选方案**：当 GPU 不可用或模型未下载时，自动退回到 `generate_fallback_script()`（基于句子拆分的简单分镜），不阻塞主流程。

### 2. CogVideoX-2b（文生视频，支持中文）

```
仓库:     THUDM/CogVideoX-2b
本地路径: backend/models/cogvideox-2b/
加载器:   diffusers.CogVideoXPipeline
配置:     backend/config.py → VIDEO_CONFIG
```

**作用**：根据分镜头脚本的中文场景描述，逐场景生成视频帧，再拼接、后处理为最终 MP4 视频。

**为什么选 CogVideoX-2b**：
- ✅ **原生支持中文** — 清华大学 THUDM 出品，与 ChatGLM 同团队
- ✅ **文生视频** — 直接文字→视频，无需中间图像
- ✅ **显存友好** — 2B 参数，FP16 仅需 6-8GB 显存（SVD 需要 16GB）
- ✅ **diffusers 集成** — 标准 Pipeline 接口

**加载代码**（`backend/services/model_loader.py:VideoModelLoader.load_model`）：
```python
from diffusers import CogVideoXPipeline
model = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
# 生成：model(prompt="孩子们在公园里玩耍", num_frames=49, ...)
```

**对比**：
| 特性 | CogVideoX-2b (新) | SVD-XT (旧) |
|------|-------------------|-------------|
| 输入方式 | 文字 → 视频 | 图片 → 视频 |
| 中文支持 | ✅ 原生 | ❌ 不支持 |
| 显存需求 | ~6-8GB (FP16) | ~16GB |
| 参数量 | 2B | ~7B |
| 推理速度 | ~30s/49帧 | ~20s/25帧 |

**备选方案**：当 GPU 不可用或模型未加载时，自动退回到 OpenCV 生成渐变色 + 文字叠加的演示视频。

## 下载模型

### 快速下载（推荐）

```bash
# 下载全部两个模型（自动使用 hf-mirror.com 国内镜像）
python scripts/download_model.py

# 只下载 LLM 模型
python scripts/download_model.py --model llm

# 只下载视频模型
python scripts/download_model.py --model video
```

### 切换下载源

```bash
# 从 ModelScope 下载（国内更快，自动过滤非模型文件）
python scripts/download_model.py --source ms

# 自定义 HuggingFace 镜像
HF_MIRROR="https://hf-mirror.com" python scripts/download_model.py
```

### 下载说明

- **不会下载整个仓库**：脚本只下载模型权重（.bin/.safetensors）、配置文件（config.json）和分词器文件（tokenizer.*），跳过 README、示例脚本、测试文件等非模型内容。
- HuggingFace 的 `from_pretrained()` 本身即只下载必需文件，无需额外过滤。
- ModelScope 的 `snapshot_download()` 通过 `ignore_file_pattern` 排除非模型文件。

## 本地存储结构

下载完成后，模型文件的目录结构：

```
backend/models/
├── chatglm3-6b/             # ChatGLM3-6B 模型
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── pytorch_model.bin    # 或 model.safetensors
│   └── ...
└── cogvideox-2b/             # CogVideoX-2b 文生视频模型
    ├── model_index.json
    ├── transformer/
    ├── vae/
    ├── scheduler/
    └── ...
```

## 镜像配置

国内用户默认通过 `hf-mirror.com` 加速下载，无需额外配置。

镜像优先级：
1. `HF_MIRROR` 环境变量（最高优先级）
2. 默认 `https://hf-mirror.com`

```bash
# 临时切换镜像
HF_MIRROR="https://hf-mirror.com" python scripts/download_model.py

# 永久设置（写入 ~/.bashrc）
export HF_MIRROR="https://hf-mirror.com"
```

## 相关文档

- [LLM 模型集成指南](LLM_INTEGRATION_GUIDE.md) — 技术选型对比、集成实现细节
- [视频模型集成指南](VIDEO_MODEL_INTEGRATION_GUIDE.md) — 视频生成方案对比、优化策略
- [显存优化指南](MEMORY_OPTIMIZATION_GUIDE.md) — FP16、注意力切片等优化配置
