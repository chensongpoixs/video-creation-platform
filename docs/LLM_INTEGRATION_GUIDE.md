# LLM 模型集成指南

## 📋 目录
1. [技术选型分析](#技术选型分析)
2. [模型下载方案](#模型下载方案)
3. [集成实现流程](#集成实现流程)
4. [优化策略](#优化策略)
5. [测试验证](#测试验证)

---

## 1. 技术选型分析

### 1.1 可选 LLM 模型对比

| 模型 | 参数量 | 显存需求 | 中文支持 | 推荐度 |
|------|--------|----------|----------|--------|
| LLaMA-2-7B | 7B | ~14GB | ⭐⭐ | ⭐⭐⭐ |
| Mistral-7B | 7B | ~14GB | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| ChatGLM3-6B | 6B | ~12GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Qwen-7B | 7B | ~14GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Baichuan2-7B | 7B | ~14GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 1.2 推荐方案

**首选: ChatGLM3-6B**
- ✅ 中文支持优秀
- ✅ 显存需求适中（12GB）
- ✅ 开源免费
- ✅ 社区活跃
- ✅ 文档完善

**备选: Qwen-7B**
- ✅ 阿里开源，质量高
- ✅ 中文能力强
- ✅ 支持长文本

---

## 2. 模型下载方案

### 2.1 方案一：Hugging Face Hub（推荐）

#### 优点
- 官方渠道，安全可靠
- 自动管理缓存
- 支持断点续传

#### 下载步骤

```bash
# 1. 安装依赖
pip install transformers torch accelerate

# 2. 设置环境变量（可选，加速下载）
export HF_ENDPOINT=https://hf-mirror.com

# 3. Python 代码下载
from transformers import AutoModel, AutoTokenizer

model_name = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# 模型会自动下载到 ~/.cache/huggingface/hub/
```

#### 手动下载（国内镜像）

```bash
# 使用 huggingface-cli
pip install huggingface_hub

# 下载模型
huggingface-cli download THUDM/chatglm3-6b \
  --local-dir ./models/chatglm3-6b \
  --local-dir-use-symlinks False
```

### 2.2 方案二：ModelScope（国内推荐）

```bash
# 1. 安装 ModelScope
pip install modelscope

# 2. 下载模型
from modelscope import snapshot_download

model_dir = snapshot_download(
    'ZhipuAI/chatglm3-6b',
    cache_dir='./models'
)
```

### 2.3 方案三：Git LFS（完整下载）

```bash
# 1. 安装 Git LFS
git lfs install

# 2. 克隆仓库
git clone https://huggingface.co/THUDM/chatglm3-6b ./models/chatglm3-6b

# 或使用镜像
git clone https://hf-mirror.com/THUDM/chatglm3-6b ./models/chatglm3-6b
```

---

## 3. 集成实现流程

### 3.1 项目结构调整

```
backend/
├── models/
│   ├── chatglm3-6b/          # 模型文件目录
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer_config.json
│   │   └── ...
│   └── __init__.py
├── services/
│   ├── llm_service.py        # 修改此文件
│   └── model_loader.py       # 修改此文件
└── config.py                 # 添加模型配置
```

### 3.2 配置文件修改

**backend/config.py**

```python
# LLM 模型配置
LLM_CONFIG = {
    "model_name": "THUDM/chatglm3-6b",
    "model_path": "./models/chatglm3-6b",  # 本地路径
    "device": "cuda",  # cuda 或 cpu
    "use_fp16": True,  # 使用半精度
    "max_length": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}
```

### 3.3 模型加载器实现

**backend/services/model_loader.py**

```python
import torch
from transformers import AutoModel, AutoTokenizer
from utils.logger import setup_logger
from config import LLM_CONFIG

logger = setup_logger(__name__)

class LLMModelLoader:
    """LLM 模型加载器"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = LLM_CONFIG["device"]
        
    def load_model(self):
        """加载 ChatGLM3 模型"""
        try:
            logger.info(f"开始加载模型: {LLM_CONFIG['model_name']}")
            
            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_CONFIG["model_path"],
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModel.from_pretrained(
                LLM_CONFIG["model_path"],
                trust_remote_code=True
            )
            
            # 移动到 GPU 并使用半精度
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                if LLM_CONFIG["use_fp16"]:
                    self.model = self.model.half()
            
            # 设置为评估模式
            self.model.eval()
            
            logger.info("模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        # 合并配置
        gen_kwargs = {
            "max_length": LLM_CONFIG["max_length"],
            "temperature": LLM_CONFIG["temperature"],
            "top_p": LLM_CONFIG["top_p"],
            "do_sample": LLM_CONFIG["do_sample"],
        }
        gen_kwargs.update(kwargs)
        
        # 生成
        response, history = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            **gen_kwargs
        )
        
        return response
    
    def unload_model(self):
        """卸载模型"""
        if self.model:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("模型已卸载")

# 全局实例
llm_loader = LLMModelLoader()
```

### 3.4 LLM 服务实现

**backend/services/llm_service.py**

```python
import json
import re
from typing import Dict, List
from services.model_loader import llm_loader
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 提示词模板
SCRIPT_GENERATION_PROMPT = """你是一个专业的视频脚本创作助手。请根据用户的创作指令，生成详细的视频脚本和分镜。

用户指令：{user_prompt}

请按照以下 JSON 格式输出：
{{
  "title": "视频标题",
  "total_duration": 总时长（秒）,
  "scenes": [
    {{
      "scene_number": 1,
      "description": "场景描述（详细的视觉描述）",
      "duration": 5,
      "camera": "镜头类型（wide shot/close up/medium shot）",
      "action": "动作描述"
    }}
  ]
}}

要求：
1. 每个场景描述要具体、生动
2. 场景之间要有连贯性
3. 每个场景时长 3-8 秒
4. 至少生成 3 个场景
5. 只输出 JSON，不要其他内容
"""

def generate_script(prompt: str) -> Dict:
    """
    使用 LLM 生成视频脚本
    
    Args:
        prompt: 用户输入的创作指令
        
    Returns:
        包含分镜信息的字典
    """
    try:
        logger.info(f"开始生成脚本，用户输入: {prompt}")
        
        # 构造完整提示词
        full_prompt = SCRIPT_GENERATION_PROMPT.format(user_prompt=prompt)
        
        # 调用 LLM 生成
        response = llm_loader.generate(
            full_prompt,
            max_length=2048,
            temperature=0.7
        )
        
        logger.info(f"LLM 原始输出: {response}")
        
        # 解析 JSON
        script = parse_llm_response(response)
        
        # 验证和修正
        script = validate_and_fix_script(script)
        
        logger.info(f"脚本生成成功，共 {len(script['scenes'])} 个场景")
        return script
        
    except Exception as e:
        logger.error(f"脚本生成失败: {str(e)}")
        # 返回备用脚本
        return generate_fallback_script(prompt)

def parse_llm_response(response: str) -> Dict:
    """解析 LLM 输出的 JSON"""
    try:
        # 提取 JSON 部分
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            raise ValueError("未找到 JSON 格式")
    except Exception as e:
        logger.warning(f"JSON 解析失败: {str(e)}")
        raise

def validate_and_fix_script(script: Dict) -> Dict:
    """验证和修正脚本格式"""
    # 确保必要字段存在
    if "title" not in script:
        script["title"] = "自动生成视频"
    
    if "scenes" not in script or not script["scenes"]:
        raise ValueError("脚本中没有场景")
    
    # 修正场景编号
    for i, scene in enumerate(script["scenes"]):
        scene["scene_number"] = i + 1
        
        # 确保必要字段
        if "description" not in scene:
            scene["description"] = f"场景 {i+1}"
        if "duration" not in scene:
            scene["duration"] = 5
        if "camera" not in scene:
            scene["camera"] = "wide shot"
    
    # 计算总时长
    script["total_duration"] = sum(s["duration"] for s in script["scenes"])
    
    return script

def generate_fallback_script(prompt: str) -> Dict:
    """生成备用脚本（当 LLM 失败时）"""
    logger.warning("使用备用脚本生成")
    
    # 简单分句
    sentences = re.split(r'[，。,.]', prompt)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    scenes = []
    for i, sentence in enumerate(sentences[:5]):  # 最多5个场景
        scenes.append({
            "scene_number": i + 1,
            "description": sentence,
            "duration": 5,
            "camera": "wide shot",
            "action": "展示场景"
        })
    
    return {
        "title": "自动生成视频",
        "total_duration": len(scenes) * 5,
        "scenes": scenes
    }

def optimize_prompt_for_video(scene_description: str) -> str:
    """
    优化场景描述为视频生成模型的 Prompt
    
    Args:
        scene_description: 场景描述
        
    Returns:
        优化后的 Prompt
    """
    # 添加视觉质量关键词
    quality_keywords = "high quality, cinematic, detailed, 4k, professional"
    
    # 构造完整 Prompt
    prompt = f"{scene_description}, {quality_keywords}"
    
    return prompt
```

### 3.5 启动时加载模型

**backend/main.py**

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from services.model_loader import llm_loader
from utils.logger import setup_logger

logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    logger.info("应用启动，开始加载模型...")
    success = llm_loader.load_model()
    if not success:
        logger.error("模型加载失败，将使用备用方案")
    
    yield
    
    # 关闭时卸载模型
    logger.info("应用关闭，卸载模型...")
    llm_loader.unload_model()

app = FastAPI(
    title="多模态视频创作平台",
    lifespan=lifespan
)
```

---

## 4. 优化策略

### 4.1 显存优化

#### 方案一：半精度（FP16）
```python
model = model.half()  # 显存减半
```

#### 方案二：INT8 量化
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModel.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)
```

#### 方案三：INT4 量化（最激进）
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
```

### 4.2 推理加速

#### 使用 Flash Attention
```bash
pip install flash-attn
```

```python
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
```

#### 批量推理
```python
# 批量生成多个场景的描述
responses = model.batch_generate(prompts)
```

### 4.3 缓存策略

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def generate_script_cached(prompt: str) -> Dict:
    """带缓存的脚本生成"""
    return generate_script(prompt)
```

---

## 5. 测试验证

### 5.1 单元测试

**tests/test_llm_service.py**

```python
import pytest
from services.llm_service import generate_script, parse_llm_response

def test_generate_script():
    """测试脚本生成"""
    prompt = "制作一段关于森林探险的短视频"
    script = generate_script(prompt)
    
    assert "scenes" in script
    assert len(script["scenes"]) > 0
    assert script["scenes"][0]["scene_number"] == 1

def test_parse_llm_response():
    """测试 JSON 解析"""
    response = '''
    {
      "title": "测试视频",
      "scenes": [
        {"scene_number": 1, "description": "场景1", "duration": 5}
      ]
    }
    '''
    script = parse_llm_response(response)
    assert script["title"] == "测试视频"
```

### 5.2 集成测试

```python
def test_full_pipeline():
    """测试完整流程"""
    from services.model_loader import llm_loader
    
    # 加载模型
    llm_loader.load_model()
    
    # 生成脚本
    script = generate_script("制作一段关于海滩日落的视频")
    
    # 验证结果
    assert len(script["scenes"]) >= 3
    
    # 卸载模型
    llm_loader.unload_model()
```

### 5.3 性能测试

```python
import time

def test_generation_speed():
    """测试生成速度"""
    start = time.time()
    script = generate_script("测试提示词")
    end = time.time()
    
    duration = end - start
    print(f"生成耗时: {duration:.2f} 秒")
    assert duration < 10  # 应在10秒内完成
```

---

## 6. 常见问题

### Q1: 显存不足怎么办？
A: 使用 INT8 或 INT4 量化，或者使用更小的模型（如 ChatGLM3-6B）

### Q2: 下载速度慢怎么办？
A: 使用国内镜像（ModelScope 或 HF-Mirror）

### Q3: 模型输出格式不对怎么办？
A: 使用更详细的提示词，或者添加后处理逻辑

### Q4: 如何切换其他模型？
A: 修改 config.py 中的 model_name 和 model_path

---

## 7. 实施时间表

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 1 | 下载模型 | 1-2 小时 |
| 2 | 修改配置文件 | 10 分钟 |
| 3 | 实现模型加载器 | 30 分钟 |
| 4 | 实现 LLM 服务 | 1 小时 |
| 5 | 集成到主程序 | 20 分钟 |
| 6 | 测试验证 | 30 分钟 |
| **总计** | | **3-4 小时** |

---

## 8. 参考资源

- ChatGLM3 官方文档: https://github.com/THUDM/ChatGLM3
- Transformers 文档: https://huggingface.co/docs/transformers
- ModelScope 文档: https://modelscope.cn/docs
- 量化技术: https://huggingface.co/docs/transformers/quantization
