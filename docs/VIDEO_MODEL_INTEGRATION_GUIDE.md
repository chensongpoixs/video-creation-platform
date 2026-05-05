# 视频生成模型集成指南

## 📋 目录
1. [技术选型分析](#技术选型分析)
2. [模型下载方案](#模型下载方案)
3. [集成实现流程](#集成实现流程)
4. [优化策略](#优化策略)
5. [测试验证](#测试验证)

---

## 1. 技术选型分析

### 1.1 可选视频生成模型对比

| 模型 | 参数量 | 显存需求 | 推理速度 | 质量 | 推荐度 |
|------|--------|----------|----------|------|--------|
| Stable Diffusion Video | 7B | ~16GB | 中 | 优秀 | ⭐⭐⭐⭐⭐ |
| ModelScope T2V | 3B | ~8GB | 快 | 良好 | ⭐⭐⭐⭐ |
| Damo Video | 5B | ~12GB | 中 | 优秀 | ⭐⭐⭐⭐ |
| Open Sora | 7B | ~16GB | 慢 | 优秀 | ⭐⭐⭐ |
| AnimateDiff | 1B | ~4GB | 快 | 一般 | ⭐⭐⭐ |

### 1.2 推荐方案

**首选: Stable Diffusion Video (SVD)**
- ✅ 质量最好
- ✅ 社区活跃
- ✅ 文档完善
- ✅ 易于集成
- ✅ 支持多种输入

**备选: ModelScope T2V**
- ✅ 显存需求低
- ✅ 推理速度快
- ✅ 中文支持好
- ✅ 国内部署友好

---

## 2. 模型下载方案

### 2.1 Stable Diffusion Video 下载

#### 方案一：Hugging Face（推荐）

```bash
# 1. 安装依赖
pip install diffusers transformers torch accelerate

# 2. Python 下载
from diffusers import StableVideoDiffusionPipeline
import torch

model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
)

# 模型会自动下载到 ~/.cache/huggingface/hub/
```

#### 方案二：ModelScope（国内推荐）

```bash
# 1. 安装 ModelScope
pip install modelscope

# 2. Python 下载
from modelscope import snapshot_download

model_dir = snapshot_download(
    'damo/text-to-video-synthesis',
    cache_dir='./models'
)
```

#### 方案三：Git LFS

```bash
# 1. 安装 Git LFS
git lfs install

# 2. 克隆仓库
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt \
  ./models/svd-xt
```

### 2.2 模型文件结构

```
backend/models/
├── svd-xt/                          # Stable Diffusion Video
│   ├── config.json
│   ├── diffusion_pytorch_model.bin
│   ├── model_index.json
│   ├── scheduler/
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── unet/
│   └── vae/
└── t2v-model/                       # ModelScope T2V（备选）
    ├── config.json
    └── pytorch_model.bin
```

---

## 3. 集成实现流程

### 3.1 项目结构调整

```
backend/
├── models/
│   ├── svd-xt/                      # 视频生成模型
│   └── chatglm3-6b/                 # LLM 模型
├── services/
│   ├── llm_service.py               # LLM 服务（已完成）
│   ├── video_service.py             # 视频生成服务（待更新）
│   ├── model_loader.py              # 模型加载器（待更新）
│   └── video_processor.py            # 视频处理器（新增）
└── config.py                        # 配置文件（待更新）
```

### 3.2 配置文件修改

**backend/config.py**

```python
# 视频生成模型配置
VIDEO_CONFIG = {
    "model_name": "stabilityai/stable-video-diffusion-img2vid-xt",
    "model_path": "./models/svd-xt",  # 本地路径
    "device": "cuda",
    "use_fp16": True,
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "height": 576,
    "width": 1024,
    "num_frames": 25,  # 生成帧数
    "fps": 6,  # 帧率
    "auto_download": True,
}

# 视频处理配置
VIDEO_PROCESSING_CONFIG = {
    "output_format": "mp4",
    "codec": "libx264",
    "bitrate": "5000k",
    "enable_interpolation": False,  # 帧插值
}
```

### 3.3 视频模型加载器实现

**backend/services/model_loader.py（更新）**

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from utils.logger import setup_logger
from config import VIDEO_CONFIG

logger = setup_logger(__name__)

class VideoModelLoader:
    """视频生成模型加载器"""
    
    def __init__(self):
        self.model = None
        self.device = VIDEO_CONFIG["device"] if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        logger.info(f"视频模型加载器初始化，使用设备: {self.device}")
    
    def load_model(self):
        """加载 Stable Diffusion Video 模型"""
        if self.is_loaded:
            logger.info("视频模型已加载，跳过")
            return True
        
        try:
            logger.info(f"开始加载视频模型: {VIDEO_CONFIG['model_name']}")
            
            model_path = VIDEO_CONFIG["model_path"]
            
            # 检查模型路径
            if not os.path.exists(model_path) and VIDEO_CONFIG.get("auto_download", False):
                logger.info(f"本地模型不存在，从 Hugging Face 下载...")
                model_path = VIDEO_CONFIG["model_name"]
            elif not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                return False
            
            # 加载模型
            logger.info("加载 Stable Diffusion Video 模型...")
            
            load_kwargs = {
                "torch_dtype": torch.float16 if VIDEO_CONFIG.get("use_fp16") else torch.float32,
            }
            
            if VIDEO_CONFIG.get("use_fp16"):
                load_kwargs["variant"] = "fp16"
            
            self.model = StableVideoDiffusionPipeline.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            # 移动到设备
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                
                # 启用内存优化
                self.model.enable_attention_slicing()
                
                # 启用 xFormers 加速（如果可用）
                try:
                    self.model.enable_xformers_memory_efficient_attention()
                    logger.info("启用 xFormers 加速")
                except:
                    logger.warning("xFormers 不可用，使用标准注意力")
            
            self.is_loaded = True
            logger.info("✅ 视频模型加载完成")
            
            # 显示显存使用
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"GPU 显存使用: {memory_allocated:.2f} GB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 视频模型加载失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def generate_video(self, prompt: str, image=None, **kwargs) -> list:
        """
        生成视频
        
        Args:
            prompt: 文本描述
            image: 输入图像（可选）
            **kwargs: 生成参数
            
        Returns:
            视频帧列表
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("视频模型未加载")
        
        try:
            # 合并配置
            gen_kwargs = {
                "num_inference_steps": VIDEO_CONFIG.get("num_inference_steps", 25),
                "guidance_scale": VIDEO_CONFIG.get("guidance_scale", 7.5),
                "height": VIDEO_CONFIG.get("height", 576),
                "width": VIDEO_CONFIG.get("width", 1024),
                "num_frames": VIDEO_CONFIG.get("num_frames", 25),
            }
            gen_kwargs.update(kwargs)
            
            logger.info(f"生成视频，参数: {gen_kwargs}")
            
            # 生成视频
            if image is not None:
                # 图像到视频
                output = self.model(
                    image=image,
                    prompt=prompt,
                    **gen_kwargs
                )
            else:
                # 文本到视频（需要先生成图像）
                logger.warning("需要输入图像，使用默认图像")
                from PIL import Image
                import numpy as np
                
                # 创建默认图像
                image = Image.new('RGB', (gen_kwargs["width"], gen_kwargs["height"]))
                output = self.model(
                    image=image,
                    prompt=prompt,
                    **gen_kwargs
                )
            
            frames = output.frames[0]  # 获取第一个视频的帧
            return frames
            
        except Exception as e:
            logger.error(f"视频生成失败: {str(e)}")
            raise
    
    def unload_model(self):
        """卸载模型释放显存"""
        if self.model:
            del self.model
            self.model = None
            self.is_loaded = False
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("视频模型已卸载")

# 全局实例
video_loader = VideoModelLoader()
```

### 3.4 视频处理器实现

**backend/services/video_processor.py（新增）**

```python
"""
视频处理模块 - 负责视频帧处理和编码
"""
import cv2
import numpy as np
from typing import List
from PIL import Image
import os
from utils.logger import setup_logger
from config import VIDEO_PROCESSING_CONFIG, VIDEO_OUTPUT_DIR

logger = setup_logger(__name__)

class VideoProcessor:
    """视频处理器"""
    
    @staticmethod
    def frames_to_video(frames: List, output_path: str, fps: int = 6) -> str:
        """
        将帧列表转换为视频文件
        
        Args:
            frames: PIL Image 列表
            output_path: 输出路径
            fps: 帧率
            
        Returns:
            视频文件路径
        """
        try:
            logger.info(f"开始转换视频，帧数: {len(frames)}, FPS: {fps}")
            
            # 转换为 numpy 数组
            frame_array = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                
                # 转换为 BGR（OpenCV 格式）
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                frame_array.append(frame)
            
            # 获取视频参数
            height, width = frame_array[0].shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 写入帧
            for frame in frame_array:
                out.write(frame)
            
            out.release()
            
            logger.info(f"✅ 视频转换完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"视频转换失败: {str(e)}")
            raise
    
    @staticmethod
    def generate_placeholder_image(width: int = 1024, height: int = 576) -> Image.Image:
        """生成占位符图像"""
        return Image.new('RGB', (width, height), color=(73, 109, 137))
    
    @staticmethod
    def interpolate_frames(frames: List, factor: int = 2) -> List:
        """
        帧插值（增加帧数）
        
        Args:
            frames: 原始帧列表
            factor: 插值因子
            
        Returns:
            插值后的帧列表
        """
        logger.info(f"执行帧插值，因子: {factor}")
        
        interpolated = []
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])
            
            # 简单的线性插值
            for j in range(1, factor):
                alpha = j / factor
                blended = Image.blend(frames[i], frames[i + 1], alpha)
                interpolated.append(blended)
        
        interpolated.append(frames[-1])
        return interpolated
```

### 3.5 视频生成服务更新

**backend/services/video_service.py（更新）**

```python
"""
视频生成服务模块 - 负责视频生成和后处理
"""
import os
from typing import List, Dict
from PIL import Image
from utils.logger import setup_logger
from config import VIDEO_OUTPUT_DIR, VIDEO_CONFIG
from services.video_processor import VideoProcessor

logger = setup_logger(__name__)

def generate_video_from_script(script: Dict, task_id: str) -> str:
    """
    根据脚本生成完整视频
    
    Args:
        script: 包含分镜信息的脚本字典
        task_id: 任务ID
        
    Returns:
        生成的视频文件路径
    """
    try:
        logger.info(f"开始生成视频，任务ID: {task_id}")
        
        video_paths = []
        
        # 为每个分镜生成视频片段
        for scene in script['scenes']:
            logger.info(f"生成场景 {scene['scene_number']}: {scene['description']}")
            video_path = generate_scene_video(scene, task_id)
            video_paths.append(video_path)
        
        # 拼接所有视频片段
        final_video_path = stitch_videos(video_paths, task_id)
        
        logger.info(f"✅ 视频生成完成: {final_video_path}")
        return final_video_path
        
    except Exception as e:
        logger.error(f"视频生成失败: {str(e)}")
        raise

def generate_scene_video(scene: Dict, task_id: str) -> str:
    """
    生成单个场景的视频片段
    
    Args:
        scene: 场景信息字典
        task_id: 任务ID
        
    Returns:
        视频片段文件路径
    """
    try:
        from services.model_loader import video_loader
        
        scene_id = scene['scene_number']
        output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{task_id}_scene_{scene_id}.mp4")
        
        # 检查模型是否加载
        if not video_loader.is_loaded:
            logger.warning("视频模型未加载，使用备用方案")
            return generate_scene_video_fallback(scene, task_id)
        
        # 优化提示词
        from services.llm_service import optimize_prompt_for_video
        prompt = optimize_prompt_for_video(scene['description'])
        
        logger.info(f"生成提示词: {prompt}")
        
        # 生成占位符图像
        image = VideoProcessor.generate_placeholder_image(
            width=VIDEO_CONFIG.get("width", 1024),
            height=VIDEO_CONFIG.get("height", 576)
        )
        
        # 生成视频帧
        frames = video_loader.generate_video(
            prompt=prompt,
            image=image,
            num_frames=VIDEO_CONFIG.get("num_frames", 25)
        )
        
        # 帧插值（可选）
        if VIDEO_CONFIG.get("enable_interpolation", False):
            frames = VideoProcessor.interpolate_frames(frames, factor=2)
        
        # 转换为视频文件
        fps = VIDEO_CONFIG.get("fps", 6)
        VideoProcessor.frames_to_video(frames, output_path, fps=fps)
        
        return output_path
        
    except Exception as e:
        logger.error(f"场景视频生成失败: {str(e)}")
        # 使用备用方案
        return generate_scene_video_fallback(scene, task_id)

def generate_scene_video_fallback(scene: Dict, task_id: str) -> str:
    """
    生成场景视频的备用方案（当模型不可用时）
    """
    logger.info(f"使用备用方案生成场景 {scene['scene_number']}")
    
    scene_id = scene['scene_number']
    duration = scene['duration']
    fps = VIDEO_CONFIG.get("fps", 6)
    width = VIDEO_CONFIG.get("width", 1024)
    height = VIDEO_CONFIG.get("height", 576)
    
    output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{task_id}_scene_{scene_id}.mp4")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 生成随机颜色的帧
    import numpy as np
    color = np.random.randint(0, 255, 3).tolist()
    total_frames = duration * fps
    
    for _ in range(total_frames):
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        
        # 添加场景描述文字
        import cv2
        text = f"Scene {scene_id}: {scene['description'][:40]}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
    
    out.release()
    return output_path

def stitch_videos(video_paths: List[str], task_id: str) -> str:
    """
    拼接多个视频片段
    
    Args:
        video_paths: 视频文件路径列表
        task_id: 任务ID
        
    Returns:
        拼接后的视频文件路径
    """
    if not video_paths:
        raise ValueError("没有视频片段可拼接")
    
    import cv2
    
    output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{task_id}_final.mp4")
    
    logger.info(f"开始拼接视频，片段数: {len(video_paths)}")
    
    # 读取第一个视频获取参数
    cap = cv2.VideoCapture(video_paths[0])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 逐个读取并写入视频片段
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
    
    out.release()
    
    logger.info(f"✅ 视频拼接完成: {output_path}")
    return output_path
```

### 3.6 主程序集成

**backend/main.py（更新）**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 60)
    logger.info("应用启动，开始初始化...")
    logger.info("=" * 60)
    
    try:
        from services.model_loader import llm_loader, video_loader
        
        # 加载 LLM 模型
        logger.info("开始加载 LLM 模型...")
        llm_success = llm_loader.load_model()
        if llm_success:
            logger.info("✅ LLM 模型加载成功")
        else:
            logger.warning("⚠️ LLM 模型加载失败")
        
        # 加载视频模型
        logger.info("开始加载视频生成模型...")
        video_success = video_loader.load_model()
        if video_success:
            logger.info("✅ 视频生成模型加载成功")
        else:
            logger.warning("⚠️ 视频生成模型加载失败，将使用备用方案")
            
    except Exception as e:
        logger.error(f"❌ 模型初始化失败: {str(e)}")
    
    logger.info("=" * 60)
    logger.info("应用启动完成")
    logger.info("=" * 60)
    
    yield
    
    # 关闭时卸载模型
    logger.info("应用关闭，卸载模型...")
    try:
        from services.model_loader import llm_loader, video_loader
        llm_loader.unload_model()
        video_loader.unload_model()
    except:
        pass
```

---

## 4. 优化策略

### 4.1 显存优化

#### 方案一：FP16 半精度
```python
VIDEO_CONFIG["use_fp16"] = True  # 显存减半
```

#### 方案二：内存高效注意力
```python
model.enable_attention_slicing()
model.enable_xformers_memory_efficient_attention()
```

#### 方案三：分块处理
```python
# 分块生成视频
chunk_size = 5  # 每次生成5帧
for i in range(0, num_frames, chunk_size):
    frames = model.generate(num_frames=chunk_size)
```

### 4.2 推理加速

#### 方案一：减少推理步数
```python
VIDEO_CONFIG["num_inference_steps"] = 15  # 从25降低到15
```

#### 方案二：使用 TensorRT
```bash
pip install tensorrt
```

#### 方案三：批量处理
```python
# 同时处理多个场景
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(generate_scene, scene) for scene in scenes]
```

### 4.3 缓存策略

```python
from functools import lru_cache

@lru_cache(maxsize=10)
def generate_video_cached(prompt: str, image_hash: str):
    """带缓存的视频生成"""
    return generate_video(prompt, image)
```

---

## 5. 测试验证

### 5.1 单元测试

**tests/test_video_service.py**

```python
import pytest
from services.video_service import generate_scene_video_fallback
from services.video_processor import VideoProcessor
from PIL import Image

def test_video_processor():
    """测试视频处理器"""
    # 生成占位符图像
    image = VideoProcessor.generate_placeholder_image()
    assert image.size == (1024, 576)

def test_generate_scene_video_fallback():
    """测试备用视频生成"""
    scene = {
        "scene_number": 1,
        "description": "测试场景",
        "duration": 2
    }
    video_path = generate_scene_video_fallback(scene, "test_task")
    assert os.path.exists(video_path)
```

### 5.2 集成测试

```python
def test_full_video_generation():
    """测试完整视频生成流程"""
    script = {
        "title": "测试视频",
        "scenes": [
            {"scene_number": 1, "description": "场景1", "duration": 2},
            {"scene_number": 2, "description": "场景2", "duration": 2}
        ]
    }
    
    video_path = generate_video_from_script(script, "test_task")
    assert os.path.exists(video_path)
```

### 5.3 性能测试

```python
import time

def test_generation_speed():
    """测试生成速度"""
    start = time.time()
    video_path = generate_scene_video(scene, "test_task")
    duration = time.time() - start
    
    print(f"生成耗时: {duration:.2f} 秒")
    assert duration < 300  # 应在5分钟内完成
```

---

## 6. 常见问题

### Q1: 显存不足怎么办？
A: 
1. 使用 FP16 半精度
2. 减少推理步数
3. 使用更小的模型（ModelScope T2V）
4. 启用内存高效注意力

### Q2: 生成速度太慢怎么办？
A:
1. 减少推理步数（从25到15）
2. 使用 TensorRT 加速
3. 启用批量处理
4. 使用更小的分辨率

### Q3: 生成质量不好怎么办？
A:
1. 优化提示词
2. 增加推理步数
3. 调整 guidance_scale
4. 使用更好的输入图像

### Q4: 如何切换其他模型？
A: 修改 config.py 中的 VIDEO_CONFIG

---

## 7. 实施时间表

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 1 | 下载模型 | 1-2 小时 |
| 2 | 修改配置文件 | 10 分钟 |
| 3 | 实现模型加载器 | 30 分钟 |
| 4 | 实现视频处理器 | 30 分钟 |
| 5 | 更新视频服务 | 1 小时 |
| 6 | 集成到主程序 | 20 分钟 |
| 7 | 测试验证 | 1 小时 |
| **总计** | | **4-5 小时** |

---

## 8. 参考资源

- Stable Diffusion Video: https://github.com/Stability-AI/generative-models
- Diffusers 文档: https://huggingface.co/docs/diffusers
- ModelScope: https://modelscope.cn/docs
- OpenCV 文档: https://docs.opencv.org/
