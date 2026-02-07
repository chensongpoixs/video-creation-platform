# 视频生成模型集成实现计划

## 📋 项目概览

**任务**: 集成视频生成模型（Stable Diffusion Video）  
**目标**: 实现从文本到视频的完整生成流程  
**预计工作量**: 4-5 小时  
**难度**: 中等

---

## 🎯 实现目标

### 核心功能
1. ✅ 模型下载和加载
2. ✅ 视频帧生成
3. ✅ 视频编码和保存
4. ✅ 显存优化
5. ✅ 备用方案

### 集成目标
1. ✅ 与 LLM 服务协同
2. ✅ 与任务队列集成
3. ✅ 与 API 接口集成
4. ✅ 完整的错误处理

---

## 📊 实现步骤

### 步骤 1: 配置文件更新 (10 分钟)

**文件**: `backend/config.py`

**修改内容**:
```python
# 添加视频模型配置
VIDEO_CONFIG = {
    "model_name": "stabilityai/stable-video-diffusion-img2vid-xt",
    "model_path": "./models/svd-xt",
    "device": "cuda",
    "use_fp16": True,
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "height": 576,
    "width": 1024,
    "num_frames": 25,
    "fps": 6,
    "auto_download": True,
}

# 添加视频处理配置
VIDEO_PROCESSING_CONFIG = {
    "output_format": "mp4",
    "codec": "libx264",
    "bitrate": "5000k",
    "enable_interpolation": False,
}
```

**检查清单**:
- [ ] 添加 VIDEO_CONFIG
- [ ] 添加 VIDEO_PROCESSING_CONFIG
- [ ] 验证配置参数

---

### 步骤 2: 视频处理器实现 (30 分钟)

**文件**: `backend/services/video_processor.py` (新增)

**功能**:
- 帧列表转视频
- 占位符图像生成
- 帧插值
- 视频编码

**关键代码**:
```python
class VideoProcessor:
    @staticmethod
    def frames_to_video(frames, output_path, fps=6):
        """将帧转换为视频"""
        # 实现视频编码逻辑
        pass
    
    @staticmethod
    def generate_placeholder_image(width=1024, height=576):
        """生成占位符图像"""
        pass
    
    @staticmethod
    def interpolate_frames(frames, factor=2):
        """帧插值"""
        pass
```

**检查清单**:
- [ ] 实现 frames_to_video
- [ ] 实现 generate_placeholder_image
- [ ] 实现 interpolate_frames
- [ ] 添加错误处理
- [ ] 添加日志记录

---

### 步骤 3: 模型加载器更新 (30 分钟)

**文件**: `backend/services/model_loader.py` (更新)

**新增类**: `VideoModelLoader`

**功能**:
- 加载 Stable Diffusion Video 模型
- 显存优化
- 视频生成
- 模型卸载

**关键代码**:
```python
class VideoModelLoader:
    def load_model(self):
        """加载视频模型"""
        # 实现模型加载逻辑
        pass
    
    def generate_video(self, prompt, image=None, **kwargs):
        """生成视频"""
        # 实现视频生成逻辑
        pass
    
    def unload_model(self):
        """卸载模型"""
        pass
```

**检查清单**:
- [ ] 实现 load_model
- [ ] 实现 generate_video
- [ ] 实现 unload_model
- [ ] 添加 FP16 支持
- [ ] 添加内存优化
- [ ] 添加错误处理

---

### 步骤 4: 视频服务更新 (1 小时)

**文件**: `backend/services/video_service.py` (更新)

**修改函数**:
- `generate_video_from_script` - 更新为真实实现
- `generate_scene_video` - 实现真实视频生成
- `stitch_videos` - 保持不变

**新增函数**:
- `generate_scene_video_fallback` - 备用方案

**关键代码**:
```python
def generate_scene_video(scene, task_id):
    """生成场景视频"""
    # 1. 检查模型是否加载
    # 2. 优化提示词
    # 3. 生成占位符图像
    # 4. 调用模型生成视频
    # 5. 转换为视频文件
    pass

def generate_scene_video_fallback(scene, task_id):
    """备用方案"""
    # 生成纯色视频+文字
    pass
```

**检查清单**:
- [ ] 更新 generate_video_from_script
- [ ] 实现 generate_scene_video
- [ ] 实现 generate_scene_video_fallback
- [ ] 添加错误处理
- [ ] 添加日志记录

---

### 步骤 5: 主程序集成 (20 分钟)

**文件**: `backend/main.py` (更新)

**修改内容**:
- 更新 lifespan 函数
- 加载视频模型
- 卸载视频模型

**关键代码**:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 加载 LLM 模型
    # 加载视频模型
    yield
    # 卸载 LLM 模型
    # 卸载视频模型
```

**检查清单**:
- [ ] 更新 lifespan 函数
- [ ] 添加视频模型加载
- [ ] 添加视频模型卸载
- [ ] 添加错误处理

---

### 步骤 6: 依赖更新 (5 分钟)

**文件**: `backend/requirements.txt` (更新)

**新增依赖**:
```
diffusers==0.24.0
pillow==10.1.0
opencv-python==4.8.1.78
xformers==0.0.22  # 可选，用于加速
```

**检查清单**:
- [ ] 添加 diffusers
- [ ] 添加 xformers（可选）
- [ ] 验证版本兼容性

---

### 步骤 7: 测试代码 (30 分钟)

**文件**: `tests/test_video_service.py` (新增)

**测试用例**:
1. 视频处理器测试
2. 备用方案测试
3. 完整流程测试
4. 性能测试

**关键代码**:
```python
def test_video_processor():
    """测试视频处理器"""
    pass

def test_generate_scene_video_fallback():
    """测试备用方案"""
    pass

def test_full_video_generation():
    """测试完整流程"""
    pass
```

**检查清单**:
- [ ] 实现 test_video_processor
- [ ] 实现 test_generate_scene_video_fallback
- [ ] 实现 test_full_video_generation
- [ ] 运行测试验证

---

### 步骤 8: 文档编写 (30 分钟)

**文件**: 
- `docs/VIDEO_MODEL_INTEGRATION_GUIDE.md` (已完成)
- `README_VIDEO_SETUP.md` (新增)
- `VIDEO_MODEL_IMPLEMENTATION_SUMMARY.md` (新增)

**内容**:
- 快速开始指南
- 配置说明
- 故障排查
- 性能优化

**检查清单**:
- [ ] 编写快速开始指南
- [ ] 编写配置说明
- [ ] 编写故障排查
- [ ] 编写性能优化

---

## 🔧 技术细节

### 关键技术点

#### 1. 模型加载
```python
from diffusers import StableVideoDiffusionPipeline

pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    variant="fp16"
)
```

#### 2. 视频生成
```python
output = pipe(
    image=image,
    prompt=prompt,
    num_inference_steps=25,
    guidance_scale=7.5,
    height=576,
    width=1024,
    num_frames=25
)
frames = output.frames[0]
```

#### 3. 视频编码
```python
import cv2

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for frame in frames:
    out.write(frame)

out.release()
```

### 显存优化

#### 方案 1: FP16 半精度
```python
model = model.half()  # 显存减半
```

#### 方案 2: 内存高效注意力
```python
model.enable_attention_slicing()
model.enable_xformers_memory_efficient_attention()
```

#### 方案 3: 分块处理
```python
# 分块生成视频
for i in range(0, num_frames, chunk_size):
    frames = model.generate(num_frames=chunk_size)
```

---

## 📈 预期成果

### 功能完成
- ✅ 视频模型加载
- ✅ 视频帧生成
- ✅ 视频编码保存
- ✅ 显存优化
- ✅ 备用方案
- ✅ 错误处理

### 代码统计
- 新增代码: ~400 行
- 修改代码: ~200 行
- 测试代码: ~100 行
- 文档: ~3000 字

### 性能指标
- 模型加载: 30-60 秒
- 视频生成: 2-5 分钟（取决于帧数和推理步数）
- 显存占用: 16GB (FP32) / 8GB (FP16)

---

## ⚠️ 风险和应对

### 风险 1: 显存不足
**应对**:
- 使用 FP16 半精度
- 减少推理步数
- 使用更小的模型

### 风险 2: 生成速度慢
**应对**:
- 减少推理步数
- 启用内存优化
- 使用 TensorRT 加速

### 风险 3: 模型下载失败
**应对**:
- 使用国内镜像
- 手动下载模型
- 使用备用方案

---

## ✅ 完成检查清单

### 代码实现
- [ ] 配置文件更新
- [ ] 视频处理器实现
- [ ] 模型加载器更新
- [ ] 视频服务更新
- [ ] 主程序集成
- [ ] 依赖更新

### 测试验证
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 性能测试通过
- [ ] 手动测试通过

### 文档编写
- [ ] 集成指南完成
- [ ] 快速指南完成
- [ ] 实施总结完成
- [ ] 代码注释完整

### 最终检查
- [ ] 代码质量检查
- [ ] 文档完整性检查
- [ ] 功能完整性检查
- [ ] 性能基准测试

---

## 📅 时间安排

| 阶段 | 任务 | 预计时间 | 状态 |
|------|------|----------|------|
| 1 | 配置文件更新 | 10 分钟 | ⏳ |
| 2 | 视频处理器实现 | 30 分钟 | ⏳ |
| 3 | 模型加载器更新 | 30 分钟 | ⏳ |
| 4 | 视频服务更新 | 1 小时 | ⏳ |
| 5 | 主程序集成 | 20 分钟 | ⏳ |
| 6 | 依赖更新 | 5 分钟 | ⏳ |
| 7 | 测试代码 | 30 分钟 | ⏳ |
| 8 | 文档编写 | 30 分钟 | ⏳ |
| **总计** | | **4-5 小时** | |

---

## 🎯 成功标准

### 功能标准
- ✅ 系统能够加载视频生成模型
- ✅ 系统能够生成视频帧
- ✅ 系统能够编码保存视频
- ✅ 系统能够处理错误和异常

### 性能标准
- ✅ 模型加载时间 < 60 秒
- ✅ 视频生成时间 < 5 分钟
- ✅ 显存占用 < 16GB
- ✅ 系统稳定运行 > 1 小时

### 质量标准
- ✅ 代码质量: 优秀
- ✅ 文档完整性: 优秀
- ✅ 测试覆盖率: >80%
- ✅ 错误处理: 完善

---

## 📝 实施说明

### 开始前准备
1. 确保 CUDA 和 PyTorch 已安装
2. 确保有足够的显存（至少 16GB）
3. 确保网络连接正常（用于下载模型）

### 实施过程
1. 按照步骤顺序进行
2. 每个步骤完成后进行测试
3. 遇到问题及时调整

### 完成后验证
1. 运行所有测试用例
2. 进行手动功能测试
3. 检查文档完整性

---

**状态**: 📋 计划已制定，准备开始实施  
**下一步**: 开始步骤 1 - 配置文件更新
