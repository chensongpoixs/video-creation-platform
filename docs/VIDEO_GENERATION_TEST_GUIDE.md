# 视频生成流程测试指南

## 📋 目录
1. [测试目标](#测试目标)
2. [测试环境准备](#测试环境准备)
3. [测试流程](#测试流程)
4. [测试用例](#测试用例)
5. [问题排查](#问题排查)

---

## 1. 测试目标

### 1.1 主要目标
- ✅ 验证 LLM 模型能否正常生成脚本
- ✅ 验证视频模型能否正常生成视频帧
- ✅ 验证视频处理器能否正常编码视频
- ✅ 验证端到端流程是否完整
- ✅ 测量性能指标（时间、显存）

### 1.2 测试范围
- 单元测试：各模块独立功能
- 集成测试：模块间协作
- 端到端测试：完整流程
- 性能测试：速度和资源占用

---

## 2. 测试环境准备

### 2.1 硬件要求检查
```bash
# 检查 GPU
nvidia-smi

# 检查 CUDA
nvcc --version

# 检查显存
nvidia-smi --query-gpu=memory.total --format=csv
```

**最低要求**:
- GPU: NVIDIA RTX 3090 或更高
- 显存: 24GB
- 内存: 32GB RAM

### 2.2 软件环境检查
```bash
# 检查 Python 版本
python --version  # 需要 3.10+

# 检查依赖
pip list | grep -E "torch|transformers|diffusers"
```

### 2.3 模型文件检查
```bash
# 检查 LLM 模型
ls -lh backend/models/chatglm3-6b/

# 检查视频模型
ls -lh backend/models/svd-xt/
```

---

## 3. 测试流程

### 3.1 阶段 1: 环境验证测试

#### 测试 1.1: 验证脚本
```bash
cd backend
python ../scripts/verify_setup.py
```

**预期输出**:
```
✅ Python 版本
✅ CUDA
✅ 依赖包
✅ 目录结构
⚠️ 模型文件（可选）
```

#### 测试 1.2: 导入测试
```python
# test_imports.py
import torch
from transformers import AutoModel, AutoTokenizer
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import cv2

print("✅ 所有依赖导入成功")
```

---

### 3.2 阶段 2: 单元测试

#### 测试 2.1: LLM 服务测试
```bash
pytest tests/test_llm_service.py -v
```

**测试内容**:
- 备用脚本生成
- JSON 解析
- 脚本验证

#### 测试 2.2: 视频处理器测试
```bash
pytest tests/test_video_service.py -v
```

**测试内容**:
- 占位符图像生成
- 帧转视频
- 帧插值

---

### 3.3 阶段 3: 模型加载测试

#### 测试 3.1: LLM 模型加载
```python
# test_llm_loading.py
from services.model_loader import llm_loader
import time

start = time.time()
success = llm_loader.load_model()
duration = time.time() - start

print(f"LLM 加载: {'成功' if success else '失败'}")
print(f"耗时: {duration:.2f} 秒")

if success:
    # 测试生成
    response = llm_loader.generate("你好")
    print(f"生成测试: {response[:50]}")
```

**预期结果**:
- 加载时间: 30-60 秒
- 显存占用: 10-12GB
- 生成成功

#### 测试 3.2: 视频模型加载
```python
# test_video_loading.py
from services.model_loader import video_loader
import time

start = time.time()
success = video_loader.load_model()
duration = time.time() - start

print(f"视频模型加载: {'成功' if success else '失败'}")
print(f"耗时: {duration:.2f} 秒")
```

**预期结果**:
- 加载时间: 60-120 秒
- 显存占用: 8-10GB
- 加载成功

---

### 3.4 阶段 4: 功能测试

#### 测试 4.1: 脚本生成测试
```python
# test_script_generation.py
from services.llm_service import generate_script

prompts = [
    "制作一段关于森林探险的短视频",
    "制作一段关于海滩日落的视频",
    "制作一段关于城市夜景的视频"
]

for prompt in prompts:
    print(f"\n测试提示词: {prompt}")
    script = generate_script(prompt)
    
    print(f"场景数: {len(script['scenes'])}")
    print(f"总时长: {script['total_duration']} 秒")
    
    for scene in script['scenes'][:2]:
        print(f"  场景 {scene['scene_number']}: {scene['description'][:40]}")
```

**预期结果**:
- 生成时间: 5-10 秒
- 场景数: 3-8 个
- 描述详细

#### 测试 4.2: 单场景视频生成测试
```python
# test_single_scene.py
from services.video_service_new import generate_scene_video
import time

scene = {
    "scene_number": 1,
    "description": "阳光明媚的森林，鸟儿在树枝上歌唱",
    "duration": 3
}

print("开始生成单场景视频...")
start = time.time()

try:
    video_path = generate_scene_video(scene, "test_single")
    duration = time.time() - start
    
    print(f"✅ 视频生成成功")
    print(f"路径: {video_path}")
    print(f"耗时: {duration:.2f} 秒")
    
    # 检查文件
    import os
    if os.path.exists(video_path):
        size = os.path.getsize(video_path) / 1024 / 1024
        print(f"文件大小: {size:.2f} MB")
    
except Exception as e:
    print(f"❌ 生成失败: {str(e)}")
```

**预期结果**:
- 生成时间: 2-5 分钟
- 文件大小: 1-5 MB
- 视频可播放

---

### 3.5 阶段 5: 端到端测试

#### 测试 5.1: 完整流程测试
```python
# test_end_to_end.py
from services.llm_service import generate_script
from services.video_service_new import generate_video_from_script
import time

print("=" * 60)
print("端到端视频生成测试")
print("=" * 60)

# 步骤 1: 生成脚本
prompt = "制作一段关于森林探险的短视频，包含河流和小动物"
print(f"\n步骤 1: 生成脚本")
print(f"提示词: {prompt}")

start_script = time.time()
script = generate_script(prompt)
script_time = time.time() - start_script

print(f"✅ 脚本生成完成")
print(f"场景数: {len(script['scenes'])}")
print(f"耗时: {script_time:.2f} 秒")

# 步骤 2: 生成视频
print(f"\n步骤 2: 生成视频")
start_video = time.time()

try:
    video_path = generate_video_from_script(script, "test_e2e")
    video_time = time.time() - start_video
    
    print(f"✅ 视频生成完成")
    print(f"路径: {video_path}")
    print(f"耗时: {video_time:.2f} 秒")
    
    # 总结
    total_time = script_time + video_time
    print(f"\n" + "=" * 60)
    print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"  - 脚本生成: {script_time:.2f} 秒")
    print(f"  - 视频生成: {video_time:.2f} 秒")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ 视频生成失败: {str(e)}")
    import traceback
    traceback.print_exc()
```

**预期结果**:
- 脚本生成: 5-10 秒
- 视频生成: 10-20 分钟
- 总时长: 10-20 分钟
- 视频可播放

---

### 3.6 阶段 6: 性能测试

#### 测试 6.1: 显存监控
```python
# test_memory_usage.py
import torch
from services.model_loader import llm_loader, video_loader

def print_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"显存: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")

print("初始状态:")
print_memory()

print("\n加载 LLM 模型...")
llm_loader.load_model()
print_memory()

print("\n加载视频模型...")
video_loader.load_model()
print_memory()

print("\n生成测试...")
# 执行生成
print_memory()

print("\n卸载模型...")
llm_loader.unload_model()
video_loader.unload_model()
torch.cuda.empty_cache()
print_memory()
```

#### 测试 6.2: 速度基准测试
```python
# test_benchmark.py
import time
from services.llm_service import generate_script
from services.video_service_new import generate_scene_video

# 测试脚本生成速度
prompts = [
    "森林探险",
    "海滩日落",
    "城市夜景"
]

script_times = []
for prompt in prompts:
    start = time.time()
    generate_script(prompt)
    duration = time.time() - start
    script_times.append(duration)

print(f"脚本生成平均时间: {sum(script_times)/len(script_times):.2f} 秒")

# 测试视频生成速度
scene = {
    "scene_number": 1,
    "description": "测试场景",
    "duration": 2
}

start = time.time()
generate_scene_video(scene, "benchmark")
video_time = time.time() - start

print(f"单场景视频生成时间: {video_time:.2f} 秒")
```

---

## 4. 测试用例

### 4.1 基础测试用例

| 测试ID | 测试内容 | 输入 | 预期输出 | 优先级 |
|--------|----------|------|----------|--------|
| T001 | 环境验证 | - | 所有检查通过 | P0 |
| T002 | LLM加载 | - | 加载成功 | P0 |
| T003 | 视频模型加载 | - | 加载成功 | P0 |
| T004 | 脚本生成 | 简单提示词 | 3-8个场景 | P0 |
| T005 | 单场景视频 | 单个场景 | MP4文件 | P0 |
| T006 | 完整流程 | 完整提示词 | 完整视频 | P0 |

### 4.2 边界测试用例

| 测试ID | 测试内容 | 输入 | 预期行为 |
|--------|----------|------|----------|
| T101 | 空提示词 | "" | 使用备用方案 |
| T102 | 超长提示词 | 500字 | 正常处理或截断 |
| T103 | 特殊字符 | "!@#$%" | 正常处理 |
| T104 | 多语言 | 英文/中文 | 正常处理 |

### 4.3 异常测试用例

| 测试ID | 测试内容 | 场景 | 预期行为 |
|--------|----------|------|----------|
| T201 | 显存不足 | 模拟显存不足 | 优雅降级 |
| T202 | 模型未加载 | 直接调用生成 | 使用备用方案 |
| T203 | 网络中断 | 下载模型时 | 错误提示 |

---

## 5. 问题排查

### 5.1 常见问题

#### 问题 1: LLM 模型加载失败
**症状**: 
```
❌ LLM 模型加载失败: No module named 'transformers'
```

**解决方案**:
```bash
pip install transformers torch accelerate
```

#### 问题 2: 视频模型加载失败
**症状**:
```
❌ 视频模型加载失败: No module named 'diffusers'
```

**解决方案**:
```bash
pip install diffusers xformers
```

#### 问题 3: 显存不足
**症状**:
```
CUDA out of memory
```

**解决方案**:
1. 使用 FP16: `VIDEO_CONFIG["use_fp16"] = True`
2. 减少帧数: `VIDEO_CONFIG["num_frames"] = 15`
3. 降低分辨率: `VIDEO_CONFIG["height"] = 512`

#### 问题 4: 生成速度慢
**症状**: 单场景生成超过 10 分钟

**解决方案**:
1. 减少推理步数: `VIDEO_CONFIG["num_inference_steps"] = 15`
2. 安装 xformers: `pip install xformers`
3. 检查 GPU 利用率: `nvidia-smi`

#### 问题 5: 视频无法播放
**症状**: 生成的 MP4 文件无法播放

**解决方案**:
1. 检查文件大小: `ls -lh video.mp4`
2. 使用 VLC 播放器
3. 检查编码器: 确保安装了 ffmpeg

---

## 6. 测试报告模板

### 6.1 测试执行记录

```markdown
# 视频生成测试报告

## 测试信息
- 测试日期: YYYY-MM-DD
- 测试人员: XXX
- 测试环境: GPU型号, 显存大小

## 测试结果

### 环境验证
- [ ] Python 版本检查
- [ ] CUDA 检查
- [ ] 依赖包检查
- [ ] 模型文件检查

### 单元测试
- [ ] LLM 服务测试
- [ ] 视频处理器测试
- [ ] 模型加载测试

### 功能测试
- [ ] 脚本生成测试
- [ ] 单场景视频生成
- [ ] 完整流程测试

### 性能测试
- LLM 加载时间: XX 秒
- 视频模型加载时间: XX 秒
- 脚本生成时间: XX 秒
- 单场景生成时间: XX 分钟
- 完整视频生成时间: XX 分钟
- 显存占用: XX GB

## 问题记录
1. 问题描述
   - 解决方案
   - 状态: 已解决/待解决

## 总结
- 通过测试: X/Y
- 主要问题: XXX
- 建议: XXX
```

---

## 7. 自动化测试脚本

### 7.1 完整测试脚本
```bash
#!/bin/bash
# run_all_tests.sh

echo "开始完整测试流程..."

# 1. 环境验证
echo "1. 环境验证..."
python scripts/verify_setup.py

# 2. 单元测试
echo "2. 单元测试..."
pytest tests/ -v

# 3. 模型加载测试
echo "3. 模型加载测试..."
python tests/test_model_loading.py

# 4. 功能测试
echo "4. 功能测试..."
python tests/test_script_generation.py
python tests/test_single_scene.py

# 5. 端到端测试
echo "5. 端到端测试..."
python tests/test_end_to_end.py

# 6. 性能测试
echo "6. 性能测试..."
python tests/test_benchmark.py

echo "测试完成！"
```

---

## 8. 持续集成

### 8.1 GitHub Actions 配置
```yaml
name: Video Generation Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r backend/requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v
```

---

## 9. 参考资源

- [PyTorch 文档](https://pytorch.org/docs/)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [Diffusers 文档](https://huggingface.co/docs/diffusers)
- [OpenCV 文档](https://docs.opencv.org/)
