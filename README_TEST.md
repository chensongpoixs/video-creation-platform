# 测试执行说明

## 🎯 快速开始

### 1. 快速测试（推荐首次运行）

```bash
# Windows 用户
scripts\run_quick_test.bat

# 或手动执行
python scripts\verify_setup.py
python tests\test_script_generation.py
python tests\test_single_scene.py
```

**预计时间**: 5-10 分钟  
**测试内容**: 环境验证 + 脚本生成 + 单场景视频

---

### 2. 完整测试流程

```bash
# Windows 用户
scripts\run_all_tests.bat

# 或手动执行所有测试
python scripts\verify_setup.py
pytest tests\test_llm_service.py tests\test_video_service.py -v
python tests\test_model_loading.py
python tests\test_script_generation.py
python tests\test_single_scene.py
python tests\test_end_to_end.py
python tests\test_benchmark.py
```

**预计时间**: 30-60 分钟  
**测试内容**: 所有功能和性能测试

---

## 📋 测试前准备

### 硬件要求
- GPU: NVIDIA RTX 3090 或更高（24GB 显存）
- 内存: 32GB RAM
- 存储: 50GB 可用空间

### 软件要求
```bash
# Python 3.10+
python --version

# 安装依赖
cd backend
pip install -r requirements.txt
```

### 模型准备（可选）

#### 方式 1: 自动下载
在 `backend/config.py` 中设置：
```python
LLM_CONFIG["auto_download"] = True
VIDEO_CONFIG["auto_download"] = True
```

#### 方式 2: 手动下载
- ChatGLM3-6B: https://huggingface.co/THUDM/chatglm3-6b
  - 下载到: `backend/models/chatglm3-6b/`
- Stable Video Diffusion: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
  - 下载到: `backend/models/svd-xt/`

---

## 🧪 测试说明

### 测试 1: 环境验证
```bash
python scripts\verify_setup.py
```
检查 Python、CUDA、依赖包、目录结构

### 测试 2: 脚本生成
```bash
python tests\test_script_generation.py
```
测试 LLM 脚本生成功能，3个测试用例

### 测试 3: 单场景视频
```bash
python tests\test_single_scene.py
```
测试单个场景的视频生成

### 测试 4: 端到端测试
```bash
python tests\test_end_to_end.py
```
测试完整流程：提示词 → 脚本 → 视频

### 测试 5: 性能测试
```bash
python tests\test_benchmark.py
```
测试性能指标和显存使用

---

## 📊 预期结果

### 脚本生成
- 耗时: 5-10 秒
- 场景数: 3-8 个
- 每个场景有详细描述

### 视频生成
- 单场景: 2-5 分钟（真实模型）或 5-10 秒（备用方案）
- 完整视频: 10-20 分钟
- 输出格式: MP4
- 文件大小: 1-5 MB/场景

### 显存使用
- LLM 模型: 10-12GB
- 视频模型: 8-10GB
- 总计: 18-22GB

---

## 🔧 常见问题

### 问题 1: CUDA 内存不足
```python
# 在 backend/config.py 中设置
LLM_CONFIG["use_fp16"] = True
VIDEO_CONFIG["use_fp16"] = True
VIDEO_CONFIG["num_frames"] = 15
```

### 问题 2: 模型加载失败
```python
# 设置自动下载
LLM_CONFIG["auto_download"] = True
VIDEO_CONFIG["auto_download"] = True
```

### 问题 3: 生成速度慢
```bash
# 安装 xformers 加速
pip install xformers
```

```python
# 减少推理步数
VIDEO_CONFIG["num_inference_steps"] = 15
```

---

## 📝 测试报告

测试完成后，请填写测试报告：

1. 复制模板
```bash
copy docs\TEST_EXECUTION_REPORT.md test_report_[日期].md
```

2. 填写测试结果
- 测试信息
- 每个测试的结果
- 性能数据
- 遇到的问题

3. 总结
- 通过率
- 主要发现
- 改进建议

---

## 📚 详细文档

- [视频生成测试指南](docs/VIDEO_GENERATION_TEST_GUIDE.md) - 详细测试指南
- [测试实施指南](docs/TEST_IMPLEMENTATION_GUIDE.md) - 实施步骤
- [测试报告模板](docs/TEST_EXECUTION_REPORT.md) - 报告模板
- [测试实现总结](VIDEO_GENERATION_TEST_IMPLEMENTATION.md) - 实现说明

---

## 🎯 测试通过标准

### 最低标准
- ✅ 环境验证通过
- ✅ 脚本生成正常
- ✅ 能生成视频文件
- ✅ 视频可播放

### 推荐标准
- ✅ 真实模型加载成功
- ✅ 性能达到预期
- ✅ 所有测试通过

---

## 💡 提示

1. **首次运行**: 建议先执行快速测试
2. **模型下载**: 首次运行会自动下载模型（需要时间）
3. **备用方案**: 如果模型不可用，会自动使用备用方案
4. **性能优化**: 根据硬件情况调整配置参数

---

**祝测试顺利！** 🚀
