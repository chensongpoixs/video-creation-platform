# 测试实施指南

本文档提供详细的测试实施步骤和说明。

---

## 📋 测试前准备

### 1. 环境检查

#### 1.1 硬件要求
```bash
# 检查 GPU
nvidia-smi

# 确认显存至少 24GB
nvidia-smi --query-gpu=memory.total --format=csv
```

#### 1.2 软件依赖
```bash
# 检查 Python 版本 (需要 3.10+)
python --version

# 检查 CUDA
nvcc --version

# 检查关键依赖
pip list | findstr "torch transformers diffusers"
```

### 2. 模型准备

#### 2.1 LLM 模型 (ChatGLM3-6B)
```bash
# 方式 1: 手动下载
# 访问: https://huggingface.co/THUDM/chatglm3-6b
# 下载到: backend/models/chatglm3-6b/

# 方式 2: 自动下载
# 在 backend/config.py 中设置:
# LLM_CONFIG["auto_download"] = True
```

#### 2.2 视频模型 (Stable Video Diffusion)
```bash
# 方式 1: 手动下载
# 访问: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
# 下载到: backend/models/svd-xt/

# 方式 2: 自动下载
# 在 backend/config.py 中设置:
# VIDEO_CONFIG["auto_download"] = True
```

---

## 🚀 测试执行步骤

### 阶段 1: 快速验证测试

**目的**: 快速验证基础功能是否正常

**执行命令**:
```bash
scripts\run_quick_test.bat
```

**预期时间**: 5-10 分钟

**包含测试**:
1. 环境验证
2. 脚本生成测试
3. 单场景视频生成测试

**通过标准**:
- 所有测试显示 ✅
- 生成的视频文件可以播放

---

### 阶段 2: 完整测试流程

**目的**: 全面测试所有功能

**执行命令**:
```bash
scripts\run_all_tests.bat
```

**预期时间**: 30-60 分钟

**包含测试**:
1. 环境验证
2. 单元测试
3. 模型加载测试
4. 脚本生成测试
5. 单场景视频生成测试
6. 端到端测试
7. 性能测试

---

### 阶段 3: 单独测试执行

如果需要单独执行某个测试：

#### 3.1 环境验证
```bash
python scripts\verify_setup.py
```

#### 3.2 单元测试
```bash
pytest tests\test_llm_service.py -v
pytest tests\test_video_service.py -v
```

#### 3.3 模型加载测试
```bash
python tests\test_model_loading.py
```

**注意事项**:
- 首次加载模型会比较慢（30-120秒）
- 需要足够的显存（至少 20GB）
- 如果显存不足，会自动使用备用方案

#### 3.4 脚本生成测试
```bash
python tests\test_script_generation.py
```

**预期结果**:
- 每个提示词生成 3-8 个场景
- 生成时间 5-10 秒
- 场景描述详细且合理

#### 3.5 单场景视频生成测试
```bash
python tests\test_single_scene.py
```

**预期结果**:
- 生成时间: 2-5 分钟（使用真实模型）或 5-10 秒（备用方案）
- 文件大小: 1-5 MB
- 视频可播放

#### 3.6 端到端测试
```bash
python tests\test_end_to_end.py
```

**预期结果**:
- 脚本生成: 5-10 秒
- 视频生成: 10-20 分钟（取决于场景数）
- 完整视频可播放

#### 3.7 性能测试
```bash
python tests\test_benchmark.py
```

**测试内容**:
- 脚本生成速度
- 视频生成速度
- 显存使用情况

---

## 📊 测试结果记录

### 1. 使用测试报告模板

复制 `docs/TEST_EXECUTION_REPORT.md` 并填写测试结果：

```bash
copy docs\TEST_EXECUTION_REPORT.md test_report_[日期].md
```

### 2. 记录关键信息

#### 2.1 环境信息
```bash
# GPU 信息
nvidia-smi > gpu_info.txt

# Python 环境
python --version > python_info.txt
pip list > pip_list.txt

# CUDA 信息
nvcc --version > cuda_info.txt
```

#### 2.2 测试输出
- 保存每个测试的控制台输出
- 截图关键错误信息
- 记录生成的视频文件路径

#### 2.3 性能数据
- 模型加载时间
- 脚本生成时间
- 视频生成时间
- 显存使用峰值

---

## 🔧 常见问题处理

### 问题 1: 模型加载失败

**症状**:
```
❌ LLM 模型加载失败: No such file or directory
```

**解决方案**:
1. 检查模型文件是否存在
2. 检查 `config.py` 中的路径配置
3. 尝试设置 `auto_download=True`

### 问题 2: CUDA 内存不足

**症状**:
```
CUDA out of memory
```

**解决方案**:
1. 在 `config.py` 中启用 FP16:
   ```python
   LLM_CONFIG["use_fp16"] = True
   VIDEO_CONFIG["use_fp16"] = True
   ```

2. 减少视频帧数:
   ```python
   VIDEO_CONFIG["num_frames"] = 15
   ```

3. 降低分辨率:
   ```python
   VIDEO_CONFIG["height"] = 512
   VIDEO_CONFIG["width"] = 768
   ```

### 问题 3: 依赖包缺失

**症状**:
```
ModuleNotFoundError: No module named 'xxx'
```

**解决方案**:
```bash
pip install -r backend\requirements.txt
```

### 问题 4: 视频无法播放

**症状**: 生成的 MP4 文件无法播放

**解决方案**:
1. 检查文件大小是否为 0
2. 使用 VLC 播放器尝试播放
3. 检查 ffmpeg 是否正确安装
4. 查看生成日志中的错误信息

### 问题 5: 测试速度慢

**症状**: 视频生成时间超过预期

**解决方案**:
1. 安装 xformers 加速:
   ```bash
   pip install xformers
   ```

2. 减少推理步数:
   ```python
   VIDEO_CONFIG["num_inference_steps"] = 15
   ```

3. 检查 GPU 利用率:
   ```bash
   nvidia-smi
   ```

---

## 📈 性能优化建议

### 1. 显存优化
- 使用 FP16 半精度
- 启用 attention slicing
- 使用 xFormers 优化

### 2. 速度优化
- 减少推理步数（15-25 步）
- 使用较小的分辨率
- 启用模型编译（PyTorch 2.0+）

### 3. 质量优化
- 增加推理步数（30-50 步）
- 使用更高的分辨率
- 调整 guidance_scale 参数

---

## 🎯 测试通过标准

### 最低标准（必须通过）
- [ ] 环境验证全部通过
- [ ] 脚本生成功能正常
- [ ] 能生成视频文件（可以是备用方案）
- [ ] 视频文件可以播放

### 推荐标准
- [ ] 真实模型加载成功
- [ ] 脚本生成时间 < 10 秒
- [ ] 单场景生成时间 < 5 分钟
- [ ] 端到端测试通过
- [ ] 显存使用 < 20GB

### 优秀标准
- [ ] 所有测试通过
- [ ] 性能达到预期
- [ ] 生成质量良好
- [ ] 无严重错误或警告

---

## 📝 测试后续工作

### 1. 整理测试报告
- 填写 `TEST_EXECUTION_REPORT.md`
- 记录所有问题和解决方案
- 总结测试发现

### 2. 问题跟踪
- 创建问题列表
- 标记优先级
- 分配解决责任人

### 3. 优化改进
- 根据测试结果优化配置
- 修复发现的 bug
- 改进性能瓶颈

### 4. 文档更新
- 更新 README
- 补充常见问题
- 完善使用说明

---

## 🔗 相关文档

- [视频生成测试指南](VIDEO_GENERATION_TEST_GUIDE.md)
- [LLM 集成指南](LLM_INTEGRATION_GUIDE.md)
- [视频模型集成指南](VIDEO_MODEL_INTEGRATION_GUIDE.md)
- [开发指南](DEVELOPMENT.md)
- [部署指南](DEPLOYMENT.md)

---

## 📞 获取帮助

如果遇到问题：

1. 查看测试指南中的常见问题部分
2. 检查日志文件 `backend/logs/app.log`
3. 查看相关文档
4. 提交 Issue 并附上详细信息

---

**祝测试顺利！** 🎉
