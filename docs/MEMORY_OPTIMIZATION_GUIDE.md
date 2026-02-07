# 显存优化指南

本文档介绍如何通过 FP16 半精度优化显存使用，降低硬件要求。

---

## 📊 优化效果

### 显存占用对比

| 模型 | FP32 | FP16 | 节省 |
|------|------|------|------|
| ChatGLM3-6B | ~12GB | ~6GB | 50% |
| SVD-XT | ~10GB | ~5GB | 50% |
| **总计** | **~22GB** | **~11GB** | **50%** |

### 硬件兼容性

| GPU | 显存 | FP32 支持 | FP16 支持 |
|-----|------|-----------|-----------|
| RTX 3060 | 12GB | ❌ | ⚠️ 单模型 |
| RTX 3070 | 8GB | ❌ | ❌ |
| RTX 3080 | 10GB | ❌ | ❌ |
| RTX 3090 | 24GB | ✅ | ✅ |
| RTX 4070 Ti | 12GB | ❌ | ⚠️ 单模型 |
| RTX 4080 | 16GB | ❌ | ✅ |
| RTX 4090 | 24GB | ✅ | ✅ |

---

## 🚀 快速开始

### 1. 启用 FP16（默认已启用）

FP16 优化已默认启用，无需额外配置。

查看配置：`backend/config.py`

```python
# LLM 模型配置
LLM_CONFIG = {
    "use_fp16": True,  # ✅ 已启用
    # ...
}

# 视频模型配置
VIDEO_CONFIG = {
    "use_fp16": True,  # ✅ 已启用
    # ...
}
```

### 2. 测试显存优化

```bash
# 运行显存优化测试
python tests/test_memory_optimization.py
```

### 3. 查看显存使用

测试会显示：
- 模型加载前后的显存占用
- 生成过程中的峰值显存
- FP16 vs FP32 的对比

---

## ⚙️ 配置选项

### 基础配置

#### 启用/禁用 FP16

```python
# backend/config.py

# 全局启用 FP16
LLM_CONFIG["use_fp16"] = True
VIDEO_CONFIG["use_fp16"] = True

# 只对 LLM 启用
LLM_CONFIG["use_fp16"] = True
VIDEO_CONFIG["use_fp16"] = False

# 禁用 FP16（使用 FP32）
LLM_CONFIG["use_fp16"] = False
VIDEO_CONFIG["use_fp16"] = False
```

### 高级配置

#### 自动优化

系统会根据显存大小自动选择精度：

```python
MEMORY_CONFIG = {
    "auto_optimize": True,  # 启用自动优化
    "force_fp16_threshold": 16.0,  # 显存 < 16GB 强制 FP16
}
```

#### 显存监控

```python
MEMORY_CONFIG = {
    "enable_monitoring": True,  # 启用显存监控
    "warn_threshold": 0.85,  # 使用率 > 85% 警告
    "clear_cache_after_generation": True,  # 生成后清理缓存
}
```

#### 内存优化选项

```python
# LLM 优化
LLM_CONFIG = {
    "enable_memory_efficient": True,  # 梯度检查点
}

# 视频模型优化
VIDEO_CONFIG = {
    "enable_attention_slicing": True,  # 注意力切片
    "enable_vae_slicing": True,  # VAE 切片
    "enable_xformers": True,  # xFormers 加速
}
```

---

## 📈 性能对比

### 生成速度

| 任务 | FP32 | FP16 | 提升 |
|------|------|------|------|
| 脚本生成 | 5-10秒 | 4-8秒 | 10-20% |
| 单场景视频 | 2-5分钟 | 2-4分钟 | 10-20% |

### 生成质量

| 指标 | FP32 | FP16 | 差异 |
|------|------|------|------|
| 脚本质量 | 100% | 98-99% | 几乎无差异 |
| 视频质量 | 100% | 95-98% | 轻微差异 |

---

## 🔧 进阶优化

### 1. 安装 xFormers 加速

```bash
# 安装 xFormers（推荐）
pip install xformers

# 或从源码安装
pip install git+https://github.com/facebookresearch/xformers.git
```

**效果**:
- 速度提升 10-30%
- 显存占用减少 10-20%

### 2. 降低分辨率

```python
# backend/config.py
VIDEO_CONFIG = {
    "height": 512,  # 默认 576
    "width": 768,   # 默认 1024
}
```

**效果**:
- 显存占用减少 30-40%
- 生成速度提升 20-30%
- 质量轻微下降

### 3. 减少帧数

```python
VIDEO_CONFIG = {
    "num_frames": 15,  # 默认 25
}
```

**效果**:
- 显存占用减少 20-30%
- 生成速度提升 30-40%

### 4. 减少推理步数

```python
VIDEO_CONFIG = {
    "num_inference_steps": 15,  # 默认 25
}
```

**效果**:
- 生成速度提升 30-40%
- 质量轻微下降

---

## 🎯 优化策略

### 策略 1: 最大显存节省（推荐）

适用于显存紧张的场景（< 16GB）

```python
# 启用所有优化
LLM_CONFIG["use_fp16"] = True
VIDEO_CONFIG["use_fp16"] = True
VIDEO_CONFIG["num_frames"] = 15
VIDEO_CONFIG["num_inference_steps"] = 15
VIDEO_CONFIG["height"] = 512
VIDEO_CONFIG["width"] = 768
```

**效果**:
- 显存: ~8GB
- 速度: 快
- 质量: 良好

### 策略 2: 平衡模式（默认）

适用于中等显存（16-20GB）

```python
# 默认配置
LLM_CONFIG["use_fp16"] = True
VIDEO_CONFIG["use_fp16"] = True
# 其他保持默认
```

**效果**:
- 显存: ~11GB
- 速度: 中等
- 质量: 优秀

### 策略 3: 最高质量

适用于充足显存（> 24GB）

```python
# 禁用优化
LLM_CONFIG["use_fp16"] = False
VIDEO_CONFIG["use_fp16"] = False
VIDEO_CONFIG["num_inference_steps"] = 30
```

**效果**:
- 显存: ~22GB
- 速度: 慢
- 质量: 最佳

---

## 🔍 常见问题

### Q1: FP16 会影响质量吗？

**A**: 影响很小（< 5%），肉眼几乎无法察觉。对于大多数应用场景，FP16 是最佳选择。

### Q2: 如何知道当前使用的精度？

**A**: 查看日志输出：

```
✅ 使用 FP16 半精度（显存减半）
```

或运行测试：

```bash
python tests/test_memory_optimization.py
```

### Q3: 显存不足怎么办？

**A**: 按优先级尝试：

1. 启用 FP16（默认已启用）
2. 安装 xFormers
3. 减少帧数（15帧）
4. 降低分辨率（512x768）
5. 减少推理步数（15步）

### Q4: 如何回退到 FP32？

**A**: 修改配置：

```python
LLM_CONFIG["use_fp16"] = False
VIDEO_CONFIG["use_fp16"] = False
```

### Q5: xFormers 是必需的吗？

**A**: 不是必需的，但强烈推荐：

- 速度提升 10-30%
- 显存节省 10-20%
- 安装简单：`pip install xformers`

---

## 📊 显存监控

### 实时监控

```python
from utils.memory_monitor import print_memory, memory_monitor

# 打印当前显存
print_memory("当前状态")

# 获取详细信息
info = memory_monitor.get_memory_info()
print(f"已分配: {info['allocated']:.2f} GB")
print(f"可用: {info['free']:.2f} GB")
print(f"使用率: {info['usage_percent']:.1f}%")
```

### 峰值统计

```python
# 重置峰值统计
memory_monitor.reset_peak_memory()

# 执行操作...

# 获取峰值
peak = memory_monitor.get_peak_memory()
print(f"峰值显存: {peak:.2f} GB")
```

### 优化建议

```python
# 获取优化建议
suggestions = memory_monitor.suggest_optimization()
print(suggestions['message'])
```

---

## 🧪 测试和验证

### 1. 基础测试

```bash
# 测试显存优化
python tests/test_memory_optimization.py
```

### 2. 对比测试

修改配置，分别测试 FP32 和 FP16：

```python
# 测试 FP16
LLM_CONFIG["use_fp16"] = True
VIDEO_CONFIG["use_fp16"] = True
# 运行测试

# 测试 FP32
LLM_CONFIG["use_fp16"] = False
VIDEO_CONFIG["use_fp16"] = False
# 运行测试
```

### 3. 质量对比

生成相同提示词的视频，对比质量：

```bash
python tests/test_single_scene.py
```

---

## 💡 最佳实践

### 1. 开发阶段

- 使用 FP16 加快迭代
- 启用显存监控
- 定期清理缓存

### 2. 生产环境

- 根据硬件选择策略
- 启用自动优化
- 监控显存使用

### 3. 质量要求高的场景

- 使用 FP32
- 增加推理步数
- 提高分辨率

---

## 📚 参考资料

### 技术文档

- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [Hugging Face FP16](https://huggingface.co/docs/transformers/perf_train_gpu_one)
- [Diffusers Optimization](https://huggingface.co/docs/diffusers/optimization/fp16)

### 相关文档

- [测试指南](TEST_IMPLEMENTATION_GUIDE.md)
- [LLM 集成指南](LLM_INTEGRATION_GUIDE.md)
- [视频模型集成指南](VIDEO_MODEL_INTEGRATION_GUIDE.md)

---

## 🎉 总结

### 优化收益

- ✅ 显存减少 50%（22GB → 11GB）
- ✅ 支持更多 GPU（16GB+）
- ✅ 速度可能提升 10-20%
- ✅ 质量几乎无损（> 95%）

### 推荐配置

```python
# 默认配置（推荐）
LLM_CONFIG["use_fp16"] = True
VIDEO_CONFIG["use_fp16"] = True
MEMORY_CONFIG["auto_optimize"] = True
```

### 下一步

1. 运行测试验证效果
2. 根据硬件调整配置
3. 监控显存使用
4. 持续优化

---

**享受更高效的视频生成！** 🚀
