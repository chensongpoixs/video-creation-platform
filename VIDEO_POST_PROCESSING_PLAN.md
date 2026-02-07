# 视频后处理功能实现方案

## 📋 目录
1. [需求分析](#需求分析)
2. [功能设计](#功能设计)
3. [技术选型](#技术选型)
4. [实现方案](#实现方案)
5. [性能优化](#性能优化)

---

## 1. 需求分析

### 1.1 当前状态

#### 已实现功能
- ✅ 视频帧生成（Stable Diffusion Video）
- ✅ 帧转视频（OpenCV）
- ✅ 视频拼接（多场景合并）
- ✅ 简单帧插值

#### 缺失功能
- ❌ 视频滤镜和特效
- ❌ 转场效果
- ❌ 字幕添加
- ❌ 背景音乐
- ❌ 视频质量优化
- ❌ 视频格式转换
- ❌ 视频压缩

### 1.2 功能需求

#### 核心功能
1. **视频滤镜**: 亮度、对比度、饱和度、锐化、模糊
2. **转场效果**: 淡入淡出、交叉溶解、滑动、缩放
3. **字幕系统**: 文字叠加、字幕样式、时间轴控制
4. **音频处理**: 背景音乐、音效、音量控制
5. **质量优化**: 去噪、稳定、色彩校正
6. **格式转换**: MP4、AVI、MOV、WebM
7. **视频压缩**: H.264、H.265 编码

#### 扩展功能
- ⏳ 水印添加
- ⏳ 画中画效果
- ⏳ 慢动作/快进
- ⏳ 视频裁剪和旋转
- ⏳ 绿幕抠图

### 1.3 非功能需求

- **性能**: 处理速度 < 2x 视频时长
- **质量**: 无明显失真
- **兼容性**: 支持主流格式
- **易用性**: 简单的 API 接口

---

## 2. 功能设计

### 2.1 系统架构

```
视频生成流程:
1. 脚本生成 (LLM)
2. 场景视频生成 (Diffusion Model)
3. 视频拼接 (OpenCV)
4. 后处理 (新增) ⭐
   ├── 滤镜处理
   ├── 转场效果
   ├── 字幕添加
   ├── 音频处理
   └── 质量优化
5. 格式转换和压缩
6. 输出最终视频
```

### 2.2 模块设计

#### 2.2.1 视频滤镜模块
```python
class VideoFilter:
    - adjust_brightness()      # 调整亮度
    - adjust_contrast()        # 调整对比度
    - adjust_saturation()      # 调整饱和度
    - sharpen()                # 锐化
    - blur()                   # 模糊
    - apply_filter()           # 应用滤镜
```

#### 2.2.2 转场效果模块
```python
class TransitionEffect:
    - fade_in_out()            # 淡入淡出
    - cross_dissolve()         # 交叉溶解
    - slide()                  # 滑动
    - zoom()                   # 缩放
    - apply_transition()       # 应用转场
```

#### 2.2.3 字幕系统模块
```python
class SubtitleSystem:
    - add_subtitle()           # 添加字幕
    - set_style()              # 设置样式
    - render_subtitles()       # 渲染字幕
```

#### 2.2.4 音频处理模块
```python
class AudioProcessor:
    - add_background_music()   # 添加背景音乐
    - add_sound_effect()       # 添加音效
    - adjust_volume()          # 调整音量
    - mix_audio()              # 混音
```

#### 2.2.5 质量优化模块
```python
class VideoOptimizer:
    - denoise()                # 去噪
    - stabilize()              # 稳定
    - color_correction()       # 色彩校正
    - enhance_quality()        # 质量增强
```

#### 2.2.6 格式转换模块
```python
class VideoConverter:
    - convert_format()         # 格式转换
    - compress_video()         # 视频压缩
    - change_resolution()      # 改变分辨率
    - change_fps()             # 改变帧率
```

### 2.3 数据流设计

```
输入: 原始视频文件
  ↓
滤镜处理 (可选)
  ↓
转场效果 (场景间)
  ↓
字幕添加 (可选)
  ↓
音频处理 (可选)
  ↓
质量优化 (可选)
  ↓
格式转换和压缩
  ↓
输出: 最终视频文件
```

---

## 3. 技术选型

### 3.1 技术方案对比

| 技术 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **OpenCV** | 快速、轻量、Python 友好 | 功能有限 | 基础处理 |
| **FFmpeg** | 功能强大、格式支持全 | 命令行调用 | 格式转换、压缩 |
| **MoviePy** | 简单易用、Python 原生 | 性能一般 | 快速开发 |
| **Pillow** | 图像处理强大 | 仅图像 | 字幕渲染 |

### 3.2 选择方案

#### 主要技术栈
1. **OpenCV**: 视频读写、滤镜、转场
2. **FFmpeg**: 格式转换、压缩、音频处理
3. **Pillow**: 字幕渲染、文字叠加
4. **NumPy**: 数值计算、图像处理

#### 理由
- ✅ OpenCV 性能好，适合实时处理
- ✅ FFmpeg 功能全，格式支持广
- ✅ Pillow 文字渲染效果好
- ✅ 组合使用，发挥各自优势

### 3.3 依赖库

```python
opencv-python==4.8.1.78      # 已有
pillow==10.1.0               # 已有
numpy==1.24.3                # 已有
ffmpeg-python==0.2.0         # 新增
pydub==0.25.1                # 新增（音频处理）
```

---

## 4. 实现方案

### 4.1 目录结构

```
backend/
├── services/
│   ├── video_processor.py          # 基础视频处理（已有）
│   ├── video_service.py            # 视频生成服务（已有）
│   ├── video_filter.py             # 视频滤镜（新增）⭐
│   ├── video_transition.py         # 转场效果（新增）⭐
│   ├── subtitle_system.py          # 字幕系统（新增）⭐
│   ├── audio_processor.py          # 音频处理（新增）⭐
│   ├── video_optimizer.py          # 质量优化（新增）⭐
│   └── video_converter.py          # 格式转换（新增）⭐
├── config.py                       # 配置文件（更新）
└── utils/
    └── ffmpeg_utils.py             # FFmpeg 工具（新增）⭐
```

### 4.2 核心实现

#### 4.2.1 视频滤镜 (video_filter.py)

```python
class VideoFilter:
    """视频滤镜处理"""
    
    @staticmethod
    def adjust_brightness(frame: np.ndarray, factor: float) -> np.ndarray:
        """调整亮度"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def adjust_contrast(frame: np.ndarray, factor: float) -> np.ndarray:
        """调整对比度"""
        return cv2.convertScaleAbs(frame, alpha=factor, beta=0)
    
    @staticmethod
    def adjust_saturation(frame: np.ndarray, factor: float) -> np.ndarray:
        """调整饱和度"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def sharpen(frame: np.ndarray) -> np.ndarray:
        """锐化"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(frame, -1, kernel)
    
    @staticmethod
    def blur(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """模糊"""
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
```

#### 4.2.2 转场效果 (video_transition.py)

```python
class TransitionEffect:
    """转场效果处理"""
    
    @staticmethod
    def fade_transition(frames1: List, frames2: List, duration: int = 10) -> List:
        """淡入淡出转场"""
        transition_frames = []
        for i in range(duration):
            alpha = i / duration
            frame1 = np.array(frames1[-1])
            frame2 = np.array(frames2[0])
            blended = cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0)
            transition_frames.append(blended)
        return transition_frames
    
    @staticmethod
    def slide_transition(frames1: List, frames2: List, 
                        direction: str = 'left', duration: int = 10) -> List:
        """滑动转场"""
        # 实现滑动效果
        pass
```

#### 4.2.3 字幕系统 (subtitle_system.py)

```python
class SubtitleSystem:
    """字幕系统"""
    
    def __init__(self, font_path: str = None, font_size: int = 32):
        self.font_path = font_path
        self.font_size = font_size
    
    def add_subtitle(self, frame: np.ndarray, text: str, 
                    position: tuple = (50, 50)) -> np.ndarray:
        """添加字幕"""
        # 使用 Pillow 渲染文字
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 加载字体
        font = ImageFont.truetype(self.font_path, self.font_size) \
               if self.font_path else ImageFont.load_default()
        
        # 绘制文字
        draw.text(position, text, font=font, fill=(255, 255, 255))
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
```

#### 4.2.4 音频处理 (audio_processor.py)

```python
class AudioProcessor:
    """音频处理"""
    
    @staticmethod
    def add_background_music(video_path: str, audio_path: str, 
                           output_path: str, volume: float = 0.5):
        """添加背景音乐"""
        import ffmpeg
        
        video = ffmpeg.input(video_path)
        audio = ffmpeg.input(audio_path)
        
        # 调整音量
        audio = audio.filter('volume', volume)
        
        # 合并视频和音频
        output = ffmpeg.output(
            video, audio, output_path,
            vcodec='copy', acodec='aac',
            shortest=None
        )
        
        ffmpeg.run(output, overwrite_output=True)
```

#### 4.2.5 质量优化 (video_optimizer.py)

```python
class VideoOptimizer:
    """视频质量优化"""
    
    @staticmethod
    def denoise(frame: np.ndarray) -> np.ndarray:
        """去噪"""
        return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    
    @staticmethod
    def color_correction(frame: np.ndarray) -> np.ndarray:
        """色彩校正"""
        # 自动白平衡
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
```

#### 4.2.6 格式转换 (video_converter.py)

```python
class VideoConverter:
    """视频格式转换"""
    
    @staticmethod
    def convert_format(input_path: str, output_path: str, 
                      codec: str = 'libx264'):
        """格式转换"""
        import ffmpeg
        
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path, vcodec=codec)
        ffmpeg.run(stream, overwrite_output=True)
    
    @staticmethod
    def compress_video(input_path: str, output_path: str, 
                      crf: int = 23):
        """视频压缩"""
        import ffmpeg
        
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(
            stream, output_path,
            vcodec='libx264',
            crf=crf,  # 质量参数，越小质量越好
            preset='medium'
        )
        ffmpeg.run(stream, overwrite_output=True)
```

### 4.3 集成到视频生成流程

#### 更新 video_service.py

```python
def generate_video_from_script(script: Dict, task_id: str, 
                               post_process_config: Dict = None) -> str:
    """
    根据脚本生成完整视频（含后处理）
    """
    # 1. 生成场景视频
    video_paths = []
    for scene in script['scenes']:
        video_path = generate_scene_video(scene, task_id)
        video_paths.append(video_path)
    
    # 2. 应用转场效果
    if post_process_config and post_process_config.get('enable_transition'):
        video_paths = apply_transitions(video_paths, task_id)
    
    # 3. 拼接视频
    stitched_video = stitch_videos(video_paths, task_id)
    
    # 4. 应用滤镜
    if post_process_config and post_process_config.get('filters'):
        stitched_video = apply_filters(stitched_video, 
                                       post_process_config['filters'])
    
    # 5. 添加字幕
    if post_process_config and post_process_config.get('subtitles'):
        stitched_video = add_subtitles(stitched_video, 
                                       post_process_config['subtitles'])
    
    # 6. 添加背景音乐
    if post_process_config and post_process_config.get('background_music'):
        stitched_video = add_background_music(stitched_video, 
                                             post_process_config['background_music'])
    
    # 7. 质量优化
    if post_process_config and post_process_config.get('optimize'):
        stitched_video = optimize_video(stitched_video)
    
    # 8. 格式转换和压缩
    final_video = convert_and_compress(stitched_video, task_id)
    
    return final_video
```

### 4.4 配置更新

#### config.py

```python
# 视频后处理配置
VIDEO_POST_PROCESSING_CONFIG = {
    # 滤镜配置
    "filters": {
        "brightness": 1.0,      # 亮度 (0.5-2.0)
        "contrast": 1.0,        # 对比度 (0.5-2.0)
        "saturation": 1.0,      # 饱和度 (0.5-2.0)
        "sharpen": False,       # 锐化
        "denoise": False,       # 去噪
    },
    
    # 转场配置
    "transition": {
        "enabled": True,
        "type": "fade",         # fade, slide, zoom
        "duration": 10,         # 帧数
    },
    
    # 字幕配置
    "subtitle": {
        "enabled": False,
        "font_size": 32,
        "font_color": (255, 255, 255),
        "position": "bottom",   # top, bottom, center
        "font_path": None,      # 字体文件路径
    },
    
    # 音频配置
    "audio": {
        "background_music": None,  # 背景音乐文件路径
        "volume": 0.5,             # 音量 (0.0-1.0)
    },
    
    # 质量优化
    "optimization": {
        "denoise": False,
        "color_correction": False,
        "stabilize": False,
    },
    
    # 输出配置
    "output": {
        "format": "mp4",
        "codec": "libx264",
        "crf": 23,              # 质量参数 (18-28)
        "preset": "medium",     # ultrafast, fast, medium, slow
    }
}
```

---

## 5. 性能优化

### 5.1 优化策略

#### 1. 多线程处理
```python
from concurrent.futures import ThreadPoolExecutor

def process_frames_parallel(frames, process_func):
    """并行处理帧"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        processed_frames = list(executor.map(process_func, frames))
    return processed_frames
```

#### 2. 批量处理
```python
def process_frames_batch(frames, batch_size=10):
    """批量处理帧"""
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        # 处理批次
        yield process_batch(batch)
```

#### 3. 内存优化
```python
def process_video_streaming(input_path, output_path, process_func):
    """流式处理视频（不加载全部到内存）"""
    cap = cv2.VideoCapture(input_path)
    # ... 逐帧处理
```

#### 4. GPU 加速
```python
# 使用 CUDA 加速（如果可用）
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    # 使用 GPU 处理
    pass
```

### 5.2 性能指标

| 操作 | 目标时间 | 优化方法 |
|------|----------|----------|
| 滤镜处理 | < 0.5x 视频时长 | 多线程 |
| 转场效果 | < 0.2x 视频时长 | 批量处理 |
| 字幕添加 | < 0.3x 视频时长 | 缓存字体 |
| 格式转换 | < 1.0x 视频时长 | FFmpeg 硬件加速 |

---

## 6. 实施步骤

### 步骤 1: 安装依赖（10分钟）
```bash
pip install ffmpeg-python pydub
```

### 步骤 2: 实现视频滤镜（30分钟）
- 创建 `video_filter.py`
- 实现基础滤镜功能

### 步骤 3: 实现转场效果（30分钟）
- 创建 `video_transition.py`
- 实现淡入淡出、交叉溶解

### 步骤 4: 实现字幕系统（40分钟）
- 创建 `subtitle_system.py`
- 实现字幕渲染和叠加

### 步骤 5: 实现音频处理（30分钟）
- 创建 `audio_processor.py`
- 实现背景音乐添加

### 步骤 6: 实现质量优化（30分钟）
- 创建 `video_optimizer.py`
- 实现去噪、色彩校正

### 步骤 7: 实现格式转换（20分钟）
- 创建 `video_converter.py`
- 实现格式转换和压缩

### 步骤 8: 集成到主流程（30分钟）
- 更新 `video_service.py`
- 添加后处理流程

### 步骤 9: 更新配置（10分钟）
- 更新 `config.py`
- 添加后处理配置

### 步骤 10: 测试（40分钟）
- 创建测试用例
- 验证功能

### 步骤 11: 文档（30分钟）
- 编写使用文档
- 更新 API 文档

**总时间**: 约 4-5 小时

---

## 7. 测试计划

### 7.1 单元测试

```python
def test_video_filter():
    """测试视频滤镜"""
    # 测试亮度调整
    # 测试对比度调整
    # 测试饱和度调整
    pass

def test_transition_effect():
    """测试转场效果"""
    # 测试淡入淡出
    # 测试交叉溶解
    pass

def test_subtitle_system():
    """测试字幕系统"""
    # 测试字幕添加
    # 测试字幕样式
    pass
```

### 7.2 集成测试

```python
def test_full_post_processing():
    """测试完整后处理流程"""
    # 生成测试视频
    # 应用所有后处理
    # 验证输出
    pass
```

---

## 8. 验收标准

### 功能验收
- ✅ 滤镜功能正常
- ✅ 转场效果流畅
- ✅ 字幕显示正确
- ✅ 音频同步准确
- ✅ 质量优化有效
- ✅ 格式转换成功

### 性能验收
- ✅ 处理速度 < 2x 视频时长
- ✅ 内存占用合理
- ✅ 无明显卡顿

### 质量验收
- ✅ 无明显失真
- ✅ 色彩准确
- ✅ 音画同步

---

## 9. 总结

### 实施收益
- ✅ **功能完整**: 完整的视频后处理能力
- ✅ **质量提升**: 视频质量显著提升
- ✅ **用户体验**: 更专业的视频输出
- ✅ **可扩展性**: 易于添加新功能

### 技术亮点
1. **模块化设计**: 各功能独立，易于维护
2. **性能优化**: 多线程、批量处理
3. **灵活配置**: 丰富的配置选项
4. **工具组合**: OpenCV + FFmpeg + Pillow

---

**准备开始实施！** 🚀

