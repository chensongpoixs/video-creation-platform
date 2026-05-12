"""
系统配置文件
"""
import os
from pathlib import Path

# ============================================================
# Hugging Face 镜像配置（国内用户使用 hf-mirror.com 加速下载）
# 设置环境变量后，transformers / diffusers / huggingface_hub 自动使用镜像
# ============================================================
HF_MIRROR = os.getenv("HF_MIRROR", "https://hf-mirror.com")
os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 视频输出目录
VIDEO_OUTPUT_DIR = Path(os.getenv("VIDEO_OUTPUT_DIR", BASE_DIR / "videos"))
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型目录（环境变量 MODELS_DIR 可覆盖，如 /data/models）
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 单个模型路径（环境变量可覆盖，不设置则默认在 MODELS_DIR 下）
_DEFAULT_LLM_PATH = MODELS_DIR / "chatglm3-6b"
_DEFAULT_VIDEO_PATH = MODELS_DIR / "cogvideox-2b"

# LLM 模型配置
LLM_CONFIG = {
    "model_name": "THUDM/chatglm3-6b",
    "model_path": str(os.getenv("LLM_MODEL_PATH", _DEFAULT_LLM_PATH)),
    "device": "cpu",           # "cuda"(GPU) 或 "cpu"，默认 cuda
    "allow_cpu_inference": False,  # CPU 下是否走 LLM 推理（6B 模型 CPU 推理 10-30min，默认跳过）
    "use_fp16": True,           # 启用 FP16 半精度，显存减半
    "use_int8": False,          # INT8 量化（更激进的优化）
    "max_length": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "auto_download": True,
    # FP16 优化配置
    "fp16_opt_level": "O1",  # 混合精度级别: O0(FP32), O1(推荐), O2(几乎FP16), O3(纯FP16)
    "enable_memory_efficient": True,  # 启用内存优化
}

# 视频生成模型配置
VIDEO_CONFIG = {
    # CogVideoX-2b: 支持中文的文生视频模型（清华大学 THUDM 出品，与 ChatGLM 同团队）
    "model_name": "THUDM/CogVideoX-2b",
    "model_path": str(os.getenv("VIDEO_MODEL_PATH", _DEFAULT_VIDEO_PATH)),
    "device": "cuda",            # CogVideoX-2b 推荐 GPU，约 12GB 显存
    "use_fp16": True,            # FP16 半精度，显存减半（约 6-8GB）
    "num_inference_steps": 50,   # CogVideoX-2b 默认推理步数
    "guidance_scale": 6.0,       # CogVideoX 支持 CFG 引导
    "height": 480,               # CogVideoX-2b 默认分辨率
    "width": 720,
    "num_frames": 49,            # 生成帧数（约 6 秒 @ 8fps）
    "fps": 8,
    "auto_download": True,
    # 内存优化
    "enable_attention_slicing": True,  # 注意力切片
    "enable_vae_slicing": True,        # VAE 切片
    "enable_xformers": False,          # CogVideoX 使用 Flash Attention，非 xFormers
    # Transformer 量化：启动默认使用 4bit 权重量化（bitsandbytes NF4，diffusers 所称 Q4 级别）
    # none: 不量化 | bnb_4bit: 默认，自动启用（无 CUDA/无 bitsandbytes/加载失败时由加载器回退 FP16）| gguf: 本地 .gguf 替换 transformer
    "transformer_quantization": os.getenv("VIDEO_TRANSFORMER_QUANTIZATION", "bnb_4bit"),
    "transformer_gguf_path": os.getenv("VIDEO_TRANSFORMER_GGUF_PATH", "").strip(),
    # GGUF 动态反量化后的计算精度（gguf 模式）
    "gguf_compute_dtype": os.getenv("VIDEO_GGUF_COMPUTE_DTYPE", "float16"),  # float16 | bfloat16
}

# 视频处理配置
VIDEO_PROCESSING_CONFIG = {
    "output_format": "mp4",
    "codec": "libx264",
    "bitrate": "5000k",
    "enable_interpolation": False,
}

# 显存优化配置
MEMORY_CONFIG = {
    "auto_optimize": True,  # 自动根据显存情况优化
    "min_free_memory": 2.0,  # 最小空闲显存 (GB)
    "enable_monitoring": True,  # 启用显存监控
    "clear_cache_after_generation": True,  # 生成后清理缓存
    "warn_threshold": 0.85,  # 显存使用警告阈值 (85%)
    "force_fp16_threshold": 16.0,  # 显存小于此值强制 FP16 (GB)
}

# 数据库配置（使用绝对路径，确保从任意目录启动都指向同一文件）
DATABASE_URL = f"sqlite:///{BASE_DIR / 'video_platform.db'}"

# API 配置
API_HOST = "0.0.0.0"
API_PORT = 8010

# 任务配置
MAX_CONCURRENT_TASKS = 3
TASK_TIMEOUT = 600  # 秒

# JWT 认证配置
JWT_CONFIG = {
    "secret_key": os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production-please"),
    "algorithm": "HS256",
    "access_token_expire_minutes": 60,  # 访问令牌过期时间（分钟）
    "refresh_token_expire_days": 7,  # 刷新令牌过期时间（天）
}

# 视频后处理配置
VIDEO_POST_PROCESSING_CONFIG = {
    # 滤镜配置
    "filters": {
        "brightness": 1.0,      # 亮度 (0.5-2.0)
        "contrast": 1.0,        # 对比度 (0.5-2.0)
        "saturation": 1.0,      # 饱和度 (0.5-2.0)
        "sharpen": False,       # 锐化
        "blur": 0,              # 模糊（0 为不模糊）
        "vignette": 0,          # 暗角 (0.0-1.0)
        "sepia": False,         # 复古棕褐色
        "grayscale": False,     # 灰度
    },
    
    # 转场配置
    "transition": {
        "enabled": True,
        "type": "fade",         # fade, cross_dissolve, slide_left, slide_right
        "duration_frames": 15,  # 转场持续帧数
    },
    
    # 字幕配置
    "subtitle": {
        "enabled": False,
        "font_size": 32,
        "font_color": (255, 255, 255),
        "bg_color": None,       # 背景颜色 (R, G, B, A)，None 为透明
        "position": "bottom",   # top, center, bottom
        "font_path": None,      # 字体文件路径
    },
    
    # 音频配置
    "audio": {
        "background_music": None,  # 背景音乐文件路径
        "volume": 0.3,             # 音量 (0.0-1.0)
    },
    
    # 质量优化
    "optimization": {
        "denoise": False,
        "denoise_strength": 8,      # 去噪强度 (1-30)
        "color_correction": False,
        "enhance_contrast": False,
        "stabilize": False,
    },
    
    # 输出配置
    "output": {
        "format": "mp4",
        "codec": "libx264",
        "crf": 23,              # 质量参数 (18-28)，越小质量越好
        "preset": "medium",     # ultrafast, fast, medium, slow, veryslow
        "compress": True,       # 是否压缩
    }
}

# 性能优化配置
PERFORMANCE_CONFIG = {
    # 并发配置
    "max_workers": 4,               # 最大工作线程数
    "thread_pool_size": 10,         # 线程池大小
    "async_enabled": True,          # 启用异步处理
    
    # 缓存配置
    "cache_enabled": True,          # 启用缓存
    "cache_ttl": 3600,              # 缓存过期时间（秒）
    "cache_max_size": 1000,         # 内存缓存最大数量
    "redis_enabled": False,         # 启用 Redis（需要安装）
    "redis_url": "redis://localhost:6379",
    
    # 数据库配置
    "db_pool_size": 10,             # 连接池大小
    "db_max_overflow": 20,          # 最大溢出连接
    "db_pool_timeout": 30,          # 连接超时（秒）
    "db_pool_recycle": 3600,        # 连接回收时间（秒）
    
    # 监控配置
    "monitoring_enabled": True,     # 启用性能监控
    "metrics_interval": 60,         # 指标采集间隔（秒）
    "slow_request_threshold": 1.0,  # 慢请求阈值（秒）
    
    # 资源限制
    "max_memory_percent": 80,       # 最大内存使用率（%）
    "max_cpu_percent": 80,          # 最大 CPU 使用率（%）
    "max_concurrent_tasks": 10,     # 最大并发任务数
}

# 日志配置
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "logs" / "app.log"
LOG_FILE.parent.mkdir(exist_ok=True)
