"""
模型加载器 - 负责加载和管理LLM和视频生成模型
支持 FP16 半精度优化，显存占用减半
按需下载：仅下载项目使用的模型权重/配置/分词器，不下载整个仓库
"""
import torch
import os
from typing import Optional, List
from pathlib import Path
from utils.logger import setup_logger
from utils.memory_monitor import memory_monitor, print_memory, clear_memory
from config import LLM_CONFIG, VIDEO_CONFIG, MEMORY_CONFIG

logger = setup_logger(__name__)

# ============================================================
# 模型下载：仅下载权重/配置/分词器等必需文件，不下载整个仓库
# ============================================================
MODEL_FILE_PATTERNS = [
    "*.json",           # config.json, tokenizer_config.json, model_index.json 等
    "*.safetensors",    # 模型权重（safetensors 格式）
    "*.bin",            # 模型权重（pytorch 格式）
    "*.model",          # sentencepiece 模型
    "*.py",             # modeling code（trust_remote_code 需要）
    "tokenizer.*",      # 分词器文件
    "vocab.*",          # 词表
    "*.txt",            # special_tokens_map, added_tokens 等
    "*.yaml",           # 部分模型配置
    "*.md",             # model card（仅 model_index 相关）
    "preprocessor_config.json",
    "scheduler/**",     # diffusers scheduler 配置
    "vae/**",           # diffusers VAE
    "unet/**",          # diffusers UNet
    "feature_extractor/**",
    "tokenizer/**",
    "text_encoder/**",
    "image_encoder/**",
]

# 明确排除的非模型文件（README、示例、图片、测试等）
MODEL_IGNORE_PATTERNS = [
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg",
    "*.ipynb", "*.cpp", "*.cu", "*.h", "*.sh",
    "assets/**", "examples/**", "docs/**", "tests/**",
    ".gitattributes", ".gitignore",
    "README.md", "LICENSE*",
]


def _download_model_files(repo_id: str, local_dir: str, description: str = "") -> bool:
    """下载模型必需文件到项目目录（跳过非模型文件，不下载整个仓库）

    使用 huggingface_hub.snapshot_download 配合 allow_patterns/ignore_patterns，
    仅下载模型权重、配置、分词器等必需文件，排除图片、示例、文档等。
    """
    try:
        from huggingface_hub import snapshot_download

        logger.info(f"开始下载 {description}...")
        logger.info(f"  仓库: {repo_id}")
        logger.info(f"  保存: {local_dir}")
        logger.info(f"  镜像: {os.environ.get('HF_ENDPOINT', '(默认)')}")

        snapshot_download(
            repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=MODEL_FILE_PATTERNS,
            ignore_patterns=MODEL_IGNORE_PATTERNS,
            resume_download=True,
        )

        logger.info(f"✅ {description} 下载完成 → {local_dir}")
        return True

    except ImportError:
        logger.error("huggingface_hub 未安装，请运行: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"❌ 下载失败 {repo_id}: {e}")
        return False

class LLMModelLoader:
    """LLM 模型加载器 - device 由 config.py 中的 LLM_CONFIG['device'] 控制"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = LLM_CONFIG["device"]           # 直接取配置值: "cuda" 或 "cpu"
        self.is_loaded = False
        self.use_fp16 = LLM_CONFIG.get("use_fp16", True)

        logger.info(f"LLM 加载器初始化，配置设备: {self.device}")
        logger.info(f"FP16 模式: {'启用' if self.use_fp16 else '禁用'}")

        # 配置校验
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "⚠️ 配置 device='cuda' 但 CUDA 不可用！"
                "模型加载将失败，请安装 CUDA 版 PyTorch 或修改 config.py 中 device='cpu'"
            )

        # 自动优化：根据显存大小决定是否使用 FP16
        if MEMORY_CONFIG.get("auto_optimize", True) and torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            force_fp16_threshold = MEMORY_CONFIG.get("force_fp16_threshold", 16.0)

            if total_memory < force_fp16_threshold and not self.use_fp16:
                logger.warning(f"显存 {total_memory:.1f}GB < {force_fp16_threshold}GB，自动启用 FP16")
                self.use_fp16 = True

    def load_model(self):
        """加载 ChatGLM3 模型 — cuda/cpu 由配置决定

        按需下载：仅当模型本地不存在且 auto_download=True 时下载。
        CPU + allow_cpu_inference=False 时跳过整个加载/下载（模型不会被使用）。
        """
        if self.is_loaded:
            logger.info("LLM 模型已加载，跳过")
            return True

        # 校验：配置 cuda 但 CUDA 不可用 → 直接失败
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.error(
                "❌ 无法加载 LLM 模型: config.py 中 device='cuda'，但 CUDA 不可用。\n"
                "   解决方案: 1) 安装 PyTorch CUDA 版: pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124\n"
                "           2) 或修改 config.py: LLM_CONFIG['device'] = 'cpu'"
            )
            return False

        # CPU 模式且未开启 CPU 推理 → 跳过加载，不下载模型
        if self.device == "cpu" and not LLM_CONFIG.get("allow_cpu_inference", False):
            logger.info(
                "⏭️  跳过 LLM 模型加载: device='cpu' 且 allow_cpu_inference=False，"
                "将使用备用脚本生成方案（不下载模型）"
            )
            return False

        try:
            logger.info(f"开始加载 LLM 模型（设备: {self.device}）: {LLM_CONFIG['model_name']}")

            # 显存监控
            if MEMORY_CONFIG.get("enable_monitoring", True):
                print_memory("加载前 - ")

            model_path = LLM_CONFIG["model_path"]
            model_name = LLM_CONFIG["model_name"]

            # 检查关键文件是否存在（config.json 是 transformers 模型的标志文件）
            model_config = os.path.join(model_path, "config.json")
            if not os.path.exists(model_config):
                # 目录存在但可能为空 → 清理后重新下载
                if os.path.exists(model_path):
                    import shutil
                    shutil.rmtree(model_path)
                if LLM_CONFIG.get("auto_download", False):
                    if not _download_model_files(
                        model_name, model_path,
                        description=f"ChatGLM3-6B（LLM 剧本生成）",
                    ):
                        logger.error("LLM 模型下载失败，将使用备用方案")
                        return False
                else:
                    logger.error(f"模型文件不存在且 auto_download=False: {model_path}")
                    logger.info("请运行 python scripts/download_model.py --model llm 或设置 auto_download=True")
                    return False

            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError:
                logger.error("transformers 未安装，请运行: pip install transformers")
                return False

            logger.info("加载 tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            logger.info("加载 LLM 模型...")
            load_kwargs = {
                "trust_remote_code": True,
            }

            # FP16 优化
            if self.use_fp16 and self.device == "cuda":
                logger.info("✅ 使用 FP16 半精度（显存减半）")
                load_kwargs["torch_dtype"] = torch.float16

            # INT8 量化（更激进）
            if LLM_CONFIG.get("use_int8", False):
                logger.info("使用 INT8 量化")
                load_kwargs["load_in_8bit"] = True
                load_kwargs["device_map"] = "auto"
            else:
                self.model = AutoModel.from_pretrained(model_path, **load_kwargs)

                if self.device == "cuda":
                    self.model = self.model.cuda()

            # 启用内存优化
            if LLM_CONFIG.get("enable_memory_efficient", True):
                try:
                    self.model.gradient_checkpointing_enable()
                    logger.info("✅ 启用梯度检查点（内存优化）")
                except:
                    logger.debug("梯度检查点不可用")

            self.model.eval()

            self.is_loaded = True
            logger.info("✅ LLM 模型加载完成")

            # 显存监控
            if MEMORY_CONFIG.get("enable_monitoring", True):
                print_memory("加载后 - ")
                peak = memory_monitor.get_peak_memory()
                logger.info(f"峰值显存: {peak:.2f} GB")

            return True

        except Exception as e:
            logger.error(f"❌ LLM 模型加载失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        if not self.is_loaded or self.model is None:
            logger.error("LLM 模型未加载")
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        try:
            # 生成前清理缓存
            if MEMORY_CONFIG.get("clear_cache_after_generation", True):
                clear_memory()
            
            gen_kwargs = {
                "max_length": LLM_CONFIG.get("max_length", 2048),
                "temperature": LLM_CONFIG.get("temperature", 0.7),
                "top_p": LLM_CONFIG.get("top_p", 0.9),
                "do_sample": LLM_CONFIG.get("do_sample", True),
            }
            gen_kwargs.update(kwargs)
            
            logger.debug(f"生成参数: {gen_kwargs}")
            
            # 显存监控
            if MEMORY_CONFIG.get("enable_monitoring", True):
                memory_monitor.record_snapshot("生成前")
            
            response, history = self.model.chat(
                self.tokenizer,
                prompt,
                history=[],
                **gen_kwargs
            )
            
            # 生成后清理缓存
            if MEMORY_CONFIG.get("clear_cache_after_generation", True):
                clear_memory()
            
            # 显存监控
            if MEMORY_CONFIG.get("enable_monitoring", True):
                memory_monitor.record_snapshot("生成后")
            
            return response
            
        except Exception as e:
            logger.error(f"生成失败: {str(e)}")
            raise
    
    def unload_model(self):
        """卸载模型释放显存"""
        if self.model:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.is_loaded = False
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("LLM 模型已卸载")

class VideoModelLoader:
    """视频生成模型加载器 — CogVideoX-2b 文生视频（原生支持中文 prompt）

    CogVideoX 由清华大学 THUDM 团队发布，与 ChatGLM 同源，通过 diffusers 集成。
    """

    def __init__(self):
        self.model = None
        self.device = VIDEO_CONFIG["device"]
        self.is_loaded = False
        self.use_fp16 = VIDEO_CONFIG.get("use_fp16", True)
        # bitsandbytes 4bit 加载后不可对整管线 .to(cuda)，需 enable_model_cpu_offload
        self._bnb4bit_pipeline = False

        logger.info(f"视频模型加载器初始化，配置设备: {self.device}")
        logger.info(f"FP16 模式: {'启用' if self.use_fp16 else '禁用'}")

        # 配置校验
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "⚠️ 配置 device='cuda' 但 CUDA 不可用！"
                "模型加载将失败，请安装 CUDA 版 PyTorch 或修改 config.py 中 device='cpu'"
            )

        # 自动优化
        if MEMORY_CONFIG.get("auto_optimize", True) and torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            force_fp16_threshold = MEMORY_CONFIG.get("force_fp16_threshold", 16.0)

            if total_memory < force_fp16_threshold and not self.use_fp16:
                logger.warning(f"显存 {total_memory:.1f}GB < {force_fp16_threshold}GB，自动启用 FP16")
                self.use_fp16 = True

    def load_model(self):
        """加载 CogVideoX-2b 文生视频模型

        按需下载：仅当模型本地不存在且 auto_download=True 时下载。
        CogVideoX-2b FP16 约需 6-8GB 显存，适合消费级 GPU。
        """
        if self.is_loaded:
            logger.info("视频模型已加载，跳过")
            return True

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.error(
                "❌ 无法加载视频模型: config.py 中 device='cuda'，但 CUDA 不可用。\n"
                "   解决方案: 1) 安装 PyTorch CUDA 版: pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124\n"
                "           2) 或修改 config.py: VIDEO_CONFIG['device'] = 'cpu'"
            )
            return False

        try:
            logger.info(f"开始加载视频模型（设备: {self.device}）: {VIDEO_CONFIG['model_name']}")

            if MEMORY_CONFIG.get("enable_monitoring", True):
                print_memory("加载前 - ")

            model_path = VIDEO_CONFIG["model_path"]
            model_name = VIDEO_CONFIG["model_name"]

            # 检查模型文件是否存在（model_index.json 是 diffusers 模型的标志文件）
            model_index = os.path.join(model_path, "model_index.json")
            if not os.path.exists(model_index):
                # 目录存在但为空 → 删除，让 snapshot_download 重新下载
                if os.path.exists(model_path):
                    import shutil
                    shutil.rmtree(model_path)
                if VIDEO_CONFIG.get("auto_download", False):
                    if not _download_model_files(
                        model_name, model_path,
                        description="CogVideoX-2b（文生视频，支持中文）",
                    ):
                        logger.error("视频模型下载失败，将使用备用方案")
                        return False
                else:
                    logger.warning(f"模型文件不存在且 auto_download=False: {model_path}")
                    logger.info("提示：首次运行时会自动下载，或使用备用方案")
                    return False

            try:
                from diffusers import CogVideoXPipeline
            except ImportError:
                logger.error(
                    "diffusers 版本过低，CogVideoX 需要 diffusers >= 0.30.0\n"
                    "升级: pip install diffusers --upgrade"
                )
                return False

            logger.info("加载 CogVideoX-2b 文生视频 Pipeline...")

            quant_mode = str(VIDEO_CONFIG.get("transformer_quantization", "bnb_4bit")).lower()
            if quant_mode not in ("none", "bnb_4bit", "gguf"):
                logger.warning("未知 transformer_quantization=%s，按 none 处理", quant_mode)
                quant_mode = "none"

            load_kwargs: dict = {}
            transformer_override = None
            attempted_bnb_4bit = False

            if quant_mode == "gguf":
                gguf_path = (VIDEO_CONFIG.get("transformer_gguf_path") or "").strip()
                if not gguf_path or not os.path.isfile(gguf_path):
                    logger.error(
                        "transformer_quantization=gguf 需要有效的 VIDEO_TRANSFORMER_GGUF_PATH（指向 .gguf 单文件）"
                    )
                    return False
                try:
                    from diffusers import CogVideoXTransformer3DModel, GGUFQuantizationConfig
                except ImportError as e:
                    logger.error("GGUF 加载需要 diffusers 版本支持 CogVideoXTransformer3DModel / GGUFQuantizationConfig: %s", e)
                    return False
                try:
                    import gguf  # noqa: F401
                except ImportError:
                    logger.error("GGUF 模式请安装: pip install gguf")
                    return False
                gdtype = str(VIDEO_CONFIG.get("gguf_compute_dtype", "float16")).lower()
                compute_dtype = torch.bfloat16 if gdtype == "bfloat16" else torch.float16
                logger.info(
                    "使用 GGUF 单文件加载 transformer: %s, compute_dtype=%s",
                    gguf_path,
                    compute_dtype,
                )
                transformer_override = CogVideoXTransformer3DModel.from_single_file(
                    gguf_path,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
                    config=model_path,
                    subfolder="transformer",
                    torch_dtype=compute_dtype,
                )
                load_kwargs["torch_dtype"] = compute_dtype

            elif quant_mode == "bnb_4bit":
                bnb_ok = self.device == "cuda"
                if not bnb_ok:
                    logger.info("transformer_quantization=bnb_4bit 但 device 非 cuda，启动时自动回退 FP16/FP32")
                else:
                    try:
                        import bitsandbytes  # noqa: F401
                    except ImportError:
                        bnb_ok = False
                        logger.warning("未安装 bitsandbytes，启动时自动回退 FP16（可 pip install bitsandbytes 启用 4bit）")
                    if bnb_ok:
                        try:
                            from diffusers.quantizers import PipelineQuantizationConfig
                        except ImportError as e:
                            bnb_ok = False
                            logger.warning("无法导入 PipelineQuantizationConfig，自动回退 FP16: %s", e)
                if bnb_ok:
                    compute_dtype = torch.float16 if self.use_fp16 else torch.float32
                    load_kwargs["quantization_config"] = PipelineQuantizationConfig(
                        quant_backend="bitsandbytes_4bit",
                        quant_kwargs={
                            "load_in_4bit": True,
                            "bnb_4bit_quant_type": "nf4",
                            "bnb_4bit_compute_dtype": compute_dtype,
                        },
                        components_to_quantize=["transformer"],
                    )
                    load_kwargs["torch_dtype"] = compute_dtype
                    attempted_bnb_4bit = True
                    logger.info("启动默认启用 transformer 4bit 权重量化（bitsandbytes NF4 / Q4 级别）")
                else:
                    if self.use_fp16 and self.device == "cuda":
                        logger.info("✅ 使用 FP16 半精度（约 6-8GB 显存）")
                        load_kwargs["torch_dtype"] = torch.float16
                    else:
                        load_kwargs["torch_dtype"] = torch.float32

            else:
                if self.use_fp16 and self.device == "cuda":
                    logger.info("✅ 使用 FP16 半精度（约 6-8GB 显存）")
                    load_kwargs["torch_dtype"] = torch.float16
                else:
                    load_kwargs["torch_dtype"] = torch.float32

            pipe_kw = dict(load_kwargs)
            if transformer_override is not None:
                pipe_kw["transformer"] = transformer_override

            self._bnb4bit_pipeline = False
            try:
                self.model = CogVideoXPipeline.from_pretrained(model_path, **pipe_kw)
                self._bnb4bit_pipeline = bool(
                    attempted_bnb_4bit and pipe_kw.get("quantization_config") is not None
                )
            except Exception as e:
                if attempted_bnb_4bit and "quantization_config" in pipe_kw:
                    logger.warning("4bit 量化加载失败，启动时自动回退 FP16: %s", e)
                    load_kwargs_fb: dict = {}
                    if self.use_fp16 and self.device == "cuda":
                        load_kwargs_fb["torch_dtype"] = torch.float16
                    else:
                        load_kwargs_fb["torch_dtype"] = torch.float32
                    fb_kw = dict(load_kwargs_fb)
                    if transformer_override is not None:
                        fb_kw["transformer"] = transformer_override
                    self.model = CogVideoXPipeline.from_pretrained(model_path, **fb_kw)
                    self._bnb4bit_pipeline = False
                else:
                    raise

            # 移动到 GPU 并启用优化（bnb 量化子模块与整管线 .to(cuda) 不兼容，见 accelerate/diffusers 限制）
            if self.device == "cuda":
                if self._bnb4bit_pipeline:
                    self.model.enable_model_cpu_offload()
                    logger.info("bitsandbytes 4bit 已启用 enable_model_cpu_offload（替代整管线 .to(cuda)）")
                else:
                    self.model = self.model.to(self.device)

                if VIDEO_CONFIG.get("enable_vae_slicing", True):
                    try:
                        self.model.vae.enable_slicing()
                        logger.info("✅ 启用 VAE 切片（内存优化）")
                    except Exception:
                        logger.debug("VAE 切片不可用")

                if VIDEO_CONFIG.get("enable_attention_slicing", True):
                    try:
                        self.model.enable_attention_slicing()
                        logger.info("✅ 启用注意力切片（内存优化）")
                    except Exception:
                        logger.debug("注意力切片不可用")

            self.is_loaded = True
            logger.info("✅ CogVideoX-2b 模型加载完成")

            if MEMORY_CONFIG.get("enable_monitoring", True):
                print_memory("加载后 - ")
                peak = memory_monitor.get_peak_memory()
                logger.info(f"峰值显存: {peak:.2f} GB")

            return True

        except Exception as e:
            logger.error(f"❌ 视频模型加载失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def generate_video(self, prompt: str = "", image=None, **kwargs) -> list:
        """
        使用 CogVideoX 文生视频（原生支持中文 prompt）

        Args:
            prompt: 中文/英文场景描述，直接传给 CogVideoX 模型
            image: 不使用（CogVideoX 是文生视频，不需要图像输入）
            **kwargs: 生成参数覆盖

        Returns:
            视频帧列表 (list of PIL Images)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("视频模型未加载")

        if not prompt or not prompt.strip():
            raise ValueError("CogVideoX 需要文本 prompt，不能为空")

        try:
            if MEMORY_CONFIG.get("clear_cache_after_generation", True):
                clear_memory()

            gen_kwargs = {
                "num_inference_steps": VIDEO_CONFIG.get("num_inference_steps", 50),
                "guidance_scale": VIDEO_CONFIG.get("guidance_scale", 6.0),
                "height": VIDEO_CONFIG.get("height", 480),
                "width": VIDEO_CONFIG.get("width", 720),
                "num_frames": VIDEO_CONFIG.get("num_frames", 49),
                "num_videos_per_prompt": 1,
            }
            gen_kwargs.update(kwargs)

            logger.info(f"文本提示词: {prompt[:80]}...")
            logger.info(f"生成参数: {gen_kwargs}")

            if MEMORY_CONFIG.get("enable_monitoring", True):
                memory_monitor.record_snapshot("生成前")
                print_memory("生成前 - ")

            output = self.model(
                prompt=prompt,
                **gen_kwargs
            )

            frames = output.frames[0]
            logger.info(f"✅ CogVideoX 视频生成完成，帧数: {len(frames)}")

            if MEMORY_CONFIG.get("clear_cache_after_generation", True):
                clear_memory()

            if MEMORY_CONFIG.get("enable_monitoring", True):
                memory_monitor.record_snapshot("生成后")
                print_memory("生成后 - ")
                peak = memory_monitor.get_peak_memory()
                logger.info(f"峰值显存: {peak:.2f} GB")

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
        self._bnb4bit_pipeline = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("视频模型已卸载")

# 全局实例
llm_loader = LLMModelLoader()
video_loader = VideoModelLoader()
