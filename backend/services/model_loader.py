"""
模型加载器 - 负责加载和管理LLM和视频生成模型
支持 FP16 半精度优化，显存占用减半
"""
import torch
import os
from typing import Optional
from pathlib import Path
from utils.logger import setup_logger
from utils.memory_monitor import memory_monitor, print_memory, clear_memory
from config import LLM_CONFIG, VIDEO_CONFIG, MEMORY_CONFIG

logger = setup_logger(__name__)

class LLMModelLoader:
    """LLM 模型加载器 - 支持 FP16 优化"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = LLM_CONFIG["device"] if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        self.use_fp16 = LLM_CONFIG.get("use_fp16", True)
        
        logger.info(f"LLM 加载器初始化，使用设备: {self.device}")
        logger.info(f"FP16 模式: {'启用' if self.use_fp16 else '禁用'}")
        
        # 自动优化：根据显存大小决定是否使用 FP16
        if MEMORY_CONFIG.get("auto_optimize", True) and torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            force_fp16_threshold = MEMORY_CONFIG.get("force_fp16_threshold", 16.0)
            
            if total_memory < force_fp16_threshold and not self.use_fp16:
                logger.warning(f"显存 {total_memory:.1f}GB < {force_fp16_threshold}GB，自动启用 FP16")
                self.use_fp16 = True
    
    def load_model(self):
        """加载 ChatGLM3 模型 - 支持 FP16 优化"""
        if self.is_loaded:
            logger.info("LLM 模型已加载，跳过")
            return True
        
        try:
            logger.info(f"开始加载 LLM 模型: {LLM_CONFIG['model_name']}")
            
            # 显存监控
            if MEMORY_CONFIG.get("enable_monitoring", True):
                print_memory("加载前 - ")
            
            model_path = LLM_CONFIG["model_path"]
            
            if not os.path.exists(model_path) and LLM_CONFIG.get("auto_download", False):
                logger.info(f"本地模型不存在，从 Hugging Face 下载...")
                model_path = LLM_CONFIG["model_name"]
            elif not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                logger.info("请先下载模型或设置 auto_download=True")
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
    """视频生成模型加载器 - 支持 FP16 优化"""
    
    def __init__(self):
        self.model = None
        self.device = VIDEO_CONFIG["device"] if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        self.use_fp16 = VIDEO_CONFIG.get("use_fp16", True)
        
        logger.info(f"视频模型加载器初始化，使用设备: {self.device}")
        logger.info(f"FP16 模式: {'启用' if self.use_fp16 else '禁用'}")
        
        # 自动优化
        if MEMORY_CONFIG.get("auto_optimize", True) and torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            force_fp16_threshold = MEMORY_CONFIG.get("force_fp16_threshold", 16.0)
            
            if total_memory < force_fp16_threshold and not self.use_fp16:
                logger.warning(f"显存 {total_memory:.1f}GB < {force_fp16_threshold}GB，自动启用 FP16")
                self.use_fp16 = True
    
    def load_model(self):
        """加载 Stable Diffusion Video 模型 - 支持 FP16 优化"""
        if self.is_loaded:
            logger.info("视频模型已加载，跳过")
            return True
        
        try:
            logger.info(f"开始加载视频模型: {VIDEO_CONFIG['model_name']}")
            
            # 显存监控
            if MEMORY_CONFIG.get("enable_monitoring", True):
                print_memory("加载前 - ")
            
            model_path = VIDEO_CONFIG["model_path"]
            
            if not os.path.exists(model_path) and VIDEO_CONFIG.get("auto_download", False):
                logger.info(f"本地模型不存在，从 Hugging Face 下载...")
                model_path = VIDEO_CONFIG["model_name"]
            elif not os.path.exists(model_path):
                logger.warning(f"模型路径不存在: {model_path}")
                logger.info("提示：首次运行时会自动下载，或使用备用方案")
                return False
            
            try:
                from diffusers import StableVideoDiffusionPipeline
            except ImportError:
                logger.error("diffusers 未安装，请运行: pip install diffusers")
                return False
            
            logger.info("加载 Stable Diffusion Video 模型...")
            
            load_kwargs = {}
            
            # FP16 优化
            if self.use_fp16 and self.device == "cuda":
                logger.info("✅ 使用 FP16 半精度（显存减半）")
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["variant"] = "fp16"
            else:
                load_kwargs["torch_dtype"] = torch.float32
            
            self.model = StableVideoDiffusionPipeline.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            # 移动到设备
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                
                # 启用内存优化
                if VIDEO_CONFIG.get("enable_attention_slicing", True):
                    logger.info("✅ 启用注意力切片（内存优化）")
                    self.model.enable_attention_slicing()
                
                if VIDEO_CONFIG.get("enable_vae_slicing", True):
                    try:
                        self.model.enable_vae_slicing()
                        logger.info("✅ 启用 VAE 切片（内存优化）")
                    except:
                        logger.debug("VAE 切片不可用")
                
                # 尝试启用 xFormers 加速
                if VIDEO_CONFIG.get("enable_xformers", True):
                    try:
                        self.model.enable_xformers_memory_efficient_attention()
                        logger.info("✅ 启用 xFormers 加速")
                    except Exception as e:
                        logger.warning(f"xFormers 不可用: {str(e)}")
                        logger.info("提示：安装 xformers 可提升性能: pip install xformers")
            
            self.is_loaded = True
            logger.info("✅ 视频模型加载完成")
            
            # 显存监控
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
            # 生成前清理缓存
            if MEMORY_CONFIG.get("clear_cache_after_generation", True):
                clear_memory()
            
            gen_kwargs = {
                "num_inference_steps": VIDEO_CONFIG.get("num_inference_steps", 25),
                "guidance_scale": VIDEO_CONFIG.get("guidance_scale", 7.5),
                "height": VIDEO_CONFIG.get("height", 576),
                "width": VIDEO_CONFIG.get("width", 1024),
                "num_frames": VIDEO_CONFIG.get("num_frames", 25),
            }
            gen_kwargs.update(kwargs)
            
            logger.info(f"生成视频，参数: {gen_kwargs}")
            
            # 显存监控
            if MEMORY_CONFIG.get("enable_monitoring", True):
                memory_monitor.record_snapshot("生成前")
                print_memory("生成前 - ")
            
            if image is not None:
                output = self.model(
                    image=image,
                    prompt=prompt,
                    **gen_kwargs
                )
            else:
                logger.warning("需要输入图像，使用默认图像")
                from PIL import Image
                image = Image.new('RGB', (gen_kwargs["width"], gen_kwargs["height"]))
                output = self.model(
                    image=image,
                    prompt=prompt,
                    **gen_kwargs
                )
            
            frames = output.frames[0]
            logger.info(f"✅ 视频生成完成，帧数: {len(frames)}")
            
            # 生成后清理缓存
            if MEMORY_CONFIG.get("clear_cache_after_generation", True):
                clear_memory()
            
            # 显存监控
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
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("视频模型已卸载")

# 全局实例
llm_loader = LLMModelLoader()
video_loader = VideoModelLoader()
