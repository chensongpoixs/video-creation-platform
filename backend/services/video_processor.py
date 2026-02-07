"""
视频处理模块 - 负责视频帧处理和编码
"""
import cv2
import numpy as np
from typing import List
from PIL import Image
import os
from utils.logger import setup_logger
from config import VIDEO_OUTPUT_DIR, VIDEO_CONFIG

logger = setup_logger(__name__)

class VideoProcessor:
    """视频处理器"""
    
    @staticmethod
    def frames_to_video(frames: List, output_path: str, fps: int = 6) -> str:
        """
        将帧列表转换为视频文件
        
        Args:
            frames: PIL Image 列表或 numpy 数组列表
            output_path: 输出路径
            fps: 帧率
            
        Returns:
            视频文件路径
        """
        try:
            logger.info(f"开始转换视频，帧数: {len(frames)}, FPS: {fps}")
            
            if not frames:
                raise ValueError("帧列表为空")
            
            # 转换为 numpy 数组
            frame_array = []
            for i, frame in enumerate(frames):
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                
                # 确保是 numpy 数组
                if not isinstance(frame, np.ndarray):
                    frame = np.array(frame)
                
                # 转换为 BGR（OpenCV 格式）
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # 检查是否是 RGB
                    if frame.dtype == np.uint8:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                frame_array.append(frame)
            
            # 获取视频参数
            height, width = frame_array[0].shape[:2]
            
            logger.info(f"视频参数: {width}x{height}, {len(frame_array)} 帧")
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise RuntimeError("无法创建视频写入器")
            
            # 写入帧
            for i, frame in enumerate(frame_array):
                # 确保帧大小正确
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                
                out.write(frame)
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"已写入 {i + 1}/{len(frame_array)} 帧")
            
            out.release()
            
            logger.info(f"✅ 视频转换完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"视频转换失败: {str(e)}")
            raise
    
    @staticmethod
    def generate_placeholder_image(width: int = 1024, height: int = 576) -> Image.Image:
        """
        生成占位符图像
        
        Args:
            width: 宽度
            height: 高度
            
        Returns:
            PIL Image
        """
        logger.debug(f"生成占位符图像: {width}x{height}")
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
        
        if len(frames) < 2:
            return frames
        
        interpolated = []
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])
            
            # 简单的线性插值
            for j in range(1, factor):
                alpha = j / factor
                
                # 转换为 numpy 数组进行插值
                frame1 = np.array(frames[i], dtype=np.float32)
                frame2 = np.array(frames[i + 1], dtype=np.float32)
                
                blended = (1 - alpha) * frame1 + alpha * frame2
                blended = np.clip(blended, 0, 255).astype(np.uint8)
                
                interpolated.append(Image.fromarray(blended))
        
        interpolated.append(frames[-1])
        
        logger.info(f"帧插值完成: {len(frames)} -> {len(interpolated)} 帧")
        return interpolated
    
    @staticmethod
    def resize_frame(frame: Image.Image, width: int, height: int) -> Image.Image:
        """
        调整帧大小
        
        Args:
            frame: 输入帧
            width: 目标宽度
            height: 目标高度
            
        Returns:
            调整后的帧
        """
        return frame.resize((width, height), Image.Resampling.LANCZOS)
