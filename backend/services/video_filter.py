"""
视频滤镜处理模块
"""
import cv2
import numpy as np
from typing import List
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VideoFilter:
    """视频滤镜处理类"""
    
    @staticmethod
    def adjust_brightness(frame: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        调整亮度
        
        Args:
            frame: 输入帧
            factor: 亮度因子 (0.5-2.0)，1.0 为原始亮度
            
        Returns:
            调整后的帧
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def adjust_contrast(frame: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        调整对比度
        
        Args:
            frame: 输入帧
            factor: 对比度因子 (0.5-2.0)，1.0 为原始对比度
            
        Returns:
            调整后的帧
        """
        return cv2.convertScaleAbs(frame, alpha=factor, beta=0)
    
    @staticmethod
    def adjust_saturation(frame: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        调整饱和度
        
        Args:
            frame: 输入帧
            factor: 饱和度因子 (0.5-2.0)，1.0 为原始饱和度
            
        Returns:
            调整后的帧
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def sharpen(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        锐化
        
        Args:
            frame: 输入帧
            strength: 锐化强度 (0.0-2.0)
            
        Returns:
            锐化后的帧
        """
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]]) * strength
        kernel[1, 1] = 9 - 8 * strength + 1
        return cv2.filter2D(frame, -1, kernel)
    
    @staticmethod
    def blur(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        高斯模糊
        
        Args:
            frame: 输入帧
            kernel_size: 核大小（奇数）
            
        Returns:
            模糊后的帧
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def apply_vignette(frame: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        应用暗角效果
        
        Args:
            frame: 输入帧
            strength: 暗角强度 (0.0-1.0)
            
        Returns:
            应用暗角后的帧
        """
        rows, cols = frame.shape[:2]
        
        # 创建径向渐变遮罩
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols / 2)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows / 2)
        
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = kernel / kernel.max()
        
        # 应用强度
        mask = 1 - (1 - mask) * strength
        
        # 应用遮罩
        output = frame.copy()
        for i in range(3):
            output[:, :, i] = output[:, :, i] * mask
        
        return output.astype(np.uint8)
    
    @staticmethod
    def apply_sepia(frame: np.ndarray) -> np.ndarray:
        """
        应用复古棕褐色滤镜
        
        Args:
            frame: 输入帧
            
        Returns:
            应用滤镜后的帧
        """
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        
        sepia = cv2.transform(frame, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_grayscale(frame: np.ndarray) -> np.ndarray:
        """
        转换为灰度
        
        Args:
            frame: 输入帧
            
        Returns:
            灰度帧（3通道）
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def apply_filter_to_video(input_path: str, output_path: str, 
                             filter_config: dict) -> str:
        """
        对整个视频应用滤镜
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            filter_config: 滤镜配置字典
            
        Returns:
            输出视频路径
        """
        logger.info(f"开始应用滤镜: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 应用滤镜
            if filter_config.get('brightness', 1.0) != 1.0:
                frame = VideoFilter.adjust_brightness(
                    frame, filter_config['brightness']
                )
            
            if filter_config.get('contrast', 1.0) != 1.0:
                frame = VideoFilter.adjust_contrast(
                    frame, filter_config['contrast']
                )
            
            if filter_config.get('saturation', 1.0) != 1.0:
                frame = VideoFilter.adjust_saturation(
                    frame, filter_config['saturation']
                )
            
            if filter_config.get('sharpen', False):
                frame = VideoFilter.sharpen(frame)
            
            if filter_config.get('blur', 0) > 0:
                frame = VideoFilter.blur(frame, filter_config['blur'])
            
            if filter_config.get('vignette', 0) > 0:
                frame = VideoFilter.apply_vignette(
                    frame, filter_config['vignette']
                )
            
            if filter_config.get('sepia', False):
                frame = VideoFilter.apply_sepia(frame)
            
            if filter_config.get('grayscale', False):
                frame = VideoFilter.apply_grayscale(frame)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                logger.debug(f"已处理 {frame_count}/{total_frames} 帧")
        
        cap.release()
        out.release()
        
        logger.info(f"✅ 滤镜应用完成: {output_path}")
        return output_path
