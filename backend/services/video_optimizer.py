"""
视频质量优化模块
"""
import cv2
import numpy as np
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VideoOptimizer:
    """视频质量优化类"""
    
    @staticmethod
    def denoise(frame: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        去噪处理
        
        Args:
            frame: 输入帧
            strength: 去噪强度 (1-30)
            
        Returns:
            去噪后的帧
        """
        return cv2.fastNlMeansDenoisingColored(
            frame, None, strength, strength, 7, 21
        )
    
    @staticmethod
    def color_correction(frame: np.ndarray) -> np.ndarray:
        """
        自动色彩校正（白平衡）
        
        Args:
            frame: 输入帧
            
        Returns:
            校正后的帧
        """
        # 转换到 LAB 色彩空间
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # 计算 A 和 B 通道的平均值
        avg_a = np.average(lab[:, :, 1])
        avg_b = np.average(lab[:, :, 2])
        
        # 调整 A 和 B 通道
        lab[:, :, 1] = lab[:, :, 1] - ((avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1)
        lab[:, :, 2] = lab[:, :, 2] - ((avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1)
        
        # 转换回 BGR
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def enhance_contrast(frame: np.ndarray) -> np.ndarray:
        """
        自适应对比度增强（CLAHE）
        
        Args:
            frame: 输入帧
            
        Returns:
            增强后的帧
        """
        # 转换到 LAB 色彩空间
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # 对 L 通道应用 CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # 转换回 BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def sharpen_frame(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        锐化帧
        
        Args:
            frame: 输入帧
            strength: 锐化强度 (0.0-2.0)
            
        Returns:
            锐化后的帧
        """
        # 使用 Unsharp Mask 算法
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        sharpened = cv2.addWeighted(frame, 1.0 + strength, gaussian, -strength, 0)
        return sharpened
    
    @staticmethod
    def reduce_noise_video(input_path: str, output_path: str,
                          strength: int = 10) -> str:
        """
        对整个视频进行去噪
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            strength: 去噪强度
            
        Returns:
            输出视频路径
        """
        logger.info(f"开始视频去噪，强度: {strength}")
        
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 去噪
            denoised = VideoOptimizer.denoise(frame, strength)
            out.write(denoised)
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.debug(f"已处理 {frame_count}/{total_frames} 帧")
        
        cap.release()
        out.release()
        
        logger.info(f"✅ 视频去噪完成: {output_path}")
        return output_path
    
    @staticmethod
    def optimize_video(input_path: str, output_path: str,
                      denoise: bool = True,
                      color_correct: bool = True,
                      enhance_contrast: bool = True) -> str:
        """
        综合优化视频质量
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            denoise: 是否去噪
            color_correct: 是否色彩校正
            enhance_contrast: 是否增强对比度
            
        Returns:
            输出视频路径
        """
        logger.info(f"开始视频质量优化")
        logger.info(f"去噪: {denoise}, 色彩校正: {color_correct}, 对比度增强: {enhance_contrast}")
        
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 应用优化
            if denoise:
                frame = VideoOptimizer.denoise(frame, strength=8)
            
            if color_correct:
                frame = VideoOptimizer.color_correction(frame)
            
            if enhance_contrast:
                frame = VideoOptimizer.enhance_contrast(frame)
            
            out.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.debug(f"已处理 {frame_count}/{total_frames} 帧")
        
        cap.release()
        out.release()
        
        logger.info(f"✅ 视频质量优化完成: {output_path}")
        return output_path
    
    @staticmethod
    def stabilize_video(input_path: str, output_path: str) -> str:
        """
        视频稳定（简单实现）
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            
        Returns:
            输出视频路径
        """
        logger.info(f"开始视频稳定处理")
        
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 读取第一帧
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            return output_path
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        out.write(prev_frame)
        
        # 累积变换矩阵
        transforms = []
        
        frame_count = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 检测特征点
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=200, qualityLevel=0.01,
                minDistance=30, blockSize=3
            )
            
            if prev_pts is not None:
                # 光流跟踪
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_pts, None
                )
                
                # 过滤有效点
                idx = np.where(status == 1)[0]
                prev_pts = prev_pts[idx]
                curr_pts = curr_pts[idx]
                
                if len(prev_pts) > 0:
                    # 估计变换矩阵
                    m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
                    
                    if m is not None:
                        # 应用平滑变换
                        dx = m[0, 2]
                        dy = m[1, 2]
                        da = np.arctan2(m[1, 0], m[0, 0])
                        
                        # 平滑参数
                        smooth_factor = 0.7
                        dx *= smooth_factor
                        dy *= smooth_factor
                        da *= smooth_factor
                        
                        # 构建平滑变换矩阵
                        m_smooth = np.array([
                            [np.cos(da), -np.sin(da), dx],
                            [np.sin(da), np.cos(da), dy]
                        ], dtype=np.float32)
                        
                        # 应用变换
                        frame_stabilized = cv2.warpAffine(
                            frame, m_smooth, (width, height)
                        )
                        out.write(frame_stabilized)
                    else:
                        out.write(frame)
                else:
                    out.write(frame)
            else:
                out.write(frame)
            
            prev_gray = gray
            frame_count += 1
            
            if frame_count % 30 == 0:
                logger.debug(f"已处理 {frame_count} 帧")
        
        cap.release()
        out.release()
        
        logger.info(f"✅ 视频稳定完成: {output_path}")
        return output_path
