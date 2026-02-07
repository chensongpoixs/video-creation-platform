"""
视频格式转换和压缩模块
"""
import os
import subprocess
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VideoConverter:
    """视频格式转换类"""
    
    @staticmethod
    def check_ffmpeg() -> bool:
        """
        检查 FFmpeg 是否可用
        
        Returns:
            是否可用
        """
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    @staticmethod
    def convert_format(input_path: str, output_path: str,
                      codec: str = 'libx264',
                      preset: str = 'medium') -> str:
        """
        转换视频格式
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            codec: 视频编码器 ('libx264', 'libx265', 'libvpx-vp9')
            preset: 编码预设 ('ultrafast', 'fast', 'medium', 'slow', 'veryslow')
            
        Returns:
            输出视频路径
        """
        logger.info(f"转换视频格式: {codec}, 预设: {preset}")
        
        if not VideoConverter.check_ffmpeg():
            logger.error("FFmpeg 未安装，无法转换格式")
            raise RuntimeError("FFmpeg 未安装")
        
        try:
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', codec,
                '-preset', preset,
                '-c:a', 'aac',
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg 错误: {result.stderr}")
                raise RuntimeError(f"FFmpeg 执行失败: {result.stderr}")
            
            logger.info(f"✅ 格式转换完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"格式转换失败: {str(e)}")
            raise
    
    @staticmethod
    def compress_video(input_path: str, output_path: str,
                      crf: int = 23,
                      preset: str = 'medium') -> str:
        """
        压缩视频
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            crf: 质量参数 (18-28)，越小质量越好，文件越大
            preset: 编码预设
            
        Returns:
            输出视频路径
        """
        logger.info(f"压缩视频: CRF={crf}, 预设={preset}")
        
        if not VideoConverter.check_ffmpeg():
            logger.error("FFmpeg 未安装，无法压缩视频")
            raise RuntimeError("FFmpeg 未安装")
        
        try:
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',
                '-crf', str(crf),
                '-preset', preset,
                '-c:a', 'aac',
                '-b:a', '128k',
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg 错误: {result.stderr}")
                raise RuntimeError(f"FFmpeg 执行失败: {result.stderr}")
            
            # 计算压缩率
            input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
            output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            compression_ratio = (1 - output_size / input_size) * 100
            
            logger.info(f"✅ 视频压缩完成: {output_path}")
            logger.info(f"   原始大小: {input_size:.2f} MB")
            logger.info(f"   压缩后: {output_size:.2f} MB")
            logger.info(f"   压缩率: {compression_ratio:.1f}%")
            
            return output_path
            
        except Exception as e:
            logger.error(f"视频压缩失败: {str(e)}")
            raise
    
    @staticmethod
    def change_resolution(input_path: str, output_path: str,
                         width: int, height: int) -> str:
        """
        改变视频分辨率
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            width: 目标宽度
            height: 目标高度
            
        Returns:
            输出视频路径
        """
        logger.info(f"改变分辨率: {width}x{height}")
        
        if not VideoConverter.check_ffmpeg():
            logger.error("FFmpeg 未安装，无法改变分辨率")
            raise RuntimeError("FFmpeg 未安装")
        
        try:
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-vf', f'scale={width}:{height}',
                '-c:a', 'copy',
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg 错误: {result.stderr}")
                raise RuntimeError(f"FFmpeg 执行失败: {result.stderr}")
            
            logger.info(f"✅ 分辨率改变完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"改变分辨率失败: {str(e)}")
            raise
    
    @staticmethod
    def change_fps(input_path: str, output_path: str, fps: int) -> str:
        """
        改变视频帧率
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            fps: 目标帧率
            
        Returns:
            输出视频路径
        """
        logger.info(f"改变帧率: {fps} FPS")
        
        if not VideoConverter.check_ffmpeg():
            logger.error("FFmpeg 未安装，无法改变帧率")
            raise RuntimeError("FFmpeg 未安装")
        
        try:
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-filter:v', f'fps={fps}',
                '-c:a', 'copy',
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg 错误: {result.stderr}")
                raise RuntimeError(f"FFmpeg 执行失败: {result.stderr}")
            
            logger.info(f"✅ 帧率改变完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"改变帧率失败: {str(e)}")
            raise
    
    @staticmethod
    def convert_to_gif(input_path: str, output_path: str,
                      fps: int = 10, width: int = 480) -> str:
        """
        转换为 GIF
        
        Args:
            input_path: 输入视频路径
            output_path: 输出 GIF 路径
            fps: 帧率
            width: 宽度（高度自动计算）
            
        Returns:
            输出 GIF 路径
        """
        logger.info(f"转换为 GIF: {fps} FPS, 宽度 {width}px")
        
        if not VideoConverter.check_ffmpeg():
            logger.error("FFmpeg 未安装，无法转换为 GIF")
            raise RuntimeError("FFmpeg 未安装")
        
        try:
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-vf', f'fps={fps},scale={width}:-1:flags=lanczos',
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg 错误: {result.stderr}")
                raise RuntimeError(f"FFmpeg 执行失败: {result.stderr}")
            
            logger.info(f"✅ GIF 转换完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"GIF 转换失败: {str(e)}")
            raise
    
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频信息字典
        """
        if not VideoConverter.check_ffmpeg():
            logger.error("FFmpeg 未安装，无法获取视频信息")
            raise RuntimeError("FFmpeg 未安装")
        
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFprobe 错误: {result.stderr}")
                raise RuntimeError(f"FFprobe 执行失败: {result.stderr}")
            
            import json
            info = json.loads(result.stdout)
            
            return info
            
        except Exception as e:
            logger.error(f"获取视频信息失败: {str(e)}")
            raise
