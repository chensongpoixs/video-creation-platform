"""
音频处理模块
"""
import os
import subprocess
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioProcessor:
    """音频处理类"""
    
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
    def add_background_music(video_path: str, audio_path: str,
                           output_path: str, volume: float = 0.3) -> str:
        """
        为视频添加背景音乐
        
        Args:
            video_path: 视频文件路径
            audio_path: 音频文件路径
            output_path: 输出文件路径
            volume: 音频音量 (0.0-1.0)
            
        Returns:
            输出文件路径
        """
        logger.info(f"添加背景音乐: {audio_path}")
        logger.info(f"音量: {volume}")
        
        if not AudioProcessor.check_ffmpeg():
            logger.error("FFmpeg 未安装，无法添加背景音乐")
            raise RuntimeError("FFmpeg 未安装")
        
        try:
            # 使用 FFmpeg 命令
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-filter_complex',
                f'[1:a]volume={volume}[a1];[0:a][a1]amix=inputs=2:duration=first',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-y',  # 覆盖输出文件
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
            
            logger.info(f"✅ 背景音乐添加完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"添加背景音乐失败: {str(e)}")
            raise
    
    @staticmethod
    def replace_audio(video_path: str, audio_path: str,
                     output_path: str) -> str:
        """
        替换视频音频
        
        Args:
            video_path: 视频文件路径
            audio_path: 音频文件路径
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        logger.info(f"替换视频音频: {audio_path}")
        
        if not AudioProcessor.check_ffmpeg():
            logger.error("FFmpeg 未安装，无法替换音频")
            raise RuntimeError("FFmpeg 未安装")
        
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
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
            
            logger.info(f"✅ 音频替换完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"替换音频失败: {str(e)}")
            raise
    
    @staticmethod
    def adjust_audio_volume(video_path: str, output_path: str,
                          volume: float = 1.0) -> str:
        """
        调整视频音量
        
        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径
            volume: 音量倍数 (0.0-2.0)
            
        Returns:
            输出文件路径
        """
        logger.info(f"调整音量: {volume}x")
        
        if not AudioProcessor.check_ffmpeg():
            logger.error("FFmpeg 未安装，无法调整音量")
            raise RuntimeError("FFmpeg 未安装")
        
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-filter:a', f'volume={volume}',
                '-c:v', 'copy',
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
            
            logger.info(f"✅ 音量调整完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"调整音量失败: {str(e)}")
            raise
    
    @staticmethod
    def extract_audio(video_path: str, output_path: str) -> str:
        """
        从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            output_path: 输出音频文件路径
            
        Returns:
            输出音频文件路径
        """
        logger.info(f"提取音频: {video_path}")
        
        if not AudioProcessor.check_ffmpeg():
            logger.error("FFmpeg 未安装，无法提取音频")
            raise RuntimeError("FFmpeg 未安装")
        
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # 不包含视频
                '-acodec', 'mp3',
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
            
            logger.info(f"✅ 音频提取完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"提取音频失败: {str(e)}")
            raise
    
    @staticmethod
    def remove_audio(video_path: str, output_path: str) -> str:
        """
        移除视频音频
        
        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        logger.info(f"移除音频: {video_path}")
        
        if not AudioProcessor.check_ffmpeg():
            logger.error("FFmpeg 未安装，无法移除音频")
            raise RuntimeError("FFmpeg 未安装")
        
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-an',  # 不包含音频
                '-c:v', 'copy',
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
            
            logger.info(f"✅ 音频移除完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"移除音频失败: {str(e)}")
            raise
