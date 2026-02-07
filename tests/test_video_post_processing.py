"""
视频后处理功能测试
"""
import pytest
import os
import cv2
import numpy as np
from PIL import Image

# 测试前需要添加 backend 到路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.video_filter import VideoFilter
from services.video_transition import TransitionEffect
from services.subtitle_system import SubtitleSystem
from services.audio_processor import AudioProcessor
from services.video_optimizer import VideoOptimizer
from services.video_converter import VideoConverter


class TestVideoFilter:
    """视频滤镜测试"""
    
    def test_adjust_brightness(self):
        """测试亮度调整"""
        # 创建测试帧
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 调整亮度
        bright_frame = VideoFilter.adjust_brightness(frame, 1.5)
        dark_frame = VideoFilter.adjust_brightness(frame, 0.5)
        
        assert bright_frame.shape == frame.shape
        assert dark_frame.shape == frame.shape
        assert np.mean(bright_frame) > np.mean(frame)
        assert np.mean(dark_frame) < np.mean(frame)
    
    def test_adjust_contrast(self):
        """测试对比度调整"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        high_contrast = VideoFilter.adjust_contrast(frame, 1.5)
        low_contrast = VideoFilter.adjust_contrast(frame, 0.5)
        
        assert high_contrast.shape == frame.shape
        assert low_contrast.shape == frame.shape
    
    def test_adjust_saturation(self):
        """测试饱和度调整"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        saturated = VideoFilter.adjust_saturation(frame, 1.5)
        desaturated = VideoFilter.adjust_saturation(frame, 0.5)
        
        assert saturated.shape == frame.shape
        assert desaturated.shape == frame.shape
    
    def test_sharpen(self):
        """测试锐化"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        sharpened = VideoFilter.sharpen(frame, strength=1.0)
        
        assert sharpened.shape == frame.shape
    
    def test_blur(self):
        """测试模糊"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        blurred = VideoFilter.blur(frame, kernel_size=5)
        
        assert blurred.shape == frame.shape
    
    def test_apply_vignette(self):
        """测试暗角效果"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        vignetted = VideoFilter.apply_vignette(frame, strength=0.5)
        
        assert vignetted.shape == frame.shape
    
    def test_apply_sepia(self):
        """测试复古滤镜"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        sepia = VideoFilter.apply_sepia(frame)
        
        assert sepia.shape == frame.shape
    
    def test_apply_grayscale(self):
        """测试灰度转换"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        gray = VideoFilter.apply_grayscale(frame)
        
        assert gray.shape == frame.shape
        # 检查是否为灰度（三通道值相同）
        assert np.allclose(gray[:, :, 0], gray[:, :, 1])
        assert np.allclose(gray[:, :, 1], gray[:, :, 2])


class TestSubtitleSystem:
    """字幕系统测试"""
    
    def test_subtitle_initialization(self):
        """测试字幕系统初始化"""
        subtitle_system = SubtitleSystem(
            font_size=32,
            font_color=(255, 255, 255),
            position='bottom'
        )
        
        assert subtitle_system.font_size == 32
        assert subtitle_system.font_color == (255, 255, 255)
        assert subtitle_system.position == 'bottom'
    
    def test_add_subtitle_to_frame(self):
        """测试添加字幕到帧"""
        subtitle_system = SubtitleSystem()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        subtitled_frame = subtitle_system.add_subtitle_to_frame(
            frame, "Test Subtitle"
        )
        
        assert subtitled_frame.shape == frame.shape
        # 字幕帧应该与原帧不同
        assert not np.array_equal(subtitled_frame, frame)
    
    def test_generate_subtitles_from_script(self):
        """测试从脚本生成字幕"""
        script = {
            'scenes': [
                {'scene_number': 1, 'description': 'Scene 1', 'duration': 5},
                {'scene_number': 2, 'description': 'Scene 2', 'duration': 3},
            ]
        }
        
        subtitles = SubtitleSystem.generate_subtitles_from_script(script)
        
        assert len(subtitles) == 2
        assert subtitles[0]['start'] == 0.0
        assert subtitles[0]['end'] == 5.0
        assert subtitles[1]['start'] == 5.0
        assert subtitles[1]['end'] == 8.0
    
    def test_parse_srt_time(self):
        """测试 SRT 时间解析"""
        time_str = "00:01:30,500"
        seconds = SubtitleSystem._parse_srt_time(time_str)
        
        assert seconds == 90.5  # 1分30.5秒


class TestAudioProcessor:
    """音频处理测试"""
    
    def test_check_ffmpeg(self):
        """测试 FFmpeg 检查"""
        result = AudioProcessor.check_ffmpeg()
        
        # 结果应该是布尔值
        assert isinstance(result, bool)


class TestVideoOptimizer:
    """视频优化测试"""
    
    def test_denoise(self):
        """测试去噪"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        denoised = VideoOptimizer.denoise(frame, strength=10)
        
        assert denoised.shape == frame.shape
    
    def test_color_correction(self):
        """测试色彩校正"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        corrected = VideoOptimizer.color_correction(frame)
        
        assert corrected.shape == frame.shape
    
    def test_enhance_contrast(self):
        """测试对比度增强"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        enhanced = VideoOptimizer.enhance_contrast(frame)
        
        assert enhanced.shape == frame.shape
    
    def test_sharpen_frame(self):
        """测试锐化"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        sharpened = VideoOptimizer.sharpen_frame(frame, strength=1.0)
        
        assert sharpened.shape == frame.shape


class TestVideoConverter:
    """视频转换测试"""
    
    def test_check_ffmpeg(self):
        """测试 FFmpeg 检查"""
        result = VideoConverter.check_ffmpeg()
        
        # 结果应该是布尔值
        assert isinstance(result, bool)


class TestIntegration:
    """集成测试"""
    
    def test_filter_config(self):
        """测试滤镜配置"""
        from config import VIDEO_POST_PROCESSING_CONFIG
        
        assert 'filters' in VIDEO_POST_PROCESSING_CONFIG
        assert 'transition' in VIDEO_POST_PROCESSING_CONFIG
        assert 'subtitle' in VIDEO_POST_PROCESSING_CONFIG
        assert 'audio' in VIDEO_POST_PROCESSING_CONFIG
        assert 'optimization' in VIDEO_POST_PROCESSING_CONFIG
        assert 'output' in VIDEO_POST_PROCESSING_CONFIG
    
    def test_filter_pipeline(self):
        """测试滤镜流水线"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 应用多个滤镜
        frame = VideoFilter.adjust_brightness(frame, 1.2)
        frame = VideoFilter.adjust_contrast(frame, 1.1)
        frame = VideoFilter.sharpen(frame)
        
        assert frame.shape == (100, 100, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
