"""
视频服务测试
"""
import pytest
import os
from services.video_service_new import (
    generate_scene_video_fallback,
    stitch_videos
)
from services.video_processor import VideoProcessor
from PIL import Image

def test_video_processor_placeholder():
    """测试占位符图像生成"""
    image = VideoProcessor.generate_placeholder_image()
    assert image.size == (1024, 576)
    assert isinstance(image, Image.Image)

def test_video_processor_placeholder_custom_size():
    """测试自定义大小的占位符图像"""
    image = VideoProcessor.generate_placeholder_image(width=512, height=512)
    assert image.size == (512, 512)

def test_generate_scene_video_fallback():
    """测试备用视频生成"""
    scene = {
        "scene_number": 1,
        "description": "测试场景描述",
        "duration": 2
    }
    
    video_path = generate_scene_video_fallback(scene, "test_task")
    
    # 验证文件存在
    assert os.path.exists(video_path)
    
    # 清理测试文件
    if os.path.exists(video_path):
        os.remove(video_path)

def test_frames_to_video():
    """测试帧转视频"""
    # 创建测试帧
    frames = [
        Image.new('RGB', (100, 100), color=(255, 0, 0)),
        Image.new('RGB', (100, 100), color=(0, 255, 0)),
        Image.new('RGB', (100, 100), color=(0, 0, 255))
    ]
    
    output_path = "test_output.mp4"
    
    try:
        result = VideoProcessor.frames_to_video(frames, output_path, fps=1)
        assert os.path.exists(result)
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)

def test_interpolate_frames():
    """测试帧插值"""
    frames = [
        Image.new('RGB', (100, 100), color=(0, 0, 0)),
        Image.new('RGB', (100, 100), color=(255, 255, 255))
    ]
    
    interpolated = VideoProcessor.interpolate_frames(frames, factor=2)
    
    # 应该从2帧变成3帧
    assert len(interpolated) == 3
