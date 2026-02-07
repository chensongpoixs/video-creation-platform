"""
单场景视频生成测试
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import time
from services.video_service import generate_scene_video
from utils.logger import setup_logger

logger = setup_logger(__name__)

def test_single_scene():
    """测试单场景视频生成"""
    print("\n" + "="*60)
    print("单场景视频生成测试")
    print("="*60)
    
    scene = {
        "scene_number": 1,
        "description": "阳光明媚的森林，鸟儿在树枝上歌唱，微风吹过树叶沙沙作响",
        "duration": 3
    }
    
    print(f"\n场景描述: {scene['description']}")
    print(f"时长: {scene['duration']} 秒")
    
    print("\n开始生成视频...")
    start = time.time()
    
    try:
        video_path = generate_scene_video(scene, "test_single")
        duration = time.time() - start
        
        print(f"\n✅ 视频生成成功")
        print(f"路径: {video_path}")
        print(f"耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
        
        # 检查文件
        if os.path.exists(video_path):
            size = os.path.getsize(video_path) / 1024 / 1024
            print(f"文件大小: {size:.2f} MB")
            print("✅ 文件存在")
        else:
            print("❌ 文件不存在")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ 生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_scene()
    exit(0 if success else 1)
