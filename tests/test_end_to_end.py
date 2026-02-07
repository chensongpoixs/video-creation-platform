"""
端到端测试 - 完整视频生成流程
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import time
from services.llm_service import generate_script
from services.video_service import generate_video_from_script
from utils.logger import setup_logger

logger = setup_logger(__name__)

def test_end_to_end():
    """端到端视频生成测试"""
    print("="*60)
    print("端到端视频生成测试")
    print("="*60)
    
    # 步骤 1: 生成脚本
    prompt = "制作一段关于森林探险的短视频，包含河流和小动物"
    print(f"\n步骤 1: 生成脚本")
    print(f"提示词: {prompt}")
    print("-" * 60)
    
    start_script = time.time()
    
    try:
        script = generate_script(prompt)
        script_time = time.time() - start_script
        
        print(f"✅ 脚本生成完成")
        print(f"场景数: {len(script['scenes'])}")
        print(f"总时长: {script['total_duration']} 秒")
        print(f"耗时: {script_time:.2f} 秒")
        
        # 显示场景
        print("\n场景列表:")
        for scene in script['scenes']:
            print(f"  场景 {scene['scene_number']}: {scene['description'][:50]}... ({scene['duration']}秒)")
        
    except Exception as e:
        print(f"❌ 脚本生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 步骤 2: 生成视频
    print(f"\n步骤 2: 生成视频")
    print("-" * 60)
    
    start_video = time.time()
    
    try:
        video_path = generate_video_from_script(script, "test_e2e")
        video_time = time.time() - start_video
        
        print(f"\n✅ 视频生成完成")
        print(f"路径: {video_path}")
        print(f"耗时: {video_time:.2f} 秒 ({video_time/60:.2f} 分钟)")
        
        # 检查文件
        if os.path.exists(video_path):
            size = os.path.getsize(video_path) / 1024 / 1024
            print(f"文件大小: {size:.2f} MB")
        else:
            print("❌ 文件不存在")
            return False
        
        # 总结
        total_time = script_time + video_time
        print(f"\n" + "="*60)
        print("测试总结")
        print("="*60)
        print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        print(f"  - 脚本生成: {script_time:.2f} 秒")
        print(f"  - 视频生成: {video_time:.2f} 秒 ({video_time/60:.2f} 分钟)")
        print(f"场景数: {len(script['scenes'])}")
        print(f"视频时长: {script['total_duration']} 秒")
        print(f"输出文件: {video_path}")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 视频生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_end_to_end()
    exit(0 if success else 1)
