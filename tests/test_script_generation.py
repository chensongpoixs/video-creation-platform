"""
脚本生成测试
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import time
from services.llm_service import generate_script
from utils.logger import setup_logger

logger = setup_logger(__name__)

def test_script_generation():
    """测试脚本生成功能"""
    print("\n" + "="*60)
    print("脚本生成测试")
    print("="*60)
    
    prompts = [
        "制作一段关于森林探险的短视频",
        "制作一段关于海滩日落的视频",
        "制作一段关于城市夜景的视频"
    ]
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n测试 {i}/{len(prompts)}: {prompt}")
        print("-" * 60)
        
        try:
            start = time.time()
            script = generate_script(prompt)
            duration = time.time() - start
            
            print(f"✅ 生成成功")
            print(f"耗时: {duration:.2f} 秒")
            print(f"场景数: {len(script['scenes'])}")
            print(f"总时长: {script['total_duration']} 秒")
            
            # 显示前两个场景
            for scene in script['scenes'][:2]:
                print(f"\n  场景 {scene['scene_number']}:")
                print(f"    描述: {scene['description'][:60]}...")
                print(f"    时长: {scene['duration']} 秒")
            
            results.append({
                "prompt": prompt,
                "success": True,
                "duration": duration,
                "scene_count": len(script['scenes'])
            })
            
        except Exception as e:
            print(f"❌ 生成失败: {str(e)}")
            results.append({
                "prompt": prompt,
                "success": False,
                "error": str(e)
            })
    
    # 统计
    print("\n" + "="*60)
    print("测试统计")
    print("="*60)
    
    success_count = sum(1 for r in results if r.get("success", False))
    print(f"成功: {success_count}/{len(results)}")
    
    if success_count > 0:
        avg_duration = sum(r["duration"] for r in results if r.get("success")) / success_count
        avg_scenes = sum(r["scene_count"] for r in results if r.get("success")) / success_count
        print(f"平均耗时: {avg_duration:.2f} 秒")
        print(f"平均场景数: {avg_scenes:.1f}")
    
    return results

if __name__ == "__main__":
    results = test_script_generation()
    
    # 返回状态码
    success_count = sum(1 for r in results if r.get("success", False))
    exit(0 if success_count == len(results) else 1)
