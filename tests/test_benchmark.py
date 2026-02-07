"""
性能基准测试
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import time
import torch
from services.llm_service import generate_script
from services.video_service import generate_scene_video
from utils.logger import setup_logger

logger = setup_logger(__name__)

def print_memory():
    """打印显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"显存: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")

def benchmark_script_generation():
    """脚本生成性能测试"""
    print("\n" + "="*60)
    print("脚本生成性能测试")
    print("="*60)
    
    prompts = [
        "森林探险",
        "海滩日落",
        "城市夜景"
    ]
    
    times = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n测试 {i}/{len(prompts)}: {prompt}")
        
        start = time.time()
        try:
            script = generate_script(prompt)
            duration = time.time() - start
            times.append(duration)
            print(f"耗时: {duration:.2f} 秒")
        except Exception as e:
            print(f"失败: {str(e)}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n统计:")
        print(f"  平均: {avg_time:.2f} 秒")
        print(f"  最快: {min_time:.2f} 秒")
        print(f"  最慢: {max_time:.2f} 秒")
        
        return avg_time
    
    return None

def benchmark_video_generation():
    """视频生成性能测试"""
    print("\n" + "="*60)
    print("视频生成性能测试")
    print("="*60)
    
    scene = {
        "scene_number": 1,
        "description": "测试场景：阳光明媚的草地",
        "duration": 2
    }
    
    print(f"\n场景: {scene['description']}")
    print(f"时长: {scene['duration']} 秒")
    
    print("\n开始生成...")
    print_memory()
    
    start = time.time()
    try:
        video_path = generate_scene_video(scene, "benchmark")
        duration = time.time() - start
        
        print(f"\n✅ 生成完成")
        print(f"耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
        print_memory()
        
        if os.path.exists(video_path):
            size = os.path.getsize(video_path) / 1024 / 1024
            print(f"文件大小: {size:.2f} MB")
        
        return duration
        
    except Exception as e:
        print(f"❌ 失败: {str(e)}")
        return None

def benchmark_memory_usage():
    """显存使用测试"""
    print("\n" + "="*60)
    print("显存使用测试")
    print("="*60)
    
    from services.model_loader import llm_loader, video_loader
    
    print("\n初始状态:")
    print_memory()
    
    if not llm_loader.is_loaded:
        print("\n加载 LLM 模型...")
        llm_loader.load_model()
        print_memory()
    else:
        print("\nLLM 模型已加载:")
        print_memory()
    
    if not video_loader.is_loaded:
        print("\n加载视频模型...")
        video_loader.load_model()
        print_memory()
    else:
        print("\n视频模型已加载:")
        print_memory()

if __name__ == "__main__":
    print("="*60)
    print("性能基准测试")
    print("="*60)
    
    # 显存使用测试
    benchmark_memory_usage()
    
    # 脚本生成性能
    script_time = benchmark_script_generation()
    
    # 视频生成性能
    video_time = benchmark_video_generation()
    
    # 总结
    print("\n" + "="*60)
    print("性能总结")
    print("="*60)
    
    if script_time:
        print(f"脚本生成平均时间: {script_time:.2f} 秒")
    
    if video_time:
        print(f"单场景视频生成时间: {video_time:.2f} 秒 ({video_time/60:.2f} 分钟)")
    
    print("="*60)
