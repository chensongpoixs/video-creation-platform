"""
显存优化测试 - 对比 FP32 和 FP16 的显存占用和性能
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import time
import torch
from services.model_loader import llm_loader, video_loader
from services.llm_service import generate_script
from utils.memory_monitor import memory_monitor, print_memory
from utils.logger import setup_logger
from config import LLM_CONFIG, VIDEO_CONFIG

logger = setup_logger(__name__)

def test_llm_memory():
    """测试 LLM 模型显存占用"""
    print("\n" + "="*60)
    print("LLM 模型显存测试")
    print("="*60)
    
    print(f"\n配置: FP16={'启用' if LLM_CONFIG.get('use_fp16') else '禁用'}")
    
    # 重置峰值统计
    memory_monitor.reset_peak_memory()
    
    print("\n初始显存:")
    print_memory()
    
    # 加载模型
    print("\n加载模型...")
    start = time.time()
    success = llm_loader.load_model()
    load_time = time.time() - start
    
    if not success:
        print("❌ 模型加载失败")
        return None
    
    print(f"✅ 加载完成，耗时: {load_time:.2f} 秒")
    print("\n加载后显存:")
    print_memory()
    
    # 测试生成
    print("\n测试生成...")
    try:
        start = time.time()
        response = llm_loader.generate("你好，请介绍一下你自己", max_length=100)
        gen_time = time.time() - start
        
        print(f"✅ 生成完成，耗时: {gen_time:.2f} 秒")
        print(f"生成内容: {response[:50]}...")
        
        print("\n生成后显存:")
        print_memory()
        
    except Exception as e:
        print(f"❌ 生成失败: {str(e)}")
        gen_time = None
    
    # 峰值显存
    peak = memory_monitor.get_peak_memory()
    print(f"\n峰值显存: {peak:.2f} GB")
    
    return {
        "load_time": load_time,
        "gen_time": gen_time,
        "peak_memory": peak,
        "fp16": LLM_CONFIG.get('use_fp16', False)
    }

def test_video_model_memory():
    """测试视频模型显存占用"""
    print("\n" + "="*60)
    print("视频模型显存测试")
    print("="*60)
    
    print(f"\n配置: FP16={'启用' if VIDEO_CONFIG.get('use_fp16') else '禁用'}")
    
    # 重置峰值统计
    memory_monitor.reset_peak_memory()
    
    print("\n当前显存:")
    print_memory()
    
    # 加载模型
    print("\n加载模型...")
    start = time.time()
    success = video_loader.load_model()
    load_time = time.time() - start
    
    if not success:
        print("❌ 模型加载失败")
        return None
    
    print(f"✅ 加载完成，耗时: {load_time:.2f} 秒")
    print("\n加载后显存:")
    print_memory()
    
    # 峰值显存
    peak = memory_monitor.get_peak_memory()
    print(f"\n峰值显存: {peak:.2f} GB")
    
    return {
        "load_time": load_time,
        "peak_memory": peak,
        "fp16": VIDEO_CONFIG.get('use_fp16', False)
    }

def test_script_generation_memory():
    """测试脚本生成的显存占用"""
    print("\n" + "="*60)
    print("脚本生成显存测试")
    print("="*60)
    
    # 重置峰值统计
    memory_monitor.reset_peak_memory()
    
    print("\n生成前显存:")
    print_memory()
    
    # 生成脚本
    prompt = "制作一段关于森林探险的短视频"
    print(f"\n提示词: {prompt}")
    
    start = time.time()
    try:
        script = generate_script(prompt)
        gen_time = time.time() - start
        
        print(f"\n✅ 生成完成，耗时: {gen_time:.2f} 秒")
        print(f"场景数: {len(script['scenes'])}")
        
        print("\n生成后显存:")
        print_memory()
        
        peak = memory_monitor.get_peak_memory()
        print(f"\n峰值显存: {peak:.2f} GB")
        
        return {
            "gen_time": gen_time,
            "peak_memory": peak,
            "scene_count": len(script['scenes'])
        }
        
    except Exception as e:
        print(f"❌ 生成失败: {str(e)}")
        return None

def test_memory_optimization_summary():
    """显存优化总结"""
    print("\n" + "="*60)
    print("显存优化测试总结")
    print("="*60)
    
    # 测试 LLM
    llm_result = test_llm_memory()
    
    # 测试视频模型
    video_result = test_video_model_memory()
    
    # 测试脚本生成
    script_result = test_script_generation_memory()
    
    # 总结
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    if llm_result:
        print(f"\nLLM 模型:")
        print(f"  FP16: {'启用' if llm_result['fp16'] else '禁用'}")
        print(f"  加载时间: {llm_result['load_time']:.2f} 秒")
        if llm_result['gen_time']:
            print(f"  生成时间: {llm_result['gen_time']:.2f} 秒")
        print(f"  峰值显存: {llm_result['peak_memory']:.2f} GB")
    
    if video_result:
        print(f"\n视频模型:")
        print(f"  FP16: {'启用' if video_result['fp16'] else '禁用'}")
        print(f"  加载时间: {video_result['load_time']:.2f} 秒")
        print(f"  峰值显存: {video_result['peak_memory']:.2f} GB")
    
    if script_result:
        print(f"\n脚本生成:")
        print(f"  生成时间: {script_result['gen_time']:.2f} 秒")
        print(f"  场景数: {script_result['scene_count']}")
        print(f"  峰值显存: {script_result['peak_memory']:.2f} GB")
    
    # 显存节省估算
    if llm_result and video_result:
        total_peak = llm_result['peak_memory'] + video_result['peak_memory']
        print(f"\n总峰值显存: {total_peak:.2f} GB")
        
        if llm_result['fp16'] and video_result['fp16']:
            estimated_fp32 = total_peak * 2
            saved = estimated_fp32 - total_peak
            print(f"FP16 优化:")
            print(f"  估算 FP32 显存: {estimated_fp32:.2f} GB")
            print(f"  节省显存: {saved:.2f} GB ({saved/estimated_fp32*100:.1f}%)")
    
    print("="*60)
    
    # 优化建议
    suggestions = memory_monitor.suggest_optimization()
    if suggestions['message']:
        print(f"\n优化建议: {suggestions['message']}")

if __name__ == "__main__":
    print("="*60)
    print("显存优化测试")
    print("="*60)
    
    # 显示 GPU 信息
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {props.name}")
        print(f"总显存: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("\n⚠️ CUDA 不可用")
    
    # 运行测试
    test_memory_optimization_summary()
