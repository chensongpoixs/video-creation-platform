"""
模型加载测试
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import time
import torch
from services.model_loader import llm_loader, video_loader
from utils.logger import setup_logger

logger = setup_logger(__name__)

def print_memory():
    """打印显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"显存: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")
    else:
        print("CUDA 不可用")

def test_llm_loading():
    """测试 LLM 模型加载"""
    print("\n" + "="*60)
    print("测试 LLM 模型加载")
    print("="*60)
    
    print("\n初始显存状态:")
    print_memory()
    
    print("\n开始加载 LLM 模型...")
    start = time.time()
    success = llm_loader.load_model()
    duration = time.time() - start
    
    print(f"\nLLM 加载: {'✅ 成功' if success else '❌ 失败'}")
    print(f"耗时: {duration:.2f} 秒")
    
    if success:
        print("\n加载后显存状态:")
        print_memory()
        
        # 测试生成
        try:
            print("\n测试生成功能...")
            response = llm_loader.generate("你好，请介绍一下你自己", max_length=100)
            print(f"生成测试: {response[:100]}")
            print("✅ 生成功能正常")
        except Exception as e:
            print(f"❌ 生成测试失败: {str(e)}")
    
    return success

def test_video_loading():
    """测试视频模型加载"""
    print("\n" + "="*60)
    print("测试视频模型加载")
    print("="*60)
    
    print("\n当前显存状态:")
    print_memory()
    
    print("\n开始加载视频模型...")
    start = time.time()
    success = video_loader.load_model()
    duration = time.time() - start
    
    print(f"\n视频模型加载: {'✅ 成功' if success else '❌ 失败'}")
    print(f"耗时: {duration:.2f} 秒")
    
    if success:
        print("\n加载后显存状态:")
        print_memory()
    
    return success

def test_memory_management():
    """测试显存管理"""
    print("\n" + "="*60)
    print("测试显存管理")
    print("="*60)
    
    print("\n卸载 LLM 模型...")
    llm_loader.unload_model()
    print_memory()
    
    print("\n卸载视频模型...")
    video_loader.unload_model()
    print_memory()
    
    print("\n清理显存缓存...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_memory()

if __name__ == "__main__":
    print("="*60)
    print("模型加载测试")
    print("="*60)
    
    # 测试 LLM 加载
    llm_success = test_llm_loading()
    
    # 测试视频模型加载
    video_success = test_video_loading()
    
    # 测试显存管理
    test_memory_management()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"LLM 模型加载: {'✅ 通过' if llm_success else '❌ 失败'}")
    print(f"视频模型加载: {'✅ 通过' if video_success else '❌ 失败'}")
    print("="*60)
