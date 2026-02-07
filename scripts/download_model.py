#!/usr/bin/env python3
"""
模型下载脚本
支持从 Hugging Face 或 ModelScope 下载模型
"""
import os
import sys
import argparse

def download_from_huggingface(model_name: str, output_dir: str):
    """从 Hugging Face 下载模型"""
    print(f"从 Hugging Face 下载模型: {model_name}")
    print(f"保存到: {output_dir}")
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        print("下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=output_dir
        )
        
        print("下载模型...")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=output_dir
        )
        
        print("✅ 下载完成！")
        print(f"模型保存在: {output_dir}")
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        sys.exit(1)

def download_from_modelscope(model_name: str, output_dir: str):
    """从 ModelScope 下载模型"""
    print(f"从 ModelScope 下载模型: {model_name}")
    
    try:
        from modelscope import snapshot_download
        
        model_dir = snapshot_download(
            model_name,
            cache_dir=output_dir
        )
        
        print("✅ 下载完成！")
        print(f"模型保存在: {model_dir}")
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 LLM 模型")
    parser.add_argument("--source", choices=["hf", "ms"], default="hf",
                       help="下载源: hf=Hugging Face, ms=ModelScope")
    parser.add_argument("--model", default="THUDM/chatglm3-6b",
                       help="模型名称")
    parser.add_argument("--output", default="./models",
                       help="输出目录")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.source == "hf":
        download_from_huggingface(args.model, args.output)
    else:
        download_from_modelscope(args.model, args.output)
