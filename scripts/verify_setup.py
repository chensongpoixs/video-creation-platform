#!/usr/bin/env python3
"""
环境验证脚本 - 检查系统是否准备就绪
"""
import sys
import os

def check_python_version():
    """检查 Python 版本"""
    print("检查 Python 版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python 版本过低: {version.major}.{version.minor}")
        print("   需要 Python 3.10+")
        return False

def check_cuda():
    """检查 CUDA"""
    print("\n检查 CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return True
        else:
            print("⚠️ CUDA 不可用，将使用 CPU 模式")
            return False
    except ImportError:
        print("❌ PyTorch 未安装")
        return False

def check_dependencies():
    """检查依赖包"""
    print("\n检查依赖包...")
    required = [
        "fastapi",
        "uvicorn",
        "transformers",
        "torch",
        "opencv-python",
        "numpy"
    ]
    
    all_ok = True
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} 未安装")
            all_ok = False
    
    return all_ok

def check_model():
    """检查模型文件"""
    print("\n检查模型文件...")
    model_path = "../backend/models/chatglm3-6b"
    
    if os.path.exists(model_path):
        files = os.listdir(model_path)
        if "config.json" in files:
            print(f"✅ 模型文件存在: {model_path}")
            return True
        else:
            print(f"⚠️ 模型文件不完整: {model_path}")
            return False
    else:
        print(f"⚠️ 模型文件不存在: {model_path}")
        print("   提示：首次运行时会自动下载")
        return False

def check_directories():
    """检查必要目录"""
    print("\n检查目录结构...")
    dirs = [
        "../backend/models",
        "../backend/videos",
        "../backend/logs"
    ]
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"⚠️ {dir_path} 不存在，正在创建...")
            os.makedirs(dir_path, exist_ok=True)
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("多模态视频创作平台 - 环境验证")
    print("=" * 60)
    
    results = []
    results.append(("Python 版本", check_python_version()))
    results.append(("CUDA", check_cuda()))
    results.append(("依赖包", check_dependencies()))
    results.append(("目录结构", check_directories()))
    results.append(("模型文件", check_model()))
    
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    for name, status in results:
        status_str = "✅ 通过" if status else "❌ 失败"
        print(f"{name:15} {status_str}")
    
    all_passed = all(status for _, status in results[:-1])  # 模型文件可选
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 环境检查通过！可以启动服务")
        print("\n启动命令:")
        print("  cd backend")
        print("  python main.py")
    else:
        print("❌ 环境检查未通过，请先解决上述问题")
        print("\n安装依赖:")
        print("  cd backend")
        print("  pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()
