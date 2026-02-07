"""
快速测试脚本
"""
import sys
import os

# 添加 backend 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

print("="*60)
print("快速环境测试")
print("="*60)

# 1. Python 版本
print("\n1. Python 版本:")
print(f"   {sys.version}")

# 2. 检查 PyTorch
print("\n2. PyTorch:")
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA 可用")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"   ⚠️ CUDA 不可用")
except ImportError:
    print("   ❌ PyTorch 未安装")

# 3. 检查其他依赖
print("\n3. 依赖包:")
packages = ["fastapi", "transformers", "diffusers", "cv2", "numpy", "PIL"]
for pkg in packages:
    try:
        if pkg == "cv2":
            import cv2
            print(f"   ✅ opencv-python")
        elif pkg == "PIL":
            from PIL import Image
            print(f"   ✅ Pillow")
        else:
            __import__(pkg)
            print(f"   ✅ {pkg}")
    except ImportError:
        print(f"   ❌ {pkg} 未安装")

# 4. 测试脚本生成
print("\n4. 测试脚本生成:")
try:
    from services.llm_service import generate_script
    script = generate_script("测试视频")
    print(f"   ✅ 脚本生成成功")
    print(f"   场景数: {len(script['scenes'])}")
except Exception as e:
    print(f"   ❌ 失败: {str(e)}")

# 5. 检查目录
print("\n5. 目录结构:")
dirs = ["backend/models", "backend/videos", "backend/logs"]
for d in dirs:
    if os.path.exists(d):
        print(f"   ✅ {d}")
    else:
        print(f"   ⚠️ {d} 不存在")

print("\n" + "="*60)
print("测试完成")
print("="*60)
