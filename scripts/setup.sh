#!/bin/bash
# 项目初始化脚本

echo "=== 多模态视频创作平台 - 环境配置 ==="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA可用:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "警告: 未检测到CUDA，将使用CPU模式"
fi

# 创建虚拟环境
echo "创建Python虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 安装依赖
echo "安装Python依赖..."
cd backend
pip install --upgrade pip
pip install -r requirements.txt

# 创建必要目录
echo "创建必要目录..."
mkdir -p videos logs

echo "=== 配置完成 ==="
echo "启动服务: python main.py"
