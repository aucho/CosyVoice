#!/bin/bash

# GPU Docker 支持诊断和修复脚本

echo "=========================================="
echo "GPU Docker 支持诊断"
echo "=========================================="
echo ""

# 1. 检查 NVIDIA 驱动
echo "[1/5] 检查 NVIDIA 驱动..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA 驱动已安装"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "✗ NVIDIA 驱动未安装"
    echo "  请先安装 NVIDIA 驱动"
    exit 1
fi
echo ""

# 2. 检查 nvidia-container-toolkit
echo "[2/5] 检查 nvidia-container-toolkit..."
if command -v nvidia-container-runtime &> /dev/null; then
    echo "✓ nvidia-container-toolkit 已安装"
    nvidia-container-runtime --version
else
    echo "✗ nvidia-container-toolkit 未安装"
    echo ""
    echo "安装步骤（Ubuntu/Debian）："
    echo "  # 添加 NVIDIA 仓库"
    echo "  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
    echo "  curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -"
    echo "  curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | \\"
    echo "    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo ""
    echo "  # 安装"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y nvidia-container-toolkit"
    echo ""
    echo "  # 配置 Docker"
    echo "  sudo nvidia-ctk runtime configure --runtime=docker"
    echo "  sudo systemctl restart docker"
    echo ""
    exit 1
fi
echo ""

# 3. 检查 Docker 配置
echo "[3/5] 检查 Docker 配置..."
if docker info 2>/dev/null | grep -i runtime | grep -q nvidia; then
    echo "✓ Docker 已配置 NVIDIA runtime"
else
    echo "✗ Docker 未配置 NVIDIA runtime"
    echo "  运行以下命令配置："
    echo "  sudo nvidia-ctk runtime configure --runtime=docker"
    echo "  sudo systemctl restart docker"
    echo ""
fi
echo ""

# 4. 测试官方 CUDA 镜像
echo "[4/5] 测试官方 CUDA 镜像..."
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✓ GPU Docker 测试成功"
else
    echo "✗ GPU Docker 测试失败"
    echo "  请检查上述配置步骤"
    exit 1
fi
echo ""

# 5. 测试你的镜像
echo "[5/5] 测试 cosyvoice 镜像..."
if docker run --rm --gpus all cosyvoice:latest nvidia-smi &> /dev/null 2>&1; then
    echo "✓ cosyvoice 镜像 GPU 支持正常"
else
    echo "⚠ cosyvoice 镜像 GPU 测试失败（可能是镜像内没有 nvidia-smi）"
    echo "  但这不影响使用，可以继续尝试运行容器"
fi
echo ""

echo "=========================================="
echo "诊断完成！如果所有检查都通过，可以运行："
echo "  docker run -it --gpus all --rm cosyvoice:latest /bin/bash"
echo "=========================================="

