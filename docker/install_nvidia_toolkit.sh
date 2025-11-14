#!/bin/bash

# NVIDIA Container Toolkit 安装脚本（适用于 Ubuntu 22.04）

set -e

echo "=========================================="
echo "安装 NVIDIA Container Toolkit"
echo "=========================================="
echo ""

# 1. 检查系统
echo "[1/6] 检查系统版本..."
if [ ! -f /etc/os-release ]; then
    echo "✗ 无法检测系统版本"
    exit 1
fi

. /etc/os-release
DISTRIBUTION=$ID$VERSION_ID
echo "✓ 检测到系统: $DISTRIBUTION"
echo ""

# 2. 检查 NVIDIA 驱动
echo "[2/6] 检查 NVIDIA 驱动..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ NVIDIA 驱动未安装，请先安装驱动"
    exit 1
fi
echo "✓ NVIDIA 驱动已安装"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
echo ""

# 3. 添加 NVIDIA 仓库（使用新方法）
echo "[3/6] 添加 NVIDIA 仓库..."
# 删除旧的仓库配置（如果存在）
sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 使用新方法添加 GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 添加仓库
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "✓ 仓库已添加"
echo ""

# 4. 更新包列表
echo "[4/6] 更新包列表..."
sudo apt-get update
echo "✓ 包列表已更新"
echo ""

# 5. 安装 nvidia-container-toolkit
echo "[5/6] 安装 nvidia-container-toolkit..."
if sudo apt-get install -y nvidia-container-toolkit; then
    echo "✓ nvidia-container-toolkit 安装成功"
else
    echo "✗ 安装失败，尝试替代方法..."
    echo ""
    echo "如果上述方法失败，可以尝试："
    echo "  1. 检查网络连接"
    echo "  2. 使用国内镜像源"
    echo "  3. 手动下载 deb 包安装"
    exit 1
fi
echo ""

# 6. 配置 Docker
echo "[6/6] 配置 Docker..."
if command -v nvidia-ctk &> /dev/null; then
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "✓ Docker 已配置并重启"
else
    echo "✗ nvidia-ctk 命令未找到"
    exit 1
fi
echo ""

# 7. 测试
echo "=========================================="
echo "测试 GPU Docker 支持..."
echo "=========================================="
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✓ GPU Docker 测试成功！"
    echo ""
    echo "现在可以运行："
    echo "  docker run -it --gpus all --rm cosyvoice:latest /bin/bash"
else
    echo "⚠ GPU Docker 测试失败，但工具已安装"
    echo "  请检查 Docker 服务状态：sudo systemctl status docker"
fi
echo ""

