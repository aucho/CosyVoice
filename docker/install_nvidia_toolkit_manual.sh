#!/bin/bash

# NVIDIA Container Toolkit 手动安装脚本（适用于网络受限环境）

set -e

echo "=========================================="
echo "NVIDIA Container Toolkit 手动安装"
echo "=========================================="
echo ""

# 检查系统架构
ARCH=$(dpkg --print-architecture)
echo "检测到系统架构: $ARCH"
echo ""

# 检查 NVIDIA 驱动
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ NVIDIA 驱动未安装，请先安装驱动"
    exit 1
fi

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "✓ NVIDIA 驱动版本: $DRIVER_VERSION"
echo ""

# 定义下载 URL（使用 GitHub 镜像或直接下载）
BASE_URL="https://github.com/NVIDIA/libnvidia-container/releases/download"
VERSION="v1.15.1"

echo "=========================================="
echo "方法 1: 直接下载并安装 deb 包"
echo "=========================================="
echo ""

# 创建临时目录
TMP_DIR=$(mktemp -d)
cd $TMP_DIR

echo "下载 nvidia-container-toolkit deb 包..."
echo ""

# 根据架构选择包
case $ARCH in
    amd64)
        PKG_NAME="libnvidia-container-toolkit_1.15.1-1_amd64.deb"
        ;;
    arm64)
        PKG_NAME="libnvidia-container-toolkit_1.15.1-1_arm64.deb"
        ;;
    *)
        echo "✗ 不支持的架构: $ARCH"
        exit 1
        ;;
esac

# 尝试多个下载源
DOWNLOAD_URLS=(
    "${BASE_URL}/${VERSION}/${PKG_NAME}"
    "https://mirrors.tuna.tsinghua.edu.cn/github-release/NVIDIA/libnvidia-container/${VERSION}/${PKG_NAME}"
)

DOWNLOADED=false
for URL in "${DOWNLOAD_URLS[@]}"; do
    echo "尝试从: $URL"
    if wget -q --timeout=10 --tries=2 "$URL" -O "$PKG_NAME" 2>/dev/null || \
       curl -L --connect-timeout 10 --max-time 30 "$URL" -o "$PKG_NAME" 2>/dev/null; then
        if [ -f "$PKG_NAME" ] && [ -s "$PKG_NAME" ]; then
            echo "✓ 下载成功: $PKG_NAME"
            DOWNLOADED=true
            break
        fi
    fi
    echo "  下载失败，尝试下一个源..."
done

if [ "$DOWNLOADED" = false ]; then
    echo ""
    echo "✗ 自动下载失败"
    echo ""
    echo "请手动下载并安装："
    echo "  1. 访问: https://github.com/NVIDIA/libnvidia-container/releases"
    echo "  2. 下载对应架构的包: $PKG_NAME"
    echo "  3. 运行: sudo dpkg -i $PKG_NAME"
    echo "  4. 运行: sudo apt-get install -f  # 修复依赖"
    echo ""
    echo "或者使用以下命令手动下载（如果网络允许）："
    echo "  wget ${BASE_URL}/${VERSION}/${PKG_NAME}"
    echo "  sudo dpkg -i $PKG_NAME"
    echo "  sudo apt-get install -f"
    exit 1
fi

echo ""
echo "安装 deb 包..."
if sudo dpkg -i "$PKG_NAME" 2>&1 | grep -q "depends on"; then
    echo "修复依赖..."
    sudo apt-get install -f -y
fi

# 验证安装
if command -v nvidia-container-runtime &> /dev/null || command -v nvidia-ctk &> /dev/null; then
    echo "✓ nvidia-container-toolkit 安装成功"
else
    echo "⚠ 安装完成，但未找到命令，可能需要重启或检查 PATH"
fi

# 清理
cd -
rm -rf $TMP_DIR

echo ""
echo "=========================================="
echo "配置 Docker"
echo "=========================================="
echo ""

if command -v nvidia-ctk &> /dev/null; then
    echo "配置 Docker runtime..."
    sudo nvidia-ctk runtime configure --runtime=docker
    
    echo "重启 Docker..."
    sudo systemctl restart docker
    
    echo "✓ Docker 已配置并重启"
else
    echo "✗ nvidia-ctk 命令未找到，请检查安装"
    exit 1
fi

echo ""
echo "=========================================="
echo "测试 GPU Docker 支持"
echo "=========================================="
echo ""

sleep 2  # 等待 Docker 重启完成

if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✓ GPU Docker 测试成功！"
    echo ""
    echo "现在可以运行："
    echo "  docker run -it --gpus all --rm cosyvoice:latest /bin/bash"
else
    echo "⚠ GPU Docker 测试失败"
    echo ""
    echo "请检查："
    echo "  1. Docker 服务状态: sudo systemctl status docker"
    echo "  2. Docker 配置: cat /etc/docker/daemon.json"
    echo "  3. 重新运行测试: docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"
fi

echo ""

