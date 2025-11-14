#!/bin/bash

# CosyVoice WebUI Docker 一键运行脚本
# 使用方法: 
#   单实例: ./run_webui.sh
#   多实例: ./run_webui.sh --ports 8000-8010
#   指定端口: ./run_webui.sh --ports 8000,8001,8002

# 注意：不使用 set -e，因为我们需要在循环中处理单个容器的启动失败

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置参数（可根据需要修改）
IMAGE_NAME="cosyvoice"
IMAGE_TAG="latest"
CONTAINER_NAME_PREFIX="cosyvoice-webui"
MODEL_DIR="pretrained_models/CosyVoice2-0.5B"
MAX_CONCURRENT=8
MAX_QUEUE_SIZE=20

# 默认端口（单实例模式）
DEFAULT_PORT=8000

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 解析命令行参数
PORTS=()
MULTI_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --ports)
            MULTI_MODE=true
            if [[ $2 =~ ^[0-9]+-[0-9]+$ ]]; then
                # 端口范围，如 8000-8010
                START_PORT=$(echo $2 | cut -d'-' -f1)
                END_PORT=$(echo $2 | cut -d'-' -f2)
                for ((port=$START_PORT; port<=$END_PORT; port++)); do
                    PORTS+=($port)
                done
            elif [[ $2 =~ ^[0-9]+(,[0-9]+)+$ ]]; then
                # 逗号分隔的端口列表
                IFS=',' read -ra PORT_ARRAY <<< "$2"
                PORTS=("${PORT_ARRAY[@]}")
            else
                print_error "无效的端口格式: $2"
                print_info "支持格式: 8000-8010 或 8000,8001,8002"
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            echo "使用方法:"
            echo "  单实例模式: $0"
            echo "  多实例模式: $0 --ports 8000-8010"
            echo "  指定端口:   $0 --ports 8000,8001,8002"
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            exit 1
            ;;
    esac
done

# 如果没有指定端口，使用默认单实例模式
if [ "$MULTI_MODE" = false ]; then
    PORTS=($DEFAULT_PORT)
fi

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 未安装，请先安装 $1"
        exit 1
    fi
}

# 检查 Docker 是否安装
print_info "检查 Docker 环境..."
check_command docker

# 检查 Docker 是否运行
if ! docker info &> /dev/null; then
    print_error "Docker 未运行，请先启动 Docker 服务"
    exit 1
fi
print_success "Docker 环境检查通过"

# 检查 GPU 支持
print_info "检查 GPU 支持..."
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    GPU_FLAG="--gpus all"
    print_success "检测到 GPU 支持，将启用 GPU"
else
    GPU_FLAG="--gpus all"
    print_warning "GPU 检测失败，但仍将尝试启用 GPU（如果失败将回退到 CPU）"
fi

# 检查镜像是否存在
print_info "检查 Docker 镜像..."
if docker images | grep -q "^${IMAGE_NAME}.*${IMAGE_TAG}"; then
    print_success "镜像 ${IMAGE_NAME}:${IMAGE_TAG} 已存在"
else
    print_error "镜像 ${IMAGE_NAME}:${IMAGE_TAG} 不存在"
    print_info "请先构建镜像: docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f \"$SCRIPT_DIR/Dockerfile\" \"$PROJECT_ROOT\""
    exit 1
fi

# 检查并停止/删除已存在的容器
print_info "检查容器状态..."
if [ "$MULTI_MODE" = true ]; then
    # 多实例模式：清理所有相关容器
    for port in "${PORTS[@]}"; do
        container_name="${CONTAINER_NAME_PREFIX}-${port}"
        if docker ps -a | grep -q "${container_name}"; then
            if docker ps | grep -q "${container_name}"; then
                print_warning "容器 ${container_name} 正在运行，正在停止..."
                docker stop ${container_name} &> /dev/null || true
            fi
            print_info "删除旧容器 ${container_name}..."
            docker rm ${container_name} &> /dev/null || true
        fi
    done
    print_success "旧容器清理完成"
else
    # 单实例模式
    container_name="${CONTAINER_NAME_PREFIX}-${DEFAULT_PORT}"
    if docker ps -a | grep -q "${container_name}"; then
        if docker ps | grep -q "${container_name}"; then
            print_warning "容器 ${container_name} 正在运行，正在停止..."
            docker stop ${container_name}
        fi
        print_info "删除旧容器..."
        docker rm ${container_name}
        print_success "旧容器已删除"
    fi
fi

# 创建模型目录（如果不存在）
MODEL_PATH="$PROJECT_ROOT/pretrained_models"
if [ ! -d "$MODEL_PATH" ]; then
    print_warning "模型目录不存在，创建目录: $MODEL_PATH"
    mkdir -p "$MODEL_PATH"
    print_info "请将模型文件下载到 $MODEL_PATH 目录"
fi

# 运行容器
SUCCESS_COUNT=0
FAILED_PORTS=()

print_info "开始启动 ${#PORTS[@]} 个容器实例..."

for port in "${PORTS[@]}"; do
    container_name="${CONTAINER_NAME_PREFIX}-${port}"
    print_info "启动容器 ${container_name} (端口 ${port})..."
    
    # 启动容器（启用端口映射和GPU支持）
    run_output=$(docker run -d \
        ${GPU_FLAG} \
        -p ${port}:${port} \
        -v "$MODEL_PATH:/workspace/CosyVoice/pretrained_models" \
        --name ${container_name} \
        --restart unless-stopped \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        bash -c "source /opt/conda/etc/profile.d/conda.sh && \
                 conda activate cosyvoice && \
                 cd /workspace/CosyVoice && \
                 if [ -d 'pretrained_models/CosyVoice-ttsfrd' ] && [ -f 'pretrained_models/CosyVoice-ttsfrd/resource.zip' ] && [ ! -f 'pretrained_models/CosyVoice-ttsfrd/ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl' ]; then \
                     echo '安装 ttsfrd...' && \
                     cd pretrained_models/CosyVoice-ttsfrd && \
                     unzip -q resource.zip -d . && \
                     pip install ttsfrd_dependency-0.1-py3-none-any.whl && \
                     pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl && \
                     cd /workspace/CosyVoice; \
                 fi && \
                 python webui.py \
                     --port ${port} \
                     --model_dir ${MODEL_DIR}" 2>&1)
    run_exit_code=$?
    
    if [ $run_exit_code -eq 0 ]; then
        # 等待一下，然后检查容器是否还在运行
        sleep 2
        if docker ps | grep -q "${container_name}"; then
            print_success "容器 ${container_name} 启动成功"
            ((SUCCESS_COUNT++))
        else
            print_error "容器 ${container_name} 启动失败（容器已退出）"
            FAILED_PORTS+=($port)
            # 获取错误日志
            print_info "容器 ${container_name} 的错误日志:"
            docker logs ${container_name} 2>&1 | tail -20
            echo ""
        fi
    else
        print_error "容器 ${container_name} 启动失败"
        print_info "Docker 运行错误:"
        echo "$run_output"
        FAILED_PORTS+=($port)
        # 如果容器被创建但启动失败，尝试获取日志
        if docker ps -a | grep -q "${container_name}"; then
            print_info "容器 ${container_name} 的错误日志:"
            docker logs ${container_name} 2>&1 | tail -20
            echo ""
        fi
    fi
done

echo ""
echo "=========================================="
print_success "启动完成！成功: ${SUCCESS_COUNT}/${#PORTS[@]} 个容器"
echo ""

if [ ${#FAILED_PORTS[@]} -gt 0 ]; then
    print_warning "以下端口启动失败: ${FAILED_PORTS[*]}"
    echo ""
fi

print_info "所有服务访问地址:"
for port in "${PORTS[@]}"; do
    container_name="${CONTAINER_NAME_PREFIX}-${port}"
    if docker ps | grep -q "${container_name}"; then
        echo "  ✓ http://localhost:${port} (${container_name})"
    else
        echo "  ✗ http://localhost:${port} (${container_name}) - 未运行"
    fi
done

echo ""
print_info "常用命令:"
echo "  查看所有容器: docker ps | grep ${CONTAINER_NAME_PREFIX}"
echo "  查看日志:     docker logs -f ${CONTAINER_NAME_PREFIX}-<端口>"
echo "  停止所有:     ./stop_webui.sh --all"
echo "  停止单个:     docker stop ${CONTAINER_NAME_PREFIX}-<端口>"
echo "=========================================="
echo ""

# 等待几秒后显示第一个容器的日志
if [ ${SUCCESS_COUNT} -gt 0 ]; then
    sleep 3
    first_port=${PORTS[0]}
    first_container="${CONTAINER_NAME_PREFIX}-${first_port}"
    print_info "显示容器 ${first_container} 的日志（按 Ctrl+C 退出日志查看，容器将继续运行）..."
    echo ""
    docker logs -f ${first_container}
fi

