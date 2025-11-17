FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEN=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update -y --fix-missing
RUN apt-get install -y git build-essential curl wget ffmpeg unzip git git-lfs sox libsox-dev && \
    apt-get clean && \
    git lfs install

# ==================================================================
# conda install and conda forge channel as default
# ------------------------------------------------------------------
# Install miniforge
# 使用清华大学镜像源加速下载
RUN wget --timeout=60 --tries=3 --progress=bar:force:noscroll \
    https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /opt/nvidia/entrypoint.d/100.conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV}" >> /opt/nvidia/entrypoint.d/110.conda_default_env.sh && \
    echo "conda activate ${VENV}" >> $HOME/.bashrc

ENV PATH=/opt/conda/bin:$PATH

# 配置 conda 使用清华大学镜像源加速，移除默认频道避免 TOS 问题
RUN conda config --remove channels defaults || true && \
    conda config --remove channels https://repo.anaconda.com/pkgs/main || true && \
    conda config --remove channels https://repo.anaconda.com/pkgs/r || true && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda config --set show_channel_urls yes
# ------------------------------------------------------------------
# ~conda
# ==================================================================

# 使用 --override-channels 强制只使用 conda-forge，避免 TOS 问题
RUN conda create -y -n ${VENV} --override-channels -c conda-forge python=3.10
ENV CONDA_DEFAULT_ENV=${VENV}
ENV PATH=/opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH

WORKDIR /workspace

ENV PYTHONPATH=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS

RUN git clone --recursive https://github.com/aucho/CosyVoice.git

RUN conda activate ${VENV} && conda install -y --override-channels -c conda-forge pynini==2.1.5
RUN conda activate ${VENV} && cd CosyVoice && \
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# 安装 ttsfrd（可选，用于更好的文本规范化性能）
# 注意：需要确保 pretrained_models/CosyVoice-ttsfrd 目录已存在且包含相应文件
RUN if [ -d "/workspace/CosyVoice/pretrained_models/CosyVoice-ttsfrd" ] && \
        [ -f "/workspace/CosyVoice/pretrained_models/CosyVoice-ttsfrd/resource.zip" ]; then \
        conda activate ${VENV} && \
        cd /workspace/CosyVoice/pretrained_models/CosyVoice-ttsfrd && \
        unzip -q resource.zip -d . && \
        pip install ttsfrd_dependency-0.1-py3-none-any.whl && \
        pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl; \
    else \
        echo "ttsfrd 模型文件不存在，跳过安装（将使用 wetext 作为默认）"; \
    fi

WORKDIR /workspace/CosyVoice
