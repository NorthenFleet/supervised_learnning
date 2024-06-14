# 使用官方的 Miniconda 镜像
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /sea_war

# 创建新的 Conda 环境并安装 Python 3.10
RUN conda create -y -n py310 python=3.10 && \
    echo "source activate py310" > ~/.bashrc && \
    conda clean -afy

# 激活 Conda 环境并安装 PyTorch
RUN echo "source activate py310" > ~/.bashrc && \
    /opt/conda/envs/py310/bin/pip install torch

# 设置容器启动时自动激活 Conda 环境
CMD ["/bin/bash", "-c", "source activate py310 && /bin/bash"]