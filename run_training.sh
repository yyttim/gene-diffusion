#!/bin/bash
# 训练启动包装脚本 - 确保环境正确设置

set -e

NODE_NAME=${1:-cyclone001-agent-149}

echo "=== GeneDiffusion训练启动 ==="
echo "目标节点: $NODE_NAME"
echo ""

# 1. 设置基础环境变量
echo "1. 设置环境变量..."
export CUDA_HOME=/soft/cuda/11.8
if [ ! -d "$CUDA_HOME" ]; then
    export CUDA_HOME=/usr/local/cuda-11.8
    if [ ! -d "$CUDA_HOME" ]; then
        export CUDA_HOME=/usr/local/cuda
    fi
fi

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:/lib64:$LD_LIBRARY_PATH

# 添加conda库路径
CONDA_ENV=/home/share/huadjyin/home/lishaoshuai/miniconda3/envs/gene-diffusion
export LD_LIBRARY_PATH=$CONDA_ENV/lib:$LD_LIBRARY_PATH

echo "   CUDA_HOME: $CUDA_HOME"
echo "   LD_LIBRARY_PATH已更新"

# 2. 激活conda环境
echo ""
echo "2. 激活conda环境..."
source /home/share/huadjyin/home/lishaoshuai/miniconda3/bin/activate gene-diffusion
echo "   环境: $CONDA_DEFAULT_ENV"

# 3. 测试PyTorch
echo ""
echo "3. 测试PyTorch环境..."
python -c "
import torch
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA可用: {torch.cuda.is_available()}')
" || {
    echo "   ⚠️ PyTorch测试失败"
    echo "   尝试使用CPU版本继续..."
}

# 4. 启动训练
echo ""
echo "4. 启动训练..."
echo "----------------------------------------"
./submit_600m_efficient.sh $NODE_NAME

echo ""
echo "=== 训练任务已提交 ===" 