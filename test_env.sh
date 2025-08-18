#!/bin/bash
# 测试环境配置脚本

echo "=== 环境测试 ==="
echo ""

# 1. 测试module系统
echo "1. 检查module系统："
if [ -f /home/HPCBase/tools/module-5.2.0/init/profile.sh ]; then
    source /home/HPCBase/tools/module-5.2.0/init/profile.sh
    module use /home/HPCBase/modulefiles/
    echo "   ✓ Module系统已初始化"
else
    echo "   ✗ Module系统未找到"
fi

if command -v module &> /dev/null; then
    echo "   ✓ module命令可用"
else
    echo "   ✗ module命令不可用"
fi

# 2. 测试CUDA
echo ""
echo "2. 检查CUDA环境："
if [ -d /soft/cuda/11.8 ]; then
    echo "   ✓ CUDA 11.8 found at /soft/cuda/11.8"
    export CUDA_HOME=/soft/cuda/11.8
elif [ -d /usr/local/cuda-11.8 ]; then
    echo "   ✓ CUDA 11.8 found at /usr/local/cuda-11.8"
    export CUDA_HOME=/usr/local/cuda-11.8
elif [ -d /usr/local/cuda ]; then
    echo "   ✓ CUDA found at /usr/local/cuda"
    export CUDA_HOME=/usr/local/cuda
else
    echo "   ✗ CUDA not found"
fi

# 3. 测试conda环境
echo ""
echo "3. 检查conda环境："
if [ -f /home/share/huadjyin/home/lishaoshuai/miniconda3/bin/activate ]; then
    echo "   ✓ Conda found"
    source /home/share/huadjyin/home/lishaoshuai/miniconda3/bin/activate gene-diffusion
    if [ "$CONDA_DEFAULT_ENV" = "gene-diffusion" ]; then
        echo "   ✓ gene-diffusion环境已激活"
    else
        echo "   ✗ gene-diffusion环境激活失败"
    fi
else
    echo "   ✗ Conda not found"
fi

# 4. 测试Python和PyTorch
echo ""
echo "4. 检查Python和PyTorch："
python -c "import sys; print(f'   Python: {sys.version.split()[0]}')" 2>/dev/null || echo "   ✗ Python not working"
python -c "import torch; print(f'   PyTorch: {torch.__version__}')" 2>/dev/null || echo "   ✗ PyTorch not found"
python -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}')" 2>/dev/null

# 5. 测试GPU
echo ""
echo "5. 检查GPU设备："
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || echo "   ✗ GPU query failed"
else
    echo "   ✗ nvidia-smi not found"
fi

echo ""
echo "=== 测试完成 ===" 