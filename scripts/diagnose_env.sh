#!/bin/bash
# GeneDiffusion环境诊断脚本

echo "=== GeneDiffusion 环境诊断 ==="
echo "运行时间: $(date)"
echo ""

# 检查当前环境
echo "1. 当前环境信息:"
echo "   Shell: $SHELL"
echo "   Python: $(which python)"
echo "   Conda环境: $CONDA_DEFAULT_ENV"
echo ""

# 检查CUDA相关
echo "2. CUDA环境:"
echo "   CUDA_HOME: $CUDA_HOME"
echo "   LD_LIBRARY_PATH: $LD_LIBRARY_PATH" | cut -c1-80
echo ""

# 检查系统信息
echo "3. 系统信息:"
echo "   内核: $(uname -r)"
echo "   架构: $(uname -m)"
echo "   GLIBC版本: $(ldd --version | head -n1)"
echo ""

# 检查Python包
echo "4. Python包版本:"
python -c "
import sys
print(f'   Python: {sys.version}')
try:
    import numpy
    print(f'   NumPy: {numpy.__version__}')
except:
    print('   NumPy: 未安装')
try:
    import pandas
    print(f'   Pandas: {pandas.__version__}')
except:
    print('   Pandas: 未安装')
"

# 尝试导入torch（可能会失败）
echo ""
echo "5. 尝试导入PyTorch:"
python -c "
try:
    import torch
    print(f'   PyTorch版本: {torch.__version__}')
    print(f'   CUDA编译版本: {torch.version.cuda}')
    print(f'   CUDA可用: {torch.cuda.is_available()}')
except Exception as e:
    print(f'   错误: {type(e).__name__}: {str(e)[:100]}')
"

echo ""
echo "6. 建议的修复方案:"
echo "   a) 重新安装PyTorch:"
echo "      conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia"
echo ""
echo "   b) 或创建新环境:"
echo "      conda create -n gene-diffusion-v2 python=3.9"
echo "      conda activate gene-diffusion-v2"
echo "      conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia"
echo ""
echo "=== 诊断完成 ===" 