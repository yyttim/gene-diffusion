#!/bin/bash
# CUDA库修复脚本

echo "=== CUDA库查找和修复 ==="
echo ""

# 1. 搜索libcurand.so
echo "1. 搜索libcurand.so.10..."
find /usr -name "libcurand.so*" 2>/dev/null | head -10
find /opt -name "libcurand.so*" 2>/dev/null | head -10
find /soft -name "libcurand.so*" 2>/dev/null | head -10
find /home/share/huadjyin/home/lishaoshuai/miniconda3 -name "libcurand.so*" 2>/dev/null | head -10

echo ""
echo "2. 搜索libcublas.so..."
find /usr -name "libcublas.so*" 2>/dev/null | head -10
find /opt -name "libcublas.so*" 2>/dev/null | head -10
find /soft -name "libcublas.so*" 2>/dev/null | head -10
find /home/share/huadjyin/home/lishaoshuai/miniconda3 -name "libcublas.so*" 2>/dev/null | head -10

echo ""
echo "3. 当前LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH | tr ':' '\n'

echo ""
echo "4. PyTorch CUDA测试:"
source /home/share/huadjyin/home/lishaoshuai/miniconda3/bin/activate gene-diffusion
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
" 2>&1

echo ""
echo "5. 建议的修复方法:"
echo "如果找到了CUDA库，请将其路径添加到LD_LIBRARY_PATH:"
echo "export LD_LIBRARY_PATH=/path/to/cuda/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "或者重新安装PyTorch的CPU版本:"
echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu" 