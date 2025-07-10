#!/bin/bash
# GeneDiffusion环境设置脚本 - 用于超算服务器

echo "===== GeneDiffusion 环境设置开始 ====="

# 设置项目路径
PROJECT_DIR="/home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion"
cd $PROJECT_DIR

# 加载module环境
source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module purge
module use /home/HPCBase/modulefiles/

# 加载必要的模块
module load compilers/gcc/12.3.0
module load compilers/cuda/11.8.0
module load libs/cudnn/8.6.0_cuda11

# 设置conda环境名称
ENV_NAME="gene-diffusion"

# 检查是否已安装conda
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda命令"
    echo "尝试加载Anaconda模块..."
    module load tools/anaconda3
fi

# 创建新的conda环境
echo "创建conda环境: $ENV_NAME"
conda create -n $ENV_NAME python=3.9 -y

# 激活环境
source activate $ENV_NAME

# 安装PyTorch (CUDA 11.8版本)
echo "安装PyTorch (CUDA 11.8)..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
echo "安装其他依赖..."
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scikit-learn>=0.24.0
pip install scipy>=1.7.0
pip install einops>=0.6.0
pip install transformers>=4.30.0
pip install biopython>=1.79
pip install pysam>=0.19.0
pip install h5py>=3.7.0
pip install tqdm>=4.62.0
pip install tensorboard>=2.9.0
pip install wandb>=0.12.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install pyyaml>=6.0
pip install jsonlines>=3.0.0
pip install click>=8.0.0

# 验证安装
echo "验证PyTorch安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
echo ""
echo "注意：CUDA支持需要在计算节点上才能正常工作。"
echo "在编译节点上CUDA可能显示为不可用，这是正常的。"

echo "===== 环境设置完成 ====="
echo "使用以下命令激活环境："
echo "source activate $ENV_NAME" 