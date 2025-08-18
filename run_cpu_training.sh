#!/bin/bash
# CPU模式训练脚本 - 当GPU不可用时的备选方案

echo "=== GeneDiffusion CPU模式训练 ==="
echo "⚠️  注意：CPU模式速度会很慢，建议仅用于测试"
echo ""

# 设置环境
PROJECT_ROOT="/home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion"
OUTPUT_BASE="${PROJECT_ROOT}/outputs/train_cpu"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
CONFIG_FILE="${PROJECT_ROOT}/configs/train_4gpu_600m.yaml"

mkdir -p ${OUTPUT_DIR}

# 禁用CUDA
export CUDA_VISIBLE_DEVICES=""

# 激活环境
source /home/share/huadjyin/home/lishaoshuai/miniconda3/bin/activate gene-diffusion

cd ${PROJECT_ROOT}

echo "开始CPU训练..."
echo "输出目录: ${OUTPUT_DIR}"
echo ""

# 单进程CPU训练
python src/train.py \
    --config ${CONFIG_FILE} \
    --output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/train.log

echo ""
echo "训练完成或已启动" 