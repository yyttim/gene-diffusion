#!/bin/bash
# 数据预处理脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=""  # 不使用GPU
export OMP_NUM_THREADS=16

# 激活环境
conda activate gene-diffusion

# 项目路径
PROJECT_DIR="/home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion"
cd $PROJECT_DIR

# 创建输出目录
OUTPUT_DIR="datasets/preprocessed"
mkdir -p $OUTPUT_DIR

# 设置日志
LOG_FILE="$OUTPUT_DIR/preprocess_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a $LOG_FILE
echo "GeneDiffusion 数据预处理" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "开始时间: $(date)" | tee -a $LOG_FILE
echo "输出目录: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# 运行预处理
python preprocess_data.py \
    --data_dir datasets/genomics_seq \
    --output_dir $OUTPUT_DIR \
    --seq_length 10000 \
    --overlap 0.8 \
    --max_sequences_per_species 15000 \
    --num_workers 8 \
    2>&1 | tee -a $LOG_FILE

echo "========================================" | tee -a $LOG_FILE
echo "预处理完成时间: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# 显示统计信息
if [ -f "$OUTPUT_DIR/dataset_stats.json" ]; then
    echo "数据集统计信息:" | tee -a $LOG_FILE
    python -m json.tool "$OUTPUT_DIR/dataset_stats.json" | tee -a $LOG_FILE
fi 