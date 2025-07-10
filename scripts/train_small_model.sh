#!/bin/bash
# 小型模型训练脚本 - 使用全量数据但参数量小

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4个GPU
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 激活环境
source ~/.bashrc
conda activate scgpt

# 项目路径
PROJECT_DIR="/home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion"
cd $PROJECT_DIR

# 创建输出目录
OUTPUT_DIR="outputs/small_model_test"
mkdir -p $OUTPUT_DIR

# 设置日志
LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a $LOG_FILE
echo "GeneDiffusion 小型模型训练" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "开始时间: $(date)" | tee -a $LOG_FILE
echo "配置文件: configs/train_config_small.yaml" | tee -a $LOG_FILE
echo "输出目录: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# 使用torchrun进行分布式训练
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    src/train.py \
    --config configs/train_config_small.yaml \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 50 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 512 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --warmup_steps 5000 \
    --save_steps 2000 \
    --eval_steps 500 \
    --logging_steps 50 \
    --dataloader_num_workers 16 \
    --fp16 \
    --run_name "small_model_full_data" \
    2>&1 | tee -a $LOG_FILE

echo "========================================" | tee -a $LOG_FILE
echo "训练完成时间: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE 