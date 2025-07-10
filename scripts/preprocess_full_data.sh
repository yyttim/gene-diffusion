#!/bin/bash
# 全量数据预处理脚本 - 后台运行版本

# 项目路径
PROJECT_DIR="/home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion"
cd $PROJECT_DIR

# 创建输出目录
OUTPUT_DIR="datasets/preprocessed"
mkdir -p $OUTPUT_DIR

# 创建日志目录
LOG_DIR="logs/preprocessing"
mkdir -p $LOG_DIR

# 设置日志文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/preprocess_full_${TIMESTAMP}.log"
ERROR_FILE="$LOG_DIR/preprocess_full_${TIMESTAMP}.err"

# 检查是否已经在处理
LOCK_FILE="$OUTPUT_DIR/.preprocessing.lock"
if [ -f "$LOCK_FILE" ]; then
    echo "错误: 预处理已经在运行中！"
    echo "如果确定没有在运行，请删除锁文件: rm $LOCK_FILE"
    exit 1
fi

# 创建锁文件
echo "PID: $$" > $LOCK_FILE
echo "开始时间: $(date)" >> $LOCK_FILE

# 记录开始信息
echo "========================================" | tee -a $LOG_FILE
echo "GeneDiffusion 全量数据预处理" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "开始时间: $(date)" | tee -a $LOG_FILE
echo "输出目录: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "日志文件: $LOG_FILE" | tee -a $LOG_FILE
echo "错误文件: $ERROR_FILE" | tee -a $LOG_FILE
echo "进程PID: $$" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# 设置环境变量
export CUDA_VISIBLE_DEVICES=""  # 不使用GPU
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1  # 实时输出Python日志

# 激活conda环境
source /home/share/huadjyin/home/lishaoshuai/miniconda3/etc/profile.d/conda.sh
conda activate gene-diffusion

# 清理函数
cleanup() {
    echo "清理中..." | tee -a $LOG_FILE
    rm -f $LOCK_FILE
    echo "完成时间: $(date)" | tee -a $LOG_FILE
}

# 设置退出时清理
trap cleanup EXIT

# 运行预处理
echo "开始预处理..." | tee -a $LOG_FILE

python preprocess_data.py \
    --data_dir datasets/genomics_seq \
    --output_dir $OUTPUT_DIR \
    --seq_length 10000 \
    --overlap 0.8 \
    --max_sequences_per_species 15000 \
    --num_workers 8 \
    >> $LOG_FILE 2>> $ERROR_FILE

# 检查退出状态
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================" | tee -a $LOG_FILE
    echo "预处理成功完成！" | tee -a $LOG_FILE
    echo "完成时间: $(date)" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    
    # 显示统计信息
    if [ -f "$OUTPUT_DIR/dataset_stats.json" ]; then
        echo "" | tee -a $LOG_FILE
        echo "数据集统计信息:" | tee -a $LOG_FILE
        python -m json.tool "$OUTPUT_DIR/dataset_stats.json" | tee -a $LOG_FILE
    fi
else
    echo "========================================" | tee -a $LOG_FILE
    echo "预处理失败！退出码: $EXIT_CODE" | tee -a $LOG_FILE
    echo "请查看错误日志: $ERROR_FILE" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
fi

# 清理锁文件
rm -f $LOCK_FILE 