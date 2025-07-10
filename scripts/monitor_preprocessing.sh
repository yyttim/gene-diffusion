#!/bin/bash
# 监控预处理进度

PROJECT_DIR="/home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion"
cd $PROJECT_DIR

OUTPUT_DIR="datasets/preprocessed"
LOG_DIR="logs/preprocessing"

# 检查是否在运行
LOCK_FILE="$OUTPUT_DIR/.preprocessing.lock"
if [ -f "$LOCK_FILE" ]; then
    echo "预处理正在运行..."
    cat $LOCK_FILE
    echo ""
else
    echo "预处理未在运行"
fi

# 查找最新的日志文件
LATEST_LOG=$(ls -t $LOG_DIR/preprocess_full_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "未找到日志文件"
    exit 1
fi

echo "日志文件: $LATEST_LOG"
echo "=========================================="

# 显示已处理的物种数
echo "处理进度:"
if [ -d "$OUTPUT_DIR" ]; then
    NUM_SPECIES=$(ls $OUTPUT_DIR/*.h5 2>/dev/null | wc -l)
    echo "已处理物种数: $NUM_SPECIES / 230"
    
    # 显示最近处理的物种
    echo ""
    echo "最近处理的物种:"
    ls -lt $OUTPUT_DIR/*.h5 2>/dev/null | head -5 | awk '{print $9}' | xargs -I {} basename {} .h5
fi

echo ""
echo "=========================================="
echo "最新日志（最后20行）:"
tail -20 $LATEST_LOG

# 检查错误
LATEST_ERR=$(echo $LATEST_LOG | sed 's/.log$/.err/')
if [ -f "$LATEST_ERR" ] && [ -s "$LATEST_ERR" ]; then
    echo ""
    echo "=========================================="
    echo "错误信息:"
    tail -10 $LATEST_ERR
fi

# 估算剩余时间
if [ -f "$LOCK_FILE" ]; then
    START_TIME=$(grep "开始时间:" $LOCK_FILE | cut -d' ' -f3-)
    if [ ! -z "$START_TIME" ]; then
        START_EPOCH=$(date -d "$START_TIME" +%s 2>/dev/null || echo 0)
        NOW_EPOCH=$(date +%s)
        if [ $START_EPOCH -ne 0 ] && [ $NUM_SPECIES -gt 0 ]; then
            ELAPSED=$((NOW_EPOCH - START_EPOCH))
            RATE=$((ELAPSED / NUM_SPECIES))
            REMAINING=$((RATE * (230 - NUM_SPECIES)))
            echo ""
            echo "估算信息:"
            echo "  已运行: $((ELAPSED / 3600))小时$((ELAPSED % 3600 / 60))分钟"
            echo "  平均速度: $RATE秒/物种"
            echo "  预计剩余: $((REMAINING / 3600))小时$((REMAINING % 3600 / 60))分钟"
        fi
    fi
fi 