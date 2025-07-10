#!/bin/bash
# 快速检查预处理状态

echo "GeneDiffusion 数据预处理状态"
echo "==============================="
echo ""

# 检查锁文件
LOCK_FILE="datasets/preprocessed/.preprocessing.lock"
if [ -f "$LOCK_FILE" ]; then
    echo "✓ 预处理正在运行"
    cat $LOCK_FILE
    echo ""
    
    # 检查进程
    PID=$(grep "PID:" $LOCK_FILE | cut -d' ' -f2)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ 进程 $PID 活跃"
    else
        echo "✗ 进程 $PID 已停止"
    fi
else
    echo "✗ 预处理未在运行"
fi

echo ""
echo "数据统计:"
echo "---------"

# 统计已处理文件
if [ -d "datasets/preprocessed" ]; then
    NUM_H5=$(ls datasets/preprocessed/*.h5 2>/dev/null | wc -l)
    echo "已处理物种: $NUM_H5 / 234"
    
    if [ $NUM_H5 -gt 0 ]; then
        TOTAL_SIZE=$(du -sh datasets/preprocessed/*.h5 | tail -1 | cut -f1)
        echo "数据大小: $TOTAL_SIZE"
    fi
fi

echo ""
echo "查看详细进度: bash scripts/monitor_preprocessing.sh"
echo "查看实时日志: tail -f logs/preprocessing/preprocess_full_*.log" 