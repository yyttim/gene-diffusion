#!/bin/bash
# GeneDiffusion多节点训练任务提交脚本

# 设置项目和日志路径
PROJECT_DIR="/home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion"
LOG_DIR="${PROJECT_DIR}/logs"

# 创建日志目录
mkdir -p $LOG_DIR

# 提交多节点训练任务（使用2个节点，每个节点4个GPU）
echo "提交多节点训练任务..."
dsub -n gene_diffusion_multinode \
     -pn 'cyclone001-agent-12 cyclone001-agent-8' \
     -N 2 \
     -A root.project.P24Z28400N0259_tmp \
     -R "cpu=8;gpu=4;mem=40000" \
     -oo ${LOG_DIR}/multinode_task.%J.out \
     -eo ${LOG_DIR}/multinode_task.%J.err \
     -s ${PROJECT_DIR}/scripts/run_multinode_training.sh

echo "多节点任务已提交！"
echo "注意：-N 2 表示使用2个节点"
echo "查看任务状态: djob"
echo "查看详细信息: djob -x -W JOB_ID"
echo "查看专用池资源: dnode -W -g tmp_P24Z28400N0259" 