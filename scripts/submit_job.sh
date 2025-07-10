#!/bin/bash
# GeneDiffusion训练任务提交脚本

# 设置项目和日志路径
PROJECT_DIR="/home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion"
LOG_DIR="${PROJECT_DIR}/logs"

# 创建日志目录
mkdir -p $LOG_DIR

# 提交单节点测试任务
echo "提交单节点测试任务..."
dsub -n gene_diffusion_test \
     -N 1 \
     -A root.project.P24Z28400N0259_tmp \
     -R "cpu=8;gpu=2;mem=40000" \
     -oo ${LOG_DIR}/test_task.%J.out \
     -eo ${LOG_DIR}/test_task.%J.err \
     -s ${PROJECT_DIR}/scripts/run_training.sh

echo "任务已提交！"
echo "查看任务状态: djob"
echo "查看任务详情: djob -x -W JOB_ID"
echo "查看日志: tail -f ${LOG_DIR}/test_task.*.out" 