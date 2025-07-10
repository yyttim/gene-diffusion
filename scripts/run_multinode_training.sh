#!/bin/bash
#DSUB -n gene_diffusion_multinode
#DSUB -N 2
#DSUB -A root.project.P24Z28400N0259_tmp
#DSUB -R "cpu=8;gpu=4;mem=40000"
#DSUB -o multinode.%J.out
#DSUB -e multinode.%J.err

echo "===== GeneDiffusion 多节点训练任务开始 ====="
echo "任务ID: $DONAU_JOB_ID"
echo "主节点: $(hostname)"
echo "时间: $(date)"

# 设置项目路径
PROJECT_DIR="/home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion"
cd $PROJECT_DIR

# 加载必要的模块
source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module purge
module use /home/HPCBase/modulefiles/
module load compilers/gcc/12.3.0
module load cuda/11.8
module load tools/anaconda3
module load mpi/hmpi

# 激活conda环境
source activate gene-diffusion

# 设置分布式训练环境变量
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$DONAU_NNODES
export RANK=$DONAU_RANK
export LOCAL_RANK=$DONAU_LOCAL_RANK

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

# 打印节点信息
echo "===== 节点信息 ====="
echo "总节点数: $WORLD_SIZE"
echo "当前节点排名: $RANK"
echo "本地排名: $LOCAL_RANK"
echo "主节点地址: $MASTER_ADDR:$MASTER_PORT"

# 设置数据路径
DATA_DIR="datasets/genomics_seq"

# 使用torchrun启动分布式训练
echo "===== 开始分布式训练 ====="
torchrun \
    --nproc_per_node=4 \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src.train \
    --config configs/train_config.json \
    --data_dir $DATA_DIR \
    --output_dir ./outputs_multinode \
    --num_epochs 50 \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --fp16 \
    --save_steps 5000 \
    --logging_steps 100 \
    --warmup_steps 2000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_total_limit 5 \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --greater_is_better False \
    --ddp_backend nccl \
    --dataloader_num_workers 4

echo "===== 多节点训练任务完成 ====="
echo "结束时间: $(date)" 