#!/bin/bash
#DSUB -n gene_diffusion_train
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp
#DSUB -R "cpu=8;gpu=2;mem=40000"
#DSUB -o stdout.%J
#DSUB -e stderr.%J

echo "===== GeneDiffusion 训练任务开始 ====="
echo "任务ID: $DONAU_JOB_ID"
echo "节点: $(hostname)"
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

# 激活conda环境
source activate gene-diffusion

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8

# 打印环境信息
echo "===== 环境信息 ====="
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
nvidia-smi

# 设置数据路径
DATA_DIR="/home/share/huadjyin/home/liwenbo/projects/geno/data/genomics_seq"

# 开始训练（使用小型配置进行测试）
echo "===== 开始训练 ====="
python -m src.train \
    --config configs/train_config_small.json \
    --data_dir $DATA_DIR \
    --output_dir ./outputs \
    --num_epochs 10 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --fp16 \
    --save_steps 1000 \
    --logging_steps 100 \
    --warmup_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --greater_is_better False

echo "===== 训练任务完成 ====="
echo "结束时间: $(date)" 