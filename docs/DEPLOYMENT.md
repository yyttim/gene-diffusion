# GeneDiffusion 超算部署指南

## 快速开始

### 1. 首次部署

```bash
# 登录编译测试节点
ssh -p 6122 username@27.18.114.42

# 进入项目目录
cd /home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion

# 运行环境设置（约需10-15分钟）
bash scripts/setup_env.sh
```

### 2. 测试运行

```bash
# 激活环境
source activate gene-diffusion

# 提交测试任务（单节点，2GPU）
bash scripts/submit_job.sh
```

### 3. 生产运行

```bash
# 提交多节点训练（2节点，8GPU）
bash scripts/submit_multinode_job.sh
```

## 环境信息

- **CUDA版本**: 11.8
- **Python版本**: 3.9
- **PyTorch版本**: 2.0.1+cu118
- **项目路径**: `/home/share/huadjyin/home/lishaoshuai/yyt/gene-diffusion`
- **数据路径**: `/home/share/huadjyin/home/liwenbo/projects/geno/data/genomics_seq`

## 资源配置

### 专用池 (P24Z28400N0259_tmp)
- 用于正式训练任务
- 需指定节点避免冲突

### 共享池 (P24Z28400N0259)
- 用于测试和小任务
- 自动分配节点

## 任务管理

### 交互式管理工具
```bash
bash scripts/job_manager.sh
```

### 常用命令速查
```bash
djob                    # 查看当前任务
djob -x -W <JOB_ID>    # 查看任务详情
djob -T <JOB_ID>       # 终止任务
dnode -W -g tmp_P24Z28400N0259  # 查看专用池资源
```

## 文件说明

### 脚本文件
- `setup_env.sh`: 环境初始化脚本
- `submit_job.sh`: 单节点任务提交
- `submit_multinode_job.sh`: 多节点任务提交
- `run_training.sh`: 单节点训练执行脚本
- `run_multinode_training.sh`: 多节点分布式训练脚本
- `job_manager.sh`: 任务管理交互工具

### 配置文件
- `configs/train_config_small.json`: 测试配置（512d, 12层）
- `configs/train_config.json`: 生产配置（1280d, 48层）

## 训练参数调整

编辑 `run_training.sh` 或 `run_multinode_training.sh` 中的参数：

```bash
--num_epochs 50          # 训练轮数
--batch_size 16          # 批次大小
--gradient_accumulation_steps 8  # 梯度累积步数
--save_steps 5000        # 保存间隔
--fp16                   # 使用混合精度训练
```

## 故障处理

### 1. 模块加载失败
```bash
source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
```

### 2. Conda环境问题
```bash
module load tools/anaconda3
conda activate gene-diffusion
```

### 3. 权限错误
```bash
chmod 600 ~/.user_cre
chmod +x scripts/*.sh
```

### 4. GPU不可用
```bash
module load cuda/11.8
nvidia-smi
```

## 监控和日志

- 标准输出日志: `logs/test_task.<JOB_ID>.out`
- 错误日志: `logs/test_task.<JOB_ID>.err`
- TensorBoard: `tensorboard --logdir outputs/`

## 联系支持

如遇到问题，请检查：
1. 日志文件中的错误信息
2. 节点资源是否充足
3. CUDA环境是否正确加载 