# GeneDiffusion 大规模训练指南

本文档说明如何使用GeneDiffusion训练生物通用大模型。

## 快速开始

### 1. 数据准备
数据已经准备好，包含230个哺乳动物物种的全基因组数据：
- **数据量**: 597 GB基因组序列
- **可生成样本**: ~297M（使用10kb片段，80%重叠）
- **覆盖范围**: 全基因组（编码区、非编码区、调控区等）

### 2. 开始训练

```bash
# 单GPU训练（用于测试）
python src/train.py --config configs/train_config_large.yaml

# 分布式训练（推荐）
torchrun --nproc_per_node=8 src/train.py --config configs/train_config_large.yaml
```

### 3. 监控训练

```bash
# 使用WandB监控
# 训练会自动记录到 genediffusion-large 项目
```

## 关键参数

### 数据参数
- **序列长度**: 10,000 bp (10kb)
- **Token长度**: ~1,666 个6-mer tokens
- **重叠率**: 80%（步长2kb）
- **批次大小**: 32（可根据GPU内存调整）

### 模型参数
- **模型大小**: 768维，12层（~110M参数）
- **词表大小**: 4,099（4096个6-mer + 3个特殊token）
- **物种数**: 230个哺乳动物物种
- **注释类型**: 5种（exon, cds, lnc_rna, gene, trna）

## 数据加载示例

```python
from src.data_processor import create_data_loader

# 创建训练数据加载器
train_loader = create_data_loader(
    data_dir="/home/share/huadjyin/home/liwenbo/projects/geno/data/genomics_seq",
    batch_size=32,
    seq_length=10000,      # 10kb
    overlap=0.8,           # 80%重叠
    shuffle=True,
    num_workers=8
)

print(f"总批次数: {len(train_loader)}")  # 约9.3M
```

## 硬件要求

### 最低配置
- GPU: 1×V100 (32GB)
- 内存: 64GB
- 存储: 2TB SSD

### 推荐配置
- GPU: 8×A100 (40GB)
- 内存: 256GB
- 存储: 4TB NVMe SSD

## 训练时间估算

| 配置 | 每个Epoch时间 | 10 Epochs |
|------|--------------|-----------|
| 1×V100 | ~20天 | ~200天 |
| 1×A100 | ~10天 | ~100天 |
| 8×A100 | ~2.5天 | ~25天 |

## 优化建议

### 1. 内存优化
```python
# 使用梯度累积减少批次大小
gradient_accumulation_steps = 4
effective_batch_size = batch_size * gradient_accumulation_steps
```

### 2. 数据并行
```python
# 使用FSDP进行大模型训练
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
```

### 3. 混合精度训练
```python
# 使用自动混合精度加速训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

## 检查点管理

训练会自动保存检查点到 `outputs/genediffusion_large/`：
- 每10,000步保存一次
- 保留最佳验证损失的模型
- 支持断点续训

## 常见问题

### Q: 内存不足怎么办？
A: 尝试以下方法：
1. 减少批次大小
2. 使用梯度累积
3. 减少序列长度到5kb
4. 使用 `cache_sequences=False`

### Q: 训练太慢怎么办？
A: 优化建议：
1. 增加数据加载进程数
2. 使用SSD存储数据
3. 启用混合精度训练
4. 使用多GPU并行

### Q: 如何调整数据量？
A: 修改重叠率：
- 50%重叠 → ~150M样本
- 70%重叠 → ~220M样本
- 90%重叠 → ~370M样本

## 评估指标

训练过程会记录以下指标：
- **损失**: 主损失、GC含量损失、生物学约束损失
- **准确率**: Mask token预测准确率
- **困惑度**: 模型的不确定性
- **物种特定指标**: 每个物种的表现

## 下游任务

训练好的模型可用于：
1. **序列生成**: 生成新的DNA序列
2. **功能预测**: 预测序列功能
3. **变异影响**: 评估突变影响
4. **跨物种迁移**: 在新物种上微调

## 联系方式

如有问题，请提交Issue或联系维护者。 