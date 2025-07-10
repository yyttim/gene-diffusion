# GeneDiffusion 项目架构分析

## 项目概述
GeneDiffusion 是一个基于 LLaDA (Language Latent Diffusion) 思想的生物序列生成模型，将 Masked Diffusion 技术应用于 DNA 序列生成。

## 核心创新

### 1. 6-mer Token系统
- **基本单位**：使用6个连续碱基作为一个token（4096种可能）
- **优势**：捕获局部序列模式、减少序列长度6倍、提高计算效率
- **编码方式**：序列按6bp切分，每个6-mer映射到唯一ID

### 2. 真正的 Masked Diffusion 实现
- **前向扩散过程**：随机mask比例 t ∈ [0,1]，每个6-mer token独立mask
- **损失函数**：包含关键的 1/t 缩放因子，确保不同mask比例下损失可比
- **逆向采样**：从全mask序列开始，逐步去噪生成完整序列

### 2. 生物学特化设计
- **注释感知mask**：对功能重要区域（CDS、exon等）降低mask概率
- **物种特异性编码**：支持230个哺乳动物物种
- **生物学约束**：GC含量控制、密码子完整性、基因结构保护

### 3. 技术架构
```
GeneDiffusionModel
├── DNASequenceEncoder (序列编码)
├── AnnotationEncoder (注释编码)  
├── SpeciesEncoder (物种编码)
├── TransformerBackbone (主模型)
├── DNAFeatureExtractor (DNA特征)
└── BiologicalConstraintLoss (生物约束)
```

## 项目结构

### 源代码 (`src/`)
- `models.py` - 核心Diffusion模型实现
- `data_processor.py` - 数据加载和预处理
- `train.py` - 分布式训练框架
- `inference.py` - 推理和生成接口

### 配置 (`configs/`)
- `train_*.yaml/json` - 不同规模模型配置（50M到1.5B参数）
- `species_mapping.py` - 物种到数据文件的映射

### 脚本 (`scripts/`)
- `submit_*.sh` - 集群任务提交脚本
- `run_*.sh` - 训练运行脚本
- `setup_env.sh` - 环境配置

### 测试 (`tests/`)
- `test_masked_diffusion.py` - Diffusion核心功能验证
- `test_data_*.py` - 数据处理测试
- `test_environment.py` - 环境测试

## 数据流程

```
原始数据 (data_stage2/3)
    ↓
GenomicDataset (滑动窗口切片)
    ↓
DataLoader (批处理)
    ↓
GeneDiffusionModel
    ├── 前向扩散 (添加mask)
    ├── 模型预测 (去噪)
    └── 损失计算 (1/t缩放)
```

## 关键参数

### 模型参数
- 词汇表大小: 4099 (4096个6-mer + MASK,PAD,UNK)
- 注释类型: 5 (exon,CDS,lnc_RNA,gene,tRNA)
- 物种数量: 230
- 序列长度: 5000bp → ~833个6-mer tokens

### 训练参数
- Mask比例: 动态调整 [0.15, 0.85]
- 损失权重: 扩散损失 + 0.1×生物约束
- 优化器: AdamW with 分层学习率

## 验证状态

✅ **已验证功能**：
1. Masked Diffusion 核心算法正确实现
2. 数据加载和格式转换
3. 注释感知的mask策略
4. 分布式训练框架

🚧 **待完成**：
1. 大规模训练实验
2. 生成质量评估
3. 下游任务验证

## 运行示例

```bash
# 激活环境
conda activate gene-diffusion

# 单节点训练
bash scripts/run_training.sh

# 多节点训练
bash scripts/submit_multinode_job.sh

# 测试Diffusion实现
python tests/test_masked_diffusion.py
```

## 技术债务

1. **内存优化**：长序列需要序列并行
2. **采样效率**：考虑实现更快的采样算法
3. **评估指标**：需要生物学相关的评估指标

## 总结

GeneDiffusion 成功将 LLaDA 的 Masked Diffusion 框架适配到生物序列生成任务，通过注释感知、物种特化等创新，构建了一个生物学意义上合理的生成模型。项目结构清晰，代码质量高，为后续的大规模训练和应用奠定了坚实基础。 