# GeneDiffusion: Masked Diffusion Model for Mammalian Genomic Sequences

## 模型架构概述

GeneDiffusion是一个创新的基于Masked Diffusion的DNA序列生成模型，专门设计用于生成具有生物学意义的哺乳动物基因组序列。该模型结合了扩散模型的生成能力与生物序列的特定约束，支持228种哺乳动物物种的条件生成。

## 核心技术创新

### 1. Masked Diffusion Framework

与传统的连续扩散模型不同，GeneDiffusion采用离散的Masked Diffusion方法：

- **前向过程**：通过随机掩码token来破坏序列，掩码比例t从均匀分布U(0,1]中采样
- **反向过程**：通过Transformer预测被掩码的token，逐步恢复原始序列
- **注释感知掩码**：对功能性区域（如CDS、外显子）降低30%的掩码概率，保护重要生物学区域

```python
# 前向扩散过程
def forward_diffusion(x0, t, annotations):
    if annotations is not None:
        base_prob = t
        base_prob[annotations > 0] *= 0.7  # 功能区域保护
        mask = rand() < base_prob
    else:
        mask = rand() < t
    xt = x0.clone()
    xt[mask] = MASK_TOKEN_ID
    return xt, mask
```

### 2. 6-mer Tokenization策略

模型采用6-mer（六联体）作为基本token单位，而非传统的单碱基表示：

- **词汇表大小**：4^6 = 4096个可能的6-mer组合
- **生物学意义**：6-mer包含了密码子（3bp）信息，更好地捕获生物学模式
- **序列压缩**：相比单碱基tokenization，序列长度缩短6倍，提高计算效率

### 3. Species-Aware Architecture

#### 3.1 Species Embedding
```python
class SpeciesAwareEmbedding(nn.Module):
    def __init__(self, num_species=228, d_model=512):
        # 物种特定嵌入
        self.species_embedding = nn.Embedding(num_species, d_model // 4)
        # FiLM-like调制参数
        self.species_scale = nn.Embedding(num_species, d_model)
        self.species_shift = nn.Embedding(num_species, d_model)
```

- 通过FiLM（Feature-wise Linear Modulation）机制调制特征
- 每个物种学习独特的scale和shift参数
- 支持228种哺乳动物物种的条件生成

#### 3.2 Species-Conditioned Layer Normalization
为每个物种学习特定的归一化参数，使模型能够捕获物种特异的序列特征。

### 4. Annotation-Guided Generation

模型集成了6种基因组注释类型的信息：

1. **none** - 无注释区域
2. **exon** - 外显子
3. **cds** - 编码序列
4. **lnc_rna** - 长链非编码RNA
5. **gene** - 基因区域
6. **trna** - 转运RNA

#### 4.1 Annotation Encoder
独立的Transformer编码器处理注释信息：
- 4层Transformer块
- 使用边界感知注意力机制
- 生成与主序列融合的注释特征

#### 4.2 Boundary-Aware Attention
```python
class BoundaryAwareAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        # 边界类型嵌入：0=none, 1=start, 2=end
        self.boundary_embedding = nn.Embedding(3, d_head)
```

检测并编码功能区域的边界，增强模型对基因结构的理解。

### 5. Multi-Scale Local Pattern Encoding

```python
class LocalSequenceEncoder(nn.Module):
    def __init__(self, d_model, kernel_sizes=[3, 5, 7]):
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model//n, kernel_size=k)
            for k in kernel_sizes
        ])
```

通过多尺度卷积捕获局部序列模式：
- 并行使用3、5、7不同大小的卷积核
- 捕获短程序列依赖和motif模式
- 与全局注意力机制互补

### 6. Biological Constraint Loss

模型集成了生物学约束损失函数：

```python
class BiologicalConstraintLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.1):
        # alpha: ORF完整性权重
        # beta: 剪接位点权重  
        # gamma: 序列复杂度权重
```

- **ORF完整性**：确保CDS区域长度为3的倍数
- **剪接位点保守性**：检测GT-AG剪接信号
- **序列复杂度**：惩罚低复杂度重复序列

## 模型架构细节

### Transformer配置

标准配置（600M参数版本）：
- **d_model**: 1024
- **n_layers**: 24
- **n_heads**: 16
- **d_ff**: 4096
- **max_seq_len**: 1024 tokens (约6kb DNA)

### 位置编码

采用Rotary Position Encoding (RoPE)：
- 支持最长16384的序列长度
- 相对位置编码，更好的长序列泛化
- base=10000的频率设置

### 训练策略

#### 动态掩码采样
- 掩码比例t ~ U(0, 1]，提供多样的训练信号
- 早期训练阶段倾向于高掩码比例
- 后期逐渐降低掩码比例，精细化预测

#### 梯度缩放与稳定性
- 混合精度训练（FP16）
- 梯度裁剪（clip=0.5）
- 梯度累积支持大批量训练

#### 学习率调度
- Warmup阶段：500步
- Cosine annealing with warm restarts
- 基础学习率随批量大小sqrt缩放

## 生成过程

### 迭代去噪采样

```python
def sample(batch_size, seq_len, annotations, species_ids, num_steps=100):
    # 从完全掩码序列开始
    x = [MASK_TOKEN] * seq_len
    
    for step in range(num_steps):
        # 计算当前掩码比例
        t = 1.0 - (step + 1) / num_steps
        
        # 预测掩码位置的token
        logits = model(x, annotations, species_ids)
        
        # 温度采样
        probs = softmax(logits / temperature)
        sampled = sample_from(probs)
        
        # 更新掩码位置
        x[mask] = sampled[mask]
        
        # 重新掩码部分位置（除最后一步）
        if step < num_steps - 1:
            remask_positions = select_low_confidence(probs, t)
            x[remask_positions] = MASK_TOKEN
    
    return x
```

### 采样策略

- **温度控制**：调节生成的多样性
- **Top-k采样**：限制候选token数量
- **置信度引导重掩码**：优先重新生成低置信度位置

## 技术优势

1. **生物学一致性**：通过注释引导和约束损失确保生成序列的生物学合理性

2. **物种特异性**：FiLM调制机制使单一模型能够生成228种不同物种的序列

3. **多尺度建模**：结合局部卷积和全局注意力，捕获从motif到染色体级别的模式

4. **计算效率**：6-mer tokenization大幅减少序列长度，chunked attention支持长序列

5. **灵活的条件生成**：支持物种、注释、GC含量等多种条件约束

## 模型评估指标

- **重建准确率**：掩码token预测准确率
- **GC含量一致性**：生成序列与目标物种GC含量的匹配度
- **注释区域保真度**：功能区域的序列模式保持
- **物种特异性评分**：生成序列与物种特征的一致性

## 理论基础

GeneDiffusion结合了以下理论基础：

1. **扩散模型理论**：基于去噪分数匹配的生成建模
2. **掩码语言模型**：BERT风格的双向上下文建模
3. **条件生成模型**：FiLM和cross-attention的条件机制
4. **生物序列分析**：密码子偏好、GC含量、基因结构约束

该模型为基因组序列生成提供了一个强大而灵活的框架，能够生成具有真实生物学特征的DNA序列。 