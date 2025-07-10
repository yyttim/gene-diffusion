# GeneDiffusion 物种间平衡采样设计

## 背景与动机

在训练多物种基因组模型时，不同物种的基因组大小差异巨大：
- 人类基因组：~3.2 Gb
- 小鼠基因组：~2.7 Gb  
- 某些哺乳动物：可能只有1-2 Gb

如果不进行平衡，模型会对数据量大的物种产生偏好，导致：
1. **过拟合问题**：模型过度学习数据量大的物种特征
2. **泛化能力差**：对数据量小的物种预测效果差
3. **物种偏见**：生成的序列倾向于模仿主导物种

## 平衡采样设计

### 核心原则
确保每个物种在训练集中贡献**相同数量**的序列片段，实现真正的物种间平衡。

### 实现方法

#### 1. 三步式平衡采样流程

```python
def _build_data_index(self):
    # 第一步：收集每个物种的所有可用片段
    species_fragments_dict = {}
    for species in species_list:
        fragments = create_fragments(species)
        species_fragments_dict[species] = fragments
    
    # 第二步：确定统一的采样数量
    min_fragments = min(len(f) for f in species_fragments_dict.values())
    samples_per_species = min(min_fragments, max_sequences_per_species)
    
    # 第三步：均匀采样并打乱
    data_index = []
    for species, fragments in species_fragments_dict.items():
        sampled = random.sample(fragments, samples_per_species)
        data_index.extend(sampled)
    random.shuffle(data_index)
```

#### 2. 自动平衡机制

- **最小值对齐**：自动找到片段数最少的物种，将所有物种对齐到这个数量
- **上限控制**：通过`max_sequences_per_species`参数控制内存使用
- **随机采样**：从每个物种的可用片段中随机采样，保证多样性

#### 3. 数据多样性保证

- **滑动窗口**：80%重叠的滑动窗口确保序列多样性
- **全基因组覆盖**：包含编码区、非编码区、调控区等所有区域
- **随机打乱**：采样后打乱整个数据集，避免批次内物种聚集

## 优势分析

### 1. 模型公平性
- 每个物种获得相同的"发言权"
- 避免大基因组物种主导训练过程
- 提升对稀有物种的预测能力

### 2. 更好的泛化能力
- 模型学习到跨物种的通用模式
- 减少物种特异性过拟合
- 提升零样本迁移能力

### 3. 训练稳定性
- 批次内物种分布均匀
- 梯度更新更加稳定
- 收敛速度更快

### 4. 评估公正性
- 在验证集上各物种权重相等
- 性能指标更能反映真实泛化能力
- 便于跨物种性能比较

## 实际应用示例

### 小规模测试（5个物种）
```
物种                 片段数量      比例      
----------------------------------------
homo_sapiens         1000        20.0%
mus_musculus         1000        20.0%
bos_taurus           1000        20.0%
sus_scrofa           1000        20.0%
equus_caballus       1000        20.0%
----------------------------------------
平衡性检查: ✓ 通过
所有物种都有 1000 个片段
```

### 大规模训练（230个物种）
```
目标：3.45M训练样本
物种数：230
每个物种样本数：15,000
实际总样本数：3,450,000

内存需求：
- 每个样本：~10KB序列 → 1,666个6-mer tokens
- 总内存：3.45M × 1,666 × 2字节 ≈ 11.5GB（仅token）
```

## 配置建议

### 1. 开发测试
```yaml
max_sequences_per_species: 1000
batch_size: 32
num_workers: 4
```

### 2. 中等规模训练
```yaml
max_sequences_per_species: 10000  
batch_size: 64
num_workers: 8
```

### 3. 大规模训练
```yaml
max_sequences_per_species: 15000  # 3.45M总样本
batch_size: 128
num_workers: 16
gradient_accumulation_steps: 4
```

## 注意事项

1. **内存管理**：平衡采样可能需要在内存中暂存所有物种的片段索引
2. **采样随机性**：每次训练会随机采样不同的片段子集
3. **小物种保护**：数据量极少的物种可能会限制整体训练规模
4. **验证集平衡**：验证集也应该使用相同的平衡策略

## 未来改进方向

1. **动态平衡**：根据物种复杂度动态调整权重
2. **分层采样**：按照物种分类学层次进行分层平衡
3. **难例挖掘**：对学习困难的物种增加采样权重
4. **课程学习**：从平衡数据开始，逐步放宽限制

## 总结

物种间平衡采样是GeneDiffusion的核心设计之一，它确保模型能够公平地学习所有230个哺乳动物物种的基因组特征，为构建真正的泛哺乳动物基因组基础模型奠定了基础。通过这种设计，我们期望模型能够：

1. 捕获跨物种的通用生物学模式
2. 保持对每个物种的良好预测能力
3. 实现更好的零样本泛化到新物种

这种平衡采样策略是实现"One Model for All Mammals"愿景的关键技术之一。 