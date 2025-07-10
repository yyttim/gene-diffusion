#!/usr/bin/env python3
"""
最小化训练测试 - 只训练几个步骤
"""

import os
import sys
import yaml
import torch
import numpy as np
from datetime import datetime
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import GeneDiffusionModel
from src.data_processor import DNATokenizer, GenomicDataset, collate_batch
from torch.utils.data import DataLoader
import torch.nn.functional as F

def create_minimal_model(config):
    """创建小模型"""
    model_config = config['model']
    
    model = GeneDiffusionModel(
        vocab_size=model_config['vocab_size'],
        num_annotation_types=5,
        num_species=model_config['num_species'],
        d_model=model_config['hidden_size'],
        n_layers=model_config['num_layers'],
        n_heads=model_config['num_heads'],
        d_ff=model_config['intermediate_size'],
        max_seq_len=1667,  # 10kb / 6
        dropout=0.1,
        use_dna_features=True
    )
    
    return model

def train_few_steps():
    """训练几个步骤"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    with open('configs/train_config_small.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    output_dir = f"outputs/test_minimal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("最小化训练测试")
    print("=" * 60)
    
    # 创建模型
    print("\n创建模型...")
    model = create_minimal_model(config)
    model = model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params/1e6:.2f}M")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    # 创建最小数据集
    print("\n创建数据集...")
    dataset = GenomicDataset(
        data_dir="datasets/genomics_seq",
        species_list=['homo_sapiens'],  # 只使用人类数据
        seq_length=1000,  # 使用短序列
        max_sequences_per_species=20,  # 只使用20个样本
        cache_sequences=True  # 缓存以加快速度
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # 单进程加载
        collate_fn=collate_batch
    )
    
    # 训练几个步骤
    print("\n开始训练...")
    model.train()
    
    num_steps = 5  # 只训练5个步骤
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        
        # 移动到设备
        input_ids = batch['input_ids'].to(device)
        annotations = batch['annotations'].to(device)
        species_ids = batch['species_ids'].to(device)
        
        # 计算损失
        outputs = model.compute_loss(input_ids, annotations, species_ids)
        loss = outputs['loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"步骤 {step+1}/{num_steps}: 损失={loss.item():.4f}, 准确率={outputs['accuracy']:.4f}")
    
    print("\n测试完成！")
    
    # 保存模型检查点
    checkpoint_path = os.path.join(output_dir, 'test_checkpoint.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, checkpoint_path)
    print(f"模型已保存到: {checkpoint_path}")
    
    return True

def main():
    """主函数"""
    try:
        # 禁用wandb
        os.environ['WANDB_MODE'] = 'disabled'
        
        # 运行训练测试
        success = train_few_steps()
        
        if success:
            print("\n" + "=" * 60)
            print("训练测试成功！小模型可以正常工作。")
            print("=" * 60)
            print("\n下一步建议:")
            print("1. 使用完整数据运行: bash scripts/train_small_model.sh")
            print("2. 或者修改配置继续测试")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    main() 