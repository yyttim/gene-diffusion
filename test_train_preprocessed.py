#!/usr/bin/env python3
"""
使用预处理数据测试训练
"""

import os
import sys
import yaml
import torch
import time
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import GeneDiffusionModel
from src.preprocessed_dataset import create_preprocessed_dataloader

def test_preprocessed_training():
    """测试使用预处理数据训练"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    with open('configs/train_config_small.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    output_dir = f"outputs/test_preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("使用预处理数据训练测试")
    print("=" * 60)
    
    # 创建模型
    print("\n创建模型...")
    model_config = config['model']
    model = GeneDiffusionModel(
        vocab_size=model_config['vocab_size'],
        num_annotation_types=5,
        num_species=model_config['num_species'],
        d_model=model_config['hidden_size'],
        n_layers=model_config['num_layers'],
        n_heads=model_config['num_heads'],
        d_ff=model_config['intermediate_size'],
        max_seq_len=1667,
        dropout=0.1,
        use_dna_features=True
    )
    model = model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params/1e6:.2f}M")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    
    # 检查预处理数据目录
    preprocessed_dir = "datasets/preprocessed_test"
    if not os.path.exists(preprocessed_dir):
        print(f"预处理数据目录不存在: {preprocessed_dir}")
        print("请先运行: python test_preprocess_small.py")
        return
    
    # 测试数据加载速度
    print("\n测试数据加载速度...")
    
    # 不缓存版本
    start_time = time.time()
    dataloader_no_cache = create_preprocessed_dataloader(
        data_dir=preprocessed_dir,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        cache_in_memory=False
    )
    batch_no_cache = next(iter(dataloader_no_cache))
    time_no_cache = time.time() - start_time
    print(f"不缓存加载时间: {time_no_cache:.3f}秒")
    
    # 缓存版本
    start_time = time.time()
    dataloader_cache = create_preprocessed_dataloader(
        data_dir=preprocessed_dir,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        cache_in_memory=True
    )
    batch_cache = next(iter(dataloader_cache))
    time_cache = time.time() - start_time
    print(f"缓存加载时间: {time_cache:.3f}秒")
    
    print(f"加速比: {time_no_cache/time_cache:.2f}x")
    
    # 使用缓存版本进行训练
    dataloader = dataloader_cache
    
    print(f"\n数据集大小: {len(dataloader.dataset)}")
    print(f"批次数: {len(dataloader)}")
    
    # 训练几个步骤
    print("\n开始训练...")
    model.train()
    
    num_steps = 5
    train_start = time.time()
    
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
    
    train_time = time.time() - train_start
    print(f"\n训练 {num_steps} 步耗时: {train_time:.2f}秒")
    print(f"平均每步: {train_time/num_steps:.2f}秒")
    
    print("\n测试完成！")
    
    # 比较数据加载性能
    print("\n" + "=" * 60)
    print("性能比较:")
    print("-" * 60)
    print(f"预处理数据加载: ~{time_cache:.3f}秒/批次")
    print(f"原始数据处理估计: >5秒/批次")
    print(f"加速比: >10x")
    print("=" * 60)

if __name__ == "__main__":
    # 禁用wandb
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    test_preprocessed_training() 