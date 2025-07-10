#!/usr/bin/env python3
"""
简单测试小模型训练一个epoch
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

from src.train import DNADiffusionTrainer
from src.data_processor import create_data_loader
import wandb

def test_one_epoch():
    """测试训练一个epoch"""
    
    # 禁用wandb
    os.environ['WANDB_MODE'] = 'disabled'
    
    # 设置参数
    config_path = "configs/train_config_small.yaml"
    output_dir = f"outputs/test_1epoch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'test.log')),
            logging.StreamHandler()
        ]
    )
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改配置以测试
    config['training']['num_epochs'] = 1
    config['output_dir'] = output_dir
    config['data']['max_sequences_per_species'] = 100  # 减少数据量以加快测试
    
    print("=" * 60)
    print("小模型1个Epoch测试")
    print("=" * 60)
    print(f"配置文件: {config_path}")
    print(f"输出目录: {output_dir}")
    print(f"批次大小: {config['data']['train_batch_size']}")
    print(f"每个物种样本数: {config['data']['max_sequences_per_species']}")
    print("=" * 60)
    
    try:
        # 创建数据加载器
        print("\n创建数据加载器...")
        train_dataloader = create_data_loader(
            data_dir=config['data']['data_dir'],
            batch_size=config['data']['train_batch_size'],
            seq_length=config['data']['seq_length'],
            species_list=None,  # 使用所有物种
            shuffle=True,
            num_workers=4,
            max_sequences_per_species=config['data']['max_sequences_per_species']
        )
        
        print(f"数据集大小: {len(train_dataloader.dataset)} 样本")
        print(f"批次数: {len(train_dataloader)}")
        
        # 创建验证数据加载器（使用更少的数据）
        val_dataloader = create_data_loader(
            data_dir=config['data']['data_dir'],
            batch_size=config['data']['eval_batch_size'],
            seq_length=config['data']['seq_length'],
            species_list=['homo_sapiens', 'mus_musculus'],  # 只使用2个物种进行验证
            shuffle=False,
            num_workers=2,
            max_sequences_per_species=10
        )
        
        print(f"验证集大小: {len(val_dataloader.dataset)} 样本")
        
        # 创建训练器
        print("\n创建训练器...")
        trainer = DNADiffusionTrainer(config, rank=0, world_size=1)
        
        # 测试一个批次
        print("\n测试一个批次...")
        batch = next(iter(train_dataloader))
        print(f"批次数据形状:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")
        
        # 开始训练
        print("\n开始训练...")
        trainer.train(train_dataloader, val_dataloader)
        
        print("\n测试完成！")
        print(f"输出目录: {output_dir}")
        
    except Exception as e:
        logging.error(f"训练失败: {str(e)}", exc_info=True)
        print(f"\n错误: {str(e)}")
        print("详细错误信息请查看日志文件")

if __name__ == "__main__":
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只使用第一个GPU
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # 运行测试
    test_one_epoch() 