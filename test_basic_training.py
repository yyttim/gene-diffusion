#!/usr/bin/env python3
"""
基础训练功能测试
"""

import os
import sys
import yaml

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入"""
    print("测试导入...")
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA设备: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"✗ PyTorch导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy版本: {np.__version__}")
    except Exception as e:
        print(f"✗ NumPy导入失败: {e}")
        return False
    
    try:
        from src.models import GeneDiffusionModel
        print("✓ 模型导入成功")
    except Exception as e:
        print(f"✗ 模型导入失败: {e}")
        return False
    
    try:
        from src.data_processor import DNATokenizer, GenomicDataset
        print("✓ 数据处理器导入成功")
    except Exception as e:
        print(f"✗ 数据处理器导入失败: {e}")
        return False
    
    return True

def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    try:
        import torch
        from src.models import GeneDiffusionModel
        
        # 加载小模型配置
        with open('configs/train_config_small.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = config['model']
        
        # 创建模型
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
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ 模型创建成功，参数量: {total_params/1e6:.2f}M")
        
        # 测试前向传播
        batch_size = 2
        seq_len = 100
        
        # 创建测试输入
        xt = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))
        annotations = torch.randint(0, 6, (batch_size, seq_len))
        species_ids = torch.randint(0, model_config['num_species'], (batch_size,))
        
        # 前向传播
        with torch.no_grad():
            outputs = model(xt, annotations, species_ids)
        
        print(f"✓ 前向传播成功")
        print(f"  - logits形状: {outputs['logits'].shape}")
        print(f"  - hidden_states形状: {outputs['hidden_states'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n测试数据加载...")
    try:
        from src.data_processor import DNATokenizer, GenomicDataset
        import torch
        
        # 创建tokenizer
        tokenizer = DNATokenizer(k=6)
        print(f"✓ Tokenizer创建成功，词汇表大小: {tokenizer.vocab_size}")
        
        # 测试编码
        test_seq = "atcgatcg"
        encoded = tokenizer.encode(test_seq)
        print(f"✓ 序列编码成功: {test_seq} -> {len(encoded)} tokens")
        
        # 创建小数据集测试
        dataset = GenomicDataset(
            data_dir="datasets/genomics_seq",
            species_list=['homo_sapiens'],  # 只使用一个物种
            seq_length=1000,  # 短序列
            max_sequences_per_species=10  # 少量样本
        )
        
        print(f"✓ 数据集创建成功，大小: {len(dataset)}")
        
        if len(dataset) > 0:
            # 获取一个样本
            sample = dataset[0]
            print(f"✓ 样本加载成功:")
            print(f"  - input_ids形状: {sample['input_ids'].shape}")
            print(f"  - annotations形状: {sample['annotations'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("GeneDiffusion 基础功能测试")
    print("=" * 60)
    
    # 测试导入
    if not test_imports():
        print("\n导入测试失败，请检查环境配置")
        return
    
    # 测试模型
    if not test_model_creation():
        print("\n模型测试失败")
        return
    
    # 测试数据
    if not test_data_loading():
        print("\n数据加载测试失败")
        return
    
    print("\n" + "=" * 60)
    print("所有基础测试通过！可以开始训练。")
    print("=" * 60)

if __name__ == "__main__":
    main() 