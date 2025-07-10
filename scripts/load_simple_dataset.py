#!/usr/bin/env python3
"""
简单的数据集加载器，用于加载10K_3M_S1_0709.txt格式的数据
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import DNATokenizer
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleSequenceDataset(Dataset):
    """简单的序列数据集，直接从文本文件加载"""
    
    def __init__(self, file_path, max_sequences=None, seq_length=10000):
        """
        Args:
            file_path: 数据文件路径
            max_sequences: 最多加载的序列数（None表示全部）
            seq_length: 使用的序列长度（默认10000）
        """
        self.sequences = []
        self.seq_length = seq_length
        self.tokenizer = DNATokenizer(k=6)
        
        print(f"Loading sequences from {file_path}...")
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if max_sequences and i >= max_sequences:
                    break
                # 截取到需要的长度
                seq = line.strip()[:seq_length]
                self.sequences.append(seq)
                
                if (i + 1) % 10000 == 0:
                    print(f"  Loaded {i + 1} sequences...")
        
        print(f"Loaded {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # 转换为6-mer tokens
        token_ids = self.tokenizer.encode(sequence)
        
        # 转换为tensor
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # 创建随机mask（用于训练）
        mask_ratio = 0.15  # 15% mask比例
        masked_ids, mask = self._create_masked_sequence(input_ids, mask_ratio)
        
        return {
            'input_ids': input_ids,
            'masked_ids': masked_ids,
            'mask': mask
        }
    
    def _create_masked_sequence(self, input_ids, mask_ratio):
        """创建masked序列用于训练"""
        length = len(input_ids)
        num_mask = int(length * mask_ratio)
        
        # 随机选择要mask的位置
        mask_indices = torch.randperm(length)[:num_mask]
        
        # 创建masked序列
        masked_ids = input_ids.clone()
        masked_ids[mask_indices] = self.tokenizer.mask_token_id
        
        # 创建mask标记
        mask = torch.zeros(length, dtype=torch.bool)
        mask[mask_indices] = True
        
        return masked_ids, mask

def main():
    """测试数据加载"""
    # 数据文件路径
    data_file = "datasets/10K_3M_S1_0709.txt"
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return
    
    # 创建数据集（只加载前1000条用于测试）
    print("\n创建数据集...")
    dataset = SimpleSequenceDataset(
        file_path=data_file,
        max_sequences=1000,
        seq_length=10000
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )
    
    # 测试加载一个批次
    print("\n测试加载一个批次...")
    batch = next(iter(dataloader))
    
    print(f"批次信息:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  masked_ids shape: {batch['masked_ids'].shape}")
    print(f"  mask shape: {batch['mask'].shape}")
    
    # 显示一个样本的统计
    sample_idx = 0
    input_ids = batch['input_ids'][sample_idx]
    masked_ids = batch['masked_ids'][sample_idx]
    mask = batch['mask'][sample_idx]
    
    print(f"\n样本 {sample_idx} 统计:")
    print(f"  序列长度: {len(input_ids)} tokens")
    print(f"  Masked tokens: {mask.sum().item()}")
    print(f"  Mask比例: {mask.sum().item() / len(mask):.2%}")
    
    # 计算词汇表使用情况
    unique_tokens = len(torch.unique(input_ids))
    print(f"  使用的unique tokens: {unique_tokens} / {dataset.tokenizer.vocab_size}")
    
    print("\n数据加载测试完成！")

if __name__ == "__main__":
    main() 