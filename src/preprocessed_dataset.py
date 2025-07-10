"""
预处理数据集 - 从HDF5文件快速加载预处理的数据
"""

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class PreprocessedDataset(Dataset):
    """预处理数据集 - 直接从HDF5文件读取处理好的数据"""
    
    def __init__(
        self,
        data_dir: str,
        species_list: Optional[List[str]] = None,
        max_samples_per_species: Optional[int] = None,
        cache_in_memory: bool = False
    ):
        """
        Args:
            data_dir: 预处理数据目录
            species_list: 要使用的物种列表，None表示使用所有物种
            max_samples_per_species: 每个物种的最大样本数
            cache_in_memory: 是否将所有数据缓存到内存
        """
        self.data_dir = Path(data_dir)
        self.cache_in_memory = cache_in_memory
        
        # 加载物种映射
        mapping_file = self.data_dir / "species_mapping.json"
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        self.species_to_id = mapping_data['species_to_id']
        self.id_to_species = {int(k): v for k, v in mapping_data['id_to_species'].items()}
        self.num_species = mapping_data['num_species']
        
        # 确定要使用的物种
        if species_list is None:
            self.species_list = list(self.species_to_id.keys())
        else:
            self.species_list = [s for s in species_list if s in self.species_to_id]
        
        # 构建数据索引
        self.data_index = []
        self.h5_files = {}
        self.cached_data = {} if cache_in_memory else None
        
        for species in self.species_list:
            h5_file = self.data_dir / f"{species}.h5"
            if not h5_file.exists():
                logger.warning(f"未找到 {species} 的数据文件")
                continue
            
            # 打开HDF5文件获取样本数
            with h5py.File(h5_file, 'r') as f:
                num_samples = f.attrs['num_fragments']
                
                # 限制样本数
                if max_samples_per_species is not None:
                    num_samples = min(num_samples, max_samples_per_species)
                
                # 添加到索引
                for i in range(num_samples):
                    self.data_index.append({
                        'species': species,
                        'species_id': self.species_to_id[species],
                        'h5_file': str(h5_file),
                        'index': i
                    })
                
                # 如果需要缓存到内存
                if cache_in_memory:
                    self._cache_species_data(species, h5_file, num_samples)
        
        logger.info(f"加载了 {len(self.species_list)} 个物种的 {len(self.data_index)} 个样本")
    
    def _cache_species_data(self, species: str, h5_file: Path, num_samples: int):
        """将物种数据缓存到内存"""
        with h5py.File(h5_file, 'r') as f:
            self.cached_data[species] = {
                'token_ids': f['token_ids'][:num_samples],
                'annotations': f['annotations'][:num_samples]
            }
        logger.info(f"缓存 {species} 的 {num_samples} 个样本到内存")
    
    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, idx):
        info = self.data_index[idx]
        species = info['species']
        species_id = info['species_id']
        h5_file = info['h5_file']
        sample_idx = info['index']
        
        # 从缓存或文件读取数据
        if self.cached_data is not None and species in self.cached_data:
            # 从内存读取
            token_ids = self.cached_data[species]['token_ids'][sample_idx]
            annotations = self.cached_data[species]['annotations'][sample_idx]
        else:
            # 从文件读取
            if h5_file not in self.h5_files:
                self.h5_files[h5_file] = h5py.File(h5_file, 'r')
            
            f = self.h5_files[h5_file]
            token_ids = f['token_ids'][sample_idx]
            annotations = f['annotations'][sample_idx]
        
        # 找到实际长度（非零部分）
        mask = token_ids > 0
        actual_length = mask.sum()
        
        # 截取有效部分
        token_ids = token_ids[:actual_length]
        annotations = annotations[:actual_length]
        
        # 转换为张量
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'annotations': torch.tensor(annotations, dtype=torch.long),
            'species_id': torch.tensor(species_id, dtype=torch.long),
            'annotation_coverage': torch.tensor(0.0, dtype=torch.float32)  # 可以在预处理时计算
        }
    
    def __del__(self):
        """关闭所有打开的HDF5文件"""
        for f in self.h5_files.values():
            f.close()


def create_preprocessed_dataloader(
    data_dir: str,
    batch_size: int,
    species_list: Optional[List[str]] = None,
    max_samples_per_species: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    cache_in_memory: bool = False,
    **kwargs
) -> DataLoader:
    """创建预处理数据加载器"""
    
    dataset = PreprocessedDataset(
        data_dir=data_dir,
        species_list=species_list,
        max_samples_per_species=max_samples_per_species,
        cache_in_memory=cache_in_memory
    )
    
    # 定义collate函数来处理不同长度的序列
    def collate_fn(batch):
        # 找到批次中的最大长度
        max_len = max(item['input_ids'].size(0) for item in batch)
        
        # 填充到相同长度
        input_ids = []
        annotations = []
        species_ids = []
        annotation_coverages = []
        
        for item in batch:
            seq_len = item['input_ids'].size(0)
            
            # 填充
            if seq_len < max_len:
                padding = max_len - seq_len
                item['input_ids'] = torch.cat([
                    item['input_ids'],
                    torch.zeros(padding, dtype=torch.long)
                ])
                item['annotations'] = torch.cat([
                    item['annotations'],
                    torch.zeros(padding, dtype=torch.long)
                ])
            
            input_ids.append(item['input_ids'])
            annotations.append(item['annotations'])
            species_ids.append(item['species_id'])
            annotation_coverages.append(item['annotation_coverage'])
        
        return {
            'input_ids': torch.stack(input_ids),
            'annotations': torch.stack(annotations),
            'species_ids': torch.stack(species_ids),
            'annotation_coverage': torch.stack(annotation_coverages)
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        **kwargs
    )


if __name__ == "__main__":
    # 测试预处理数据集
    import time
    
    data_dir = "datasets/preprocessed"
    
    print("测试预处理数据集...")
    
    # 创建数据集
    dataset = PreprocessedDataset(
        data_dir=data_dir,
        species_list=['homo_sapiens', 'mus_musculus'],
        max_samples_per_species=100,
        cache_in_memory=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 测试加载速度
    start_time = time.time()
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        print(f"样本 {i}: input_ids形状={sample['input_ids'].shape}")
    
    elapsed = time.time() - start_time
    print(f"加载10个样本耗时: {elapsed:.3f}秒")
    
    # 测试数据加载器
    dataloader = create_preprocessed_dataloader(
        data_dir=data_dir,
        batch_size=4,
        species_list=['homo_sapiens'],
        max_samples_per_species=20,
        shuffle=True,
        num_workers=0
    )
    
    print(f"\n数据加载器批次数: {len(dataloader)}")
    
    # 测试一个批次
    batch = next(iter(dataloader))
    print(f"批次数据形状:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}") 