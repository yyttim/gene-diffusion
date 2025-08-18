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
        mapping_file = self.data_dir / "species_id_mapping.json"
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
                if max_samples_per_species is not None and max_samples_per_species > 0:
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
        
        # Only log on rank 0 (main process)
        if int(os.environ.get('RANK', 0)) == 0:
            logger.info(f"Loaded {len(self.species_list)} species, {len(self.data_index)} samples")
    
    def _cache_species_data(self, species: str, h5_file: Path, num_samples: int):
        """将物种数据缓存到内存"""
        with h5py.File(h5_file, 'r') as f:
            self.cached_data[species] = {
                'token_ids': f['token_ids'][:num_samples],
                'annotations': f['annotations'][:num_samples]
            }
        # Only log on rank 0
        if int(os.environ.get('RANK', 0)) == 0:
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
    """
    创建预处理数据加载器
    
    Args:
        data_dir: 预处理数据目录
        batch_size: 批次大小
        species_list: 要使用的物种列表
        max_samples_per_species: 每个物种的最大样本数
        shuffle: 是否打乱
        num_workers: 数据加载线程数
        cache_in_memory: 是否缓存到内存
        **kwargs: 其他DataLoader参数
    
    Returns:
        DataLoader实例
    """
    dataset = PreprocessedDataset(
        data_dir=data_dir,
        species_list=species_list,
        max_samples_per_species=max_samples_per_species,
        cache_in_memory=cache_in_memory
    )
    
    # 处理可变长度的collate函数
    def collate_fn(batch):
        # 获取最大长度
        max_len = max(item['input_ids'].size(0) for item in batch)
        
        # 创建padded张量
        batch_size = len(batch)
        padded_input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        padded_annotations = torch.zeros(batch_size, max_len, dtype=torch.long)
        species_ids = torch.zeros(batch_size, dtype=torch.long)
        annotation_coverages = torch.zeros(batch_size, dtype=torch.float32)
        
        # 填充数据
        for i, item in enumerate(batch):
            seq_len = item['input_ids'].size(0)
            padded_input_ids[i, :seq_len] = item['input_ids']
            padded_annotations[i, :seq_len] = item['annotations']
            species_ids[i] = item['species_id']
            annotation_coverages[i] = item['annotation_coverage']
        
        return {
            'input_ids': padded_input_ids,
            'annotations': padded_annotations,
            'species_ids': species_ids,
            'annotation_coverage': annotation_coverages
        }
    
    # 设置默认参数
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': kwargs.get('pin_memory', True),
        'persistent_workers': kwargs.get('persistent_workers', num_workers > 0),
        'prefetch_factor': kwargs.get('prefetch_factor', 2),
        'drop_last': kwargs.get('drop_last', True)
    }
    
    # 更新用户提供的参数
    dataloader_kwargs.update(kwargs)
    
    return DataLoader(dataset, **dataloader_kwargs)


if __name__ == "__main__":
    # 测试预处理数据集
    import sys
    from pathlib import Path
    
    # 添加项目根目录到路径
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # 测试加载
    data_dir = project_root / "datasets" / "preprocessed"
    
    if data_dir.exists():
        print(f"测试加载预处理数据: {data_dir}")
        
        # 创建数据加载器
        dataloader = create_preprocessed_dataloader(
            data_dir=str(data_dir),
            batch_size=4,
            species_list=None,  # 使用所有物种
            max_samples_per_species=10,  # 每个物种只加载10个样本用于测试
            shuffle=True,
            num_workers=2,
            cache_in_memory=False
        )
        
        print(f"数据集大小: {len(dataloader.dataset)}")
        print(f"批次数: {len(dataloader)}")
        
        # 测试一个批次
        for i, batch in enumerate(dataloader):
            if i >= 1:  # 只测试1个批次
                break
            
            print(f"\n批次 {i}:")
            print(f"  input_ids shape: {batch['input_ids'].shape}")
            print(f"  annotations shape: {batch['annotations'].shape}")
            print(f"  species_ids shape: {batch['species_ids'].shape}")
            print(f"  species_ids values: {batch['species_ids']}")
            
            # 显示第一个序列的前20个token
            seq = batch['input_ids'][0]
            ann = batch['annotations'][0]
            print(f"  第一个序列前20个token: {seq[:20].tolist()}")
            print(f"  对应注释: {ann[:20].tolist()}")
    else:
        print(f"预处理数据目录不存在: {data_dir}") 