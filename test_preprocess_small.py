#!/usr/bin/env python3
"""
测试小规模数据预处理
"""

import os
import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess_data import DataPreprocessor

def test_small_preprocess():
    """测试小规模数据预处理"""
    
    # 设置参数
    data_dir = "datasets/genomics_seq"
    output_dir = "datasets/preprocessed_test"
    
    print("=" * 60)
    print("测试小规模数据预处理")
    print("=" * 60)
    
    # 创建预处理器
    preprocessor = DataPreprocessor(
        data_dir=data_dir,
        output_dir=output_dir,
        seq_length=1000,  # 使用短序列
        overlap=0.5,      # 较小的重叠
        max_sequences_per_species=10,  # 每个物种只处理10个样本
        num_workers=1     # 单进程
    )
    
    # 只处理几个物种
    test_species = ['homo_sapiens', 'mus_musculus']
    
    for species in test_species:
        print(f"\n处理 {species}...")
        result = preprocessor.process_species(species)
        
        if result is not None:
            species_id = list(preprocessor.species_mapping.keys()).index(species)
            preprocessor.save_to_hdf5(result, species_id)
            print(f"✓ {species} 处理完成")
    
    # 保存物种映射
    species_to_id = {s: i for i, s in enumerate(test_species)}
    mapping_file = Path(output_dir) / "species_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump({
            'species_to_id': species_to_id,
            'id_to_species': {str(v): k for k, v in species_to_id.items()},
            'num_species': len(test_species)
        }, f, indent=2)
    
    print(f"\n预处理完成！数据保存在: {output_dir}")
    
    # 测试读取
    print("\n测试读取预处理数据...")
    from src.preprocessed_dataset import PreprocessedDataset
    
    dataset = PreprocessedDataset(
        data_dir=output_dir,
        cache_in_memory=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"样本0:")
        print(f"  input_ids: {sample['input_ids'].shape}")
        print(f"  annotations: {sample['annotations'].shape}")
        print(f"  species_id: {sample['species_id']}")
    
    print("\n测试成功！")

if __name__ == "__main__":
    test_small_preprocess() 