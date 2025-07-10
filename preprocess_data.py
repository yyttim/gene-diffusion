#!/usr/bin/env python3
"""
数据预处理脚本 - 将基因组数据预处理并保存为高效格式
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
import json
from collections import defaultdict
import multiprocessing as mp
from functools import partial

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processor import DNATokenizer, AnnotationProcessor, GenomicDataset
from configs.species_mapping import SPECIES_MAPPING

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        seq_length: int = 10000,
        overlap: float = 0.8,
        max_sequences_per_species: int = 15000,
        num_workers: int = 8
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.seq_length = seq_length
        self.overlap = overlap
        self.stride = int(seq_length * (1 - overlap))
        self.max_sequences_per_species = max_sequences_per_species
        self.num_workers = num_workers
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化tokenizer和处理器
        self.tokenizer = DNATokenizer(k=6)
        self.annotation_processor = AnnotationProcessor()
        
        # 物种映射
        self.species_mapping = SPECIES_MAPPING
        
        logger.info(f"数据预处理器初始化完成")
        logger.info(f"  序列长度: {seq_length}")
        logger.info(f"  重叠率: {overlap}")
        logger.info(f"  步长: {self.stride}")
        logger.info(f"  每个物种最大序列数: {max_sequences_per_species}")
    
    def process_species(self, species: str) -> dict:
        """处理单个物种的数据"""
        logger.info(f"处理物种: {species}")
        
        # 文件路径
        ann_file = self.data_dir / "data_stage2" / f"{species}_annotate.csv"
        seq_file = self.data_dir / "data_stage3" / self.species_mapping[species]
        
        if not ann_file.exists() or not seq_file.exists():
            logger.warning(f"跳过 {species}: 文件不存在")
            return None
        
        # 加载注释
        annotations_dict = self._load_annotations(ann_file)
        
        # 处理序列
        fragments = []
        
        with open(seq_file, 'r') as f:
            reader = pd.read_csv(f, chunksize=100)  # 分块读取
            
            for chunk in reader:
                for _, row in chunk.iterrows():
                    seq_id = row['Sequence ID']
                    sequence = row['Sequence']
                    seq_len = len(sequence)
                    
                    # 获取该序列的注释
                    seq_annotations = annotations_dict.get(seq_id, [])
                    
                    # 滑动窗口切分
                    for start in range(0, seq_len - self.seq_length + 1, self.stride):
                        end = start + self.seq_length
                        
                        # 提取片段
                        fragment_seq = sequence[start:end].lower()
                        
                        # 编码序列
                        token_ids = self.tokenizer.encode(fragment_seq)
                        
                        # 处理注释
                        fragment_annotations = []
                        for ann in seq_annotations:
                            if ann['type'].lower() in ['exon', 'cds', 'lnc_rna', 'gene', 'trna']:
                                if ann['start'] < end and ann['end'] > start:
                                    fragment_annotations.append({
                                        'type': ann['type'],
                                        'start': max(ann['start'] - start, 1),
                                        'end': min(ann['end'] - start, self.seq_length)
                                    })
                        
                        # 处理注释为6-mer级别
                        annotation_array = self.annotation_processor.process_annotations_for_kmers(
                            fragment_annotations, 
                            self.seq_length,
                            k=6
                        )
                        
                        # 确保长度匹配
                        if len(annotation_array) > len(token_ids):
                            annotation_array = annotation_array[:len(token_ids)]
                        elif len(annotation_array) < len(token_ids):
                            padding = np.zeros(len(token_ids) - len(annotation_array), dtype=np.int32)
                            annotation_array = np.concatenate([annotation_array, padding])
                        
                        fragments.append({
                            'token_ids': np.array(token_ids, dtype=np.int32),
                            'annotations': annotation_array,
                            'seq_id': seq_id,
                            'start': start,
                            'end': end
                        })
                        
                        # 达到最大数量则停止
                        if len(fragments) >= self.max_sequences_per_species:
                            break
                
                if len(fragments) >= self.max_sequences_per_species:
                    break
        
        # 随机采样
        if len(fragments) > self.max_sequences_per_species:
            indices = np.random.choice(len(fragments), self.max_sequences_per_species, replace=False)
            fragments = [fragments[i] for i in indices]
        
        logger.info(f"{species}: 生成了 {len(fragments)} 个片段")
        
        return {
            'species': species,
            'fragments': fragments
        }
    
    def _load_annotations(self, ann_file: Path) -> dict:
        """加载注释文件"""
        annotations = defaultdict(list)
        
        df = pd.read_csv(ann_file)
        for _, row in df.iterrows():
            seq_id = row['Sequence ID']
            try:
                ann = {
                    'type': row['GFF_Type'],
                    'start': int(row['GFF_Start']),
                    'end': int(row['GFF_End'])
                }
                annotations[seq_id].append(ann)
            except (KeyError, ValueError):
                continue
        
        return dict(annotations)
    
    def save_to_hdf5(self, species_data: dict, species_id: int):
        """保存单个物种的数据到HDF5文件"""
        if species_data is None:
            return
        
        species = species_data['species']
        fragments = species_data['fragments']
        
        if not fragments:
            return
        
        # 创建HDF5文件
        output_file = self.output_dir / f"{species}.h5"
        
        with h5py.File(output_file, 'w') as f:
            # 保存元数据
            f.attrs['species'] = species
            f.attrs['species_id'] = species_id
            f.attrs['num_fragments'] = len(fragments)
            f.attrs['seq_length'] = self.seq_length
            f.attrs['tokenizer_k'] = 6
            
            # 创建数据集
            max_token_len = max(len(frag['token_ids']) for frag in fragments)
            
            # 创建数据集
            token_ids_dataset = f.create_dataset(
                'token_ids',
                shape=(len(fragments), max_token_len),
                dtype='i4',
                compression='gzip',
                compression_opts=4
            )
            
            annotations_dataset = f.create_dataset(
                'annotations',
                shape=(len(fragments), max_token_len),
                dtype='i4',
                compression='gzip',
                compression_opts=4
            )
            
            # 元信息
            seq_ids = []
            starts = []
            ends = []
            
            # 保存数据
            for i, frag in enumerate(fragments):
                token_len = len(frag['token_ids'])
                token_ids_dataset[i, :token_len] = frag['token_ids']
                annotations_dataset[i, :token_len] = frag['annotations']
                
                seq_ids.append(frag['seq_id'])
                starts.append(frag['start'])
                ends.append(frag['end'])
            
            # 保存元信息
            f.create_dataset('seq_ids', data=np.array(seq_ids, dtype='S100'))
            f.create_dataset('starts', data=np.array(starts, dtype='i4'))
            f.create_dataset('ends', data=np.array(ends, dtype='i4'))
        
        logger.info(f"保存 {species} 数据到 {output_file}")
    
    def process_all_species(self):
        """处理所有物种"""
        species_list = list(self.species_mapping.keys())
        logger.info(f"准备处理 {len(species_list)} 个物种")
        
        # 创建物种ID映射
        species_to_id = {species: i for i, species in enumerate(species_list)}
        
        # 保存物种映射
        mapping_file = self.output_dir / "species_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump({
                'species_to_id': species_to_id,
                'id_to_species': {str(v): k for k, v in species_to_id.items()},
                'num_species': len(species_list)
            }, f, indent=2)
        
        # 使用多进程处理
        if self.num_workers > 1:
            with mp.Pool(self.num_workers) as pool:
                # 处理每个物种
                process_func = partial(self.process_species)
                results = []
                
                with tqdm(total=len(species_list), desc="处理物种") as pbar:
                    for result in pool.imap(process_func, species_list):
                        if result is not None:
                            species = result['species']
                            species_id = species_to_id[species]
                            self.save_to_hdf5(result, species_id)
                        pbar.update(1)
        else:
            # 单进程处理
            for species in tqdm(species_list, desc="处理物种"):
                result = self.process_species(species)
                if result is not None:
                    species_id = species_to_id[species]
                    self.save_to_hdf5(result, species_id)
        
        # 统计信息
        self._generate_stats()
    
    def _generate_stats(self):
        """生成统计信息"""
        stats = {
            'total_species': 0,
            'total_fragments': 0,
            'species_stats': {}
        }
        
        # 遍历所有HDF5文件
        for h5_file in self.output_dir.glob("*.h5"):
            with h5py.File(h5_file, 'r') as f:
                species = f.attrs['species']
                num_fragments = f.attrs['num_fragments']
                
                stats['total_species'] += 1
                stats['total_fragments'] += num_fragments
                stats['species_stats'][species] = {
                    'num_fragments': int(num_fragments),
                    'file_size_mb': h5_file.stat().st_size / (1024 * 1024)
                }
        
        # 保存统计信息
        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"数据集统计:")
        logger.info(f"  总物种数: {stats['total_species']}")
        logger.info(f"  总片段数: {stats['total_fragments']:,}")
        logger.info(f"  平均每个物种: {stats['total_fragments'] / max(stats['total_species'], 1):,.0f} 个片段")


def main():
    parser = argparse.ArgumentParser(description="预处理基因组数据")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/genomics_seq",
        help="原始数据目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/preprocessed",
        help="预处理数据输出目录"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=10000,
        help="序列长度"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.8,
        help="滑动窗口重叠率"
    )
    parser.add_argument(
        "--max_sequences_per_species",
        type=int,
        default=15000,
        help="每个物种的最大序列数"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="并行处理的进程数"
    )
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = DataPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seq_length=args.seq_length,
        overlap=args.overlap,
        max_sequences_per_species=args.max_sequences_per_species,
        num_workers=args.num_workers
    )
    
    # 处理所有物种
    preprocessor.process_all_species()
    
    logger.info("数据预处理完成！")


if __name__ == "__main__":
    main() 