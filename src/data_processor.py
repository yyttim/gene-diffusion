"""
GeneDiffusion数据处理模块
处理哺乳动物基因组序列和注释数据
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import os
import csv
import sys
import random
from collections import defaultdict
from pathlib import Path
import logging

# 增加CSV字段大小限制
csv.field_size_limit(sys.maxsize)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DNATokenizer:
    """DNA序列标记器 - 使用6-mer作为token"""
    
    def __init__(self, k: int = 6):
        self.k = k  # k-mer长度
        
        # 构建所有可能的k-mer词汇表
        bases = ['a', 'c', 'g', 't']
        self.vocab = {}
        self.id_to_token = {}
        
        # 生成所有可能的k-mer
        from itertools import product
        idx = 0
        for kmer_tuple in product(bases, repeat=k):
            kmer = ''.join(kmer_tuple)
            self.vocab[kmer] = idx
            self.id_to_token[idx] = kmer
            idx += 1
        
        # 添加特殊token
        self.mask_token = '[MASK]'
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        
        self.mask_token_id = idx
        self.pad_token_id = idx + 1
        self.unk_token_id = idx + 2
        
        self.vocab[self.mask_token] = self.mask_token_id
        self.vocab[self.pad_token] = self.pad_token_id
        self.vocab[self.unk_token] = self.unk_token_id
        
        self.id_to_token[self.mask_token_id] = self.mask_token
        self.id_to_token[self.pad_token_id] = self.pad_token
        self.id_to_token[self.unk_token_id] = self.unk_token
        
        self.vocab_size = len(self.vocab)
        logger.info(f"创建了{k}-mer词汇表，大小: {self.vocab_size}")
        
    def encode(self, sequence: str) -> List[int]:
        """编码DNA序列为k-mer token IDs"""
        sequence = sequence.lower()
        token_ids = []
        
        # 将序列切分为k-mer
        for i in range(0, len(sequence) - self.k + 1, self.k):
            kmer = sequence[i:i+self.k]
            
            # 处理包含N或其他非标准碱基的k-mer
            if 'n' in kmer or any(base not in 'acgt' for base in kmer):
                token_ids.append(self.unk_token_id)
            else:
                token_ids.append(self.vocab.get(kmer, self.unk_token_id))
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """解码token IDs为DNA序列"""
        decoded = []
        for tid in token_ids:
            if tid < len(self.id_to_token) - 3:  # 排除特殊token
                decoded.append(self.id_to_token.get(tid, ''))
            # 跳过特殊token
        return ''.join(decoded)
    
    def mask_sequence(self, token_ids: List[int], mask_ratio: float) -> Tuple[List[int], List[int]]:
        """对序列进行掩码"""
        length = len(token_ids)
        num_mask = int(length * mask_ratio)
        
        # 随机选择要掩码的位置
        mask_indices = random.sample(range(length), num_mask)
        masked_sequence = token_ids.copy()
        
        # 创建掩码
        mask = [0] * length
        for idx in mask_indices:
            masked_sequence[idx] = self.mask_token_id
            mask[idx] = 1
            
        return masked_sequence, mask
    
    def calculate_gc_content(self, sequence: str) -> float:
        """计算序列的GC含量"""
        sequence = sequence.lower()
        gc_count = sequence.count('g') + sequence.count('c')
        total_count = len(sequence.replace('n', ''))
        return gc_count / total_count if total_count > 0 else 0.0


class AnnotationProcessor:
    """注释处理器"""
    
    def __init__(self):
        # 目标注释类型映射
        self.annotation_types = {
            'none': 0,      # 无注释
            'exon': 1,      
            'cds': 2,       
            'lnc_rna': 3,   
            'gene': 4,      
            'trna': 5       
        }
        self.num_types = len(self.annotation_types)
        
        # 注释优先级（数字越大优先级越高）
        self.priority = {
            'cds': 5,      # CDS优先级最高
            'exon': 4,     
            'trna': 3,     
            'lnc_rna': 2,  
            'gene': 1,     
            'none': 0
        }
    
    def process_annotations(self, annotations: List[Dict], seq_length: int) -> np.ndarray:
        """处理注释，返回每个位置的注释类型"""
        # 初始化为无注释
        annotation_array = np.zeros(seq_length, dtype=np.int32)
        
        # 按优先级排序注释
        sorted_annotations = sorted(annotations, 
                                  key=lambda x: self.priority.get(x['type'].lower(), 0))
        
        # 应用注释
        for ann in sorted_annotations:
            ann_type = ann['type'].lower()
            if ann_type in self.annotation_types:
                start = max(0, ann['start'] - 1)  # GFF是1-based
                end = min(seq_length, ann['end'])
                annotation_array[start:end] = self.annotation_types[ann_type]
        
        return annotation_array

    def process_annotations_for_kmers(self, annotations: List[Dict], seq_length: int, k: int = 6) -> np.ndarray:
        """处理注释，返回每个k-mer位置的注释类型"""
        # 首先获取碱基级别的注释
        base_annotations = self.process_annotations(annotations, seq_length)
        
        # 转换为k-mer级别的注释
        kmer_length = (seq_length - k + 1) // k + 1  # 计算k-mer序列长度
        kmer_annotations = np.zeros(kmer_length, dtype=np.int32)
        
        # 为每个k-mer分配注释（使用该k-mer覆盖区域的主要注释）
        for i in range(kmer_length):
            start = i * k
            end = min(start + k, seq_length)
            
            # 获取这个k-mer覆盖的碱基注释
            kmer_region = base_annotations[start:end]
            
            # 使用众数作为k-mer的注释（如果有多个注释类型，选择优先级最高的）
            unique_types, counts = np.unique(kmer_region, return_counts=True)
            if len(unique_types) > 0:
                # 如果有多个类型相同数量，选择优先级最高的
                max_count = counts.max()
                candidates = unique_types[counts == max_count]
                
                # 根据优先级选择
                best_type = candidates[0]
                best_priority = self.priority.get(list(self.annotation_types.keys())[list(self.annotation_types.values()).index(best_type)], 0)
                
                for candidate in candidates[1:]:
                    type_name = list(self.annotation_types.keys())[list(self.annotation_types.values()).index(candidate)]
                    priority = self.priority.get(type_name, 0)
                    if priority > best_priority:
                        best_type = candidate
                        best_priority = priority
                
                kmer_annotations[i] = best_type
        
        return kmer_annotations


class GenomicDataset(Dataset):
    """基因组数据集"""
    
    def __init__(
        self,
        data_dir: str,
        species_list: Optional[List[str]] = None,
        seq_length: int = 10000,  # 改为10kb
        overlap: float = 0.5,
        min_annotation_ratio: float = 0.0,  # 改为0，不过滤无注释区域
        cache_sequences: bool = False,
        max_sequences_per_species: int = 15000  # 每个物种15000个样本，230物种总计3.45M
    ):
        """
        Args:
            data_dir: 数据根目录
            species_list: 要使用的物种列表，None表示使用所有物种
            seq_length: 序列片段长度（默认10kb）
            overlap: 滑动窗口重叠比例
            min_annotation_ratio: 最小注释比例（设为0表示包含所有区域）
            cache_sequences: 是否缓存序列数据
            max_sequences_per_species: 每个物种最多使用的序列数
        """
        self.data_dir = Path(data_dir)
        self.annotation_dir = self.data_dir / "data_stage2"
        self.sequence_dir = self.data_dir / "data_stage3"
        self.seq_length = seq_length
        self.overlap = overlap
        self.stride = int(seq_length * (1 - overlap))
        self.min_annotation_ratio = min_annotation_ratio
        self.cache_sequences = cache_sequences
        self.max_sequences_per_species = max_sequences_per_species
        
        self.tokenizer = DNATokenizer()
        self.annotation_processor = AnnotationProcessor()
        
        # 加载物种映射
        self.species_mapping = self._load_species_mapping()
        
        # 确定要使用的物种
        if species_list is None:
            self.species_list = list(self.species_mapping.keys())
        else:
            self.species_list = [s for s in species_list if s in self.species_mapping]
        
        logger.info(f"使用 {len(self.species_list)} 个物种的数据")
        
        # 构建数据索引
        self.data_index = self._build_data_index()
        logger.info(f"总共 {len(self.data_index)} 个数据片段")
        
        # 缓存
        self._sequence_cache = {} if cache_sequences else None
    
    def _load_species_mapping(self) -> Dict[str, str]:
        """加载物种文件映射"""
        # 从配置文件加载预定义的映射
        try:
            # 导入物种映射配置
            import sys
            sys.path.append(str(self.data_dir.parent))  # 添加gene-diffusion目录到路径
            from configs.species_mapping import SPECIES_MAPPING
            
            # 验证文件是否存在
            valid_mapping = {}
            for species, seq_file in SPECIES_MAPPING.items():
                ann_file = self.annotation_dir / f"{species}_annotate.csv"
                seq_path = self.sequence_dir / seq_file
                
                if ann_file.exists() and seq_path.exists():
                    valid_mapping[species] = seq_file
                else:
                    if not ann_file.exists():
                        logger.warning(f"未找到 {species} 的注释文件")
                    if not seq_path.exists():
                        logger.warning(f"未找到 {species} 的序列文件: {seq_file}")
            
            logger.info(f"成功加载 {len(valid_mapping)} 个物种的映射")
            return valid_mapping
            
        except ImportError:
            logger.warning("无法加载物种映射配置，使用动态扫描")
            # 回退到原始的动态扫描方法
            return self._load_species_mapping_fallback()
    
    def _load_species_mapping_fallback(self) -> Dict[str, str]:
        """备用的物种文件映射方法"""
        mapping = {}
        annotation_files = list(self.annotation_dir.glob("*_annotate.csv"))
        
        for ann_file in annotation_files:
            species = ann_file.stem.replace("_annotate", "")
            seq_files = list(self.sequence_dir.glob("*.csv"))
            
            matched = False
            for seq_file in seq_files:
                if species.replace('_', '') in seq_file.stem.lower():
                    mapping[species] = seq_file.name
                    matched = True
                    break
            
            if not matched:
                logger.warning(f"未找到 {species} 的序列文件")
        
        return mapping
    
    def _build_data_index(self) -> List[Dict]:
        """构建数据索引 - 确保每个物种使用相同数量的数据"""
        # 第一步：收集每个物种的所有片段
        species_fragments_dict = {}
        
        for species in self.species_list:
            if species not in self.species_mapping:
                continue
                
            ann_file = self.annotation_dir / f"{species}_annotate.csv"
            seq_file = self.sequence_dir / self.species_mapping[species]
            
            if not ann_file.exists() or not seq_file.exists():
                logger.warning(f"跳过 {species}：文件不存在")
                continue
            
            # 加载注释
            annotations = self._load_annotations(ann_file)
            
            # 扫描序列文件，创建片段索引
            species_fragments = self._create_fragments(
                seq_file, annotations, species
            )
            
            if len(species_fragments) > 0:
                species_fragments_dict[species] = species_fragments
                logger.info(f"{species}: 生成了 {len(species_fragments)} 个可用片段")
            else:
                logger.warning(f"{species}: 没有生成任何片段")
        
        # 第二步：确定每个物种应该使用的片段数量
        if not species_fragments_dict:
            logger.error("没有任何物种生成了数据片段")
            return []
        
        # 找到所有物种中的最小片段数
        min_fragments = min(len(fragments) for fragments in species_fragments_dict.values())
        
        # 使用min_fragments和max_sequences_per_species中的较小值
        samples_per_species = min(min_fragments, self.max_sequences_per_species)
        
        logger.info(f"每个物种将使用 {samples_per_species} 个片段（最小可用: {min_fragments}）")
        
        # 第三步：从每个物种中随机采样相同数量的片段
        data_index = []
        for species, fragments in species_fragments_dict.items():
            # 随机采样
            sampled_fragments = random.sample(fragments, samples_per_species)
            data_index.extend(sampled_fragments)
            logger.info(f"{species}: 采样了 {len(sampled_fragments)} 个片段")
        
        # 打乱整个数据集
        random.shuffle(data_index)
        
        logger.info(f"数据集构建完成: {len(species_fragments_dict)} 个物种，每个物种 {samples_per_species} 个片段，总计 {len(data_index)} 个片段")
        
        return data_index
    
    def _load_annotations(self, ann_file: Path) -> Dict[str, List[Dict]]:
        """加载注释文件"""
        annotations = defaultdict(list)
        
        with open(ann_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
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
    
    def _create_fragments(
        self, 
        seq_file: Path, 
        annotations: Dict[str, List[Dict]], 
        species: str
    ) -> List[Dict]:
        """为一个物种创建序列片段 - 使用全基因组滑动窗口策略"""
        fragments = []
        
        logger.info(f"处理 {species} 的序列文件（全基因组模式）...")
        processed_sequences = 0
        total_fragments = 0
        
        with open(seq_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                seq_id = row['Sequence ID']
                sequence = row['Sequence']
                seq_len = len(sequence)
                
                # 获取该序列的注释
                seq_annotations = annotations.get(seq_id, [])
                
                # 对所有序列使用滑动窗口策略，不管有没有注释
                for start in range(0, seq_len - self.seq_length + 1, self.stride):
                    end = start + self.seq_length
                    
                    # 获取这个片段覆盖的注释
                    fragment_annotations = []
                    for ann in seq_annotations:
                        # 只关注5种目标注释类型
                        if ann['type'].lower() not in ['exon', 'cds', 'lnc_rna', 'gene', 'trna']:
                            continue
                            
                        # 检查注释是否与片段重叠
                        if ann['start'] < end and ann['end'] > start:
                            fragment_annotations.append({
                                'type': ann['type'],
                                'start': max(ann['start'] - start, 1),
                                'end': min(ann['end'] - start, self.seq_length)
                            })
                    
                    # 计算注释覆盖率
                    annotation_coverage = self._calculate_annotation_coverage(
                        fragment_annotations, self.seq_length
                    )
                    
                    fragments.append({
                        'species': species,
                        'seq_id': seq_id,
                        'start': start,
                        'end': end,
                        'annotations': fragment_annotations,
                        'annotation_coverage': annotation_coverage,
                        'seq_file': seq_file
                    })
                    total_fragments += 1
                
                processed_sequences += 1
                
                # 定期日志
                if processed_sequences % 100 == 0:
                    logger.info(f"{species}: 已处理 {processed_sequences} 条序列，{total_fragments} 个片段")
        
        logger.info(f"{species}: 总共处理 {processed_sequences} 条序列，生成 {total_fragments} 个片段")
        return fragments
    
    def _calculate_annotation_coverage(
        self, 
        annotations: List[Dict], 
        seq_length: int
    ) -> float:
        """计算注释覆盖率"""
        covered = np.zeros(seq_length, dtype=bool)
        
        for ann in annotations:
            start = max(0, ann['start'] - 1)
            end = min(seq_length, ann['end'])
            covered[start:end] = True
        
        return np.mean(covered)
    
    def _load_sequence_fragment(self, fragment_info: Dict) -> str:
        """加载序列片段"""
        seq_file = fragment_info['seq_file']
        seq_id = fragment_info['seq_id']
        start = fragment_info['start']
        end = fragment_info['end']
        
        # 检查缓存
        cache_key = f"{seq_file}:{seq_id}:{start}:{end}"
        if self._sequence_cache is not None and cache_key in self._sequence_cache:
            return self._sequence_cache[cache_key]
        
        # 从文件加载特定片段
        with open(seq_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Sequence ID'] == seq_id:
                    # 只提取需要的片段，避免加载整条序列到内存
                    full_sequence = row['Sequence']
                    
                    # 流式读取片段
                    fragment = full_sequence[start:end].lower()
                    
                    # 缓存片段（而不是整条序列）
                    if self._sequence_cache is not None:
                        self._sequence_cache[cache_key] = fragment
                    
                    return fragment
        
        raise ValueError(f"未找到序列 {seq_id}")
    
    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, idx):
        fragment_info = self.data_index[idx]
        
        # 加载序列片段
        sequence = self._load_sequence_fragment(fragment_info)
        
        # 编码序列为6-mer
        token_ids = self.tokenizer.encode(sequence)
        
        # 处理注释 - 使用6-mer级别的注释
        annotations = self.annotation_processor.process_annotations_for_kmers(
            fragment_info['annotations'], 
            len(sequence),
            k=self.tokenizer.k
        )
        
        # 确保注释长度与token长度匹配
        if len(annotations) > len(token_ids):
            annotations = annotations[:len(token_ids)]
        elif len(annotations) < len(token_ids):
            # 填充
            padding = np.zeros(len(token_ids) - len(annotations), dtype=np.int32)
            annotations = np.concatenate([annotations, padding])
        
        # 获取物种ID
        species_id = self.species_list.index(fragment_info['species'])
        
        # 转换为张量
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'annotations': torch.tensor(annotations, dtype=torch.long),
            'species_id': torch.tensor(species_id, dtype=torch.long),
            'annotation_coverage': torch.tensor(
                fragment_info['annotation_coverage'], 
                dtype=torch.float32
            )
        }

    def export_to_csv(self, output_dir: str, max_samples: Optional[int] = None):
        """导出数据集为CSV格式（符合文档要求）
        
        Args:
            output_dir: 输出目录
            max_samples: 最大样本数（用于测试）
        """
        import os
        import csv
        from pathlib import Path
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备输出数据
        samples_by_species = defaultdict(list)
        
        # 收集所有样本
        total_samples = len(self.data_index) if max_samples is None else min(len(self.data_index), max_samples)
        
        logger.info(f"准备导出 {total_samples} 个样本到CSV...")
        
        for idx in range(total_samples):
            fragment_info = self.data_index[idx]
            species = fragment_info['species']
            
            # 加载序列片段
            sequence = self._load_sequence_fragment(fragment_info)
            
            # 创建标签序列
            labels = ['<unk_token>'] * len(sequence)  # 默认无注释
            
            # 应用注释
            for ann in fragment_info['annotations']:
                ann_type = ann['type'].lower()
                label = f"<t_{ann_type}>"
                
                # 确保边界正确
                start = max(0, ann['start'] - 1)  # 转换为0-based
                end = min(len(sequence), ann['end'] - 1)
                
                # 设置标签
                for i in range(start, end):
                    labels[i] = label
            
            # 确保序列和标签长度一致
            assert len(sequence) == len(labels), f"序列长度({len(sequence)})与标签长度({len(labels)})不一致"
            
            samples_by_species[species].append({
                'sequence_id': f"{fragment_info['seq_id']}_{fragment_info['start']}_{fragment_info['end']}",
                'sequence': sequence,
                'labels': labels
            })
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"已处理 {idx + 1}/{total_samples} 个样本")
        
        # 为每个物种创建CSV文件
        for species, samples in samples_by_species.items():
            output_file = os.path.join(output_dir, f"{species}_labeled.csv")
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sequence_id', 'sequence', 'labels'])
                
                for sample in samples:
                    writer.writerow([
                        sample['sequence_id'],
                        sample['sequence'],
                        str(sample['labels'])  # 转换为字符串表示的列表
                    ])
            
            logger.info(f"导出 {species}: {len(samples)} 个样本到 {output_file}")
        
        logger.info("CSV导出完成！")


def create_data_loader(
    data_dir: str,
    batch_size: int,
    seq_length: int = 10000,  # 改为10kb
    species_list: Optional[List[str]] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """创建数据加载器"""
    
    dataset = GenomicDataset(
        data_dir=data_dir,
        species_list=species_list,
        seq_length=seq_length,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch
    )


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """批处理整理函数"""
    # 提取各个字段
    input_ids = torch.stack([item['input_ids'] for item in batch])
    annotations = torch.stack([item['annotations'] for item in batch])
    species_ids = torch.stack([item['species_id'] for item in batch])
    annotation_coverage = torch.stack([item['annotation_coverage'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'annotations': annotations,
        'species_ids': species_ids,
        'annotation_coverage': annotation_coverage
    }


# 数据增强函数
def reverse_complement(sequence: str) -> str:
    """生成反向互补序列"""
    complement = {'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'}
    return ''.join(complement.get(base, 'n') for base in sequence[::-1])


def augment_sequence(sequence: str, annotations: np.ndarray) -> Tuple[str, np.ndarray]:
    """数据增强：反向互补"""
    if random.random() < 0.5:
        # 反向互补
        aug_seq = reverse_complement(sequence)
        # 注释也需要反转
        aug_ann = annotations[::-1].copy()
        return aug_seq, aug_ann
    return sequence, annotations


if __name__ == "__main__":
    # 测试数据处理
    import sys
    
    data_dir = "/home/share/huadjyin/home/liwenbo/projects/geno/data/genomics_seq"
    
    # 测试tokenizer
    tokenizer = DNATokenizer()
    test_seq = "ATCGATCGN"
    encoded = tokenizer.encode(test_seq)
    decoded = tokenizer.decode(encoded)
    print(f"原始序列: {test_seq}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")
    
    # 测试数据集
    print("\n创建数据集...")
    dataset = GenomicDataset(
        data_dir=data_dir,
        species_list=['homo_sapiens', 'mus_musculus'],
        seq_length=1000,  # 使用较短的长度进行测试
        max_sequences_per_species=10
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 测试一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n样本信息:")
        print(f"  输入形状: {sample['input_ids'].shape}")
        print(f"  注释形状: {sample['annotations'].shape}")
        print(f"  物种ID: {sample['species_id']}")
        print(f"  注释覆盖率: {sample['annotation_coverage']:.2f}") 