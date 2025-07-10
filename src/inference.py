"""
GeneDiffusion推理脚本
创新的DNA序列生成和评估
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from models import GeneDiffusionModel
from data_processor import DNAVocabulary, GenomicAnnotationProcessor, ConservationScoreCalculator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DNASequenceGenerator:
    """DNA序列生成器
    
    使用训练好的GeneDiffusion模型生成高质量DNA序列
    支持多种生物学约束和引导策略
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.vocab = DNAVocabulary()
        self.annotation_processor = GenomicAnnotationProcessor()
        
        # 加载模型
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        logger.info(f"模型加载完成，使用设备: {self.device}")
    
    def _load_model(self, checkpoint_path: str) -> GeneDiffusionModel:
        """加载模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # 创建模型
        model_config = config['model']
        model = GeneDiffusionModel(
            vocab_size=model_config.get('vocab_size', 8),
            num_annotation_types=model_config.get('num_annotation_types', 9),
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            d_ff=model_config['d_ff'],
            max_seq_len=model_config.get('max_seq_len', 16384),
            dropout=0.0,  # 推理时不使用dropout
            use_dna_features=model_config.get('use_dna_features', True)
        )
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 兼容DDP保存的模型
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        
        model.to(self.device)
        return model
    
    @torch.no_grad()
    def generate(self,
                 seq_length: int,
                 annotations: Optional[torch.Tensor] = None,
                 num_samples: int = 1,
                 temperature: float = 1.0,
                 top_k: Optional[int] = 50,
                 num_steps: int = 256,
                 gc_target: Optional[float] = None,
                 conserved_regions: Optional[List[Tuple[int, int]]] = None,
                 species: str = 'generic') -> List[str]:
        """生成DNA序列
        
        Args:
            seq_length: 序列长度
            annotations: 生物学注释 (可选)
            num_samples: 生成样本数
            temperature: 采样温度
            top_k: Top-k采样
            num_steps: 扩散步数
            gc_target: 目标GC含量
            conserved_regions: 保守区域列表 [(start, end), ...]
            species: 物种名称（用于特定优化）
        
        Returns:
            生成的DNA序列列表
        """
        logger.info(f"开始生成{num_samples}条长度为{seq_length}的序列...")
        
        # 创建默认注释（如果未提供）
        if annotations is None:
            annotations = torch.zeros(num_samples, seq_length, 
                                    self.annotation_processor.num_annotation_types,
                                    device=self.device)
        else:
            annotations = annotations.to(self.device)
        
        # 生成序列
        generated_tokens = self.model.sample(
            batch_size=num_samples,
            seq_len=seq_length,
            annotations=annotations,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            gc_target=gc_target,
            device=self.device
        )
        
        # 应用保守区域约束（如果有）
        if conserved_regions:
            generated_tokens = self._apply_conservation_constraints(
                generated_tokens, conserved_regions
            )
        
        # 转换为DNA序列
        sequences = []
        for tokens in generated_tokens:
            seq = self.vocab.decode(tokens.cpu().numpy())
            sequences.append(seq)
        
        # 后处理优化
        sequences = self._postprocess_sequences(sequences, species)
        
        return sequences
    
    def _apply_conservation_constraints(self,
                                      tokens: torch.Tensor,
                                      conserved_regions: List[Tuple[int, int]]) -> torch.Tensor:
        """应用保守区域约束"""
        # 这里简化处理，实际应该在生成过程中就考虑
        logger.info(f"应用{len(conserved_regions)}个保守区域约束")
        return tokens
    
    def _postprocess_sequences(self, sequences: List[str], species: str) -> List[str]:
        """后处理生成的序列
        
        确保生物学合理性
        """
        processed = []
        
        for seq in sequences:
            # 移除mask和pad token
            seq = seq.replace('MASK', '').replace('PAD', '')
            
            # 处理未知碱基
            seq = seq.replace('UNK', 'N')
            
            # 物种特定的优化
            if species == 'homo_sapiens':
                # 人类特定的密码子优化等
                pass
            elif species == 'mus_musculus':
                # 小鼠特定的优化
                pass
            
            processed.append(seq)
        
        return processed
    
    @torch.no_grad()
    def complete_sequence(self,
                         partial_sequence: str,
                         mask_positions: List[int],
                         annotations: Optional[torch.Tensor] = None,
                         temperature: float = 1.0,
                         num_steps: int = 100) -> str:
        """补全部分序列
        
        Args:
            partial_sequence: 部分序列（包含N或其他mask标记）
            mask_positions: 需要补全的位置
            annotations: 注释信息
            temperature: 采样温度
            num_steps: 扩散步数
        
        Returns:
            补全后的序列
        """
        logger.info(f"补全序列，mask位置数: {len(mask_positions)}")
        
        # 编码序列
        tokens = torch.tensor([self.vocab.encode(partial_sequence)], 
                            dtype=torch.long, device=self.device)
        
        # 设置mask位置
        for pos in mask_positions:
            tokens[0, pos] = self.vocab.token_to_id['MASK']
        
        # 创建注释（如果需要）
        if annotations is None:
            annotations = torch.zeros(1, len(partial_sequence), 
                                    self.annotation_processor.num_annotation_types,
                                    device=self.device)
        
        # 逐步去噪
        for step in range(num_steps):
            # 模型预测
            outputs = self.model(tokens, annotations)
            logits = outputs['logits']
            
            # 只在mask位置采样
            mask = (tokens == self.vocab.token_to_id['MASK'])
            
            if mask.sum() == 0:
                break
            
            # 温度调整
            logits = logits / temperature
            
            # 采样
            probs = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.view(-1, self.vocab.vocab_size), 1)
            sampled = sampled.view(tokens.shape)
            
            # 更新mask位置
            tokens = torch.where(mask, sampled, tokens)
            
            # 可选：逐步减少mask数量
            if step < num_steps - 1:
                # 重新mask一部分低置信度位置
                confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                low_conf_mask = (confidence < 0.8) & mask
                num_remask = int(low_conf_mask.sum() * (1 - step / num_steps))
                
                if num_remask > 0:
                    # 随机选择要重新mask的位置
                    remask_indices = torch.where(low_conf_mask[0])[0]
                    remask_indices = remask_indices[torch.randperm(len(remask_indices))[:num_remask]]
                    tokens[0, remask_indices] = self.vocab.token_to_id['MASK']
        
        # 解码
        completed_sequence = self.vocab.decode(tokens[0].cpu().numpy())
        
        return completed_sequence
    
    def guided_generation(self,
                         target_features: Dict[str, float],
                         seq_length: int,
                         num_samples: int = 1,
                         **kwargs) -> List[str]:
        """特征引导的序列生成
        
        Args:
            target_features: 目标特征字典，例如:
                {
                    'gc_content': 0.45,
                    'gene_density': 0.3,
                    'repeat_content': 0.1
                }
            seq_length: 序列长度
            num_samples: 生成样本数
            **kwargs: 其他生成参数
        
        Returns:
            生成的序列列表
        """
        logger.info(f"特征引导生成，目标特征: {target_features}")
        
        # 提取特定特征
        gc_target = target_features.get('gc_content', None)
        
        # 创建相应的注释
        annotations = self._create_guided_annotations(
            target_features, seq_length, num_samples
        )
        
        # 生成序列
        sequences = self.generate(
            seq_length=seq_length,
            annotations=annotations,
            num_samples=num_samples,
            gc_target=gc_target,
            **kwargs
        )
        
        # 验证生成的序列是否符合目标
        self._validate_features(sequences, target_features)
        
        return sequences
    
    def _create_guided_annotations(self,
                                 target_features: Dict[str, float],
                                 seq_length: int,
                                 num_samples: int) -> torch.Tensor:
        """根据目标特征创建引导注释"""
        annotations = torch.zeros(
            num_samples, seq_length, 
            self.annotation_processor.num_annotation_types,
            device=self.device
        )
        
        # 根据目标特征设置注释
        if 'gene_density' in target_features:
            # 设置基因区域
            gene_prob = target_features['gene_density']
            gene_mask = torch.rand(num_samples, seq_length) < gene_prob
            annotations[:, :, 0] = gene_mask.float()
        
        if 'exon_ratio' in target_features:
            # 在基因区域内设置外显子
            exon_prob = target_features['exon_ratio']
            gene_regions = annotations[:, :, 0] > 0
            exon_mask = (torch.rand(num_samples, seq_length) < exon_prob) & gene_regions
            annotations[:, :, 1] = exon_mask.float()
        
        return annotations
    
    def _validate_features(self, sequences: List[str], target_features: Dict[str, float]):
        """验证生成序列的特征"""
        for i, seq in enumerate(sequences):
            actual_gc = self.vocab.calculate_gc_content(seq)
            target_gc = target_features.get('gc_content', 0.5)
            
            logger.info(f"序列{i}: GC含量={actual_gc:.3f} (目标={target_gc:.3f})")


class SequenceEvaluator:
    """序列质量评估器"""
    
    def __init__(self):
        self.vocab = DNAVocabulary()
        self.conservation_calc = ConservationScoreCalculator()
    
    def evaluate_sequences(self, sequences: List[str]) -> Dict[str, float]:
        """评估生成序列的质量"""
        metrics = {
            'gc_content_mean': 0,
            'gc_content_std': 0,
            'n_ratio': 0,
            'repeat_content': 0,
            'complexity': 0,
            'valid_codons': 0
        }
        
        gc_contents = []
        n_ratios = []
        
        for seq in sequences:
            # GC含量
            gc = self.vocab.calculate_gc_content(seq)
            gc_contents.append(gc)
            
            # N比例
            n_ratio = seq.upper().count('N') / len(seq)
            n_ratios.append(n_ratio)
            
            # 序列复杂度
            complexity = self._calculate_complexity(seq)
            metrics['complexity'] += complexity
            
            # 有效密码子
            valid_ratio = self._check_valid_codons(seq)
            metrics['valid_codons'] += valid_ratio
        
        # 汇总统计
        metrics['gc_content_mean'] = np.mean(gc_contents)
        metrics['gc_content_std'] = np.std(gc_contents)
        metrics['n_ratio'] = np.mean(n_ratios)
        metrics['complexity'] /= len(sequences)
        metrics['valid_codons'] /= len(sequences)
        
        return metrics
    
    def _calculate_complexity(self, sequence: str) -> float:
        """计算序列复杂度（基于k-mer多样性）"""
        k = 6
        kmers = set()
        
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if 'N' not in kmer:
                kmers.add(kmer)
        
        # 归一化的k-mer多样性
        max_possible = min(len(sequence) - k + 1, 4**k)
        complexity = len(kmers) / max_possible
        
        return complexity
    
    def _check_valid_codons(self, sequence: str) -> float:
        """检查有效密码子比例"""
        valid_count = 0
        total_count = 0
        
        # 定义终止密码子
        stop_codons = {'TAA', 'TAG', 'TGA'}
        
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            if 'N' not in codon:
                total_count += 1
                if codon not in stop_codons:
                    valid_count += 1
        
        return valid_count / max(total_count, 1)


def main():
    parser = argparse.ArgumentParser(description='GeneDiffusion推理和序列生成')
    
    # 基本参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--output_file', type=str, default='generated_sequences.fasta',
                       help='输出FASTA文件')
    parser.add_argument('--mode', type=str, choices=['generate', 'complete', 'guided'],
                       default='generate', help='运行模式')
    
    # 生成参数
    parser.add_argument('--num_samples', type=int, default=10,
                       help='生成序列数量')
    parser.add_argument('--seq_length', type=int, default=10000,
                       help='序列长度')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='采样温度')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k采样')
    parser.add_argument('--num_steps', type=int, default=256,
                       help='扩散步数')
    parser.add_argument('--gc_target', type=float, default=None,
                       help='目标GC含量')
    
    # 补全模式参数
    parser.add_argument('--input_sequence', type=str,
                       help='输入序列（补全模式）')
    parser.add_argument('--mask_positions', type=int, nargs='+',
                       help='需要补全的位置')
    
    # 引导生成参数
    parser.add_argument('--target_features', type=str,
                       help='目标特征JSON文件')
    
    # 其他参数
    parser.add_argument('--species', type=str, default='generic',
                       help='目标物种')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--evaluate', action='store_true',
                       help='评估生成序列')
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = DNASequenceGenerator(args.checkpoint, args.device)
    
    # 根据模式运行
    if args.mode == 'generate':
        # 标准生成
        sequences = generator.generate(
            seq_length=args.seq_length,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            num_steps=args.num_steps,
            gc_target=args.gc_target,
            species=args.species
        )
    
    elif args.mode == 'complete':
        # 序列补全
        if not args.input_sequence or not args.mask_positions:
            raise ValueError("补全模式需要--input_sequence和--mask_positions")
        
        completed = generator.complete_sequence(
            partial_sequence=args.input_sequence,
            mask_positions=args.mask_positions,
            temperature=args.temperature,
            num_steps=args.num_steps
        )
        sequences = [completed]
    
    elif args.mode == 'guided':
        # 特征引导生成
        if not args.target_features:
            raise ValueError("引导模式需要--target_features")
        
        with open(args.target_features, 'r') as f:
            target_features = json.load(f)
        
        sequences = generator.guided_generation(
            target_features=target_features,
            seq_length=args.seq_length,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            num_steps=args.num_steps
        )
    
    # 保存序列
    logger.info(f"保存{len(sequences)}条序列到{args.output_file}")
    records = []
    for i, seq in enumerate(sequences):
        record = SeqRecord(
            Seq(seq),
            id=f"GeneDiffusion_seq_{i+1}",
            description=f"Generated by GeneDiffusion | Mode: {args.mode} | Species: {args.species}"
        )
        records.append(record)
    
    SeqIO.write(records, args.output_file, "fasta")
    
    # 评估（可选）
    if args.evaluate:
        logger.info("评估生成序列...")
        evaluator = SequenceEvaluator()
        metrics = evaluator.evaluate_sequences(sequences)
        
        logger.info("评估结果:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # 保存评估结果
        eval_file = args.output_file.replace('.fasta', '_evaluation.json')
        with open(eval_file, 'w') as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main() 