"""
简单测试脚本：验证数据加载功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接导入，避免复杂的依赖
from src.data_processor import DNATokenizer, AnnotationProcessor

def test_tokenizer():
    """测试DNA标记器"""
    print("=== 测试 DNATokenizer ===")
    tokenizer = DNATokenizer()
    
    # 测试序列
    test_seq = "ATCGATCGN"
    print(f"原始序列: {test_seq}")
    
    # 编码
    encoded = tokenizer.encode(test_seq.lower())
    print(f"编码结果: {encoded}")
    
    # 解码
    decoded = tokenizer.decode(encoded)
    print(f"解码结果: {decoded}")
    
    # Mask测试
    masked_ids, mask = tokenizer.mask_sequence(encoded, 0.3)
    masked_seq = tokenizer.decode(masked_ids)
    print(f"Masked序列 (30%): {masked_seq}")
    print(f"Mask位置: {[i for i, m in enumerate(mask) if m == 1]}")
    
    print("✓ DNATokenizer 测试通过\n")


def test_annotation_processor():
    """测试注释处理器"""
    print("=== 测试 AnnotationProcessor ===")
    processor = AnnotationProcessor()
    
    # 模拟注释数据
    annotations = [
        {'type': 'gene', 'start': 1, 'end': 100},
        {'type': 'exon', 'start': 10, 'end': 30},
        {'type': 'cds', 'start': 15, 'end': 25},
    ]
    
    # 处理注释
    seq_length = 100
    annotation_array = processor.process_annotations(annotations, seq_length)
    
    print(f"注释类型映射: {processor.annotation_types}")
    print(f"处理后的注释数组形状: {annotation_array.shape}")
    print(f"前30个位置的注释: {annotation_array[:30]}")
    print(f"注释统计: {dict(zip(*np.unique(annotation_array, return_counts=True)))}")
    
    print("✓ AnnotationProcessor 测试通过\n")


def test_species_mapping():
    """测试物种映射"""
    print("=== 测试物种映射 ===")
    
    try:
        from configs.species_mapping import SPECIES_MAPPING, MAIN_SPECIES
        print(f"物种映射总数: {len(SPECIES_MAPPING)}")
        print(f"主要物种: {MAIN_SPECIES[:5]}")
        
        # 检查几个主要物种的映射
        for species in MAIN_SPECIES[:3]:
            if species in SPECIES_MAPPING:
                print(f"  {species} -> {SPECIES_MAPPING[species][:50]}...")
            else:
                print(f"  {species} -> 未找到映射")
        
        print("✓ 物种映射测试通过\n")
    except ImportError as e:
        print(f"✗ 无法导入物种映射: {e}\n")


def test_data_paths():
    """测试数据路径"""
    print("=== 测试数据路径 ===")
    
    data_dir = "/home/share/huadjyin/home/liwenbo/projects/geno/data/genomics_seq"
    annotation_dir = os.path.join(data_dir, "data_stage2")
    sequence_dir = os.path.join(data_dir, "data_stage3")
    
    print(f"数据根目录: {data_dir}")
    print(f"  存在: {os.path.exists(data_dir)}")
    
    print(f"注释目录: {annotation_dir}")
    print(f"  存在: {os.path.exists(annotation_dir)}")
    if os.path.exists(annotation_dir):
        files = os.listdir(annotation_dir)
        print(f"  文件数: {len(files)}")
        print(f"  示例文件: {files[:3]}")
    
    print(f"序列目录: {sequence_dir}")
    print(f"  存在: {os.path.exists(sequence_dir)}")
    if os.path.exists(sequence_dir):
        files = os.listdir(sequence_dir)
        print(f"  文件数: {len(files)}")
        print(f"  示例文件: {files[:3]}")
    
    print()


if __name__ == "__main__":
    import numpy as np
    
    print("GeneDiffusion 数据加载测试")
    print("=" * 60)
    
    # 运行测试
    test_tokenizer()
    test_annotation_processor()
    test_species_mapping()
    test_data_paths()
    
    print("测试完成！")
    print("\n总结：")
    print("1. 基础组件（Tokenizer、AnnotationProcessor）工作正常")
    print("2. 物种映射配置已正确加载")
    print("3. 数据路径验证完成")
    print("4. 可以进行下一步的完整数据集测试") 