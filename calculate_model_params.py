#!/usr/bin/env python3
"""
计算GeneDiffusion模型的参数量（不需要实际创建模型）
"""

import yaml

def calculate_params(config):
    """根据配置计算模型参数量"""
    
    # 从配置获取参数
    vocab_size = config['model']['vocab_size']  # 4099
    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['num_layers']
    num_heads = config['model']['num_heads']
    intermediate_size = config['model']['intermediate_size']
    num_species = config['model']['num_species']  # 230
    num_annotation_types = 5
    
    # 计算各部分参数
    params = {}
    
    # Token embeddings
    params['token_embedding'] = vocab_size * hidden_size
    
    # Position encoding (RotaryPositionEncoding 不包含可训练参数)
    params['position_encoding'] = 0
    
    # Species embeddings (SpeciesAwareEmbedding)
    species_embed_dim = hidden_size  # 简化：使用hidden_size作为物种嵌入维度
    params['species_embedding'] = num_species * species_embed_dim
    if config['model'].get('use_film', False):
        # FiLM parameters: scale and shift for each species
        params['species_film'] = 2 * num_species * hidden_size
    else:
        params['species_film'] = 0
    
    # Annotation encoder
    # 包含embedding层和4层transformer
    params['annotation_embedding'] = (num_annotation_types + 1) * hidden_size
    ann_encoder_layers = 4
    params['annotation_encoder'] = ann_encoder_layers * calculate_transformer_block_params(
        hidden_size, num_heads, intermediate_size, num_species, use_species_cond=True
    )
    
    # Main transformer layers
    params['transformer_layers'] = num_layers * calculate_transformer_block_params(
        hidden_size, num_heads, intermediate_size, num_species, use_species_cond=True
    )
    
    # Output layers
    params['output_norm'] = 2 * hidden_size  # LayerNorm: scale + bias
    params['output_projection'] = hidden_size * vocab_size + vocab_size  # weight + bias
    
    # GC predictor
    params['gc_predictor'] = (
        hidden_size * (hidden_size // 2) + (hidden_size // 2) +  # 第一层
        (hidden_size // 2) * 1 + 1  # 第二层
    )
    
    # 计算总参数量
    total_params = sum(params.values())
    
    return total_params, params

def calculate_transformer_block_params(hidden_size, num_heads, intermediate_size, num_species, use_species_cond=False):
    """计算单个transformer block的参数量"""
    params = 0
    
    # Self-attention
    # Q, K, V projections + output projection
    params += 4 * (hidden_size * hidden_size + hidden_size)
    
    # 如果使用物种条件化的LayerNorm
    if use_species_cond:
        # SpeciesConditionedLayerNorm: 每个物种有独立的scale和bias
        params += 2 * (2 * num_species * hidden_size)  # 两个LayerNorm
    else:
        # 标准LayerNorm
        params += 2 * (2 * hidden_size)  # 两个LayerNorm
    
    # Feed-forward network
    params += hidden_size * intermediate_size + intermediate_size  # 第一层
    params += intermediate_size * hidden_size + hidden_size  # 第二层
    
    # Local sequence encoder (3个不同kernel size的Conv1d)
    kernel_sizes = [3, 5, 7]
    for k in kernel_sizes:
        params += k * hidden_size * (hidden_size // len(kernel_sizes))
    params += hidden_size  # 最后的projection bias
    
    return params

def format_params(num_params):
    """格式化参数数量"""
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    else:
        return str(num_params)

def main():
    """主函数"""
    print("GeneDiffusion 模型参数量计算")
    print("=" * 60)
    
    configs = {
        'small': 'configs/train_config_small.yaml',
        'balanced': 'configs/train_config_balanced.yaml',
        '2b': 'configs/train_config_2b.yaml'
    }
    
    results = {}
    
    for name, config_path in configs.items():
        print(f"\n{name.upper()} 模型配置:")
        print("-" * 40)
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            model_config = config['model']
            print(f"Hidden size: {model_config['hidden_size']}")
            print(f"Num layers: {model_config['num_layers']}")
            print(f"Num heads: {model_config['num_heads']}")
            print(f"Intermediate size: {model_config['intermediate_size']}")
            
            total_params, param_breakdown = calculate_params(config)
            results[name] = total_params
            
            print(f"\n参数分布:")
            for component, count in param_breakdown.items():
                if count > 0:
                    print(f"  {component}: {format_params(count)}")
            
            print(f"\n总参数量: {format_params(total_params)} ({total_params:,})")
            
        except FileNotFoundError:
            print(f"配置文件不存在: {config_path}")
        except Exception as e:
            print(f"计算出错: {e}")
    
    # 总结
    print("\n" + "=" * 60)
    print("模型参数量对比:")
    print("-" * 60)
    for name, params in results.items():
        print(f"{name.capitalize():10s}: {format_params(params):>10s}")
    
    # 训练信息
    print("\n" + "=" * 60)
    print("训练规模信息 (全量数据: 3.45M样本):")
    print("-" * 60)
    
    # 小型模型的训练设置
    if 'small' in results:
        batch_size = 256
        grad_accum = 2
        effective_batch = batch_size * grad_accum
        total_samples = 3_450_000
        steps_per_epoch = total_samples // batch_size
        
        print(f"批次大小: {batch_size}")
        print(f"梯度累积: {grad_accum}")
        print(f"有效批次大小: {effective_batch}")
        print(f"每个epoch步数: {steps_per_epoch:,}")
        print(f"50个epochs总步数: {steps_per_epoch * 50:,}")
        
        # 估算训练时间（假设每步0.5秒）
        seconds_per_step = 0.5
        total_seconds = steps_per_epoch * 50 * seconds_per_step
        hours = total_seconds / 3600
        print(f"\n估计训练时间: {hours:.1f} 小时 (假设每步{seconds_per_step}秒)")

if __name__ == "__main__":
    main() 