"""
GeneDiffusion模型架构
基于Evo2思想的哺乳动物DNA序列Masked Diffusion Model
专注于实际可实现的创新功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math
import numpy as np
from einops import rearrange, repeat


class RotaryPositionEncoding(nn.Module):
    """旋转位置编码(RoPE)实现"""
    
    def __init__(self, dim: int, max_seq_len: int = 16384, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码
        t = torch.arange(self.max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class SpeciesAwareEmbedding(nn.Module):
    """物种感知的嵌入层
    基于230个哺乳动物物种数据的创新
    """
    
    def __init__(self, num_species: int, d_model: int):
        super().__init__()
        self.species_embedding = nn.Embedding(num_species, d_model // 4)
        self.fusion = nn.Linear(d_model + d_model // 4, d_model)
        
        # 添加物种特定的调制参数
        self.species_scale = nn.Embedding(num_species, d_model)
        self.species_shift = nn.Embedding(num_species, d_model)
        
        # 初始化scale为1，shift为0
        nn.init.ones_(self.species_scale.weight)
        nn.init.zeros_(self.species_shift.weight)
    
    def forward(self, x: torch.Tensor, species_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 序列特征 (batch_size, seq_len, d_model)
            species_ids: 物种ID (batch_size,)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 获取物种嵌入并扩展到序列长度
        species_emb = self.species_embedding(species_ids).unsqueeze(1)
        species_emb = species_emb.expand(-1, seq_len, -1)
        
        # 融合物种信息
        combined = torch.cat([x, species_emb], dim=-1)
        x_fused = self.fusion(combined)
        
        # 应用物种特定的缩放和偏移（类似于FiLM）
        scale = self.species_scale(species_ids).unsqueeze(1)  # (batch_size, 1, d_model)
        shift = self.species_shift(species_ids).unsqueeze(1)  # (batch_size, 1, d_model)
        
        x_modulated = x_fused * scale + shift
        
        return x_modulated


class SpeciesConditionedLayerNorm(nn.Module):
    """物种条件的层归一化
    不同物种可能有不同的序列统计特性
    """
    
    def __init__(self, d_model: int, num_species: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        # 每个物种的归一化参数
        self.species_gamma = nn.Embedding(num_species, d_model)
        self.species_beta = nn.Embedding(num_species, d_model)
        
        # 初始化
        nn.init.ones_(self.species_gamma.weight)
        nn.init.zeros_(self.species_beta.weight)
    
    def forward(self, x: torch.Tensor, species_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            species_ids: (batch_size,)
        """
        # 标准层归一化
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x_norm = (x - mean) / (std + self.eps)
        
        # 应用物种特定的缩放和偏移
        gamma = self.species_gamma(species_ids).unsqueeze(1)  # (batch_size, 1, d_model)
        beta = self.species_beta(species_ids).unsqueeze(1)    # (batch_size, 1, d_model)
        
        return gamma * x_norm + beta


class SpeciesGuidedAttention(nn.Module):
    """物种引导的注意力机制
    学习物种特定的注意力模式
    """
    
    def __init__(self, d_model: int, n_heads: int, num_species: int, dropout: float = 0.1):
        super().__init__()
        self.attention = BoundaryAwareAttention(d_model, n_heads, dropout)
        
        # 物种特定的注意力偏置
        self.species_attn_bias = nn.Embedding(num_species, n_heads)
        
    def forward(self, 
                x: torch.Tensor,
                species_ids: torch.Tensor,
                boundary_info: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # 获取物种特定的注意力偏置
        species_bias = self.species_attn_bias(species_ids)  # (batch_size, n_heads)
        
        # 应用注意力（需要修改BoundaryAwareAttention以接受species_bias）
        return self.attention(x, boundary_info=boundary_info, mask=mask)


class BoundaryAwareAttention(nn.Module):
    """边界感知的注意力机制
    基于GFF_Start和GFF_End信息学习功能元件边界
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 边界类型嵌入
        self.boundary_embedding = nn.Embedding(3, self.d_head)  # 0: none, 1: start, 2: end
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)
    
    def forward(self, 
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                boundary_info: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size, seq_len, _ = query.shape
        
        if key is None:
            key = query
        if value is None:
            value = query
        
        # Linear projections
        q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(key).view(batch_size, key.size(1), self.n_heads, self.d_head)
        v = self.v_proj(value).view(batch_size, value.size(1), self.n_heads, self.d_head)
        
        # 添加边界信息（如果提供）
        if boundary_info is not None:
            boundary_bias = self.boundary_embedding(boundary_info).unsqueeze(2)
            q = q + boundary_bias
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


class LocalSequenceEncoder(nn.Module):
    """局部序列特征编码器
    学习密码子使用偏好、剪接位点等局部模式
    """
    
    def __init__(self, d_model: int, kernel_sizes: List[int] = [3, 5, 7]):
        super().__init__()
        # 计算每个卷积的输出通道数，确保总和等于d_model
        n_kernels = len(kernel_sizes)
        conv_dims = [d_model // n_kernels] * n_kernels
        # 将余数分配给前面的卷积
        for i in range(d_model % n_kernels):
            conv_dims[i] += 1
            
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, conv_dims[i], 
                     kernel_size=k, padding=k//2)
            for i, k in enumerate(kernel_sizes)
        ])
        self.fusion = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        # 转换为卷积格式
        x_conv = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        
        # 多尺度卷积
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x_conv))
        
        # 连接并转换回原格式
        combined = torch.cat(conv_outputs, dim=1)  # (batch_size, d_model, seq_len)
        combined = combined.transpose(1, 2)  # (batch_size, seq_len, d_model)
        
        # 融合
        output = self.fusion(combined)
        return self.norm(output + x)


class AnnotationEncoder(nn.Module):
    """注释编码器
    处理5种注释类型：exon, cds, lnc_rna, gene, trna
    """
    
    def __init__(self,
                 num_annotation_types: int = 5,
                 d_model: int = 512,
                 n_layers: int = 4,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 注释类型嵌入
        self.annotation_embedding = nn.Embedding(num_annotation_types + 1, d_model)  # +1 for no annotation
        
        # 位置编码
        self.position_encoding = RotaryPositionEncoding(d_model)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_boundary_attention=True)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, annotations: torch.Tensor, boundary_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            annotations: (batch_size, seq_len) - 每个位置的注释类型ID
            boundary_info: (batch_size, seq_len) - 边界信息
        Returns:
            encoded_annotations: (batch_size, seq_len, d_model)
        """
        # 嵌入注释
        x = self.annotation_embedding(annotations)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, boundary_info=boundary_info)
        
        x = self.norm(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer块，支持边界感知注意力和物种条件"""
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 use_boundary_attention: bool = False,
                 num_species: Optional[int] = None,
                 use_species_conditioning: bool = False):
        super().__init__()
        
        self.use_species_conditioning = use_species_conditioning and num_species is not None
        
        if use_boundary_attention:
            self.attention = BoundaryAwareAttention(d_model, n_heads, dropout)
        else:
            # 使用标准注意力
            self.attention = BoundaryAwareAttention(d_model, n_heads, dropout)
        
        # 使用物种条件层归一化或标准层归一化
        if self.use_species_conditioning:
            assert num_species is not None, "num_species must be provided when use_species_conditioning is True"
            self.norm1 = SpeciesConditionedLayerNorm(d_model, num_species)
            self.norm2 = SpeciesConditionedLayerNorm(d_model, num_species)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 局部序列编码器
        self.local_encoder = LocalSequenceEncoder(d_model)
    
    def forward(self,
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                boundary_info: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                species_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Self-attention with boundary awareness
        residual = x
        if self.use_species_conditioning and species_ids is not None:
            x = self.norm1(x, species_ids)
        else:
            x = self.norm1(x)
        x = self.attention(x, boundary_info=boundary_info, mask=mask)
        x = residual + x
        
        # Local sequence patterns
        x = self.local_encoder(x)
        
        # Feed-forward
        residual = x
        if self.use_species_conditioning and species_ids is not None:
            x = self.norm2(x, species_ids)
        else:
            x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class GeneDiffusionModel(nn.Module):
    """
    GeneDiffusion: 基于实际数据的DNA序列Masked Diffusion Model
    
    核心创新（均基于可用数据）：
    1. 哺乳动物特化的序列模式学习
    2. 注释引导的扩散过程
    3. 局部序列特征增强
    4. 边界感知机制
    5. 生物学约束的损失函数
    """
    
    def __init__(self,
                 vocab_size: int = 4099,  # 4096个6-mer + 3个特殊token (MASK, PAD, UNK)
                 num_annotation_types: int = 5,  # exon, cds, lnc_rna, gene, trna
                 num_species: int = 230,  # 230个哺乳动物物种
                 d_model: int = 512,
                 n_layers: int = 8,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 max_seq_len: int = 5120,  # 基于5k数据限制（注意：现在是6-mer的数量）
                 dropout: float = 0.1,
                 use_dna_features: bool = True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token_id = 4096  # 6-mer词表后的第一个特殊token
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = RotaryPositionEncoding(d_model, max_seq_len)
        
        # 物种感知嵌入
        self.species_encoder = SpeciesAwareEmbedding(num_species, d_model)
        
        # 注释编码器
        self.annotation_encoder = AnnotationEncoder(
            num_annotation_types, d_model, n_layers=4, n_heads=n_heads, d_ff=d_ff, dropout=dropout
        )
        
        # Main transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, 
                           use_boundary_attention=(i % 2 == 0),  # 交替使用边界感知
                           num_species=num_species,
                           use_species_conditioning=True)  # 启用物种条件
            for i in range(n_layers)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # GC含量预测头（用于约束）
        self.gc_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def compute_boundary_info(self, annotations: torch.Tensor) -> torch.Tensor:
        """从注释计算边界信息"""
        batch_size, seq_len = annotations.shape
        boundary = torch.zeros_like(annotations)
        
        # 检测边界：0=none, 1=start, 2=end
        for i in range(seq_len):
            if i == 0:
                if annotations[:, i].any():
                    boundary[:, i] = 1  # start
            elif i == seq_len - 1:
                if annotations[:, i].any():
                    boundary[:, i] = 2  # end
            else:
                # 检测转换
                prev_ann = annotations[:, i-1]
                curr_ann = annotations[:, i]
                next_ann = annotations[:, i+1] if i < seq_len-1 else torch.zeros_like(curr_ann)
                
                # 开始边界
                start_mask = (prev_ann == 0) & (curr_ann > 0)
                boundary[start_mask, i] = 1
                
                # 结束边界
                end_mask = (curr_ann > 0) & (next_ann == 0)
                boundary[end_mask, i] = 2
        
        return boundary
    
    def forward_diffusion(self, x0: torch.Tensor, t: float, 
                         annotations: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        注释感知的前向扩散过程
        
        Args:
            x0: 原始序列 (batch_size, seq_len)
            t: mask比例 [0, 1]
            annotations: 注释信息 (batch_size, seq_len)
        
        Returns:
            xt: mask后的序列
            mask: 布尔mask，True表示被mask的位置
        """
        batch_size, seq_len = x0.shape
        
        if annotations is None:
            # 标准随机mask
            mask = torch.rand(batch_size, seq_len, device=x0.device) < t
        else:
            # 功能区域优先保护：注释区域mask概率降低
            base_prob = torch.full((batch_size, seq_len), t, device=x0.device)
            # 有注释的区域降低mask概率
            annotated = annotations > 0
            base_prob[annotated] *= 0.7  # 降低30%的mask概率
            
            mask = torch.rand(batch_size, seq_len, device=x0.device) < base_prob
        
        # 创建mask后的序列
        xt = x0.clone()
        xt[mask] = self.mask_token_id
        
        return xt, mask
    
    def forward(self,
                xt: torch.Tensor,
                annotations: torch.Tensor,
                species_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播：预测被mask的token
        
        Args:
            xt: mask后的序列 (batch_size, seq_len)
            annotations: 注释信息 (batch_size, seq_len)
            species_ids: 物种ID (batch_size,)
            attention_mask: 注意力mask
        
        Returns:
            Dict containing:
                - logits: (batch_size, seq_len, vocab_size)
                - hidden_states: (batch_size, seq_len, d_model)
                - gc_content: (batch_size, seq_len) - 预测的GC含量
        """
        batch_size, seq_len = xt.shape
        
        # Embed tokens
        x = self.token_embedding(xt)
        
        # 添加物种信息
        x = self.species_encoder(x, species_ids)
        
        # 计算边界信息
        boundary_info = self.compute_boundary_info(annotations)
        
        # 编码注释
        annotation_features = self.annotation_encoder(annotations, boundary_info)
        
        # Apply transformer layers with annotation conditioning
        for i, layer in enumerate(self.layers):
            # 周期性融合注释信息
            if i % 2 == 0:
                # 使用加性融合
                x = x + 0.1 * annotation_features
            
            x = layer(x, boundary_info=boundary_info, mask=attention_mask, species_ids=species_ids)
        
        # Final normalization and projection
        hidden_states = self.norm(x)
        logits = self.output_projection(hidden_states)
        
        # 预测GC含量（用于约束）
        gc_content = self.gc_predictor(hidden_states).squeeze(-1)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'gc_content': gc_content
        }
    
    def compute_loss(self,
                     x0: torch.Tensor,
                     annotations: torch.Tensor,
                     species_ids: torch.Tensor,
                     t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算扩散损失，包含生物学约束
        
        Args:
            x0: 原始序列 (batch_size, seq_len)
            annotations: 注释信息
            species_ids: 物种ID
            t: mask比例，如果None则随机采样
        
        Returns:
            Dict containing losses and metrics
        """
        batch_size = x0.shape[0]
        
        # 随机采样t
        if t is None:
            t = torch.rand(batch_size, device=x0.device)
        
        # 存储每个样本的损失
        losses = []
        gc_losses = []
        
        for i in range(batch_size):
            # 前向扩散
            xt, mask = self.forward_diffusion(x0[i:i+1], t[i].item(), annotations[i:i+1])
            
            # 模型预测
            outputs = self.forward(xt, annotations[i:i+1], species_ids[i:i+1])
            logits = outputs['logits']
            gc_pred = outputs['gc_content']
            
            # Masked语言模型损失
            loss = F.cross_entropy(
                logits[0][mask[0]], 
                x0[i][mask[0]], 
                reduction='mean'
            )
            
            # 除以t以获得正确的损失缩放
            if t[i] > 0:
                loss = loss / t[i]
            
            losses.append(loss)
            
            # GC含量损失（仅在CDS区域）
            if (annotations[i] == 1).any():  # 假设1是CDS的ID
                cds_mask = annotations[i] == 1
                actual_gc = ((x0[i] == 1) | (x0[i] == 2)).float()[cds_mask].mean()  # C=1, G=2
                pred_gc = gc_pred[0][cds_mask].mean()
                gc_loss = F.mse_loss(pred_gc, actual_gc)
                gc_losses.append(gc_loss)
        
        # 聚合损失
        main_loss = torch.stack(losses).mean()
        gc_loss = torch.stack(gc_losses).mean() if gc_losses else torch.tensor(0.0)
        
        total_loss = main_loss + 0.1 * gc_loss
        
        # 计算准确率和困惑度（用于监控训练）
        with torch.no_grad():
            # 简单的准确率估算：计算模型预测的最大概率类别是否正确
            accuracy = 0.0
            total_predictions = 0
            
            for i in range(batch_size):
                xt, mask = self.forward_diffusion(x0[i:i+1], t[i].item(), annotations[i:i+1])
                outputs = self.forward(xt, annotations[i:i+1], species_ids[i:i+1])
                logits = outputs['logits']
                
                # 只在mask位置计算准确率
                if mask[0].sum() > 0:
                    pred_tokens = logits[0][mask[0]].argmax(dim=-1)
                    true_tokens = x0[i][mask[0]]
                    accuracy += (pred_tokens == true_tokens).float().sum()
                    total_predictions += mask[0].sum()
            
            accuracy = accuracy / max(total_predictions, 1.0)
            perplexity = torch.exp(main_loss).item()
        
        return {
            'loss': total_loss,
            'main_loss': main_loss,
            'gc_loss': gc_loss,
            'accuracy': accuracy.item() if torch.is_tensor(accuracy) else accuracy,
            'perplexity': perplexity
        }
    
    @torch.no_grad()
    def sample(self,
               batch_size: int,
               seq_len: int,
               annotations: torch.Tensor,
               species_ids: torch.Tensor,
               num_steps: int = 100,
               temperature: float = 1.0,
               top_k: Optional[int] = None,
               device: str = 'cuda') -> torch.Tensor:
        """
        从全mask序列生成DNA序列
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            annotations: 注释信息
            species_ids: 物种ID
            num_steps: 去噪步数
            temperature: 采样温度
            top_k: top-k采样
            device: 设备
        
        Returns:
            生成的序列 (batch_size, seq_len)
        """
        # 初始化为全mask序列
        x = torch.full((batch_size, seq_len), self.mask_token_id, device=device)
        
        # 逐步去噪
        for step in range(num_steps):
            # 计算当前mask比例
            t = 1.0 - (step + 1) / num_steps
            
            # 获取模型预测
            outputs = self.forward(x, annotations, species_ids)
            logits = outputs['logits']
            
            # 找到当前被mask的位置
            mask = x == self.mask_token_id
            
            if not mask.any():
                break
            
            # 对mask位置进行采样
            probs = F.softmax(logits / temperature, dim=-1)
            
            if top_k is not None:
                # Top-k采样
                values, indices = probs.topk(top_k, dim=-1)
                probs = torch.zeros_like(probs).scatter_(-1, indices, values)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # 采样
            sampled = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)
            
            # 只更新mask位置
            x[mask] = sampled[mask]
            
            # 重新mask一部分（模拟扩散过程）
            if step < num_steps - 1:
                # 计算需要重新mask的数量
                num_remask = int(mask.sum() * t)
                if num_remask > 0:
                    # 随机选择一些位置重新mask
                    unmask_indices = torch.where(mask)
                    remask_indices = torch.randperm(len(unmask_indices[0]))[:num_remask]
                    for idx in remask_indices:
                        x[unmask_indices[0][idx], unmask_indices[1][idx]] = self.mask_token_id
        
        return x


class BiologicalConstraintLoss(nn.Module):
    """生物学约束损失
    基于实际可验证的生物学规则
    """
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.1, gamma: float = 0.1):
        super().__init__()
        self.alpha = alpha  # ORF完整性权重
        self.beta = beta   # 剪接位点权重
        self.gamma = gamma # 序列复杂度权重
        
        # 定义剪接位点模式 (GT-AG)
        self.donor_pattern = torch.tensor([2, 3])  # GT: G=2, T=3
        self.acceptor_pattern = torch.tensor([0, 2])  # AG: A=0, G=2
    
    def compute_orf_loss(self, sequences: torch.Tensor, annotations: torch.Tensor) -> torch.Tensor:
        """计算开放阅读框损失"""
        # 在CDS区域检查是否有终止密码子
        batch_size, seq_len = sequences.shape
        cds_mask = annotations == 1  # 假设1是CDS
        
        # 简化：检查CDS区域的长度是否为3的倍数
        orf_losses = []
        for i in range(batch_size):
            if cds_mask[i].any():
                cds_length = cds_mask[i].sum()
                orf_loss = (cds_length % 3) / 3.0  # 归一化到[0,1]
                orf_losses.append(orf_loss)
        
        return torch.stack(orf_losses).mean() if orf_losses else torch.tensor(0.0)
    
    def compute_splice_loss(self, sequences: torch.Tensor, boundary_info: torch.Tensor) -> torch.Tensor:
        """计算剪接位点损失"""
        # 在exon边界检查GT-AG规则
        # 简化实现
        return torch.tensor(0.0)  # 占位符
    
    def compute_complexity_loss(self, sequences: torch.Tensor) -> torch.Tensor:
        """计算序列复杂度损失，避免过度重复"""
        # 计算6-mer的多样性
        batch_size, seq_len = sequences.shape
        complexity_losses = []
        
        for i in range(batch_size):
            seq = sequences[i]
            # 计算unique 6-mers的比例
            kmers = []
            k = 6  # 使用6-mer
            for j in range(seq_len - k + 1):
                # 构建k-mer的数值表示
                kmer_val = 0
                for pos in range(k):
                    kmer_val = kmer_val * 10 + seq[j + pos].item()
                kmers.append(kmer_val)
            
            unique_ratio = len(set(kmers)) / len(kmers) if kmers else 1.0
            complexity_loss = 1.0 - unique_ratio  # 低复杂度产生高损失
            complexity_losses.append(complexity_loss)
        
        return torch.tensor(complexity_losses).mean()
    
    def forward(self,
                predictions: torch.Tensor,
                annotations: torch.Tensor,
                boundary_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算总的生物学约束损失"""
        orf_loss = self.compute_orf_loss(predictions, annotations)
        splice_loss = self.compute_splice_loss(predictions, boundary_info) if boundary_info is not None else 0
        complexity_loss = self.compute_complexity_loss(predictions)
        
        total_loss = (self.alpha * orf_loss + 
                     self.beta * splice_loss + 
                     self.gamma * complexity_loss)
        
        return total_loss


if __name__ == "__main__":
    # 测试模型
    batch_size = 2
    seq_len = 1024
    vocab_size = 8
    num_annotation_types = 5
    num_species = 230
    
    # 创建模型
    model = GeneDiffusionModel(
        vocab_size=vocab_size,
        num_annotation_types=num_annotation_types,
        num_species=num_species,
        d_model=512,  # 使用较小的模型进行测试
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        max_seq_len=seq_len,
        use_dna_features=True
    )
    
    # 创建测试数据
    x0 = torch.randint(0, 4096, (batch_size, seq_len))  # 6-mer tokens
    annotations = torch.randint(0, num_annotation_types + 1, (batch_size, seq_len))  # 注释ID
    species_ids = torch.randint(0, num_species, (batch_size,))
    
    # 测试前向扩散
    t = 0.3
    xt, mask = model.forward_diffusion(x0, t)
    print(f"原始序列: {x0[0, :20]}")
    print(f"Mask后序列: {xt[0, :20]}")
    print(f"Mask比例: {mask.float().mean():.2f}")
    
    # 测试模型前向传播
    outputs = model(xt, annotations, species_ids)
    print(f"\n输出logits形状: {outputs['logits'].shape}")
    print(f"GC含量形状: {outputs['gc_content'].shape}")
    
    # 测试损失计算
    losses = model.compute_loss(x0, annotations, species_ids)
    print(f"\n损失: {losses['loss'].item():.4f}")
    print(f"GC含量损失: {losses['gc_loss'].item():.4f}")
    
    # 测试采样
    print("\n测试采样...")
    generated = model.sample(1, 100, annotations[:1, :100], species_ids[:1], num_steps=50, device='cpu')
    print(f"生成序列: {generated[0, :20]}") 