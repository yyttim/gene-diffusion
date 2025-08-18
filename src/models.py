"""
GeneDiffusion Model - Masked Diffusion for DNA sequences
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from typing import Optional, Tuple, Dict, List
import math
import numpy as np  # type: ignore
from einops import rearrange, repeat  # type: ignore
import logging

# Gradient checkpointing compatibility
try:
    from torch.utils.checkpoint import checkpoint
    GRADIENT_CHECKPOINTING_AVAILABLE = True
except ImportError:
    GRADIENT_CHECKPOINTING_AVAILABLE = False
    checkpoint = None
    logging.warning("torch.utils.checkpoint not available. Gradient checkpointing will be disabled.")


class RotaryPositionEncoding(nn.Module):
    """Rotary position encoding (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 16384, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(self.max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class SpeciesAwareEmbedding(nn.Module):
    """Species-specific embedding layer"""
    
    def __init__(self, num_species: int, d_model: int):
        super().__init__()
        self.species_embedding = nn.Embedding(num_species, d_model // 4)
        self.fusion = nn.Linear(d_model + d_model // 4, d_model)
        
        # Species-specific modulation (FiLM-like)
        self.species_scale = nn.Embedding(num_species, d_model)
        self.species_shift = nn.Embedding(num_species, d_model)
        
        nn.init.ones_(self.species_scale.weight)
        nn.init.zeros_(self.species_shift.weight)
    
    def forward(self, x: torch.Tensor, species_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        species_emb = self.species_embedding(species_ids).unsqueeze(1)
        species_emb = species_emb.expand(-1, seq_len, -1)
        
        combined = torch.cat([x, species_emb], dim=-1)
        x_fused = self.fusion(combined)
        
        scale = self.species_scale(species_ids).unsqueeze(1)
        shift = self.species_shift(species_ids).unsqueeze(1)
        
        x_modulated = x_fused * scale + shift
        
        return x_modulated


class SpeciesConditionedLayerNorm(nn.Module):
    """Species-conditioned layer normalization"""
    
    def __init__(self, d_model: int, num_species: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        self.species_gamma = nn.Embedding(num_species, d_model)
        self.species_beta = nn.Embedding(num_species, d_model)
        
        nn.init.ones_(self.species_gamma.weight)
        nn.init.zeros_(self.species_beta.weight)
    
    def forward(self, x: torch.Tensor, species_ids: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x_norm = (x - mean) / (std + self.eps)
        
        gamma = self.species_gamma(species_ids).unsqueeze(1)
        beta = self.species_beta(species_ids).unsqueeze(1)
        
        return gamma * x_norm + beta


class SpeciesGuidedAttention(nn.Module):
    """Species-guided attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, num_species: int, dropout: float = 0.1):
        super().__init__()
        self.attention = BoundaryAwareAttention(d_model, n_heads, dropout)
        self.species_attn_bias = nn.Embedding(num_species, n_heads)
        
    def forward(self, 
                x: torch.Tensor,
                species_ids: torch.Tensor,
                boundary_info: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        species_bias = self.species_attn_bias(species_ids)
        return self.attention(x, boundary_info=boundary_info, mask=mask)


class BoundaryAwareAttention(nn.Module):
    """Attention with functional boundary awareness"""
    
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
        
        # Boundary type embedding: 0=none, 1=start, 2=end
        self.boundary_embedding = nn.Embedding(3, self.d_head)
        
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
        k = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.d_head)
        v = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Add boundary information
        if boundary_info is not None:
            boundary_bias = self.boundary_embedding(boundary_info).unsqueeze(2)
            q = q + boundary_bias
        
        # Reshape for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention
        if seq_len > 512 and self.training:  # Chunked attention for long sequences
            attn_output = self._chunked_attention(q, k, v, mask)
        else:
            if hasattr(F, 'scaled_dot_product_attention'):
                # PyTorch 2.0+ optimized attention
                q_scaled = q * self.scale
                attn_output = F.scaled_dot_product_attention(
                    q_scaled, k, v,
                    attn_mask=mask,
                    dropout_p=self.dropout.p if self.training else 0.0
                )
            else:
                # Fallback implementation
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                
                # Log-sum-exp trick for stability
                scores_max = scores.max(dim=-1, keepdim=True)[0]
                scores = scores - scores_max
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                
                attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(attn_output)
    
    def _chunked_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          mask: Optional[torch.Tensor] = None, chunk_size: int = 256) -> torch.Tensor:
        """Memory-efficient chunked attention"""
        batch_size, n_heads, seq_len, d_head = q.shape
        
        attn_output = torch.zeros_like(v)
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i]
            
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                mask_chunk = mask[:, :, i:end_i] if mask.dim() == 3 else mask
                scores = scores.masked_fill(mask_chunk == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output[:, :, i:end_i] = torch.matmul(attn_weights, v)
        
        return attn_output


class LocalSequenceEncoder(nn.Module):
    """Multi-scale convolutional encoder for local patterns"""
    
    def __init__(self, d_model: int, kernel_sizes: List[int] = [3, 5, 7]):
        super().__init__()
        n_kernels = len(kernel_sizes)
        conv_dims = [d_model // n_kernels] * n_kernels
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
        batch_size, seq_len, d_model = x.shape
        
        if seq_len == 0:
            return x
            
        x_conv = x.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            try:
                conv_out = conv(x_conv)
                conv_outputs.append(conv_out)
            except RuntimeError as e:
                print(f"Conv error: input_shape={x_conv.shape}, conv={conv}")
                raise e
        
        combined = torch.cat(conv_outputs, dim=1)
        
        if combined.shape[1] != d_model:
            raise ValueError(f"Conv output dimension mismatch: expected {d_model}, got {combined.shape[1]}")
            
        combined = combined.transpose(1, 2)
        
        output = self.fusion(combined)
        return self.norm(output + x)


class AnnotationEncoder(nn.Module):
    """Encodes genomic annotations (none, exon, cds, lnc_rna, gene, trna)"""
    
    def __init__(self,
                 num_annotation_types: int = 6,
                 d_model: int = 512,
                 n_layers: int = 4,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 use_gradient_checkpointing: bool = False):
        super().__init__()
        
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.annotation_embedding = nn.Embedding(num_annotation_types, d_model)
        self.position_encoding = RotaryPositionEncoding(d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_boundary_attention=True)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, annotations: torch.Tensor, boundary_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.annotation_embedding(annotations)
        
        for layer in self.layers:
            if self.training and self.use_gradient_checkpointing and GRADIENT_CHECKPOINTING_AVAILABLE:
                x = checkpoint(
                    layer,
                    x,
                    None,  # context
                    boundary_info,
                    None,  # mask
                    None,  # species_ids
                    use_reentrant=False
                )
            else:
                x = layer(x, boundary_info=boundary_info)
        
        x = self.norm(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block with optional boundary awareness and species conditioning"""
    
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
            self.attention = BoundaryAwareAttention(d_model, n_heads, dropout)
        
        if self.use_species_conditioning:
            assert num_species is not None
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
        
        self.local_encoder = LocalSequenceEncoder(d_model)
    
    def forward(self,
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                boundary_info: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                species_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Self-attention
        residual = x
        if self.use_species_conditioning and species_ids is not None:
            x = self.norm1(x, species_ids)
        else:
            x = self.norm1(x)
        x = self.attention(x, boundary_info=boundary_info, mask=mask)
        x = residual + x
        
        # Local patterns
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
    Masked Diffusion Model for DNA sequences
    - 6-mer tokenization (4096 tokens + special tokens)
    - Species-aware modeling (228 mammalian species)
    - Annotation-guided generation (6 annotation types)
    """
    
    def __init__(self,
                 vocab_size: int = 4099,  # 4096 6-mers + 3 special tokens
                 num_annotation_types: int = 6,
                 num_species: int = 228,
                 d_model: int = 512,
                 n_layers: int = 8,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 max_seq_len: int = 5120,
                 dropout: float = 0.1,
                 use_dna_features: bool = True,
                 use_gradient_checkpointing: bool = False):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token_id = 4096  # First special token after 6-mer vocabulary
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if use_gradient_checkpointing and not GRADIENT_CHECKPOINTING_AVAILABLE:
            logging.warning(
                "Gradient checkpointing was requested but torch.utils.checkpoint is not available. "
                "Please upgrade PyTorch to version 1.11+ to use gradient checkpointing."
            )
            self.use_gradient_checkpointing = False
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = RotaryPositionEncoding(d_model, max_seq_len)
        
        # Species-aware embedding
        self.species_encoder = SpeciesAwareEmbedding(num_species, d_model)
        
        # Annotation encoder
        self.annotation_encoder = AnnotationEncoder(
            num_annotation_types, d_model, n_layers=4, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Main transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, 
                           use_boundary_attention=(i % 2 == 0),  # Alternate boundary awareness
                           num_species=num_species,
                           use_species_conditioning=True)
            for i in range(n_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # GC content predictor (for biological constraints)
        self.gc_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def compute_boundary_info(self, annotations: torch.Tensor) -> torch.Tensor:
        """Detect annotation boundaries"""
        batch_size, seq_len = annotations.shape
        boundary = torch.zeros_like(annotations)
        
        for i in range(seq_len):
            if i == 0:
                if annotations[:, i].any():
                    boundary[:, i] = 1  # start
            elif i == seq_len - 1:
                if annotations[:, i].any():
                    boundary[:, i] = 2  # end
            else:
                prev_ann = annotations[:, i-1]
                curr_ann = annotations[:, i]
                next_ann = annotations[:, i+1] if i < seq_len-1 else torch.zeros_like(curr_ann)
                
                # Start boundary
                start_mask = (prev_ann == 0) & (curr_ann > 0)
                boundary[start_mask, i] = 1
                
                # End boundary
                end_mask = (curr_ann > 0) & (next_ann == 0)
                boundary[end_mask, i] = 2
        
        return boundary
    
    def forward_diffusion(self, x0: torch.Tensor, t: float, 
                         annotations: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: mask tokens with annotation-aware protection
        
        Args:
            x0: Original sequences
            t: Mask ratio [0, 1]
            annotations: Annotation labels
        
        Returns:
            xt: Masked sequences
            mask: Boolean mask
        """
        batch_size, seq_len = x0.shape
        
        if annotations is None:
            mask = torch.rand(batch_size, seq_len, device=x0.device) < t
        else:
            # Reduce mask probability for annotated regions
            base_prob = torch.full((batch_size, seq_len), t, device=x0.device)
            annotated = annotations > 0
            base_prob[annotated] *= 0.7  # 30% protection for functional regions
            
            mask = torch.rand(batch_size, seq_len, device=x0.device) < base_prob
        
        xt = x0.clone()
        xt[mask] = self.mask_token_id
        
        return xt, mask
    
    def forward(self,
                xt: torch.Tensor,
                annotations: torch.Tensor,
                species_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass: predict masked tokens
        
        Returns dict with:
            - logits: Token predictions
            - hidden_states: Final hidden states
            - gc_content: Predicted GC content
        """
        batch_size, seq_len = xt.shape
        
        # Embed tokens
        x = self.token_embedding(xt)
        
        # Add species information
        x = self.species_encoder(x, species_ids)
        
        # Compute boundaries
        boundary_info = self.compute_boundary_info(annotations)
        
        # Encode annotations
        annotation_features = self.annotation_encoder(annotations, boundary_info)
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            # Periodically fuse annotation information
            if i % 2 == 0:
                x = x + 0.1 * annotation_features
            
            # Use gradient checkpointing if enabled
            if self.training and self.use_gradient_checkpointing and GRADIENT_CHECKPOINTING_AVAILABLE:
                x = checkpoint(
                    layer,
                    x,
                    None,  # context
                    boundary_info,
                    attention_mask,
                    species_ids,
                    use_reentrant=False
                )
            else:
                x = layer(x, boundary_info=boundary_info, mask=attention_mask, species_ids=species_ids)
        
        # Final projection
        hidden_states = self.norm(x)
        logits = self.output_projection(hidden_states)
        
        # Predict GC content
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
        Compute diffusion loss
        
        Args:
            x0: Original sequences
            annotations: Annotation labels
            species_ids: Species IDs
            t: Mask ratio (sampled from U(0,1] if None)
        
        Returns dict with loss and metrics
        """
        batch_size, seq_len = x0.shape
        
        # Sample mask ratio from U(0,1]
        if t is None:
            t = 1.0 - torch.rand(batch_size, device=x0.device)
        
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x0.device)
            
        # Apply forward diffusion
        if t is not None and isinstance(t, torch.Tensor) and t.dim() == 1:
            # Per-sample mask ratios
            xt = x0.clone()
            mask = torch.zeros_like(x0, dtype=torch.bool)
            
            for i in range(batch_size):
                mask_prob = t[i].item()
                if annotations is not None:
                    # Annotation-aware masking
                    base_prob = torch.full((seq_len,), mask_prob, device=x0.device)
                    annotated = annotations[i] > 0
                    base_prob[annotated] *= 0.7
                    mask[i] = torch.rand(seq_len, device=x0.device) < base_prob
                else:
                    mask[i] = torch.rand(seq_len, device=x0.device) < mask_prob
                
                xt[i][mask[i]] = self.mask_token_id
        else:
            # Uniform mask ratio
            if t is not None and isinstance(t, torch.Tensor):
                t_value = t.mean().item()
            else:
                t_value = 0.3
            xt, mask = self.forward_diffusion(x0, t_value, annotations)
        
        # Model prediction
        outputs = self.forward(xt, annotations, species_ids)
        logits = outputs['logits']
        
        # Compute loss on masked positions
        masked_logits = logits[mask]
        masked_targets = x0[mask]
        
        if masked_logits.numel() > 0:
            ce_loss = F.cross_entropy(masked_logits, masked_targets, reduction='none')
            main_loss = ce_loss.mean()
            raw_loss = ce_loss.mean()
        else:
            main_loss = torch.tensor(0.0, device=x0.device)
            raw_loss = torch.tensor(0.0, device=x0.device)
        
        gc_loss = torch.tensor(0.0, device=x0.device)  # Placeholder
        total_loss = main_loss
        
        # Compute accuracy
        with torch.no_grad():
            if masked_logits.numel() > 0:
                pred_tokens = masked_logits.argmax(dim=-1)
                accuracy = (pred_tokens == masked_targets).float().mean().item()
            else:
                accuracy = 0.0
            
            perplexity = 0.0  # Removed perplexity calculation
        
        return {
            'loss': total_loss,
            'main_loss': main_loss,
            'gc_loss': gc_loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
            'mask_ratio': t.mean().item() if isinstance(t, torch.Tensor) else t
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
        Generate sequences from fully masked input
        
        Args:
            batch_size: Number of sequences to generate
            seq_len: Sequence length
            annotations: Annotation labels
            species_ids: Species IDs
            num_steps: Denoising steps
            temperature: Sampling temperature
            top_k: Top-k sampling
            device: Device
        
        Returns:
            Generated sequences
        """
        # Start with fully masked sequence
        x = torch.full((batch_size, seq_len), self.mask_token_id, device=device)
        
        # Progressive denoising
        for step in range(num_steps):
            t = 1.0 - (step + 1) / num_steps
            
            outputs = self.forward(x, annotations, species_ids)
            logits = outputs['logits']
            
            mask = x == self.mask_token_id
            
            if not mask.any():
                break
            
            # Sample from predictions
            probs = F.softmax(logits / temperature, dim=-1)
            
            if top_k is not None:
                values, indices = probs.topk(top_k, dim=-1)
                probs = torch.zeros_like(probs).scatter_(-1, indices, values)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            sampled = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)
            
            # Update masked positions
            x[mask] = sampled[mask]
            
            # Remask some positions for next step
            if step < num_steps - 1:
                num_remask = int(mask.sum() * t)
                if num_remask > 0:
                    unmask_indices = torch.where(mask)
                    remask_indices = torch.randperm(len(unmask_indices[0]))[:num_remask]
                    for idx in remask_indices:
                        x[unmask_indices[0][idx], unmask_indices[1][idx]] = self.mask_token_id
        
        return x


class BiologicalConstraintLoss(nn.Module):
    """Biological constraints for DNA generation"""
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.1, gamma: float = 0.1):
        super().__init__()
        self.alpha = alpha  # ORF integrity weight
        self.beta = beta   # Splice site weight
        self.gamma = gamma # Sequence complexity weight
        
        # Splice site patterns
        self.donor_pattern = torch.tensor([2, 3])  # GT
        self.acceptor_pattern = torch.tensor([0, 2])  # AG
    
    def compute_orf_loss(self, sequences: torch.Tensor, annotations: torch.Tensor) -> torch.Tensor:
        """Check ORF integrity in CDS regions"""
        batch_size, seq_len = sequences.shape
        cds_mask = annotations == 2  # CDS annotation
        
        orf_loss = torch.tensor(0.0, device=sequences.device)
        num_cds_samples = 0
        
        for i in range(batch_size):
            if cds_mask[i].any():
                cds_length_in_tokens = cds_mask[i].sum()
                cds_length_in_nucleotides = cds_length_in_tokens * 6  # 6-mer tokens
                loss = (cds_length_in_nucleotides % 3) / 3.0
                orf_loss = orf_loss + loss
                num_cds_samples += 1
        
        if num_cds_samples > 0:
            orf_loss = orf_loss / num_cds_samples
            
        return orf_loss
    
    def compute_splice_loss(self, sequences: torch.Tensor, boundary_info: Optional[torch.Tensor]) -> torch.Tensor:
        """Check splice site patterns"""
        return torch.tensor(0.0, device=sequences.device)  # Placeholder
    
    def compute_complexity_loss(self, sequences: torch.Tensor) -> torch.Tensor:
        """Penalize low complexity sequences"""
        batch_size, seq_len = sequences.shape
        
        if seq_len == 0:
            return torch.tensor(0.0, device=sequences.device)
            
        complexity_losses = []
        
        for i in range(batch_size):
            seq = sequences[i]
            unique_tokens = len(torch.unique(seq))
            unique_ratio = unique_tokens / seq_len
            complexity_loss = 1.0 - unique_ratio
            complexity_losses.append(complexity_loss)
        
        return torch.tensor(complexity_losses, device=sequences.device).mean()
    
    def forward(self,
                predictions: torch.Tensor,
                annotations: torch.Tensor,
                boundary_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute total biological constraint loss"""
        orf_loss = self.compute_orf_loss(predictions, annotations)
        splice_loss = self.compute_splice_loss(predictions, boundary_info) if boundary_info is not None else 0
        complexity_loss = self.compute_complexity_loss(predictions)
        
        total_loss = (self.alpha * orf_loss + 
                     self.beta * splice_loss + 
                     self.gamma * complexity_loss)
        
        return total_loss


if __name__ == "__main__":
    # Test model
    batch_size = 2
    seq_len = 1024
    vocab_size = 4099
    num_annotation_types = 6
    num_species = 228
    
    model = GeneDiffusionModel(
        vocab_size=vocab_size,
        num_annotation_types=num_annotation_types,
        num_species=num_species,
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        max_seq_len=seq_len,
        use_dna_features=True,
        use_gradient_checkpointing=True
    )
    
    # Test data
    x0 = torch.randint(0, 4096, (batch_size, seq_len))
    annotations = torch.randint(0, num_annotation_types, (batch_size, seq_len))
    species_ids = torch.randint(0, num_species, (batch_size,))
    
    # Test forward diffusion
    t = 0.3
    xt, mask = model.forward_diffusion(x0, t)
    print(f"Original: {x0[0, :20]}")
    print(f"Masked: {xt[0, :20]}")
    print(f"Mask ratio: {mask.float().mean():.2f}")
    
    # Test forward pass
    outputs = model(xt, annotations, species_ids)
    print(f"\nLogits shape: {outputs['logits'].shape}")
    print(f"GC content shape: {outputs['gc_content'].shape}")
    
    # Test loss computation
    losses = model.compute_loss(x0, annotations, species_ids)
    print(f"\nLoss: {losses['loss'].item():.4f}")
    print(f"GC loss: {losses['gc_loss'].item():.4f}")
    
    # Test sampling
    print("\nTesting sampling...")
    generated = model.sample(1, 100, annotations[:1, :100], species_ids[:1], num_steps=50, device='cpu')
    print(f"Generated: {generated[0, :20]}") 