# GeneDiffusion: A Language Diffusion Model for Genomic Sequences

## Model Overview

GeneDiffusion is a discrete language diffusion model specifically designed for genomic sequence generation. The model treats DNA sequences as a discrete language and applies masked diffusion techniques to generate biologically meaningful mammalian genomic sequences with species-specific and annotation-guided constraints.

## Core Architecture

### 1. Discrete Diffusion Framework

The model employs a masked language diffusion approach, fundamentally different from continuous diffusion models:

- **Forward Process**: Corruption through discrete token masking with mask ratio t sampled from uniform distribution U(0,1]
- **Reverse Process**: Iterative denoising via masked token prediction using a Transformer-based architecture
- **Annotation-Aware Masking**: Differential masking probabilities with 30% reduction for functionally annotated regions (CDS, exons) to preserve biological importance during training

### 2. Sequence Representation

#### 2.1 6-mer Tokenization
The model adopts a 6-nucleotide tokenization scheme:
- **Vocabulary Size**: 4^6 = 4,096 distinct 6-mer tokens plus 3 special tokens ([MASK], [PAD], [UNK])
- **Biological Rationale**: 6-mer units encompass two codons, capturing codon usage patterns and reading frame information
- **Computational Efficiency**: 6-fold sequence length reduction compared to single-nucleotide tokenization
- **Pattern Capture**: Natural alignment with regulatory motifs and transcription factor binding sites

#### 2.2 Positional Encoding
- **Method**: Rotary Position Encoding (RoPE)
- **Maximum Sequence Length**: 16,384 tokens (approximately 98kb of genomic sequence)
- **Advantages**: Superior extrapolation to longer sequences through relative position encoding

## Model Components

### 3. Multi-Modal Conditioning Architecture

#### 3.1 Species-Specific Modeling
The model incorporates species information through multiple mechanisms:

- **Species Embedding Layer**: Projects species ID to a dense representation (d_model/4 dimensions)
- **Feature-wise Linear Modulation (FiLM)**: Species-specific scale and shift parameters modulate hidden representations throughout the network
- **Species-Conditioned Layer Normalization**: Learnable normalization parameters per species, enabling capture of species-specific sequence statistics
- **Coverage**: Supports 228 mammalian species with distinct genomic characteristics

#### 3.2 Genomic Annotation Integration
Six annotation types are integrated as conditional information:
1. None (unannotated regions)
2. Exon (expressed sequences)
3. CDS (protein-coding sequences)
4. lncRNA (long non-coding RNA)
5. Gene (gene bodies)
6. tRNA (transfer RNA genes)

**Annotation Processing Pipeline**:
- Independent 4-layer Transformer encoder for annotation sequences
- Boundary-aware attention mechanism detecting functional region transitions
- Hierarchical fusion with main sequence representations at alternating layers

### 4. Attention Mechanisms

#### 4.1 Boundary-Aware Attention
- Explicit modeling of functional region boundaries (start/end/internal positions)
- Learned boundary embeddings (3-dimensional: none, start, end)
- Enhanced structural understanding of gene architecture

#### 4.2 Efficient Long-Range Attention
- Chunked attention for sequences exceeding 512 tokens
- Memory-efficient implementation maintaining O(n) memory complexity
- Preserves global context while managing computational resources

### 5. Multi-Scale Feature Extraction

#### 5.1 Local Pattern Encoder
- Parallel convolutional layers with kernel sizes {3, 5, 7}
- Captures DNA motifs, regulatory elements, and local sequence dependencies
- Complementary to global attention patterns

#### 5.2 Hierarchical Feature Integration
- Alternating global (attention) and local (convolution) processing
- Progressive feature refinement through 24 transformer layers
- Cross-scale information flow through residual connections

## Training Methodology

### 6. Loss Functions

#### 6.1 Primary Reconstruction Loss
- Cross-entropy loss on masked token predictions
- Weighted by annotation importance and masking difficulty

#### 6.2 Biological Constraint Losses
- **ORF Integrity**: Ensures CDS regions maintain reading frame (length % 3 = 0)
- **Splice Site Conservation**: Rewards canonical GT-AG splice signals
- **Sequence Complexity**: Penalizes low-complexity repetitive sequences
- **GC Content Matching**: Species-specific GC content constraints

### 7. Training Dynamics

#### 7.1 Dynamic Masking Strategy
- Mask ratio sampling from continuous distribution enables curriculum learning
- Progressive difficulty adjustment through training epochs
- Balanced exploration of easy (low mask ratio) and hard (high mask ratio) scenarios

#### 7.2 Optimization Configuration
- **Optimizer**: AdamW with differential learning rates for model components
- **Learning Rate Schedule**: Cosine annealing with warm restarts
- **Gradient Management**: Gradient clipping (0.5) and accumulation for stable training
- **Mixed Precision**: FP16 training with dynamic loss scaling

## Generation Process

### 8. Iterative Refinement Sampling

The generation process follows an iterative denoising schedule:

1. **Initialization**: Start with fully masked sequence
2. **Progressive Unmasking**: Iteratively predict and unmask tokens over T steps
3. **Confidence-Based Remasking**: Low-confidence predictions are remasked for refinement
4. **Annealing Schedule**: Mask ratio decreases linearly from 1.0 to 0.0 over generation steps

### 9. Sampling Strategies

- **Temperature Scaling**: Controls generation diversity (typical range: 0.7-1.2)
- **Top-k Filtering**: Restricts sampling to k most probable tokens
- **Nucleus Sampling**: Alternative to top-k with dynamic vocabulary truncation
- **Conditional Guidance**: Annotation and species constraints guide token selection

## Model Configurations

### 10. Architecture Specifications

**Standard Configuration (600M parameters)**:
- Embedding Dimension: 1,024
- Transformer Layers: 24
- Attention Heads: 16
- Feed-Forward Dimension: 4,096
- Total Parameters: ~600M

**Component Distribution**:
- Token Embeddings: 4.2M parameters
- Species Conditioning: 2.3M parameters
- Annotation Encoder: 50M parameters
- Main Transformer: 540M parameters
- Output Projection: 4.2M parameters

### 11. Computational Requirements

- **Memory Footprint**: 18-22GB GPU memory for batch size 16
- **Training Throughput**: ~500 sequences/second on 4x A100 GPUs
- **Inference Speed**: ~50 sequences/second for 1kb generation
- **Scalability**: Distributed training support via DDP

## Technical Innovations

### 12. Key Contributions

1. **Discrete Diffusion for Genomics**: First application of masked language diffusion to genomic sequences
2. **Multi-Species Unified Model**: Single model supporting 228 species through conditional generation
3. **Annotation-Guided Generation**: Explicit integration of functional genomic annotations
4. **Hierarchical Multi-Scale Modeling**: Combination of local convolutions and global attention
5. **Biologically-Informed Training**: Incorporation of domain-specific constraints and priors

### 13. Advantages over Existing Approaches

- **Biological Consistency**: Hard constraints ensure generated sequences respect biological rules
- **Computational Efficiency**: 6-mer tokenization reduces computational requirements by 6x
- **Flexible Conditioning**: Multiple conditioning modalities (species, annotations, GC content)
- **Long-Range Dependencies**: Effective modeling of sequences up to 98kb
- **Interpretability**: Discrete tokens and attention weights provide interpretable generation process

## Evaluation Metrics

### 14. Performance Measures

- **Reconstruction Accuracy**: Masked token prediction accuracy (typical: 85-90%)
- **Perplexity**: Average per-token perplexity on held-out sequences
- **Biological Validity**: Percentage of generated sequences with valid ORFs
- **Species Fidelity**: Classification accuracy of generated sequences to target species
- **Annotation Consistency**: Overlap with predicted functional annotations
- **Diversity Metrics**: Unique k-mer coverage and sequence diversity indices 