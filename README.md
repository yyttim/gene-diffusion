# GeneDiffusion: Masked Diffusion Model for Mammalian Genomic Sequences

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## 🧬 Overview

GeneDiffusion is a state-of-the-art masked diffusion model designed for generating and understanding mammalian genomic sequences. By adapting the LLaDA (Language Latent Diffusion) framework to biological sequences, it learns from 230 mammalian species to capture universal genomic patterns.

### Key Features

- **🔤 6-mer Tokenization**: Efficient representation using 6-nucleotide tokens (4,096 vocabulary)
- **🦁 Multi-Species Support**: Trained on 230 mammalian species with balanced sampling
- **🎯 Annotation-Aware**: Incorporates functional annotations (genes, exons, CDS, etc.)
- **⚖️ Species-Balanced Training**: Equal representation for all species (15,000 samples each)
- **🧪 Biological Constraints**: Enforces GC content, codon integrity, and sequence complexity
- **📊 Large-Scale**: 3.45M training samples from 597GB genomic data

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gene-diffusion.git
cd gene-diffusion

# Create conda environment
conda create -n gene-diffusion python=3.8
conda activate gene-diffusion

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.models import GeneDiffusionModel
from src.data_processor import DNATokenizer

# Initialize model
model = GeneDiffusionModel(
    vocab_size=4099,
    num_species=230,
    d_model=768,
    n_layers=12,
    n_heads=12
)

# Generate DNA sequence
generated_seq = model.sample(
    seq_len=1000,  # Generate 1000 tokens (~6kb)
    species_id=0,  # Human
    num_steps=100
)
```

### Training

```bash
# Single GPU training
python src/train.py --config configs/train_config_balanced.yaml

# Multi-GPU training
torchrun --nproc_per_node=8 src/train.py --config configs/train_config_balanced.yaml
```

## 📁 Project Structure

```
gene-diffusion/
├── src/                    # Core source code
│   ├── models.py          # Model architecture
│   ├── data_processor.py  # Data loading and processing
│   ├── train.py           # Training script
│   └── inference.py       # Generation and inference
├── configs/               # Configuration files
│   ├── train_config_balanced.yaml
│   └── species_mapping.py
├── docs/                  # Documentation
│   ├── PROJECT_ARCHITECTURE.md
│   ├── BALANCED_SAMPLING_DESIGN.md
│   └── ...
├── tests/                 # Unit tests
├── scripts/               # Utility scripts
└── demo.py               # Demo script
```

## 🔬 Model Architecture

GeneDiffusion implements a masked diffusion approach tailored for genomic sequences:

1. **Tokenization**: DNA sequences → 6-mer tokens
2. **Diffusion Process**: Progressive masking with learnable schedules
3. **Denoising**: Transformer-based architecture predicts masked tokens
4. **Species Conditioning**: FiLM layers for species-specific generation

<details>
<summary>Technical Details</summary>

- **Vocabulary**: 4,096 6-mers + 3 special tokens
- **Model Size**: ~110M parameters (768d, 12 layers)
- **Context Length**: 10kb sequences (1,666 tokens)
- **Loss Function**: Diffusion loss with 1/t scaling + biological constraints

</details>

## 📊 Training Data

- **230 Mammalian Species**: From human to platypus
- **3.45M Sequences**: 15,000 balanced samples per species
- **10kb Fragments**: Sliding window with 80% overlap
- **Full Genome Coverage**: Coding, non-coding, and regulatory regions

## 🎯 Applications

1. **Sequence Generation**: Create novel DNA sequences with specific properties
2. **Functional Prediction**: Predict gene functions and regulatory elements
3. **Variant Analysis**: Assess the impact of genetic mutations
4. **Cross-Species Transfer**: Apply knowledge across species boundaries

## 📈 Performance

| Metric | Value |
|--------|-------|
| Training Samples | 3.45M |
| Validation Perplexity | TBD |
| Generation Quality | TBD |
| Species Balance | Perfect (15k each) |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use GeneDiffusion in your research, please cite:

```bibtex
@software{genediffusion2024,
  title={GeneDiffusion: Masked Diffusion Model for Mammalian Genomic Sequences},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/gene-diffusion}
}
```

## 🙏 Acknowledgments

- Thanks to the [LLaDA](https://arxiv.org/abs/2410.02388) authors for the masked diffusion framework
- Genomic data provided by NCBI RefSeq
- Computational resources from [Your Institution]

## 📧 Contact

For questions and feedback:
- Open an issue on GitHub
- Email: your.email@institution.edu

---

<p align="center">
  Made with ❤️ for the genomics community
</p> 