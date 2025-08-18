"""
GeneDiffusion Inference Script
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
from data_processor import DNATokenizer, AnnotationProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DNASequenceGenerator:
    """DNA sequence generator using trained GeneDiffusion model"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DNATokenizer()
        self.annotation_processor = AnnotationProcessor()
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        logger.info(f"Model loaded on device: {self.device}")
    
    def _load_model(self, checkpoint_path: str) -> GeneDiffusionModel:
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # Create model
        model_config = config['model']
        model = GeneDiffusionModel(
            vocab_size=model_config.get('vocab_size', 4099),
            num_annotation_types=model_config.get('num_annotation_types', 6),
            num_species=model_config.get('num_species', 228),
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            d_ff=model_config['d_ff'],
            max_seq_len=model_config.get('max_seq_len', 1024),
            dropout=0.0,  # No dropout during inference
            use_dna_features=model_config.get('use_dna_features', True),
            use_gradient_checkpointing=False  # No gradient checkpointing during inference
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Handle DDP saved models
            state_dict = checkpoint.get('state_dict', checkpoint)
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
                 species_id: int = 0,
                 num_samples: int = 1,
                 temperature: float = 1.0,
                 top_k: Optional[int] = 50,
                 num_steps: int = 100) -> List[str]:
        """
        Generate DNA sequences
        
        Args:
            seq_length: Sequence length in tokens
            annotations: Annotation tensor [batch, seq_len]
            species_id: Species ID for generation
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            num_steps: Number of denoising steps
        
        Returns:
            List of generated DNA sequences
        """
        logger.info(f"Generating {num_samples} sequences of length {seq_length}...")
        
        # Create default annotations if not provided
        if annotations is None:
            annotations = torch.zeros(num_samples, seq_length, dtype=torch.long, device=self.device)
        else:
            annotations = annotations.to(self.device)
        
        # Create species IDs tensor
        species_ids = torch.full((num_samples,), species_id, dtype=torch.long, device=self.device)
        
        # Generate sequences
        generated_tokens = self.model.sample(
            batch_size=num_samples,
            seq_len=seq_length,
            annotations=annotations,
            species_ids=species_ids,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            device=self.device
        )
        
        # Decode to DNA sequences
        sequences = []
        for tokens in generated_tokens:
            seq = self.tokenizer.decode(tokens.cpu().numpy().tolist())
            sequences.append(seq)
        
        return sequences
    
    @torch.no_grad()
    def complete_sequence(self,
                         partial_sequence: str,
                         mask_positions: List[int],
                         annotations: Optional[torch.Tensor] = None,
                         species_id: int = 0,
                         temperature: float = 1.0,
                         num_steps: int = 50) -> str:
        """
        Complete a partial sequence
        
        Args:
            partial_sequence: Partial DNA sequence
            mask_positions: Positions to complete
            annotations: Annotation information
            species_id: Species ID
            temperature: Sampling temperature
            num_steps: Number of denoising steps
        
        Returns:
            Completed sequence
        """
        logger.info(f"Completing sequence with {len(mask_positions)} masked positions")
        
        # Encode sequence
        tokens = torch.tensor([self.tokenizer.encode(partial_sequence)], 
                            dtype=torch.long, device=self.device)
        
        # Set mask positions
        for pos in mask_positions:
            if pos < tokens.shape[1]:
                tokens[0, pos] = self.tokenizer.mask_token_id
        
        # Create annotations if needed
        if annotations is None:
            annotations = torch.zeros(1, tokens.shape[1], dtype=torch.long, device=self.device)
        
        # Create species ID
        species_ids = torch.tensor([species_id], dtype=torch.long, device=self.device)
        
        # Iterative denoising
        for step in range(num_steps):
            # Model prediction
            outputs = self.model(tokens, annotations, species_ids)
            logits = outputs['logits']
            
            # Only sample at masked positions
            mask = (tokens == self.tokenizer.mask_token_id)
            
            if mask.sum() == 0:
                break
            
            # Temperature sampling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample new tokens
            sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), 1)
            sampled = sampled.view(tokens.shape)
            
            # Update masked positions
            tokens = torch.where(mask, sampled, tokens)
            
            # Optionally remask low-confidence positions
            if step < num_steps - 1:
                confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                low_conf_mask = (confidence < 0.8) & mask
                num_remask = int(low_conf_mask.sum() * (1 - step / num_steps))
                
                if num_remask > 0:
                    remask_indices = torch.where(low_conf_mask[0])[0]
                    remask_indices = remask_indices[torch.randperm(len(remask_indices))[:num_remask]]
                    tokens[0, remask_indices] = self.tokenizer.mask_token_id
        
        # Decode
        completed_sequence = self.tokenizer.decode(tokens[0].cpu().numpy().tolist())
        
        return completed_sequence
    
    def evaluate_sequences(self, sequences: List[str]) -> Dict[str, float]:
        """Evaluate generated sequence quality"""
        metrics = {
            'gc_content_mean': 0,
            'gc_content_std': 0,
            'n_ratio': 0,
            'sequence_diversity': 0,
            'avg_length': 0
        }
        
        gc_contents = []
        n_ratios = []
        
        for seq in sequences:
            # GC content
            seq_upper = seq.upper()
            gc_count = seq_upper.count('G') + seq_upper.count('C')
            gc = gc_count / len(seq) if len(seq) > 0 else 0
            gc_contents.append(gc)
            
            # N ratio
            n_ratio = seq_upper.count('N') / len(seq) if len(seq) > 0 else 0
            n_ratios.append(n_ratio)
        
        # Calculate statistics
        metrics['gc_content_mean'] = np.mean(gc_contents)
        metrics['gc_content_std'] = np.std(gc_contents)
        metrics['n_ratio'] = np.mean(n_ratios)
        metrics['avg_length'] = np.mean([len(s) for s in sequences])
        
        # Sequence diversity (unique k-mers)
        all_kmers = set()
        for seq in sequences:
            for i in range(len(seq) - 5):
                all_kmers.add(seq[i:i+6])
        metrics['sequence_diversity'] = len(all_kmers)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='GeneDiffusion inference')
    
    # Basic arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Model checkpoint path')
    parser.add_argument('--output_file', type=str, default='generated_sequences.fasta',
                       help='Output FASTA file')
    parser.add_argument('--mode', type=str, choices=['generate', 'complete'],
                       default='generate', help='Run mode')
    
    # Generation arguments
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of sequences to generate')
    parser.add_argument('--seq_length', type=int, default=1000,
                       help='Sequence length in tokens')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Number of denoising steps')
    parser.add_argument('--species_id', type=int, default=0,
                       help='Species ID for generation')
    
    # Completion mode arguments
    parser.add_argument('--input_sequence', type=str,
                       help='Input sequence for completion')
    parser.add_argument('--mask_positions', type=int, nargs='+',
                       help='Positions to complete')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Compute device')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate generated sequences')
    
    args = parser.parse_args()
    
    # Create generator
    generator = DNASequenceGenerator(args.checkpoint, args.device)
    
    # Run generation
    if args.mode == 'generate':
        sequences = generator.generate(
            seq_length=args.seq_length,
            species_id=args.species_id,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            num_steps=args.num_steps
        )
    
    elif args.mode == 'complete':
        if not args.input_sequence or not args.mask_positions:
            raise ValueError("Completion mode requires --input_sequence and --mask_positions")
        
        completed = generator.complete_sequence(
            partial_sequence=args.input_sequence,
            mask_positions=args.mask_positions,
            species_id=args.species_id,
            temperature=args.temperature,
            num_steps=args.num_steps
        )
        sequences = [completed]
    
    # Save sequences
    logger.info(f"Saving {len(sequences)} sequences to {args.output_file}")
    records = []
    for i, seq in enumerate(sequences):
        record = SeqRecord(
            Seq(seq),
            id=f"GeneDiffusion_seq_{i+1}",
            description=f"Generated by GeneDiffusion | Mode: {args.mode} | Species: {args.species_id}"
        )
        records.append(record)
    
    SeqIO.write(records, args.output_file, "fasta")
    
    # Evaluate if requested
    if args.evaluate:
        logger.info("Evaluating sequences...")
        metrics = generator.evaluate_sequences(sequences)
        
        logger.info("Evaluation results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save evaluation results
        eval_file = args.output_file.replace('.fasta', '_evaluation.json')
        with open(eval_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation saved to {eval_file}")


if __name__ == "__main__":
    main() 