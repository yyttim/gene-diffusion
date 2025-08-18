"""
GeneDiffusion Training Script
"""

import os
import sys
import argparse
import yaml
import json
from datetime import datetime
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.distributed as dist  # type: ignore
from torch.nn.parallel import DistributedDataParallel as DDP  # type: ignore
from torch.utils.data import DataLoader, DistributedSampler  # type: ignore
from torch.optim import AdamW  # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  # type: ignore
from torch.cuda.amp import autocast, GradScaler  # type: ignore
import wandb  # type: ignore

# Disable unnecessary torch.distributed logs
import logging
logging.getLogger("torch.distributed").setLevel(logging.WARNING)

# Safe wandb wrappers
def safe_wandb_log(data):
    try:
        if hasattr(wandb, 'run') and getattr(wandb, 'run', None) is not None:
            getattr(wandb, 'log', lambda x: None)(data)
    except:
        pass

def safe_wandb_init(**kwargs):
    try:
        return getattr(wandb, 'init', lambda **kw: None)(**kwargs)
    except:
        return None

def safe_wandb_finish():
    try:
        getattr(wandb, 'finish', lambda: None)()
    except:
        pass

from tqdm import tqdm
from typing import Dict, Optional, Tuple

from src.models import GeneDiffusionModel, BiologicalConstraintLoss
from src.data_processor import GenomicDataset, create_data_loader, collate_batch
from src.preprocessed_dataset import PreprocessedDataset, create_preprocessed_dataloader


def setup_logging(output_dir: str, rank: int = 0):
    os.makedirs(output_dir, exist_ok=True)
    
    if rank == 0:
        log_file = os.path.join(output_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',  # Simplified format
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        # Other ranks don't print to stdout
        logging.basicConfig(
            level=logging.WARNING,
            handlers=[logging.NullHandler()]
        )


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return 0, 1, 0, False

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    return rank, world_size, gpu, True


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml'):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    return config


class DNADiffusionTrainer:
    """Trainer for GeneDiffusion model"""
    
    def __init__(self, config: dict, rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'
        
        # Initialize components
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Loss functions
        self.diffusion_loss = nn.CrossEntropyLoss(reduction='none')
        self.biological_constraint_loss = BiologicalConstraintLoss(
            alpha=config.get('codon_loss_weight', 0.1),
            beta=config.get('gc_loss_weight', 0.1),
            gamma=config.get('pattern_loss_weight', 0.1)
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Training config
        self.use_conservation = config.get('use_conservation_guidance', True)
        self.conservation_warmup_epochs = config.get('conservation_warmup_epochs', 10)
        
        # Mixed precision
        self.use_amp = config['training'].get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # GC content targets by species
        self.gc_targets = config.get('gc_targets', {
            'homo_sapiens': 0.41,
            'mus_musculus': 0.42,
            'default': 0.45
        })
    
    def _build_model(self) -> nn.Module:
        model_config = self.config['model']
        model = GeneDiffusionModel(
            vocab_size=model_config.get('vocab_size', 4099),
            num_annotation_types=model_config.get('num_annotation_types', 6),
            num_species=model_config.get('num_species', 228),
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            d_ff=model_config['d_ff'],
            max_seq_len=model_config.get('max_seq_len', 16384),
            dropout=model_config.get('dropout', 0.1),
            use_dna_features=model_config.get('use_dna_features', True),
            use_gradient_checkpointing=model_config.get('use_gradient_checkpointing', False)
        )
        
        model = model.to(self.device)
        
        # Wrap for distributed training
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.local_rank])
        
        return model
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_config = self.config['optimizer']
        
        # Learning rate scaling for multi-node training
        if 'base_lr' in opt_config:
            base_lr = opt_config['base_lr']
            lr_scaling = opt_config.get('lr_scaling', 'sqrt')
            
            if lr_scaling == 'sqrt':
                actual_lr = base_lr * np.sqrt(self.world_size)
            elif lr_scaling == 'linear':
                actual_lr = base_lr * self.world_size
            else:
                actual_lr = base_lr
                
            if self.rank == 0:  # Only main process logs
                logging.info(f"LR: {actual_lr:.4f} (base={base_lr}, {lr_scaling} scaling)")
        else:
            actual_lr = opt_config['lr']
        
        # Parameter groups with different learning rates
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'annotation_encoder' in n],
                'lr': actual_lr * 0.1,
                'name': 'annotation_encoder'
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'dna' in n.lower()],
                'lr': actual_lr * 0.5,
                'name': 'dna_modules'
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'annotation_encoder' not in n and 'dna' not in n.lower()],
                'lr': actual_lr,
                'name': 'main_model'
            }
        ]
        
        optimizer = AdamW(
            param_groups,
            lr=opt_config['lr'],
            betas=(opt_config.get('beta1', 0.9), opt_config.get('beta2', 0.999)),
            eps=float(opt_config.get('eps', 1e-8)),
            weight_decay=opt_config.get('weight_decay', 0.01)
        )
        
        return optimizer
    
    def _build_scheduler(self):
        scheduler_config = self.config.get('scheduler', {})
        
        if scheduler_config.get('type') == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config.get('T_0', 10),
                T_mult=scheduler_config.get('T_mult', 2),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train one epoch"""
        self.model.train()
        
        use_conservation = self.use_conservation and epoch >= self.conservation_warmup_epochs
        
        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        # Metrics collection
        train_batch_metrics = []
        train_mask_dist = []
        train_loss_dist = []
        train_accuracy_dist = []
        train_main_loss_total = 0
        train_gc_loss_total = 0
        
        # Progress bar - only on main process
        if self.rank == 0:
            pbar = tqdm(dataloader, desc=f'Epoch {epoch}', 
                       mininterval=1.0,  # Update every second
                       ncols=100)  # Narrower progress bar
        else:
            pbar = dataloader
        
        gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        memory_cleanup_interval = 100  # Increased from 50
        checkpoint_every_n_steps = self.config['training'].get('checkpoint_every_n_steps', None)
        
        for i, batch in enumerate(pbar):
            # GPU memory monitoring - reduced frequency
            if self.rank == 0 and torch.cuda.is_available() and i % 500 == 0 and i > 0:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                logging.info(f"Step {i}: GPU mem {gpu_memory:.1f}GB")
            
            # Move data to device
            sequences = batch['input_ids'].to(self.device)
            annotations = batch['annotations'].to(self.device)
            species_ids = batch['species_ids'].to(self.device)
            
            batch_size = sequences.shape[0]
            
            # Sample mask ratio from U(0,1]
            t = 1.0 - torch.rand(batch_size, device=self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    model = self.model.module if hasattr(self.model, 'module') else self.model
                    outputs = model.compute_loss(sequences, annotations, species_ids, t)
                    diffusion_loss = outputs['loss']
            else:
                model = self.model.module if hasattr(self.model, 'module') else self.model
                outputs = model.compute_loss(sequences, annotations, species_ids, t)
                diffusion_loss = outputs['loss']
            
            # Biological constraint loss (if enabled)
            constraint_loss = None
            if epoch >= self.config['training'].get('constraint_start_epoch', 10):
                with torch.no_grad():
                    model = self.model.module if hasattr(self.model, 'module') else self.model
                    t_value = t.mean().item() if isinstance(t, torch.Tensor) else float(t)
                    
                    xt, mask = model.forward_diffusion(sequences, t_value, annotations)
                    model_outputs = model(xt, annotations, species_ids)
                    predictions = model_outputs['logits'].argmax(dim=-1)
                    
                    constraint_loss = self.biological_constraint_loss(predictions, annotations)
                    
                    del xt, mask, model_outputs, predictions
                
                gc_loss = torch.tensor(0.0, device=self.device)
                total_loss_value = diffusion_loss + constraint_loss
            else:
                total_loss_value = diffusion_loss
            
            # Gradient accumulation
            total_loss_value = total_loss_value / gradient_accumulation_steps
            
            # Backward pass
            if i % gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(total_loss_value).backward()
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                total_loss_value.backward()
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
            
            # Update statistics
            total_loss += total_loss_value.item() * batch_size * gradient_accumulation_steps
            total_accuracy += outputs['accuracy'] * batch_size
            total_samples += batch_size
            
            # Collect metrics - original frequency
            if i % 10 == 0:  # Restored to original
                batch_metric = {
                    'batch_idx': i,
                    'loss': total_loss_value.item() * gradient_accumulation_steps,
                    'main_loss': outputs.get('main_loss', torch.tensor(0.0)).item() if torch.is_tensor(outputs.get('main_loss', 0)) else outputs.get('main_loss', 0),
                    'accuracy': outputs['accuracy'],
                    'mask_ratio': t.mean().item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                train_batch_metrics.append(batch_metric)
                train_loss_dist.append(total_loss_value.item() * gradient_accumulation_steps)
                train_accuracy_dist.append(outputs['accuracy'])
                train_mask_dist.append(t.mean().item())
            
            train_main_loss_total += outputs.get('main_loss', torch.tensor(0.0)).item() * batch_size if torch.is_tensor(outputs.get('main_loss', 0)) else outputs.get('main_loss', 0) * batch_size
            train_gc_loss_total += outputs.get('gc_loss', torch.tensor(0.0)).item() * batch_size if torch.is_tensor(outputs.get('gc_loss', 0)) else outputs.get('gc_loss', 0) * batch_size
            
            outputs_accuracy = outputs['accuracy']
            del outputs
            
            # Update progress bar - simplified
            if self.rank == 0:
                pbar.set_postfix({
                    'L': f'{total_loss_value.item():.3f}',
                    'A': f'{outputs_accuracy:.4f}',
                    'M': f'{t.mean().item():.2f}'  # Also show mask ratio
                })
            
            # Clear GPU cache periodically
            if i % memory_cleanup_interval == 0 and i > 0:
                torch.cuda.empty_cache()
            
            if constraint_loss is not None:
                del constraint_loss
            
            self.global_step += 1
            
            # Save checkpoint periodically
            if checkpoint_every_n_steps and self.global_step % checkpoint_every_n_steps == 0 and self.rank == 0:
                checkpoint_path = os.path.join(
                    self.config['output_dir'], 
                    f'checkpoint_step_{self.global_step}.pt'
                )
                self.save_checkpoint(checkpoint_path)
                logging.info(f"Checkpoint saved: step {self.global_step}")
                
                # Keep only recent checkpoints
                keep_n = self.config['training'].get('keep_last_n_checkpoints', 5)
                if keep_n > 0:
                    checkpoint_pattern = os.path.join(self.config['output_dir'], 'checkpoint_step_*.pt')
                    import glob
                    checkpoints = sorted(glob.glob(checkpoint_pattern), 
                                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    
                    if len(checkpoints) > keep_n:
                        for old_checkpoint in checkpoints[:-keep_n]:
                            try:
                                os.remove(old_checkpoint)
                            except Exception as e:
                                pass
            
            # Log to wandb - can keep reduced frequency for wandb
            if self.rank == 0 and self.global_step % 100 == 0:  # More frequent than 500
                try:
                    if hasattr(wandb, "run") and getattr(wandb, "run", None) is not None:
                        safe_wandb_log({
                            'train/loss': total_loss_value.item(),
                            'train/accuracy': outputs_accuracy,
                            'train/mask_ratio_mean': t.mean().item(),
                            'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                            'train/step': self.global_step
                        })
                except AttributeError:
                    pass
        
        # Clear cache at epoch end
        torch.cuda.empty_cache()
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        
        # Save training metrics to JSON - only minimal logging
        if self.rank == 0 and len(train_batch_metrics) > 0:
            train_metrics = {
                'timestamp': datetime.now().isoformat(),
                'epoch': epoch,
                'global_step': self.global_step,
                'dataset_type': 'training',
                
                'overall': {
                    'avg_loss': avg_loss,
                    'avg_accuracy': avg_accuracy,
                    'avg_main_loss': train_main_loss_total / total_samples,
                    'avg_gc_loss': train_gc_loss_total / total_samples,
                    'total_samples': total_samples,
                    'total_batches': len(dataloader)
                },
                
                'distributions': {
                    'loss': {
                        'mean': np.mean(train_loss_dist),
                        'std': np.std(train_loss_dist),
                        'min': np.min(train_loss_dist),
                        'max': np.max(train_loss_dist),
                        'median': np.median(train_loss_dist)
                    },
                    'accuracy': {
                        'mean': np.mean(train_accuracy_dist),
                        'std': np.std(train_accuracy_dist),
                        'min': np.min(train_accuracy_dist),
                        'max': np.max(train_accuracy_dist),
                        'median': np.median(train_accuracy_dist)
                    },
                    'mask_ratio': {
                        'mean': np.mean(train_mask_dist),
                        'std': np.std(train_mask_dist),
                        'min': np.min(train_mask_dist),
                        'max': np.max(train_mask_dist),
                        'median': np.median(train_mask_dist)
                    }
                },
                
                'training_config': {
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'use_amp': self.use_amp,
                    'gradient_accumulation_steps': gradient_accumulation_steps,
                    'use_conservation': use_conservation
                },
                
                'sampled_batch_details': train_batch_metrics
            }
            
            metrics_dir = os.path.join(self.config['output_dir'], 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            
            train_metrics_filename = os.path.join(
                metrics_dir,
                f'train_metrics_epoch_{epoch}_step_{self.global_step}.json'
            )
            
            with open(train_metrics_filename, 'w') as f:
                json.dump(train_metrics, f, indent=2, default=str)
            
            # Save latest link
            latest_train_link = os.path.join(metrics_dir, 'latest_train_metrics.json')
            if os.path.exists(latest_train_link):
                os.remove(latest_train_link)
            with open(latest_train_link, 'w') as f:
                json.dump(train_metrics, f, indent=2, default=str)
        
        return avg_loss, avg_accuracy

    def validate(self, dataloader: DataLoader):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        # Metrics collection
        all_batch_metrics = []
        mask_ratio_dist = []
        loss_dist = []
        accuracy_dist = []
        main_loss_total = 0
        gc_loss_total = 0
        
        # Per-annotation and per-species statistics
        annotation_accuracy = {i: {'correct': 0, 'total': 0} for i in range(6)}
        species_accuracy = {}
        
        with torch.no_grad():
            # Simple iterator for validation
            iterator = dataloader if self.rank != 0 else tqdm(dataloader, desc='Validation', ncols=80)
            
            for batch_idx, batch in enumerate(iterator):
                sequences = batch['input_ids'].to(self.device)
                annotations = batch['annotations'].to(self.device)
                species_ids = batch['species_ids'].to(self.device)
                
                batch_size = sequences.shape[0]
                
                # Fixed mask ratio for validation
                t = torch.ones(batch_size, device=self.device) * 0.3
                
                # Compute loss
                model = self.model.module if hasattr(self.model, 'module') else self.model
                outputs = model.compute_loss(sequences, annotations, species_ids, t)
                
                # Collect metrics
                batch_metrics = {
                    'batch_idx': batch_idx,
                    'loss': outputs['loss'].item(),
                    'main_loss': outputs['main_loss'].item(),
                    'gc_loss': outputs['gc_loss'].item(),
                    'accuracy': outputs['accuracy'],
                    'mask_ratio': outputs['mask_ratio'],
                    'batch_size': batch_size
                }
                all_batch_metrics.append(batch_metrics)
                
                loss_dist.append(outputs['loss'].item())
                accuracy_dist.append(outputs['accuracy'])
                mask_ratio_dist.append(outputs['mask_ratio'])
                
                loss_value = outputs['loss'].item()
                accuracy_value = outputs['accuracy']
                main_loss_total += outputs['main_loss'].item() * batch_size
                gc_loss_total += outputs['gc_loss'].item() * batch_size
                
                # Per-species statistics
                for species_id in species_ids.unique():
                    species_mask = species_ids == species_id
                    if species_mask.any():
                        species_id_int = species_id.item()
                        if species_id_int not in species_accuracy:
                            species_accuracy[species_id_int] = {'correct': 0, 'total': 0, 'accuracy_sum': 0, 'count': 0}
                        species_accuracy[species_id_int]['accuracy_sum'] += outputs['accuracy']
                        species_accuracy[species_id_int]['count'] += 1
                
                del outputs
                
                total_loss += loss_value * batch_size
                total_accuracy += accuracy_value * batch_size
                total_samples += batch_size
                
                # Clear memory periodically
                if batch_idx % 50 == 0 and batch_idx > 0:  # Increased from 20
                    torch.cuda.empty_cache()
        
        torch.cuda.empty_cache()
        
        # Calculate overall statistics
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        avg_main_loss = main_loss_total / total_samples
        avg_gc_loss = gc_loss_total / total_samples
        
        # Distribution statistics
        loss_stats = {
            'mean': np.mean(loss_dist),
            'std': np.std(loss_dist),
            'min': np.min(loss_dist),
            'max': np.max(loss_dist),
            'median': np.median(loss_dist)
        }
        
        accuracy_stats = {
            'mean': np.mean(accuracy_dist),
            'std': np.std(accuracy_dist),
            'min': np.min(accuracy_dist),
            'max': np.max(accuracy_dist),
            'median': np.median(accuracy_dist)
        }
        
        # Per-species average accuracy
        for species_id, stats in species_accuracy.items():
            if stats['count'] > 0:
                stats['avg_accuracy'] = stats['accuracy_sum'] / stats['count']
        
        # Complete metrics dictionary
        complete_metrics = {
            'timestamp': datetime.now().isoformat(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'dataset_type': 'validation',
            
            'overall': {
                'avg_loss': avg_loss,
                'avg_accuracy': avg_accuracy,
                'avg_main_loss': avg_main_loss,
                'avg_gc_loss': avg_gc_loss,
                'total_samples': total_samples,
                'total_batches': len(all_batch_metrics)
            },
            
            'distributions': {
                'loss': loss_stats,
                'accuracy': accuracy_stats,
                'mask_ratio': {
                    'mean': np.mean(mask_ratio_dist),
                    'std': np.std(mask_ratio_dist)
                }
            },
            
            'per_species': species_accuracy,
            
            'config': {
                'mask_ratio_fixed': 0.3,
                'batch_size': self.config['data']['batch_size'],
                'model_params': {
                    'd_model': self.config['model']['d_model'],
                    'n_layers': self.config['model']['n_layers'],
                    'n_heads': self.config['model']['n_heads']
                }
            },
            
            'batch_details': all_batch_metrics if len(all_batch_metrics) < 100 else None
        }
        
        # Save to JSON
        if self.rank == 0:
            metrics_dir = os.path.join(self.config['output_dir'], 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            
            metrics_filename = os.path.join(
                metrics_dir,
                f'validation_metrics_epoch_{self.epoch}_step_{self.global_step}.json'
            )
            
            with open(metrics_filename, 'w') as f:
                json.dump(complete_metrics, f, indent=2, default=str)
            
            # Save latest link
            latest_link = os.path.join(metrics_dir, 'latest_validation_metrics.json')
            if os.path.exists(latest_link):
                os.remove(latest_link)
            with open(latest_link, 'w') as f:
                json.dump(complete_metrics, f, indent=2, default=str)
        
        return avg_loss, avg_accuracy

    def save_checkpoint(self, path: str, is_best: bool = False):
        if self.rank != 0:
            return
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        if self.rank == 0:
            logging.info(f'Resumed from epoch {self.epoch}')
    
    def test(self, test_dataloader: DataLoader, checkpoint_path: Optional[str] = None):
        """Final test evaluation with detailed metrics"""
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
            if self.rank == 0:
                logging.info(f"Testing with checkpoint: {checkpoint_path}")
        
        self.model.eval()
        
        # Collection arrays
        all_predictions = []
        all_targets = []
        all_species = []
        all_annotations = []
        all_losses = []
        all_accuracies = []
        
        # Performance by mask ratio
        mask_ratio_performance = {}
        
        # Detailed batch metrics
        detailed_batch_metrics = []
        
        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        # Test different mask ratios
        test_mask_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        with torch.no_grad():
            # Main test with standard mask ratio
            if self.rank == 0:
                logging.info("Testing...")
            iterator = test_dataloader if self.rank != 0 else tqdm(test_dataloader, desc='Test', ncols=80)
            
            for batch_idx, batch in enumerate(iterator):
                sequences = batch['input_ids'].to(self.device)
                annotations = batch['annotations'].to(self.device)
                species_ids = batch['species_ids'].to(self.device)
                
                batch_size = sequences.shape[0]
                
                # Standard mask ratio
                t = torch.ones(batch_size, device=self.device) * 0.3
                
                # Compute loss and predictions
                model = self.model.module if hasattr(self.model, 'module') else self.model
                outputs = model.compute_loss(sequences, annotations, species_ids, t)
                
                # Get detailed predictions
                xt, mask = model.forward_diffusion(sequences, 0.3, annotations)
                model_outputs = model(xt, annotations, species_ids)
                logits = model_outputs['logits']
                predictions = logits.argmax(dim=-1)
                
                # Collect masked predictions
                masked_predictions = predictions[mask].cpu().numpy()
                masked_targets = sequences[mask].cpu().numpy()
                
                all_predictions.extend(masked_predictions)
                all_targets.extend(masked_targets)
                
                # Collect species and annotations
                for i in range(batch_size):
                    species_id = species_ids[i].item()
                    batch_annotations = annotations[i][mask[i]].cpu().numpy()
                    all_species.extend([species_id] * mask[i].sum().item())
                    all_annotations.extend(batch_annotations)
                
                # Collect metrics
                loss_value = outputs['loss'].item()
                accuracy_value = outputs['accuracy']
                
                all_losses.append(loss_value)
                all_accuracies.append(accuracy_value)
                
                total_loss += loss_value * batch_size
                total_accuracy += accuracy_value * batch_size
                total_samples += batch_size
                
                # Batch details
                detailed_batch_metrics.append({
                    'batch_idx': batch_idx,
                    'loss': loss_value,
                    'accuracy': accuracy_value,
                    'main_loss': outputs['main_loss'].item(),
                    'gc_loss': outputs['gc_loss'].item(),
                    'mask_ratio': 0.3,
                    'batch_size': batch_size,
                    'num_masked': mask.sum().item()
                })
                
                # Clear memory
                del xt, mask, model_outputs, logits, predictions
                
                if batch_idx % 50 == 0 and batch_idx > 0:  # Increased from 20
                    torch.cuda.empty_cache()
            
            # Test different mask ratios (sample only)
            if self.rank == 0:
                for mask_ratio in test_mask_ratios:
                    ratio_losses = []
                    ratio_accuracies = []
                    
                    # Test first 10 batches only
                    for batch_idx, batch in enumerate(test_dataloader):
                        if batch_idx >= 10:
                            break
                        
                        sequences = batch['input_ids'].to(self.device)
                        annotations = batch['annotations'].to(self.device)
                        species_ids = batch['species_ids'].to(self.device)
                        
                        batch_size = sequences.shape[0]
                        t = torch.ones(batch_size, device=self.device) * mask_ratio
                        
                        model = self.model.module if hasattr(self.model, 'module') else self.model
                        outputs = model.compute_loss(sequences, annotations, species_ids, t)
                        
                        ratio_losses.append(outputs['loss'].item())
                        ratio_accuracies.append(outputs['accuracy'])
                    
                    mask_ratio_performance[mask_ratio] = {
                        'avg_loss': np.mean(ratio_losses),
                        'avg_accuracy': np.mean(ratio_accuracies),
                        'std_loss': np.std(ratio_losses),
                        'std_accuracy': np.std(ratio_accuracies)
                    }
        
        torch.cuda.empty_cache()
        
        # Calculate overall statistics
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        
        # Performance by annotation type
        annotation_performance = {}
        for ann_type in range(6):
            ann_mask = np.array(all_annotations) == ann_type
            if ann_mask.any():
                ann_predictions = np.array(all_predictions)[ann_mask]
                ann_targets = np.array(all_targets)[ann_mask]
                ann_accuracy = (ann_predictions == ann_targets).mean()
                annotation_performance[ann_type] = {
                    'accuracy': float(ann_accuracy),
                    'count': int(ann_mask.sum()),
                    'name': ['none', 'exon', 'cds', 'lnc_rna', 'gene', 'trna'][ann_type]
                }
        
        # Performance by species
        species_performance = {}
        unique_species = np.unique(all_species)
        for species_id in unique_species:
            species_mask = np.array(all_species) == species_id
            if species_mask.any():
                species_predictions = np.array(all_predictions)[species_mask]
                species_targets = np.array(all_targets)[species_mask]
                species_accuracy = (species_predictions == species_targets).mean()
                species_performance[int(species_id)] = {
                    'accuracy': float(species_accuracy),
                    'count': int(species_mask.sum())
                }
        
        # Token distribution
        from collections import Counter
        prediction_distribution = Counter(all_predictions)
        target_distribution = Counter(all_targets)
        
        # Complete test metrics
        test_metrics = {
            'timestamp': datetime.now().isoformat(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'dataset_type': 'test',
            'checkpoint_used': checkpoint_path if checkpoint_path else 'current_model',
            
            'overall_performance': {
                'avg_loss': avg_loss,
                'avg_accuracy': avg_accuracy,
                'total_samples': total_samples,
                'total_predictions': len(all_predictions),
                'loss_statistics': {
                    'mean': np.mean(all_losses),
                    'std': np.std(all_losses),
                    'min': np.min(all_losses),
                    'max': np.max(all_losses),
                    'median': np.median(all_losses)
                },
                'accuracy_statistics': {
                    'mean': np.mean(all_accuracies),
                    'std': np.std(all_accuracies),
                    'min': np.min(all_accuracies),
                    'max': np.max(all_accuracies),
                    'median': np.median(all_accuracies)
                }
            },
            
            'mask_ratio_performance': mask_ratio_performance,
            'annotation_performance': annotation_performance,
            'species_performance': dict(list(species_performance.items())[:10]),  # Top 10 species
            'total_species_tested': len(species_performance),
            
            'token_distribution': {
                'top_predicted_tokens': dict(prediction_distribution.most_common(20)),
                'top_target_tokens': dict(target_distribution.most_common(20)),
                'unique_predicted_tokens': len(prediction_distribution),
                'unique_target_tokens': len(target_distribution)
            },
            
            'model_config': {
                'd_model': self.config['model']['d_model'],
                'n_layers': self.config['model']['n_layers'],
                'n_heads': self.config['model']['n_heads'],
                'vocab_size': self.config['model'].get('vocab_size', 4099)
            },
            
            'test_config': {
                'batch_size': self.config['data']['batch_size'],
                'standard_mask_ratio': 0.3,
                'tested_mask_ratios': test_mask_ratios
            },
            
            'batch_details': detailed_batch_metrics[:100]  # First 100 batches
        }
        
        # Save to JSON
        if self.rank == 0:
            metrics_dir = os.path.join(self.config['output_dir'], 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            
            test_metrics_filename = os.path.join(
                metrics_dir,
                f'test_metrics_epoch_{self.epoch}_step_{self.global_step}_final.json'
            )
            
            with open(test_metrics_filename, 'w') as f:
                json.dump(test_metrics, f, indent=2, default=str)
            
            # Create summary report
            summary_filename = os.path.join(metrics_dir, 'test_summary.txt')
            with open(summary_filename, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("GeneDiffusion Model Test Report\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Test time: {test_metrics['timestamp']}\n")
                f.write(f"Model checkpoint: {test_metrics['checkpoint_used']}\n")
                f.write(f"Total samples: {test_metrics['overall_performance']['total_samples']}\n")
                f.write(f"Total predictions: {test_metrics['overall_performance']['total_predictions']}\n\n")
                
                f.write("Main performance metrics:\n")
                f.write(f"  Average Loss: {avg_loss:.4f}\n")
                f.write(f"  Average Accuracy: {avg_accuracy:.4f}\n\n")
                
                f.write("Accuracy by annotation type:\n")
                for ann_type, perf in annotation_performance.items():
                    f.write(f"  {perf['name']:10s}: {perf['accuracy']:.4f} (n={perf['count']})\n")
                
                f.write("\nPerformance by mask ratio:\n")
                for ratio, perf in sorted(mask_ratio_performance.items()):
                    f.write(f"  {ratio:.1f}: acc={perf['avg_accuracy']:.4f}, loss={perf['avg_loss']:.4f}\n")
                
                f.write("\n" + "=" * 60 + "\n")
            
            # Print summary
            logging.info(f"Test Complete: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}")
        
        return avg_loss, avg_accuracy
    
    def _manage_checkpoints(self, keep_last_n: int):
        """Keep only the most recent N checkpoints"""
        if keep_last_n <= 0:
            return
            
        checkpoint_dir = self.config['output_dir']
        checkpoint_files = []
        for f in os.listdir(checkpoint_dir):
            if f.startswith('checkpoint_step_') and f.endswith('.pt'):
                try:
                    step = int(f.replace('checkpoint_step_', '').replace('.pt', ''))
                    checkpoint_files.append((step, os.path.join(checkpoint_dir, f)))
                except:
                    continue
        
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        
        if len(checkpoint_files) > keep_last_n:
            for _, filepath in checkpoint_files[keep_last_n:]:
                try:
                    os.remove(filepath)
                except Exception as e:
                    pass
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Main training loop"""
        
        # GPU info - only once at start
        if self.rank == 0:
            if torch.cuda.is_available():
                logging.info(f"Training on {torch.cuda.device_count()} GPUs ({torch.cuda.get_device_name()})")
                logging.info(f"Model: {self.config['model']['n_layers']}L-{self.config['model']['d_model']}D ({sum(p.numel() for p in self.model.parameters())/1e6:.0f}M params)")
                logging.info(f"Batch: {self.config['data']['batch_size']} x {self.config['training']['gradient_accumulation_steps']} accum = {self.config['data']['batch_size'] * self.config['training']['gradient_accumulation_steps']} effective")
            else:
                logging.warning("No GPU detected, using CPU")
        
        # Initialize wandb
        if self.rank == 0:
            wandb_mode = self.config.get('wandb_mode', 'online')
            if wandb_mode == 'disabled':
                os.environ['WANDB_MODE'] = 'disabled'
            else:
                safe_wandb_init(
                    project=self.config.get('wandb_project', 'gene-diffusion'),
                    name=self.config.get('experiment_name', f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                    config=self.config
                )
        
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_dataloader, epoch)
            
            # Validate
            if val_dataloader is not None and epoch % self.config['training'].get('val_interval', 1) == 0:
                val_loss, val_acc = self.validate(val_dataloader)
                
                # Log validation metrics
                if self.rank == 0:
                    safe_wandb_log({
                        'val/loss': val_loss,
                        'val/accuracy': val_acc,
                        'epoch': epoch
                    })
                    logging.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')
                    
                    # Save best model
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint(
                            os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch}.pt'),
                            is_best=True
                        )
                        logging.info(f'New best model: loss={val_loss:.4f}')
            else:
                if self.rank == 0:
                    logging.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}')
            
            # Save checkpoint periodically
            if epoch % self.config['training'].get('save_interval', 10) == 0:
                self.save_checkpoint(
                    os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch}.pt')
                )
            
            # Log training metrics
            if self.rank == 0:
                safe_wandb_log({
                    'train/epoch_loss': train_loss,
                    'train/epoch_accuracy': train_acc,
                    'epoch': epoch
                })
        
        # Save final model
        self.save_checkpoint(
            os.path.join(self.config['output_dir'], 'final_checkpoint.pt')
        )
        
        if self.rank == 0:
            safe_wandb_finish()
            logging.info('Training complete!')


def main():
    parser = argparse.ArgumentParser(description='Train GeneDiffusion Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode with small data')
    parser.add_argument('--test', action='store_true', help='Run test evaluation only')
    parser.add_argument('--test_checkpoint', type=str, help='Checkpoint path for testing')
    # Distributed training
    parser.add_argument('--local-rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # cuDNN settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Setup distributed
    rank, world_size, gpu, is_distributed = setup_distributed()
    
    # Load config
    config = load_config(args.config)
    
    # Override config
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.debug:
        config['data']['max_samples'] = 100
        config['training']['num_epochs'] = 2
    
    # Setup output and logging
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir, rank)
    
    # Log config - only main process, simplified
    if rank == 0:
        logging.info(f"Config: {args.config}")
        logging.info(f"Output: {output_dir}")
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    # Create data loaders
    use_preprocessed = config['data'].get('use_preprocessed', False)
    
    if use_preprocessed:
        # Use preprocessed dataset
        if rank == 0:
            logging.info("Loading preprocessed dataset...")
        species_list = config['data'].get('species_list', None)
        max_samples = config['data'].get('max_samples', -1)
        max_samples_val = config['data'].get('max_samples_val', -1)
        
        if max_samples is None:
            max_samples = -1
        if max_samples_val is None:
            max_samples_val = -1
        
        train_dataloader = create_preprocessed_dataloader(
            data_dir=config['data']['data_dir'],
            batch_size=config['data']['batch_size'],
            species_list=species_list,
            max_samples_per_species=max_samples if max_samples > 0 else None,
            shuffle=True,
            num_workers=config['data'].get('num_workers', 4),
            cache_in_memory=False
        )
        
        val_dataloader = create_preprocessed_dataloader(
            data_dir=config['data']['data_dir'],
            batch_size=config['data']['batch_size'],
            species_list=species_list,
            max_samples_per_species=max_samples_val if max_samples_val > 0 else None,
            shuffle=False,
            num_workers=config['data'].get('num_workers', 4),
            cache_in_memory=False
        )
        
        if args.test:
            test_dataloader = create_preprocessed_dataloader(
                data_dir=config['data']['data_dir'],
                batch_size=config['data']['batch_size'],
                species_list=species_list,
                max_samples_per_species=config['data'].get('max_samples_test', max_samples_val) if config['data'].get('max_samples_test', max_samples_val) > 0 else None,
                shuffle=False,
                num_workers=config['data'].get('num_workers', 4),
                cache_in_memory=False
            )
    else:
        # Use raw dataset
        train_dataset = GenomicDataset(
            data_dir=config['data']['data_dir'],
            species_list=None,
            seq_length=config['model'].get('max_seq_len', 5000),
            overlap=0.5,
            min_annotation_ratio=0.1,
            cache_sequences=config['data'].get('use_cache', True),
            max_sequences_per_species=config['data'].get('max_samples', 1000) if config['data'].get('max_samples') else 1000
        )
        
        val_dataset = GenomicDataset(
            data_dir=config['data']['data_dir'],
            species_list=None,
            seq_length=config['model'].get('max_seq_len', 5000),
            overlap=0.8,
            min_annotation_ratio=0.1,
            cache_sequences=config['data'].get('use_cache', True),
            max_sequences_per_species=config['data'].get('max_samples_val', 100)
        )
        
        # Samplers
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
        
        # Data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=config['data'].get('num_workers', 4),
            collate_fn=collate_batch,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            sampler=val_sampler,
            shuffle=False,
            num_workers=config['data'].get('num_workers', 4),
            collate_fn=collate_batch,
            pin_memory=True
        )
        
        if args.test:
            test_dataloader = val_dataloader
    
    # Create trainer
    trainer = DNADiffusionTrainer(config, rank, world_size)
    
    # Resume or load checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif args.test and args.test_checkpoint:
        trainer.load_checkpoint(args.test_checkpoint)
    
    # Test or train
    if args.test:
        if rank == 0:
            logging.info("Starting test...")
        if 'test_dataloader' not in locals():
            test_dataloader = val_dataloader
        
        test_loss, test_acc = trainer.test(
            test_dataloader, 
            checkpoint_path=args.test_checkpoint
        )
        
        if rank == 0:
            logging.info(f"Test complete: Loss={test_loss:.4f}, Acc={test_acc:.4f}")
    else:
        # Start training
        if rank == 0:
            logging.info(f"Starting training...")
        trainer.train(train_dataloader, val_dataloader)
    
    # Cleanup distributed
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main() 