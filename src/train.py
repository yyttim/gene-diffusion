"""
GeneDiffusion训练脚本
创新的DNA序列Masked Diffusion Model训练流程
"""

import os
import sys
import argparse
import yaml
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from tqdm import tqdm
import logging
from typing import Dict, Optional, Tuple

from models import GeneDiffusionModel, BiologicalConstraintLoss
from data_processor import GenomicDataset, create_data_loader, collate_batch


def setup_logging(output_dir: str, rank: int = 0):
    """设置日志系统"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 只在主进程设置文件日志
    if rank == 0:
        log_file = os.path.join(output_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=f'[Rank {rank}] %(levelname)s: %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )


def setup_distributed():
    """设置分布式训练环境"""
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
    """加载配置文件"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml'):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    return config


class DNADiffusionTrainer:
    """GeneDiffusion模型训练器
    
    实现DNA序列特有的训练策略：
    - 动态mask比例调整
    - 进化保守性引导
    - 生物学约束优化
    """
    
    def __init__(self, config: dict, rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        
        # 初始化模型
        self.model = self._build_model()
        
        # 初始化优化器
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # 损失函数
        self.diffusion_loss = nn.CrossEntropyLoss(reduction='none')
        self.biological_constraint_loss = BiologicalConstraintLoss(
            alpha=config.get('codon_loss_weight', 0.1),
            beta=config.get('gc_loss_weight', 0.1),
            gamma=config.get('pattern_loss_weight', 0.1)
        )
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # 进化保守性策略
        self.use_conservation = config.get('use_conservation_guidance', True)
        self.conservation_warmup_epochs = config.get('conservation_warmup_epochs', 10)
        
        # GC含量目标（可根据物种调整）
        self.gc_targets = config.get('gc_targets', {
            'homo_sapiens': 0.41,
            'mus_musculus': 0.42,
            'default': 0.45
        })
    
    def _build_model(self) -> nn.Module:
        """构建模型"""
        model_config = self.config['model']
        model = GeneDiffusionModel(
            vocab_size=model_config.get('vocab_size', 8),
            num_annotation_types=model_config.get('num_annotation_types', 9),
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            d_ff=model_config['d_ff'],
            max_seq_len=model_config.get('max_seq_len', 16384),
            dropout=model_config.get('dropout', 0.1),
            use_dna_features=model_config.get('use_dna_features', True)
        )
        
        model = model.to(self.device)
        
        # 分布式训练包装
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.rank])
        
        return model
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """构建优化器"""
        opt_config = self.config['optimizer']
        
        # 参数分组：对不同模块使用不同学习率
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'annotation_encoder' in n],
                'lr': opt_config['lr'] * 0.1,  # 注释编码器使用较小学习率
                'name': 'annotation_encoder'
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'dna' in n.lower()],  # DNA特有模块
                'lr': opt_config['lr'] * 0.5,
                'name': 'dna_modules'
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'annotation_encoder' not in n and 'dna' not in n.lower()],
                'lr': opt_config['lr'],
                'name': 'main_model'
            }
        ]
        
        optimizer = AdamW(
            param_groups,
            lr=opt_config['lr'],
            betas=(opt_config.get('beta1', 0.9), opt_config.get('beta2', 0.999)),
            eps=opt_config.get('eps', 1e-8),
            weight_decay=opt_config.get('weight_decay', 0.01)
        )
        
        return optimizer
    
    def _build_scheduler(self):
        """构建学习率调度器"""
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
    
    def get_dynamic_mask_ratio(self, epoch: int) -> Tuple[float, float]:
        """获取动态mask比例
        
        训练初期使用高mask比例，逐渐降低
        """
        warmup_epochs = self.config['training'].get('mask_warmup_epochs', 20)
        min_ratio = self.config['training'].get('min_mask_ratio', 0.15)
        max_ratio = self.config['training'].get('max_mask_ratio', 0.85)
        
        if epoch < warmup_epochs:
            # 线性递减
            progress = epoch / warmup_epochs
            current_max = max_ratio - (max_ratio - 0.5) * progress
            return min_ratio, current_max
        else:
            return min_ratio, 0.5
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        # 获取当前epoch的mask比例范围
        min_mask, max_mask = self.get_dynamic_mask_ratio(epoch)
        
        # 是否使用进化保守性
        use_conservation = self.use_conservation and epoch >= self.conservation_warmup_epochs
        
        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        # 进度条（仅主进程显示）
        if self.rank == 0:
            pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        else:
            pbar = dataloader
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            sequences = batch['input_ids'].to(self.device)
            annotations = batch['annotations'].to(self.device)
            species_ids = batch['species_ids'].to(self.device)
            annotation_coverage = batch['annotation_coverage'].to(self.device)
            
            batch_size = sequences.shape[0]
            
            # 采样mask比例
            t = torch.rand(batch_size, device=self.device) * (max_mask - min_mask) + min_mask
            
            # 计算损失
            model = self.model.module if hasattr(self.model, 'module') else self.model
            outputs = model.compute_loss(
                sequences, 
                annotations, 
                species_ids,
                t
            )
            
            # 主扩散损失
            diffusion_loss = outputs['loss']
            
            # DNA约束损失（仅在后期加入）
            if epoch >= self.config['training'].get('constraint_start_epoch', 10):
                # 获取模型预测用于约束损失
                with torch.no_grad():
                    # 创建mask序列
                    model = self.model.module if hasattr(self.model, 'module') else self.model
                    xt, mask = model.forward_diffusion(
                        sequences, 
                        t.mean().item(),
                        annotations
                    )
                    model_outputs = model(xt, annotations, species_ids)
                    predictions = model_outputs['logits']
                
                # 计算生物学约束损失
                constraint_loss = self.biological_constraint_loss(
                    predictions, 
                    annotations
                )
                
                # 根据物种调整GC含量目标（简化版本，使用默认目标）
                target_gc = self.gc_targets['default']
                    gc_loss = 0
                for i in range(batch_size):
                        pred_gc = torch.softmax(predictions[i], dim=-1)[:, 1:3].sum(dim=-1).mean()
                        gc_loss += (pred_gc - target_gc).abs()
                gc_loss = gc_loss / batch_size
                    constraint_loss = constraint_loss + 0.1 * gc_loss
                
                total_loss_value = diffusion_loss + constraint_loss
            else:
                total_loss_value = diffusion_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss_value.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training'].get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            
            # 更新统计
            total_loss += total_loss_value.item() * batch_size
            total_accuracy += outputs['accuracy'] * batch_size  # accuracy已经是float
            total_samples += batch_size
            
            # 更新进度条
            if self.rank == 0:
                pbar.set_postfix({
                    'loss': total_loss_value.item(),
                    'acc': outputs['accuracy'],
                    'perp': outputs['perplexity'],
                    'mask': f'{min_mask:.2f}-{max_mask:.2f}'
                })
            
            self.global_step += 1
            
            # 定期记录到wandb
            if self.rank == 0 and self.global_step % 100 == 0:
                wandb.log({
                    'train/loss': total_loss_value.item(),
                    'train/accuracy': outputs['accuracy'],
                    'train/perplexity': outputs['perplexity'],
                    'train/mask_ratio_min': min_mask,
                    'train/mask_ratio_max': max_mask,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/step': self.global_step
                })
        
        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()
        
        # 返回epoch平均指标
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        
        return avg_loss, avg_accuracy
    
    def validate(self, dataloader: DataLoader):
        """验证模型"""
        self.model.eval()
        
        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation', disable=self.rank != 0):
                sequences = batch['input_ids'].to(self.device)
                annotations = batch['annotations'].to(self.device)
                species_ids = batch['species_ids'].to(self.device)
                
                batch_size = sequences.shape[0]
                
                # 使用固定的mask比例进行验证
                t = torch.ones(batch_size, device=self.device) * 0.3
                
                # 计算损失
                model = self.model.module if hasattr(self.model, 'module') else self.model
                outputs = model.compute_loss(
                    sequences, 
                    annotations, 
                    species_ids,
                    t
                )
                
                total_loss += outputs['loss'].item() * batch_size
                total_accuracy += outputs['accuracy'] * batch_size  # accuracy已经是float
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        
        return avg_loss, avg_accuracy
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """保存检查点"""
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
        logging.info(f'Saved checkpoint to {path}')
        
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            logging.info(f'Saved best checkpoint to {best_path}')
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
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
        
        logging.info(f'Loaded checkpoint from {path} (epoch {self.epoch})')
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """主训练循环"""
        num_epochs = self.config['training']['num_epochs']
        
        # 初始化wandb（仅主进程）
        if self.rank == 0:
            wandb.init(
                project=self.config.get('wandb_project', 'gene-diffusion'),
                name=self.config.get('experiment_name', f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                config=self.config
            )
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # 设置分布式采样器的epoch
            if hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_dataloader, epoch)
            
            # 验证
            if val_dataloader is not None and epoch % self.config['training'].get('val_interval', 1) == 0:
                val_loss, val_acc = self.validate(val_dataloader)
                
                # 记录验证指标
                if self.rank == 0:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/accuracy': val_acc,
                        'val/perplexity': np.exp(val_loss),
                        'epoch': epoch
                    })
                    
                    # 保存最佳模型
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint(
                            os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch}.pt'),
                            is_best=True
                        )
            
            # 定期保存检查点
            if epoch % self.config['training'].get('save_interval', 10) == 0:
                self.save_checkpoint(
                    os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch}.pt')
                )
            
            # 记录训练指标
            if self.rank == 0:
                logging.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}')
                wandb.log({
                    'train/epoch_loss': train_loss,
                    'train/epoch_accuracy': train_acc,
                    'train/epoch_perplexity': np.exp(train_loss),
                    'epoch': epoch
                })
        
        # 训练结束，保存最终模型
        self.save_checkpoint(
            os.path.join(self.config['output_dir'], 'final_checkpoint.pt')
        )
        
        if self.rank == 0:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train GeneDiffusion Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode with small data')
    
    args = parser.parse_args()
    
    # 设置分布式训练
    rank, world_size, gpu, is_distributed = setup_distributed()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.debug:
        config['data']['max_samples'] = 100
        config['training']['num_epochs'] = 2
    
    # 设置输出目录和日志
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir, rank)
    
    # 记录配置
    if rank == 0:
        logging.info(f"Configuration: {json.dumps(config, indent=2)}")
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    # 创建数据加载器
    train_dataset = GenomicDataset(
        data_dir=config['data']['data_dir'],
        species_list=None,  # 使用所有物种
        seq_length=config['model'].get('max_seq_len', 5000),
        overlap=0.5,
        min_annotation_ratio=0.1,
        cache_sequences=config['data'].get('use_cache', True),
        max_sequences_per_species=config['data'].get('max_samples', 1000) if config['data'].get('max_samples') else 1000
    )
    
    val_dataset = GenomicDataset(
        data_dir=config['data']['data_dir'],
        species_list=None,  # 使用所有物种
        seq_length=config['model'].get('max_seq_len', 5000),
        overlap=0.8,  # 验证集使用更少的重叠
        min_annotation_ratio=0.1,
        cache_sequences=config['data'].get('use_cache', True),
        max_sequences_per_species=config['data'].get('max_samples_val', 100)
    )
    
    # 数据采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
    
    # 数据加载器
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
    
    # 创建训练器
    trainer = DNADiffusionTrainer(config, rank, world_size)
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    logging.info(f"Starting training on rank {rank}/{world_size}")
    trainer.train(train_dataloader, val_dataloader)
    
    # 清理分布式训练
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main() 