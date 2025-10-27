import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from pathlib import Path
import logging
from collections import defaultdict

from utils.logger import setup_logger
from utils.config import save_config, load_config
from evaluation.metrics import AudioMetrics

class TTSTrainer:
    """Training pipeline for TTS models"""
    
    def __init__(self, model, config=None, device=None):
        self.model = model
        self.config = config or {}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Training parameters
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 16)
        self.num_epochs = self.config.get('num_epochs', 100)
        self.gradient_clip_val = self.config.get('gradient_clip_val', 1.0)
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=1e-6
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.get('lr_step_size', 50000),
            gamma=self.config.get('lr_gamma', 0.5)
        )
        
        # Loss weights
        self.mel_loss_weight = self.config.get('mel_loss_weight', 1.0)
        self.postnet_loss_weight = self.config.get('postnet_loss_weight', 1.0)
        self.stop_loss_weight = self.config.get('stop_loss_weight', 50.0)
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        self.training_history = defaultdict(list)
        
        # Setup logging
        self.logger = setup_logger('trainer')
        
        # Metrics
        self.metrics = AudioMetrics()
        
        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def train(self, train_loader, val_loader=None, resume_from=None):
        """Main training loop"""
        
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.logger.info(f"Starting training from epoch {self.current_epoch}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_losses = self._train_epoch(train_loader)
            
            # Validation phase
            if val_loader is not None:
                val_losses = self._validate_epoch(val_loader)
            else:
                val_losses = {}
            
            # Update learning rate
            self.scheduler.step()
            
            # Log epoch results
            self._log_epoch_results(train_losses, val_losses)
            
            # Save checkpoint
            if epoch % 10 == 0 or val_losses.get('total_loss', float('inf')) < self.best_loss:
                self.save_checkpoint(epoch, val_losses.get('total_loss', train_losses['total_loss']))
            
            # Early stopping check
            if self._should_stop_early():
                self.logger.info("Early stopping triggered")
                break
        
        self.logger.info("Training completed")
        
    def _train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = defaultdict(float)
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()
            
            # Forward pass
            losses = self._train_step(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.optimizer.step()
            self.current_step += 1
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            
            # Log step
            if batch_idx % 10 == 0:
                step_time = time.time() - start_time
                self.logger.info(
                    f"Epoch {self.current_epoch:3d} | "
                    f"Step {batch_idx:4d}/{num_batches:4d} | "
                    f"Loss: {losses['total_loss'].item():.4f} | "
                    f"Time: {step_time:.2f}s"
                )
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            self.training_history[f'train_{key}'].append(epoch_losses[key])
        
        return epoch_losses
    
    def _validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = defaultdict(float)
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                losses = self._validation_step(batch)
                
                # Accumulate losses
                for key, value in losses.items():
                    epoch_losses[key] += value.item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            self.training_history[f'val_{key}'].append(epoch_losses[key])
        
        return epoch_losses
    
    def _train_step(self, batch):
        """Single training step"""
        # Move batch to device
        text_inputs = batch['text'].to(self.device)
        mel_targets = batch['mel'].to(self.device)
        stop_targets = batch['stop_tokens'].to(self.device)
        
        # Forward pass
        outputs = self.model(text_inputs, mel_targets)
        
        # Calculate losses
        targets = {
            'mel_targets': mel_targets,
            'stop_targets': stop_targets
        }
        
        losses = self.model.calculate_loss(outputs, targets)
        
        return losses
    
    def _validation_step(self, batch):
        """Single validation step"""
        # Move batch to device
        text_inputs = batch['text'].to(self.device)
        mel_targets = batch['mel'].to(self.device)
        stop_targets = batch['stop_tokens'].to(self.device)
        
        # Forward pass
        outputs = self.model(text_inputs, mel_targets)
        
        # Calculate losses
        targets = {
            'mel_targets': mel_targets,
            'stop_targets': stop_targets
        }
        
        losses = self.model.calculate_loss(outputs, targets)
        
        return losses
    
    def train_step(self):
        """Simplified training step for demo purposes"""
        # Simulate a training step with random loss
        loss = np.random.uniform(0.5, 2.0) * np.exp(-self.current_step * 0.001)
        self.current_step += 1
        return loss
    
    def _log_epoch_results(self, train_losses, val_losses):
        """Log epoch results"""
        train_loss = train_losses['total_loss']
        
        log_str = f"Epoch {self.current_epoch:3d} | Train Loss: {train_loss:.4f}"
        
        if val_losses:
            val_loss = val_losses['total_loss']
            log_str += f" | Val Loss: {val_loss:.4f}"
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                log_str += " (Best)"
        
        self.logger.info(log_str)
    
    def _should_stop_early(self):
        """Check if early stopping criteria are met"""
        patience = self.config.get('patience', 10)
        
        if len(self.training_history['val_total_loss']) < patience:
            return False
        
        recent_losses = self.training_history['val_total_loss'][-patience:]
        return all(loss >= self.best_loss for loss in recent_losses)
    
    def save_checkpoint(self, epoch, loss):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'current_step': self.current_step,
            'training_history': dict(self.training_history),
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if loss <= self.best_loss:
            best_checkpoint_path = self.checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_checkpoint_path)
        
        # Save periodic checkpoint
        if epoch % 50 == 0:
            periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, periodic_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.current_step = checkpoint['current_step']
        self.best_loss = checkpoint['best_loss']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")
    
    def get_training_stats(self):
        """Get training statistics"""
        return {
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_loss': self.best_loss,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'device': str(self.device),
            'learning_rate': self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.learning_rate
        }
