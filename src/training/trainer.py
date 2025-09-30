"""
Training utilities for Two Tower Model

Trainer Input/Output:
====================
Input:
- model: TwoTowerModel - The two-tower model to train
- train_loader: DataLoader - Training data with positive/negative samples
- val_loader: DataLoader - Validation data
- config: dict - Training configuration

Output:
- training_results: dict - Training results including losses and metrics
  {
      'train_losses': List[float],
      'val_losses': List[float], 
      'val_metrics': List[dict],
      'best_val_loss': float
  }

Example:
--------
trainer = TwoTowerTrainer(model, train_loader, val_loader, config)
results = trainer.train()
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..models.two_tower_model import TwoTowerModel
from ..utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class TwoTowerTrainer:
    """Trainer class for Two Tower Model"""
    
    def __init__(self,
                 model: TwoTowerModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict):
        """
        Initialize trainer
        
        Args:
            model: Two Tower Model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        if self.config['optimizer'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        if self.config.get('scheduler') == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 10),
                gamma=self.config.get('gamma', 0.1)
            )
        elif self.config.get('scheduler') == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs']
            )
        else:
            return None
    
    def train_epoch(self) -> float:
        """Train for one epoch using positive-negative samples"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            user_embeddings, item_embeddings = self.model(
                user_features=batch['user_features'],
                item_features=batch['item_features'],
                text_features=batch['text_features']
            )
            
            # Compute positive-negative loss
            loss = self._compute_positive_negative_loss(
                user_embeddings, item_embeddings, batch['labels']
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip_norm']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate_epoch(self) -> Tuple[float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                user_embeddings, item_embeddings = self.model(
                    user_features=batch['user_features'],
                    item_features=batch['item_features'],
                    text_features=batch['text_features']
                )
                
                # Compute similarity scores
                similarity_scores = self.model.compute_similarity(
                    user_embeddings, item_embeddings
                )
                
                # Compute loss
                batch_size = similarity_scores.size(0)
                targets = batch['targets'].unsqueeze(1).expand(-1, batch_size)
                loss = self.criterion(similarity_scores, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions and targets for metrics
                all_predictions.extend(similarity_scores.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
        
        # Calculate metrics
        metrics = calculate_metrics(
            np.array(all_predictions),
            np.array(all_targets)
        )
        
        return total_loss / num_batches, metrics
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device"""
        device_batch = {}
        
        # Move user features
        device_batch['user_features'] = {}
        for key, value in batch['user_features'].items():
            device_batch['user_features'][key] = value.to(self.device)
        
        # Move item features
        device_batch['item_features'] = {}
        for key, value in batch['item_features'].items():
            device_batch['item_features'][key] = value.to(self.device)
        
        # Move other tensors
        device_batch['text_features'] = batch['text_features'].to(self.device)
        device_batch['labels'] = batch['labels'].to(self.device)
        device_batch['ratings'] = batch['ratings'].to(self.device)
        
        return device_batch
    
    def train(self) -> Dict:
        """Train the model"""
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(self.config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Val Metrics: {val_metrics}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                logger.info(f"New best model saved (Val Loss: {val_loss:.4f})")
            
            # Early stopping
            if self.config.get('early_stopping_patience'):
                if epoch >= self.config['early_stopping_patience']:
                    recent_val_losses = self.val_losses[-self.config['early_stopping_patience']:]
                    if all(val_loss >= recent_val_losses[i] for i in range(1, len(recent_val_losses))):
                        logger.info("Early stopping triggered")
                        break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model")
        
        # Plot training history
        self._plot_training_history()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_val_loss': best_val_loss
        }
    
    def _plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot metrics
        if self.val_metrics:
            metrics_names = list(self.val_metrics[0].keys())
            for metric_name in metrics_names:
                metric_values = [metrics[metric_name] for metrics in self.val_metrics]
                ax2.plot(metric_values, label=metric_name)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Metric Value')
            ax2.set_title('Validation Metrics')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/plots/training_history.png')
        plt.close()
    
    def save_model(self, path: str):
        """Save model and training history"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        logger.info(f"Model loaded from {path}")
    
    
    def _compute_positive_negative_loss(self, 
                                      user_emb: torch.Tensor, 
                                      item_emb: torch.Tensor, 
                                      labels: torch.Tensor) -> torch.Tensor:
        """Compute positive-negative sample loss"""
        # Compute similarity scores
        similarity = torch.sum(user_emb * item_emb, dim=1)  # [batch_size]
        
        # Binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(similarity, labels)
        
        return loss
