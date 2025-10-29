#!/usr/bin/env python3
"""
Main training script for Two Tower Model
"""

import argparse
import yaml
import logging
import torch
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.two_tower_model import TwoTowerModel
from src.data.data_loader import DataProcessor, create_data_loader
from src.training.trainer import TwoTowerTrainer


def setup_logging(log_config):
    """Setup logging configuration"""
    # Create logs directory
    log_dir = Path(log_config.get('file', 'logs/training.log')).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_config.get('file', 'logs/training.log')),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Two Tower Model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logging_config = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'logs/training.log'
    }
    setup_logging(logging_config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Two Tower Model training")
    logger.info(f"Configuration: {args.config}")
    
    # Create output directories
    output_dir = Path(config.get('paths', {}).get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (results_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Initialize data processor
        logger.info("Initializing data processor...")
        data_config = config.get('data', {})
        data_processor = DataProcessor(data_config)
        
        # 2. Process data
        logger.info("Processing data...")
        data_path = data_config.get('data_path', 'data/processed')
        train_dataset, val_dataset, test_dataset = data_processor.process_data(data_path)
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # 3. Create data loaders
        logger.info("Creating data loaders...")
        training_config = config.get('training', {})
        batch_size = training_config.get('batch_size', 256)
        
        train_loader = create_data_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = create_data_loader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        test_loader = create_data_loader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # 4. Initialize model
        logger.info("Initializing model...")
        logger.info(f"User feature dims: {data_processor.user_feature_dims}")
        logger.info(f"Item feature dims: {data_processor.item_feature_dims}")
        logger.info(f"Text feature dim: {data_processor.emb_dim}")
        
        model_config = config.get('model', {})
        model = TwoTowerModel(
            user_feature_dims=data_processor.user_feature_dims,
            item_feature_dims=data_processor.item_feature_dims,
            text_feature_dim=data_processor.emb_dim,  # Use actual embedding dimension
            embedding_dim=model_config.get('embedding_dim', 128),
            user_hidden_dims=model_config.get('user_mlp_hidden', [256, 128]),
            item_hidden_dims=model_config.get('item_mlp_hidden', [256, 128]),
            dropout_rate=model_config.get('dropout', 0.1),
            logit_scale=model_config.get('logit_scale', 10.0)  # Learnable temperature parameter
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized with {total_params:,} parameters")
        
        # 5. Prepare training config
        trainer_config = {
            'epochs': training_config.get('num_epochs', 20),
            'learning_rate': training_config.get('learning_rate', 1e-3),
            'weight_decay': training_config.get('weight_decay', 1e-5),
            'optimizer': 'adam',
            'scheduler': None,
            'grad_clip_norm': None,
            'early_stopping_patience': None
        }
        
        # 6. Initialize trainer
        logger.info("Initializing trainer...")
        trainer = TwoTowerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config
        )
        
        # 7. Train model
        logger.info("Starting training...")
        training_results = trainer.train()
        
        # 8. Save model
        model_path = results_dir / 'best_model.pth'
        trainer.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # 9. Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = evaluate_on_test_set(trainer, test_loader)
        logger.info(f"Test results: {test_results}")
        
        # 10. Save results
        results_data = {
            'training_results': training_results,
            'test_results': test_results,
            'config': config
        }
        
        results_path = results_dir / 'training_results.yaml'
        with open(results_path, 'w') as f:
            yaml.dump(results_data, f, default_flow_style=False)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Model saved to: {model_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        print(f"Total parameters: {total_params:,}")
        print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        print(f"Final train loss: {training_results['train_losses'][-1]:.4f}")
        print(f"Final val loss: {training_results['val_losses'][-1]:.4f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


def evaluate_on_test_set(trainer, test_loader):
    """Evaluate model on test set"""
    import numpy as np
    
    trainer.model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = trainer._move_batch_to_device(batch)
            
            # Forward pass
            user_embeddings, item_embeddings = trainer.model(
                user_features=batch['user_features'],
                item_features=batch['item_features'],
                text_features=batch['text_features']
            )
            
            # Compute similarity scores (dot product)
            similarity_scores = torch.sum(user_embeddings * item_embeddings, dim=1)
            
            # Collect predictions and targets
            all_predictions.extend(similarity_scores.cpu().numpy())
            all_targets.extend(batch['labels'].cpu().numpy())
    
    # Calculate metrics
    from src.utils.metrics import calculate_metrics
    metrics = calculate_metrics(
        np.array(all_targets),
        np.array(all_predictions)
    )
    
    return metrics


if __name__ == '__main__':
    main()
