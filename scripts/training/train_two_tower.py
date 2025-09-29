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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.two_tower_model import TwoTowerModel
from data.data_loader import DataProcessor, create_data_loader
from training.trainer import TwoTowerTrainer
from utils.metrics import evaluate_recommendation_system


def setup_logging(config):
    """Setup logging configuration"""
    log_config = config['logging']
    
    # Create logs directory
    Path(log_config['file']).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config['level']),
        format=log_config['format'],
        handlers=[
            logging.FileHandler(log_config['file']),
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
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Two Tower Model training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['model_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['plot_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['metrics_dir']).mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize data processor
        logger.info("Initializing data processor...")
        data_processor = DataProcessor(config['data'])
        
        # Process data
        logger.info("Processing data...")
        train_dataset, val_dataset, test_dataset = data_processor.process_data(args.data_path)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader = create_data_loader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True
        )
        val_loader = create_data_loader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False
        )
        test_loader = create_data_loader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False
        )
        
        # Initialize model
        logger.info("Initializing model...")
        model = TwoTowerModel(
            user_feature_dims=data_processor.user_feature_dims,
            item_feature_dims=data_processor.item_feature_dims,
            text_feature_dim=config['model']['text_feature_dim'],
            embedding_dim=config['model']['embedding_dim'],
            user_hidden_dims=config['model']['user_hidden_dims'],
            item_hidden_dims=config['model']['item_hidden_dims'],
            dropout_rate=config['model']['dropout_rate']
        )
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = TwoTowerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config['training']
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_model(args.resume)
        
        # Train model
        logger.info("Starting training...")
        training_results = trainer.train()
        
        # Save model
        model_path = Path(args.output_dir) / 'best_model.pth'
        trainer.save_model(str(model_path))
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = evaluate_model_on_test_set(trainer, test_loader, config)
        
        # Save results
        results = {
            'training_results': training_results,
            'test_results': test_results,
            'config': config
        }
        
        results_path = Path(args.output_dir) / 'training_results.yaml'
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Model saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


def evaluate_model_on_test_set(trainer, test_loader, config):
    """Evaluate model on test set"""
    logger = logging.getLogger(__name__)
    
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
            
            # Compute similarity scores
            similarity_scores = trainer.model.compute_similarity(
                user_embeddings, item_embeddings
            )
            
            # Collect predictions and targets
            all_predictions.extend(similarity_scores.cpu().numpy().flatten())
            all_targets.extend(batch['targets'].cpu().numpy().flatten())
    
    # Calculate metrics
    import numpy as np
    test_metrics = evaluate_recommendation_system(
        np.array(all_targets),
        np.array(all_predictions),
        k_values=config['evaluation']['k_values']
    )
    
    logger.info(f"Test metrics: {test_metrics}")
    return test_metrics


if __name__ == '__main__':
    main()
