#!/usr/bin/env python3
"""
Test script to verify the training pipeline works
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import yaml
from src.models.two_tower_model import TwoTowerModel
from src.data.data_loader import DataProcessor, create_data_loader
from src.training.trainer import TwoTowerTrainer

def main():
    print("=" * 60)
    print("Testing Training Pipeline")
    print("=" * 60)
    
    # Load config
    config_path = 'config/training_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n[1] Testing DataProcessor...")
    data_config = config.get('data', {})
    data_processor = DataProcessor(data_config)
    
    try:
        data_path = data_config.get('data_path', 'data/processed')
        train_dataset, val_dataset, test_dataset = data_processor.process_data(data_path)
        print(f"✅ Data loaded successfully")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    print("\n[2] Testing DataLoader creation...")
    try:
        batch_size = config.get('training', {}).get('batch_size', 256)
        train_loader = create_data_loader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Test one batch
        for batch in train_loader:
            print(f"✅ Batch loaded successfully")
            print(f"   Batch keys: {list(batch.keys())}")
            print(f"   User features: {list(batch['user_features'].keys())}")
            print(f"   Item features: {list(batch['item_features'].keys())}")
            print(f"   Text features shape: {batch['text_features'].shape}")
            print(f"   Labels shape: {batch['labels'].shape}")
            break
    except Exception as e:
        print(f"❌ DataLoader creation failed: {e}")
        return False
    
    print("\n[3] Testing Model initialization...")
    try:
        model_config = config.get('model', {})
        model = TwoTowerModel(
            user_feature_dims=data_processor.user_feature_dims,
            item_feature_dims=data_processor.item_feature_dims,
            text_feature_dim=data_processor.emb_dim,
            embedding_dim=model_config.get('embedding_dim', 128),
            user_hidden_dims=model_config.get('user_mlp_hidden', [256, 128]),
            item_hidden_dims=model_config.get('item_mlp_hidden', [256, 128]),
            dropout_rate=model_config.get('dropout', 0.1),
            logit_scale=model_config.get('logit_scale', 10.0)
        )
        print(f"✅ Model initialized successfully")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[4] Testing Forward pass...")
    try:
        model.eval()
        batch_count = 0
        for batch in train_loader:
            # Debug: print actual values
            if batch_count == 0:
                print("\nDebug: Batch values")
                print(f"  User features:")
                for k, v in batch['user_features'].items():
                    print(f"    {k}: shape={v.shape}, min={v.min()}, max={v.max()}")
                print(f"  Item features:")
                for k, v in batch['item_features'].items():
                    print(f"    {k}: shape={v.shape}, min={v.min()}, max={v.max()}")
                    vocab_size = data_processor.item_feature_dims.get(k, 'unknown')
                    print(f"      vocab_size={vocab_size}")
                    
            user_embeddings, item_embeddings = model(
                user_features=batch['user_features'],
                item_features=batch['item_features'],
                text_features=batch['text_features']
            )
            if batch_count == 0:
                print(f"✅ Forward pass successful")
                print(f"   User embeddings shape: {user_embeddings.shape}")
                print(f"   Item embeddings shape: {item_embeddings.shape}")
            batch_count += 1
            if batch_count >= 2:
                break
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[5] Testing Trainer initialization...")
    try:
        val_loader = create_data_loader(val_dataset, batch_size=batch_size, shuffle=False)
        
        trainer_config = {
            'epochs': 1,  # Just test with 1 epoch
            'learning_rate': config.get('training', {}).get('learning_rate', 1e-3),
            'weight_decay': config.get('training', {}).get('weight_decay', 1e-5),
            'optimizer': 'adam',
            'scheduler': None,
            'grad_clip_norm': None,
            'early_stopping_patience': None
        }
        
        trainer = TwoTowerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config
        )
        print(f"✅ Trainer initialized successfully")
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[6] Testing one training step...")
    try:
        model.train()
        trainer.optimizer.zero_grad()
        
        for batch in train_loader:
            batch = trainer._move_batch_to_device(batch)
            
            user_emb, item_emb = model(
                user_features=batch['user_features'],
                item_features=batch['item_features'],
                text_features=batch['text_features']
            )
            
            loss = trainer._compute_positive_negative_loss(
                user_emb, item_emb, batch['labels']
            )
            
            loss.backward()
            trainer.optimizer.step()
            
            print(f"✅ Training step successful")
            print(f"   Loss: {loss.item():.4f}")
            break
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Pipeline is ready.")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

