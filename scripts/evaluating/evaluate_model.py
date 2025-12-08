#!/usr/bin/env python3
"""
Evaluation script: Evaluate Two Tower recommendation model using HR@K and Recall@K
Usage: python scripts/evaluating/evaluate_model.py --model_path outputs/results/best_model.pth --k_values 5,10,20,50
"""

'''
python scripts/evaluating/evaluate_model.py \
    --model_path outputs/results/best_model.pth \
    --data_path data/processed \
    --k_values 5,10,20,50
    --output_dir results/my_evaluation
'''

import argparse
import sys
import os
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
sys.path.insert(0, str(ROOT))

from src.models.two_tower_model import TwoTowerModel
from src.data.data_loader import DataProcessor, create_data_loader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")


def load_model(
    model_path: str,
    user_feature_dims: Dict[str, int],
    item_feature_dims: Dict[str, int],
    text_feature_dim: int = 768,
    embedding_dim: int = 128,
    device: str = None
) -> TwoTowerModel:
    """Load trained model"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading model from {model_path}")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = TwoTowerModel(
        user_feature_dims=user_feature_dims,
        item_feature_dims=item_feature_dims,
        text_feature_dim=text_feature_dim,
        embedding_dim=embedding_dim,
        user_hidden_dims=[256, 128],
        item_hidden_dims=[256, 128],
        dropout_rate=0.1,
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")
    
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def calculate_hr_at_k(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    k: int
) -> float:
    """
    Calculate Hit Rate@K
    
    Args:
        scores: Predicted score array (n_items,)
        ground_truth: Ground truth label array (n_items,), 1 for positive samples, 0 for negative samples
        k: top-K value
        
    Returns:
        HR@K value
    """
    if len(scores) == 0:
        return 0.0
    
    # Sort by scores in descending order
    top_k_indices = np.argsort(scores)[::-1][:k]
    
    # Check if there's at least one positive sample in top-K
    hit = np.sum(ground_truth[top_k_indices]) > 0
    
    return 1.0 if hit else 0.0


def calculate_recall_at_k(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    k: int
) -> float:
    """
    Calculate Recall@K
    
    Args:
        scores: Predicted score array (n_items,)
        ground_truth: Ground truth label array (n_items,), 1 for positive samples, 0 for negative samples
        k: top-K value
        
    Returns:
        Recall@K value
    """
    if len(scores) == 0:
        return 0.0
    
    total_relevant = np.sum(ground_truth)
    if total_relevant == 0:
        return 0.0
    
    # Sort by scores in descending order
    top_k_indices = np.argsort(scores)[::-1][:k]
    
    # Count number of positive samples in top-K
    relevant_in_topk = np.sum(ground_truth[top_k_indices])
    
    # Recall@K = number of positive samples in top-K / total number of positive samples
    recall = relevant_in_topk / total_relevant
    
    return recall


def fuse_text_features(
    item_id: str,
    item_text_emb: Dict[str, List[float]],
    review_emb: List[float] = None,
    emb_dim: int = 768
) -> torch.Tensor:
    """
    Fuse item text embedding and review embedding (consistent with data loader logic)
    
    Args:
        item_id: Item ID
        item_text_emb: Item text embedding dictionary
        review_emb: Review embedding (optional, None for negative samples)
        emb_dim: Embedding dimension
        
    Returns:
        Fused text feature tensor
    """
    zero_emb = torch.zeros(emb_dim, dtype=torch.float32)
    
    # Get item embedding
    e_item = item_text_emb.get(str(item_id))
    v_item = torch.tensor(e_item, dtype=torch.float32) if e_item is not None else zero_emb
    
    # Get review embedding
    if review_emb is None:
        v_rev = zero_emb
    else:
        v_rev = torch.tensor(review_emb, dtype=torch.float32)
    
    # Ensure dimensions match
    if v_rev.numel() != v_item.numel():
        return v_item
    
    # Fuse: beta * v_item + alpha * v_rev
    alpha = 0.6
    beta = 0.4
    return beta * v_item + alpha * v_rev


def evaluate_user(
    model: TwoTowerModel,
    user_features: Dict[str, torch.Tensor],
    candidate_items: List[Dict],
    item_text_emb: Dict[str, List[float]],
    iid2idx: Dict[str, int],
    item_review_emb_map: Dict[str, List[float]],
    device: torch.device,
    k_values: List[int],
    emb_dim: int = 768
) -> Dict[str, float]:
    """
    Evaluate HR@K and Recall@K for a single user
    
    Args:
        model: Trained model
        user_features: User feature dictionary
        candidate_items: List of candidate items, each element contains item_id and label
        item_text_emb: Item text embedding dictionary
        iid2idx: Item ID to index mapping
        item_review_emb_map: Dictionary mapping item_id to review_emb (for fast lookup)
        device: Device
        k_values: List of K values
        emb_dim: Embedding dimension
        
    Returns:
        Dictionary containing HR@K and Recall@K
    """
    batch_size = 256
    n_candidates = len(candidate_items)
    
    # Prepare candidate item features
    item_ids = [item['item_id'] for item in candidate_items]
    labels = np.array([item['label'] for item in candidate_items])
    
    # Get user embedding (only need to compute once)
    with torch.no_grad():
        user_emb = model.get_user_embeddings(user_features)  # Shape: (1, embedding_dim)
        user_emb = user_emb.squeeze(0)  # Shape: (embedding_dim,)
    
    # Batch compute item embeddings and scores
    scores = []
    
    for i in range(0, n_candidates, batch_size):
        batch_items = item_ids[i:i+batch_size]
        batch_item_features = []
        batch_text_features = []
        
        for item_id in batch_items:
            item_idx = iid2idx.get(str(item_id), 0)
            item_feat = {"item_id": torch.tensor([item_idx], dtype=torch.long)}
            batch_item_features.append(item_feat)
            
            # Get review embedding (if positive sample, get from map; None for negative samples)
            review_emb = item_review_emb_map.get(str(item_id), None)
            
            # Fuse text features
            text_tensor = fuse_text_features(
                item_id=str(item_id),
                item_text_emb=item_text_emb,
                review_emb=review_emb,
                emb_dim=emb_dim
            )
            batch_text_features.append(text_tensor)
        
        # Merge batch
        batch_item_dict = {}
        for key in batch_item_features[0].keys():
            batch_item_dict[key] = torch.cat([feat[key] for feat in batch_item_features], dim=0).to(device)
        
        batch_text_tensor = torch.stack(batch_text_features, dim=0).to(device)
        
        # Compute item embeddings
        with torch.no_grad():
            item_emb = model.get_item_embeddings(batch_item_dict, batch_text_tensor)  # Shape: (batch_size, embedding_dim)
            
            # Compute similarity scores (dot product)
            # user_emb: (embedding_dim,), item_emb: (batch_size, embedding_dim)
            similarity = torch.sum(user_emb.unsqueeze(0) * item_emb, dim=1)  # Shape: (batch_size,)
            scores.extend(similarity.cpu().numpy())
    
    scores = np.array(scores)
    
    # Calculate HR@K and Recall@K
    results = {}
    for k in k_values:
        results[f'HR@{k}'] = calculate_hr_at_k(scores, labels, k)
        results[f'Recall@{k}'] = calculate_recall_at_k(scores, labels, k)
    
    return results


def evaluate_model(
    model: TwoTowerModel,
    test_dataset,
    data_processor: DataProcessor,
    k_values: List[int] = [5, 10, 20, 50],
    num_negatives: int = 1000,
    device: str = None,
    max_users: int = None
) -> Dict[str, float]:
    """
    Evaluate HR@K and Recall@K on test set
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        data_processor: Data processor
        k_values: List of K values
        num_negatives: Number of negative samples per user
        device: Device
        
    Returns:
        Dictionary containing average HR@K and Recall@K across all users
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    logger.info(f"Evaluating model on test set...")
    logger.info(f"Number of test samples: {len(test_dataset)}")
    logger.info(f"K values: {k_values}")
    logger.info(f"Number of negatives per user: {num_negatives}")
    
    # Get necessary data
    df = test_dataset.df
    uid2idx = test_dataset.uid2idx
    iid2idx = test_dataset.iid2idx
    item_text_emb = test_dataset.item_text_emb
    all_items = test_dataset.all_items
    user_pos = test_dataset.user_pos
    
    # Pre-build item_id to review_emb mapping for fast lookup (performance optimization)
    logger.info("Building item_id to review_emb mapping...")
    # Use pandas groupby to efficiently create mapping (takes first review_emb for each item_id)
    df_item_id_str = df['item_id'].astype(str)
    df_with_str = df.copy()
    df_with_str['item_id_str'] = df_item_id_str
    # Group by item_id and take first review_emb (drop duplicates)
    item_review_df = df_with_str[['item_id_str', 'review_emb']].drop_duplicates(subset='item_id_str', keep='first')
    item_review_emb_map = {}
    for _, row in item_review_df.iterrows():
        review_emb = row['review_emb']
        # Check if review_emb is valid (not None, not NaN, and not empty)
        # Handle both list/array and scalar values safely
        try:
            if review_emb is not None:
                # If it's a list/array, check if it has elements
                if isinstance(review_emb, (list, np.ndarray)):
                    if len(review_emb) > 0:
                        item_review_emb_map[row['item_id_str']] = review_emb
                else:
                    # For scalar values, check if not NaN
                    if pd.notna(review_emb):
                        item_review_emb_map[row['item_id_str']] = review_emb
        except (TypeError, ValueError, AttributeError):
            # Skip if there's any issue with the value
            pass
    logger.info(f"Built mapping for {len(item_review_emb_map)} items")
    
    # Group test samples by user
    test_df = df.iloc[test_dataset.rows].copy()
    test_df['row_idx'] = test_dataset.rows
    
    # Collect evaluation results for all users
    all_results = {f'HR@{k}': [] for k in k_values}
    all_results.update({f'Recall@{k}': [] for k in k_values})
    
    # Evaluate for each user
    user_groups = test_df.groupby('user_id')
    total_users = len(user_groups)
    
    # Limit number of users if max_users is specified
    if max_users is not None and max_users > 0:
        total_users = min(total_users, max_users)
        logger.info(f"Evaluating {total_users} users (limited from {len(user_groups)} total users)...")
    else:
        logger.info(f"Evaluating {total_users} users...")
    
    for user_idx, (user_id, user_data) in enumerate(user_groups):
        # Stop if max_users limit is reached
        if max_users is not None and user_idx >= max_users:
            break
            
        if (user_idx + 1) % 5000 == 0:
            logger.info(f"Processed {user_idx + 1}/{total_users} users...")
        
        # Get positive samples for this user (items in test set)
        positive_items = user_data['item_id'].astype(str).tolist()
        
        # Generate negative samples for this user
        user_pos_set = user_pos.get(str(user_id), set())
        negative_candidates = [item for item in all_items if item not in user_pos_set]
        
        # Randomly select negative samples
        rng = random.Random(42)  # Fixed random seed for reproducibility
        num_negs = min(num_negatives, len(negative_candidates))
        selected_negatives = rng.sample(negative_candidates, num_negs)
        
        # Build candidate item list (1 positive sample + num_negatives negative samples)
        candidate_items = []
        for item_id in positive_items:
            candidate_items.append({'item_id': item_id, 'label': 1})
        for item_id in selected_negatives:
            candidate_items.append({'item_id': item_id, 'label': 0})
        
        # Prepare user features
        user_idx_val = uid2idx.get(str(user_id), 0)
        user_features = {
            'user_id': torch.tensor([user_idx_val], dtype=torch.long).to(device)
        }
        
        # Evaluate this user
        user_results = evaluate_user(
            model=model,
            user_features=user_features,
            candidate_items=candidate_items,
            item_text_emb=item_text_emb,
            iid2idx=iid2idx,
            item_review_emb_map=item_review_emb_map,
            device=device,
            k_values=k_values,
            emb_dim=data_processor.emb_dim
        )
        
        # Collect results
        for key in all_results.keys():
            all_results[key].append(user_results[key])
    
    # Calculate average
    avg_results = {key: np.mean(values) for key, values in all_results.items()}
    
    logger.info("Evaluation completed!")
    return avg_results


def plot_metrics(
    results: Dict[str, float],
    k_values: List[int],
    output_dir: Path
):
    """Plot visualization charts for HR@K and Recall@K"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract HR@K and Recall@K values
    hr_values = [results[f'HR@{k}'] for k in k_values]
    recall_values = [results[f'Recall@{k}'] for k in k_values]
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # HR@K plot
    axes[0].plot(k_values, hr_values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    axes[0].set_xlabel('K', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Hit Rate@K', fontsize=12, fontweight='bold')
    axes[0].set_title('Hit Rate@K Evaluation', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.1])
    
    # Add value labels
    for k, hr in zip(k_values, hr_values):
        axes[0].annotate(f'{hr:.3f}', (k, hr), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    # Recall@K plot
    axes[1].plot(k_values, recall_values, marker='s', linewidth=2, markersize=8, color='#A23B72')
    axes[1].set_xlabel('K', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Recall@K', fontsize=12, fontweight='bold')
    axes[1].set_title('Recall@K Evaluation', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.1])
    
    # Add value labels
    for k, recall in zip(k_values, recall_values):
        axes[1].annotate(f'{recall:.3f}', (k, recall), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'evaluation_metrics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {plot_path}")
    plt.close()
    
    # Create combined plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(k_values, hr_values, marker='o', linewidth=2, markersize=8, 
            label='HR@K', color='#2E86AB')
    ax.plot(k_values, recall_values, marker='s', linewidth=2, markersize=8, 
            label='Recall@K', color='#A23B72')
    ax.set_xlabel('K', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('HR@K and Recall@K Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels
    for k, hr, recall in zip(k_values, hr_values, recall_values):
        ax.annotate(f'HR:{hr:.3f}', (k, hr), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=8, color='#2E86AB')
        ax.annotate(f'R:{recall:.3f}', (k, recall), textcoords="offset points", 
                   xytext=(0,-15), ha='center', fontsize=8, color='#A23B72')
    
    plt.tight_layout()
    
    # Save combined plot
    combined_plot_path = output_dir / 'evaluation_metrics_combined.png'
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Combined plot saved to {combined_plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Two Tower Model with HR@K and Recall@K')
    parser.add_argument(
        '--model_path',
        type=str,
        default='outputs/results/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/processed',
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--k_values',
        type=str,
        default='5,10,20,50',
        help='Comma-separated list of K values (e.g., 5,10,20,50)'
    )
    parser.add_argument(
        '--num_negatives',
        type=int,
        default=1000,
        help='Number of negative samples per user for evaluation'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/evaluation',
        help='Output directory for results and plots'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--max_users',
        type=int,
        default=None,
        help='Maximum number of users to evaluate (None for all users)'
    )
    
    args = parser.parse_args()
    
    # Parse K values
    k_values = [int(k.strip()) for k in args.k_values.split(',')]
    k_values = sorted(k_values)
    
    logger.info("=" * 70)
    logger.info("Two Tower Model Evaluation")
    logger.info("=" * 70)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"K values: {k_values}")
    logger.info(f"Number of negatives: {args.num_negatives}")
    logger.info(f"Max users: {args.max_users if args.max_users else 'All users'}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\nLoading data...")
    data_config = {
        'data_path': args.data_path,
        'neg_ratio': 0,  # Test set doesn't need negative samples
        'seed': 42,
        'emb_dim': 768
    }
    data_processor = DataProcessor(data_config)
    train_dataset, val_dataset, test_dataset = data_processor.process_data(args.data_path)
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Load model
    logger.info("\nLoading model...")
    model = load_model(
        model_path=args.model_path,
        user_feature_dims=data_processor.user_feature_dims,
        item_feature_dims=data_processor.item_feature_dims,
        text_feature_dim=data_processor.emb_dim,
        embedding_dim=128,
        device=args.device
    )
    
    # Evaluate model
    logger.info("\nEvaluating model...")
    results = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        data_processor=data_processor,
        k_values=k_values,
        num_negatives=args.num_negatives,
        device=args.device,
        max_users=args.max_users
    )
    
    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("Evaluation Results")
    logger.info("=" * 70)
    for k in k_values:
        logger.info(f"HR@{k}: {results[f'HR@{k}']:.4f}")
        logger.info(f"Recall@{k}: {results[f'Recall@{k}']:.4f}")
    logger.info("=" * 70)
    
    # Save results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    plot_metrics(results, k_values, output_dir)
    
    logger.info("\nEvaluation completed successfully!")


if __name__ == '__main__':
    main()
