"""
Evaluation metrics for recommendation systems
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, ndcg_score
)
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Regression metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Additional regression metrics
    metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    metrics['smape'] = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    
    return metrics


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error"""
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100


def calculate_ranking_metrics(y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Calculate ranking metrics for recommendation systems
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted scores
        k_values: List of k values for top-k metrics
        
    Returns:
        Dictionary of ranking metrics
    """
    metrics = {}
    
    # Convert to binary relevance (threshold-based)
    threshold = np.median(y_true)  # Use median as threshold
    y_true_binary = (y_true >= threshold).astype(int)
    
    # Sort by predicted scores
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true_binary[sorted_indices]
    
    # Calculate metrics for different k values
    for k in k_values:
        y_true_k = y_true_sorted[:k]
        
        # Precision@k
        precision_k = np.sum(y_true_k) / k if k > 0 else 0
        metrics[f'precision@{k}'] = precision_k
        
        # Recall@k
        total_relevant = np.sum(y_true_binary)
        recall_k = np.sum(y_true_k) / total_relevant if total_relevant > 0 else 0
        metrics[f'recall@{k}'] = recall_k
        
        # F1@k
        if precision_k + recall_k > 0:
            f1_k = 2 * precision_k * recall_k / (precision_k + recall_k)
        else:
            f1_k = 0
        metrics[f'f1@{k}'] = f1_k
        
        # NDCG@k
        ndcg_k = calculate_ndcg(y_true_sorted, k)
        metrics[f'ndcg@{k}'] = ndcg_k
    
    return metrics


def calculate_ndcg(y_true_sorted: np.ndarray, k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k
    
    Args:
        y_true_sorted: True relevance scores sorted by predicted scores
        k: Number of top items to consider
        
    Returns:
        NDCG@k value
    """
    if k == 0:
        return 0.0
    
    # DCG@k
    dcg = 0.0
    for i in range(min(k, len(y_true_sorted))):
        dcg += y_true_sorted[i] / np.log2(i + 2)
    
    # IDCG@k (ideal DCG)
    y_true_sorted_ideal = np.sort(y_true_sorted)[::-1]
    idcg = 0.0
    for i in range(min(k, len(y_true_sorted_ideal))):
        idcg += y_true_sorted_ideal[i] / np.log2(i + 2)
    
    # NDCG@k
    if idcg == 0:
        return 0.0
    else:
        return dcg / idcg


def calculate_hit_rate(y_true: np.ndarray, 
                      y_pred: np.ndarray, 
                      k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Calculate hit rate at different k values
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted scores
        k_values: List of k values
        
    Returns:
        Dictionary of hit rates
    """
    metrics = {}
    
    # Convert to binary relevance
    threshold = np.median(y_true)
    y_true_binary = (y_true >= threshold).astype(int)
    
    # Sort by predicted scores
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true_binary[sorted_indices]
    
    for k in k_values:
        # Hit rate@k: whether there's at least one relevant item in top-k
        hit_rate = 1.0 if np.sum(y_true_sorted[:k]) > 0 else 0.0
        metrics[f'hit_rate@{k}'] = hit_rate
    
    return metrics


def calculate_coverage(y_pred: np.ndarray, 
                      unique_items: int, 
                      k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Calculate catalog coverage at different k values
    
    Args:
        y_pred: Predicted scores
        unique_items: Number of unique items in catalog
        k_values: List of k values
        
    Returns:
        Dictionary of coverage metrics
    """
    metrics = {}
    
    # Sort by predicted scores
    sorted_indices = np.argsort(y_pred)[::-1]
    
    for k in k_values:
        # Coverage@k: fraction of unique items in top-k
        top_k_items = sorted_indices[:k]
        unique_top_k = len(np.unique(top_k_items))
        coverage = unique_top_k / unique_items if unique_items > 0 else 0.0
        metrics[f'coverage@{k}'] = coverage
    
    return metrics


def calculate_diversity_metrics(y_pred: np.ndarray, 
                               item_features: np.ndarray,
                               k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Calculate diversity metrics for recommendations
    
    Args:
        y_pred: Predicted scores
        item_features: Item feature matrix
        k_values: List of k values
        
    Returns:
        Dictionary of diversity metrics
    """
    metrics = {}
    
    # Sort by predicted scores
    sorted_indices = np.argsort(y_pred)[::-1]
    
    for k in k_values:
        top_k_indices = sorted_indices[:k]
        top_k_features = item_features[top_k_indices]
        
        # Intra-list diversity: average pairwise distance
        if len(top_k_features) > 1:
            distances = []
            for i in range(len(top_k_features)):
                for j in range(i + 1, len(top_k_features)):
                    dist = np.linalg.norm(top_k_features[i] - top_k_features[j])
                    distances.append(dist)
            intra_list_diversity = np.mean(distances)
        else:
            intra_list_diversity = 0.0
        
        metrics[f'intra_list_diversity@{k}'] = intra_list_diversity
    
    return metrics


def evaluate_recommendation_system(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 item_features: np.ndarray = None,
                                 unique_items: int = None,
                                 k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Comprehensive evaluation of recommendation system
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted scores
        item_features: Item feature matrix for diversity calculation
        unique_items: Number of unique items for coverage calculation
        k_values: List of k values for ranking metrics
        
    Returns:
        Dictionary of all metrics
    """
    all_metrics = {}
    
    # Basic regression metrics
    regression_metrics = calculate_metrics(y_true, y_pred)
    all_metrics.update(regression_metrics)
    
    # Ranking metrics
    ranking_metrics = calculate_ranking_metrics(y_true, y_pred, k_values)
    all_metrics.update(ranking_metrics)
    
    # Hit rate metrics
    hit_rate_metrics = calculate_hit_rate(y_true, y_pred, k_values)
    all_metrics.update(hit_rate_metrics)
    
    # Coverage metrics
    if unique_items is not None:
        coverage_metrics = calculate_coverage(y_pred, unique_items, k_values)
        all_metrics.update(coverage_metrics)
    
    # Diversity metrics
    if item_features is not None:
        diversity_metrics = calculate_diversity_metrics(y_pred, item_features, k_values)
        all_metrics.update(diversity_metrics)
    
    return all_metrics
