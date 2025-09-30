"""
Two Tower Neural Network Model for Amazon Product Recommendation

This module implements the two-tower architecture where:
- User Tower: Encodes user features and behavior into user embeddings
- Item Tower: Encodes item features into item embeddings
- Similarity: Computes cosine similarity between user and item embeddings

Model Input/Output:
==================
Input:
- user_features: Dict[str, torch.Tensor] - User feature tensors
- item_features: Dict[str, torch.Tensor] - Item feature tensors  
- text_features: torch.Tensor - Text embeddings (768-dim)

Output:
- user_embeddings: torch.Tensor - User embeddings (batch_size, embedding_dim)
- item_embeddings: torch.Tensor - Item embeddings (batch_size, embedding_dim)

Example:
--------
user_emb, item_emb = model(user_features, item_features, text_features)
similarity = model.compute_similarity(user_emb, item_emb)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class UserTower(nn.Module):
    """User Tower for encoding user features into embeddings"""
    
    def __init__(self, 
                 user_feature_dims: Dict[str, int],
                 embedding_dim: int = 128,
                 hidden_dims: List[int] = [256, 128],
                 dropout_rate: float = 0.2):
        super(UserTower, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.user_feature_dims = user_feature_dims
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict()
        for feature_name, vocab_size in user_feature_dims.items():
            self.embeddings[feature_name] = nn.Embedding(vocab_size, embedding_dim)
        
        # Dense layers
        layers = []
        input_dim = len(user_feature_dims) * embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Final projection layer
        layers.append(nn.Linear(input_dim, embedding_dim))
        self.dense_layers = nn.Sequential(*layers)
        
    def forward(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for user tower
        
        Args:
            user_features: Dictionary of user feature tensors
            
        Returns:
            User embedding tensor of shape (batch_size, embedding_dim)
        """
        embeddings = []
        
        for feature_name, feature_values in user_features.items():
            if feature_name in self.embeddings:
                emb = self.embeddings[feature_name](feature_values)
                embeddings.append(emb)
        
        # Concatenate all embeddings
        if embeddings:
            user_emb = torch.cat(embeddings, dim=-1)
        else:
            # Handle case with no categorical features
            batch_size = next(iter(user_features.values())).size(0)
            user_emb = torch.zeros(batch_size, 0, device=next(iter(user_features.values())).device)
        
        # Pass through dense layers
        user_embedding = self.dense_layers(user_emb)
        
        # L2 normalization
        user_embedding = F.normalize(user_embedding, p=2, dim=1)
        
        return user_embedding


class ItemTower(nn.Module):
    """Item Tower for encoding item features into embeddings"""
    
    def __init__(self,
                 item_feature_dims: Dict[str, int],
                 text_feature_dim: int = 768,  # BERT embedding dimension
                 embedding_dim: int = 128,
                 hidden_dims: List[int] = [256, 128],
                 dropout_rate: float = 0.2):
        super(ItemTower, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.item_feature_dims = item_feature_dims
        self.text_feature_dim = text_feature_dim
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict()
        for feature_name, vocab_size in item_feature_dims.items():
            self.embeddings[feature_name] = nn.Embedding(vocab_size, embedding_dim)
        
        # Text feature processing
        self.text_projection = nn.Linear(text_feature_dim, embedding_dim)
        
        # Dense layers
        layers = []
        input_dim = (len(item_feature_dims) + 1) * embedding_dim  # +1 for text features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Final projection layer
        layers.append(nn.Linear(input_dim, embedding_dim))
        self.dense_layers = nn.Sequential(*layers)
        
    def forward(self, 
                item_features: Dict[str, torch.Tensor],
                text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for item tower
        
        Args:
            item_features: Dictionary of item feature tensors
            text_features: Text features tensor of shape (batch_size, text_feature_dim)
            
        Returns:
            Item embedding tensor of shape (batch_size, embedding_dim)
        """
        embeddings = []
        
        # Process categorical features
        for feature_name, feature_values in item_features.items():
            if feature_name in self.embeddings:
                emb = self.embeddings[feature_name](feature_values)
                embeddings.append(emb)
        
        # Process text features
        text_emb = self.text_projection(text_features)
        embeddings.append(text_emb)
        
        # Concatenate all embeddings
        if embeddings:
            item_emb = torch.cat(embeddings, dim=-1)
        else:
            # Handle case with no features
            batch_size = text_features.size(0)
            item_emb = torch.zeros(batch_size, 0, device=text_features.device)
        
        # Pass through dense layers
        item_embedding = self.dense_layers(item_emb)
        
        # L2 normalization
        item_embedding = F.normalize(item_embedding, p=2, dim=1)
        
        return item_embedding


class TwoTowerModel(nn.Module):
    """Two Tower Model for recommendation"""
    
    def __init__(self,
                 user_feature_dims: Dict[str, int],
                 item_feature_dims: Dict[str, int],
                 text_feature_dim: int = 768,
                 embedding_dim: int = 128,
                 user_hidden_dims: List[int] = [256, 128],
                 item_hidden_dims: List[int] = [256, 128],
                 dropout_rate: float = 0.2):
        super(TwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Initialize towers
        self.user_tower = UserTower(
            user_feature_dims=user_feature_dims,
            embedding_dim=embedding_dim,
            hidden_dims=user_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        self.item_tower = ItemTower(
            item_feature_dims=item_feature_dims,
            text_feature_dim=text_feature_dim,
            embedding_dim=embedding_dim,
            hidden_dims=item_hidden_dims,
            dropout_rate=dropout_rate
        )
        
    def forward(self,
                user_features: Dict[str, torch.Tensor],
                item_features: Dict[str, torch.Tensor],
                text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for two tower model
        
        Args:
            user_features: Dictionary of user feature tensors
            item_features: Dictionary of item feature tensors
            text_features: Text features tensor
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        user_embeddings = self.user_tower(user_features)
        item_embeddings = self.item_tower(item_features, text_features)
        
        return user_embeddings, item_embeddings
    
    def compute_similarity(self,
                          user_embeddings: torch.Tensor,
                          item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between user and item embeddings
        
        Args:
            user_embeddings: User embeddings tensor
            item_embeddings: Item embeddings tensor
            
        Returns:
            Similarity scores tensor
        """
        return torch.mm(user_embeddings, item_embeddings.t())
    
    def get_user_embeddings(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get user embeddings for inference"""
        return self.user_tower(user_features)
    
    def get_item_embeddings(self, 
                           item_features: Dict[str, torch.Tensor],
                           text_features: torch.Tensor) -> torch.Tensor:
        """Get item embeddings for inference"""
        return self.item_tower(item_features, text_features)
