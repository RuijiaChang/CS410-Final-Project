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
                # Clamp indices to valid range to avoid out-of-bounds errors
                vocab_size = self.user_feature_dims.get(feature_name, 1)
                clamped_values = torch.clamp(feature_values, min=0, max=vocab_size-1)
                emb = self.embeddings[feature_name](clamped_values)
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
                # Clamp indices to valid range to avoid out-of-bounds errors
                vocab_size = self.item_feature_dims.get(feature_name, 1)
                clamped_values = torch.clamp(feature_values, min=0, max=vocab_size-1)
                emb = self.embeddings[feature_name](clamped_values)
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
                 dropout_rate: float = 0.2,
                 logit_scale: float = 1.0):
        super(TwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Learnable logit scale parameter
        # This helps the model learn better representations by amplifying the similarity scores
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))
        
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
        Compute scaled dot product similarity between user and item embeddings
        
        Args:
            user_embeddings: User embeddings tensor
            item_embeddings: Item embeddings tensor
            
        Returns:
            Similarity scores tensor scaled by logit_scale
        """
        # Clamp logit_scale to reasonable range
        self.logit_scale.data = torch.clamp(self.logit_scale.data, min=0.01, max=100.0)
        similarity = torch.mm(user_embeddings, item_embeddings.t())
        return similarity * self.logit_scale
    
    def compute_similarity_dot(self,
                              user_embeddings: torch.Tensor,
                              item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled dot product for element-wise similarity (for BCE loss)
        
        Args:
            user_embeddings: User embeddings tensor
            item_embeddings: Item embeddings tensor
            
        Returns:
            Scaled dot product similarity scores for each pair
        """
        # Clamp logit_scale to reasonable range
        self.logit_scale.data = torch.clamp(self.logit_scale.data, min=0.01, max=100.0)
        similarity = torch.sum(user_embeddings * item_embeddings, dim=1)
        return similarity * self.logit_scale
    
    def get_user_embeddings(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get user embeddings for inference"""
        return self.user_tower(user_features)
    
    def get_item_embeddings(self, 
                           item_features: Dict[str, torch.Tensor],
                           text_features: torch.Tensor) -> torch.Tensor:
        """Get item embeddings for inference"""
        return self.item_tower(item_features, text_features)


def main():
    """Test function for Two Tower Model"""
    print("Testing Two Tower Model...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define feature dimensions (based on data_loader.py output format)
    user_feature_dims = {
        'user_id': 1000,      # 1000 unique users
        'age_group': 5,        # 5 age groups (0-4)
        'gender': 2,           # 2 genders (0=female, 1=male)
        'location': 50         # 50 locations
    }
    
    item_feature_dims = {
        'item_id': 5000,       # 5000 unique items
        'category': 20,        # 20 categories
        'brand': 100,          # 100 brands
        'price_range': 10      # 10 price ranges
    }
    
    # Model parameters
    text_feature_dim = 768     # BERT embedding dimension
    embedding_dim = 128
    batch_size = 32
    
    # Create model
    model = TwoTowerModel(
        user_feature_dims=user_feature_dims,
        item_feature_dims=item_feature_dims,
        text_feature_dim=text_feature_dim,
        embedding_dim=embedding_dim,
        user_hidden_dims=[256, 128],
        item_hidden_dims=[256, 128],
        dropout_rate=0.2
    )
    
    print(f"Model created successfully!")
    print(f"User tower parameters: {sum(p.numel() for p in model.user_tower.parameters()):,}")
    print(f"Item tower parameters: {sum(p.numel() for p in model.item_tower.parameters()):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate test data following data_loader.py output format
    # user_features: Dict[str, torch.Tensor] - User feature tensors
    user_features = {
        'user_id': torch.randint(0, user_feature_dims['user_id'], (batch_size,)),
        'age_group': torch.randint(0, user_feature_dims['age_group'], (batch_size,)),
        'gender': torch.randint(0, user_feature_dims['gender'], (batch_size,)),
        'location': torch.randint(0, user_feature_dims['location'], (batch_size,))
    }
    
    # item_features: Dict[str, torch.Tensor] - Item feature tensors
    item_features = {
        'item_id': torch.randint(0, item_feature_dims['item_id'], (batch_size,)),
        'category': torch.randint(0, item_feature_dims['category'], (batch_size,)),
        'brand': torch.randint(0, item_feature_dims['brand'], (batch_size,)),
        'price_range': torch.randint(0, item_feature_dims['price_range'], (batch_size,))
    }
    
    # text_features: torch.Tensor - Text embeddings (batch_size, 768) # BERT
    text_features = torch.randn(batch_size, text_feature_dim)
    
    # labels: torch.Tensor - Binary labels (1=positive, 0=negative)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    # ratings: torch.Tensor - Ratings (positive samples have rating, negative samples are 0)
    ratings = torch.where(labels == 1, 
                         torch.rand(batch_size) * 4 + 1,  # Random rating 1-5 for positive samples
                         torch.zeros(batch_size))         # 0 for negative samples
    
    print(f"\nTest data shapes (following data_loader.py format):")
    print(f"User features: {[(k, v.shape) for k, v in user_features.items()]}")
    print(f"Item features: {[(k, v.shape) for k, v in item_features.items()]}")
    print(f"Text features: {text_features.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Ratings: {ratings.shape}")
    
    # Display sample data to verify format
    print(f"\nSample data (first 5 samples):")
    print(f"User IDs: {user_features['user_id'][:5]}")
    print(f"Age groups: {user_features['age_group'][:5]}")
    print(f"Genders: {user_features['gender'][:5]}")
    print(f"Item IDs: {item_features['item_id'][:5]}")
    print(f"Categories: {item_features['category'][:5]}")
    print(f"Labels: {labels[:5]}")
    print(f"Ratings: {ratings[:5]}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model(user_features, item_features, text_features)
    
    print(f"User embeddings shape: {user_embeddings.shape}")
    print(f"Item embeddings shape: {item_embeddings.shape}")
    
    # Test similarity computation
    print(f"\nTesting similarity computation...")
    similarity_matrix = model.compute_similarity(user_embeddings, item_embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity matrix sample (first 5x5):")
    print(similarity_matrix[:5, :5])
    
    # Test individual tower outputs
    print(f"\nTesting individual tower outputs...")
    user_emb_only = model.get_user_embeddings(user_features)
    item_emb_only = model.get_item_embeddings(item_features, text_features)
    
    print(f"User embeddings only shape: {user_emb_only.shape}")
    print(f"Item embeddings only shape: {item_emb_only.shape}")
    
    # Verify embeddings are normalized
    print(f"\nVerifying L2 normalization...")
    user_norms = torch.norm(user_embeddings, p=2, dim=1)
    item_norms = torch.norm(item_embeddings, p=2, dim=1)
    
    print(f"User embedding norms (should be ~1.0): {user_norms[:5]}")
    print(f"Item embedding norms (should be ~1.0): {item_norms[:5]}")
    
    # Test with different batch sizes
    print(f"\nTesting with different batch sizes...")
    test_batch_sizes = [1, 16, 64]
    for bs in test_batch_sizes:
        test_user_features = {k: v[:bs] for k, v in user_features.items()}
        test_item_features = {k: v[:bs] for k, v in item_features.items()}
        test_text_features = text_features[:bs]
        
        with torch.no_grad():
            u_emb, i_emb = model(test_user_features, test_item_features, test_text_features)
        
        print(f"Batch size {bs}: User emb {u_emb.shape}, Item emb {i_emb.shape}")
    
    # Test gradient computation with realistic loss
    print(f"\nTesting gradient computation...")
    model.train()
    user_embeddings, item_embeddings = model(user_features, item_features, text_features)
    
    # Compute similarity scores for each user-item pair
    similarity_scores = torch.sum(user_embeddings * item_embeddings, dim=1)  # Element-wise dot product
    
    # Create a simple binary cross-entropy loss
    # Convert labels to probabilities for BCE loss
    target_probs = labels
    loss = F.binary_cross_entropy_with_logits(similarity_scores, target_probs)
    
    print(f"Loss value: {loss.item():.4f}")
    loss.backward()
    
    # Check if gradients exist
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"Gradients computed: {has_gradients}")
    
    # Test data format compatibility with training loop
    print(f"\nTesting data format compatibility...")
    batch_data = {
        'user_features': user_features,
        'item_features': item_features,
        'text_features': text_features,
        'labels': labels,
        'ratings': ratings
    }
    
    print(f"Batch data keys: {list(batch_data.keys())}")
    print(f"All required keys present: {all(key in batch_data for key in ['user_features', 'item_features', 'text_features', 'labels', 'ratings'])}")
    


if __name__ == "__main__":
    main()
