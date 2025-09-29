"""
Unit tests for Two Tower Model
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.two_tower_model import TwoTowerModel, UserTower, ItemTower


class TestUserTower(unittest.TestCase):
    """Test cases for UserTower"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.user_feature_dims = {
            'user_id': 1000,
            'age_group': 5,
            'gender': 3
        }
        self.embedding_dim = 64
        self.batch_size = 32
        
        self.user_tower = UserTower(
            user_feature_dims=self.user_feature_dims,
            embedding_dim=self.embedding_dim,
            hidden_dims=[128, 64],
            dropout_rate=0.1
        )
        
        # Create sample user features
        self.user_features = {
            'user_id': torch.randint(0, 1000, (self.batch_size,)),
            'age_group': torch.randint(0, 5, (self.batch_size,)),
            'gender': torch.randint(0, 3, (self.batch_size,))
        }
    
    def test_user_tower_initialization(self):
        """Test UserTower initialization"""
        self.assertEqual(self.user_tower.embedding_dim, self.embedding_dim)
        self.assertEqual(len(self.user_tower.embeddings), len(self.user_feature_dims))
        self.assertEqual(len(self.user_tower.dense_layers), 7)  # 3 layers * 2 + 1 projection
    
    def test_user_tower_forward(self):
        """Test UserTower forward pass"""
        user_embeddings = self.user_tower(self.user_features)
        
        # Check output shape
        self.assertEqual(user_embeddings.shape, (self.batch_size, self.embedding_dim))
        
        # Check that embeddings are normalized
        norms = torch.norm(user_embeddings, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
    
    def test_user_tower_empty_features(self):
        """Test UserTower with empty features"""
        empty_features = {}
        user_embeddings = self.user_tower(empty_features)
        
        # Should still return valid embeddings
        self.assertEqual(user_embeddings.shape, (0, self.embedding_dim))


class TestItemTower(unittest.TestCase):
    """Test cases for ItemTower"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.item_feature_dims = {
            'item_id': 5000,
            'category': 20,
            'brand': 100
        }
        self.text_feature_dim = 768
        self.embedding_dim = 64
        self.batch_size = 32
        
        self.item_tower = ItemTower(
            item_feature_dims=self.item_feature_dims,
            text_feature_dim=self.text_feature_dim,
            embedding_dim=self.embedding_dim,
            hidden_dims=[128, 64],
            dropout_rate=0.1
        )
        
        # Create sample item features
        self.item_features = {
            'item_id': torch.randint(0, 5000, (self.batch_size,)),
            'category': torch.randint(0, 20, (self.batch_size,)),
            'brand': torch.randint(0, 100, (self.batch_size,))
        }
        self.text_features = torch.randn(self.batch_size, self.text_feature_dim)
    
    def test_item_tower_initialization(self):
        """Test ItemTower initialization"""
        self.assertEqual(self.item_tower.embedding_dim, self.embedding_dim)
        self.assertEqual(self.item_tower.text_feature_dim, self.text_feature_dim)
        self.assertEqual(len(self.item_tower.embeddings), len(self.item_feature_dims))
    
    def test_item_tower_forward(self):
        """Test ItemTower forward pass"""
        item_embeddings = self.item_tower(self.item_features, self.text_features)
        
        # Check output shape
        self.assertEqual(item_embeddings.shape, (self.batch_size, self.embedding_dim))
        
        # Check that embeddings are normalized
        norms = torch.norm(item_embeddings, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
    
    def test_item_tower_empty_features(self):
        """Test ItemTower with empty features"""
        empty_features = {}
        item_embeddings = self.item_tower(empty_features, self.text_features)
        
        # Should still return valid embeddings
        self.assertEqual(item_embeddings.shape, (self.batch_size, self.embedding_dim))


class TestTwoTowerModel(unittest.TestCase):
    """Test cases for TwoTowerModel"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.user_feature_dims = {
            'user_id': 1000,
            'age_group': 5,
            'gender': 3
        }
        self.item_feature_dims = {
            'item_id': 5000,
            'category': 20,
            'brand': 100
        }
        self.text_feature_dim = 768
        self.embedding_dim = 64
        self.batch_size = 32
        
        self.model = TwoTowerModel(
            user_feature_dims=self.user_feature_dims,
            item_feature_dims=self.item_feature_dims,
            text_feature_dim=self.text_feature_dim,
            embedding_dim=self.embedding_dim,
            user_hidden_dims=[128, 64],
            item_hidden_dims=[128, 64],
            dropout_rate=0.1
        )
        
        # Create sample features
        self.user_features = {
            'user_id': torch.randint(0, 1000, (self.batch_size,)),
            'age_group': torch.randint(0, 5, (self.batch_size,)),
            'gender': torch.randint(0, 3, (self.batch_size,))
        }
        self.item_features = {
            'item_id': torch.randint(0, 5000, (self.batch_size,)),
            'category': torch.randint(0, 20, (self.batch_size,)),
            'brand': torch.randint(0, 100, (self.batch_size,))
        }
        self.text_features = torch.randn(self.batch_size, self.text_feature_dim)
    
    def test_model_initialization(self):
        """Test TwoTowerModel initialization"""
        self.assertEqual(self.model.embedding_dim, self.embedding_dim)
        self.assertIsInstance(self.model.user_tower, UserTower)
        self.assertIsInstance(self.model.item_tower, ItemTower)
    
    def test_model_forward(self):
        """Test TwoTowerModel forward pass"""
        user_embeddings, item_embeddings = self.model(
            self.user_features, self.item_features, self.text_features
        )
        
        # Check output shapes
        self.assertEqual(user_embeddings.shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(item_embeddings.shape, (self.batch_size, self.embedding_dim))
        
        # Check that embeddings are normalized
        user_norms = torch.norm(user_embeddings, dim=1)
        item_norms = torch.norm(item_embeddings, dim=1)
        self.assertTrue(torch.allclose(user_norms, torch.ones_like(user_norms), atol=1e-6))
        self.assertTrue(torch.allclose(item_norms, torch.ones_like(item_norms), atol=1e-6))
    
    def test_compute_similarity(self):
        """Test similarity computation"""
        user_embeddings, item_embeddings = self.model(
            self.user_features, self.item_features, self.text_features
        )
        
        similarity_scores = self.model.compute_similarity(user_embeddings, item_embeddings)
        
        # Check output shape
        self.assertEqual(similarity_scores.shape, (self.batch_size, self.batch_size))
        
        # Check that diagonal elements are close to 1 (since embeddings are normalized)
        diagonal_similarities = torch.diag(similarity_scores)
        self.assertTrue(torch.allclose(diagonal_similarities, torch.ones_like(diagonal_similarities), atol=1e-6))
    
    def test_get_user_embeddings(self):
        """Test getting user embeddings"""
        user_embeddings = self.model.get_user_embeddings(self.user_features)
        
        self.assertEqual(user_embeddings.shape, (self.batch_size, self.embedding_dim))
        
        # Check that embeddings are normalized
        norms = torch.norm(user_embeddings, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))
    
    def test_get_item_embeddings(self):
        """Test getting item embeddings"""
        item_embeddings = self.model.get_item_embeddings(self.item_features, self.text_features)
        
        self.assertEqual(item_embeddings.shape, (self.batch_size, self.embedding_dim))
        
        # Check that embeddings are normalized
        norms = torch.norm(item_embeddings, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))


class TestModelIntegration(unittest.TestCase):
    """Integration tests for the complete model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.user_feature_dims = {'user_id': 100, 'age_group': 5}
        self.item_feature_dims = {'item_id': 200, 'category': 10}
        self.text_feature_dim = 768
        self.embedding_dim = 32
        self.batch_size = 16
        
        self.model = TwoTowerModel(
            user_feature_dims=self.user_feature_dims,
            item_feature_dims=self.item_feature_dims,
            text_feature_dim=self.text_feature_dim,
            embedding_dim=self.embedding_dim
        )
    
    def test_end_to_end_training_step(self):
        """Test a complete training step"""
        # Create sample data
        user_features = {
            'user_id': torch.randint(0, 100, (self.batch_size,)),
            'age_group': torch.randint(0, 5, (self.batch_size,))
        }
        item_features = {
            'item_id': torch.randint(0, 200, (self.batch_size,)),
            'category': torch.randint(0, 10, (self.batch_size,))
        }
        text_features = torch.randn(self.batch_size, self.text_feature_dim)
        targets = torch.randn(self.batch_size)
        
        # Forward pass
        user_embeddings, item_embeddings = self.model(
            user_features, item_features, text_features
        )
        
        # Compute similarity and loss
        similarity_scores = self.model.compute_similarity(user_embeddings, item_embeddings)
        loss = torch.nn.functional.mse_loss(similarity_scores, targets.unsqueeze(1).expand(-1, self.batch_size))
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
    
    def test_model_consistency(self):
        """Test that model produces consistent outputs"""
        user_features = {
            'user_id': torch.randint(0, 100, (self.batch_size,)),
            'age_group': torch.randint(0, 5, (self.batch_size,))
        }
        item_features = {
            'item_id': torch.randint(0, 200, (self.batch_size,)),
            'category': torch.randint(0, 10, (self.batch_size,))
        }
        text_features = torch.randn(self.batch_size, self.text_feature_dim)
        
        # Set model to eval mode
        self.model.eval()
        
        with torch.no_grad():
            # First forward pass
            user_emb1, item_emb1 = self.model(user_features, item_features, text_features)
            
            # Second forward pass
            user_emb2, item_emb2 = self.model(user_features, item_features, text_features)
            
            # Check that outputs are identical
            self.assertTrue(torch.allclose(user_emb1, user_emb2))
            self.assertTrue(torch.allclose(item_emb1, item_emb2))


if __name__ == '__main__':
    unittest.main()
