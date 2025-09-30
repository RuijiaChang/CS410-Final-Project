"""
Data loading and preprocessing utilities for Amazon recommendation dataset

DataLoader Input/Output:
===========================================================
Input:
- data_path: str - Path to processed CSV file
- config: dict - Configuration with user_features, item_features, text_feature_columns, batch_size, neg_ratio

Output per batch:
- user_features: Dict[str, torch.Tensor] - User feature tensors
- item_features: Dict[str, torch.Tensor] - Item feature tensors
- text_features: torch.Tensor - Text embeddings (batch_size, 768) # BERT
- labels: torch.Tensor - Binary labels (1=positive, 0=negative)
- ratings: torch.Tensor - Ratings (positive samples have rating, negative samples are 0)

Example:
--------
Input:
user_id,item_id,rating,age_group,gender,category,brand ... (other features)
123,1001,4.5,25-35,male,Electronics,Apple
123,1002,3.8,25-35,male,Books,Random House
456,1001,4.2,18-25,female,Electronics,Apple
456,1003,3.0,18-25,female,Clothing,Nike
789,1002,4.1,35-45,male,Books,Random House


if batch_size = 2
Output:
batch = {
    'user_features': {
        'user_id': tensor([123, 456]),
        'age_group': tensor([2, 1]),
        'gender': tensor([1, 0])
    },
    'item_features': {
        'item_id': tensor([1001, 1002]),
        'category': tensor([3, 1]),
        'brand': tensor([15, 8])
    },
    'text_features': tensor([[0.1, 0.2, ...], [0.3, 0.4, ...]]),  # (2, 768)
    'labels': tensor([1, 0]),
    'ratings': tensor([4.5, 0])
}

"""
