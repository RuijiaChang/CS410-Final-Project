from __future__ import annotations

from pathlib import Path
import json
from typing import List, Dict, Optional
from schemas import ProductCard, ProductDetails, Review, UserProfile

import numpy as np
import pandas as pd
import faiss

"""
NOTE: This module assumes that the following files are present in the 'model' directory:
- user_emb.npy: Numpy array of user embeddings.
- item_emb.npy: Numpy array of item embeddings.
- uid2idx.json: JSON mapping from user IDs to embedding indices.
- iid2idx.json: JSON mapping from item IDs to embedding indices.
- items_meta.parquet: Parquet file containing item metadata.
- loader_ready.csv: CSV file containing reviews/interactions.
"""
class RecommenderService:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.user_emb: np.ndarray
        self.item_emb: np.ndarray
        self.uid2idx: Dict[str, int]
        self.iid2idx: Dict[str, int]
        self.idx2iid: Dict[int, str]
        self.items_meta: pd.DataFrame
        self.reviews: pd.DataFrame
        self.index: faiss.Index

        self._load_artifacts()
        self._build_faiss_index()

    def _load_artifacts(self) -> None:
        self.user_emb = np.load("model/user_emb.npy").astype("float32")
        self.item_emb = np.load("model/item_emb.npy").astype("float32")
        with open("model/uid2idx.json", "r") as f:
            self.uid2idx = json.load(f)
        with open("model/iid2idx.json", "r") as f:
            self.iid2idx = json.load(f)

        # Invert item mapping: row index -> original item_id
        self.idx2iid = {idx: iid for iid, idx in self.iid2idx.items()}
        # Item metadata (row order matches item_emb.npy)
        self.items_meta = pd.read_parquet("model/items_meta.parquet")
        self.items_meta_by_id = self.items_meta.set_index("item_id")
        # Reviews / interactions
        self.reviews = pd.read_csv("model/loader_ready.csv")

        if "item_text" in self.reviews.columns:
            split_df = self.reviews["item_text"].str.split(
                "[SEP]", n=1, expand=True, regex=False
            )
            if split_df.shape[1] == 2:
                self.reviews["review_title"] = split_df[0].fillna("").str.strip()
                self.reviews["review_body"] = split_df[1].fillna("").str.strip()
            else:
                self.reviews["review_title"] = ""
                self.reviews["review_body"] = self.reviews["item_text"].fillna("")
        else:
            self.reviews["review_title"] = ""
            self.reviews["review_body"] = ""

    def _build_faiss_index(self) -> None:
        """
        Build a FAISS index over item embeddings.

        We use inner-product similarity on L2-normalized vectors,
        which is equivalent to cosine similarity.
        """
        d = int(self.item_emb.shape[1])

        # Normalize item embeddings
        norms = np.linalg.norm(self.item_emb, axis=1, keepdims=True) + 1e-10
        self.item_emb_norm = (self.item_emb / norms).astype("float32")

        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.item_emb_norm)

    def _get_user_vector(self, user_id: str) -> np.ndarray:
        if user_id not in self.uid2idx:
            raise KeyError(f"Unknown user_id: {user_id}")
        u_idx = self.uid2idx[user_id]
        vec = self.user_emb[u_idx]

        # Normalize to match item_emb_norm
        norm = np.linalg.norm(vec) + 1e-10
        return (vec / norm).astype("float32")

    def _get_item_row_by_idx(self, item_idx: int) -> pd.Series:
        return self.items_meta.iloc[item_idx]

    def _get_item_row_by_id(self, item_id: str) -> Optional[pd.Series]:
        try:
            return self.items_meta_by_id.loc[item_id]
        except KeyError:
            return None

