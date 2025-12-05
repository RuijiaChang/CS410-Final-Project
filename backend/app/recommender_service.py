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

    def recommend_for_user(
        self,
        user_id: str,
        top_k: int = 20,
        filter_seen: bool = False,
    ) -> List[ProductCard]:
        """
        Returns a list of product-card-style dicts for the given user.
        """
        user_vec = self._get_user_vector(user_id)
        user_vec = user_vec.reshape(1, -1)

        scores, indices = self.index.search(user_vec, top_k * 2)
        item_indices = indices[0].tolist()
        scores_list = scores[0].tolist()

        # Optional: filter out items this user already interacted with
        if filter_seen:
            seen_items = set(
                self.reviews.loc[self.reviews["user_id"] == user_id, "item_id"].dropna()
            )
        else:
            seen_items = set()

        results: List[ProductCard] = []
        for rank, (idx, score) in enumerate(zip(item_indices, scores_list)):
            iid = self.idx2iid.get(idx)
            if iid is None:
                continue

            if iid in seen_items:
                continue

            row = self._get_item_row_by_id(iid)
            if row is None:
                continue

            # Handle missing fields gracefully
            images = (
                json.loads(row.get("images", "[]"))
                if isinstance(row.get("images"), str)
                else []
            )
            image0 = images.get("hi_res", [None])[0] if images else None

            results.append(
                ProductCard(
                    item_id=iid,
                    title=row.get("title"),
                    cover_image=image0,
                    price=(
                        float(row.get("price")) if row.get("price") != "None" else 0.0
                    ),
                    average_rating=(
                        float(row.get("average_rating"))
                        if row.get("average_rating") is not None
                        else None
                    ),
                    rating_number=(
                        int(row.get("rating_number"))
                        if row.get("rating_number") is not None
                        else None
                    ),
                    main_category=row.get("main_category"),
                )
            )

            if len(results) >= top_k:
                break

        return results

    def get_item_detail(self, item_id: str) -> Optional[ProductDetails]:
        """
        Returns full item metadata (for Product Detail Page) as dict.
        """
        data = self._get_item_row_by_id(item_id)
        if data is None:
            return None

        return ProductDetails(
            item_id=item_id,
            title=data["title"],
            price=float(data["price"]) if (not isinstance(data["price"], str) and data["price"] is not None) else 0.0,
            images=[x for x in list(json.loads(data["images"]).get("hi_res", "")) if x is not None],
            main_category=data["main_category"],
            average_rating=data["average_rating"],
            rating_number=data["rating_number"],
            description=data["description"],
            features=json.loads(data["features"]),
            categories=json.loads(data["categories"]),
            store=data["store"],
            details=json.loads(data["details"]),
        )

    def get_item_reviews(self, item_id: str, limit: int = 50) -> List[Review]:
        """
        Returns list of reviews for this item_id, formatted as:
        """
        df = self.reviews.loc[self.reviews["item_id"] == item_id]

        if df.empty:
            return []

        # Sort by timestamp descending (most recent first), if available
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp", ascending=False)

        if limit is not None:
            df = df.head(limit)

        results: List[Review] = []
        for _, r in df.iterrows():
            review = Review()
            review.user_id = r.get("user_id")
            review.item_id = r.get("item_id")
            review.rating = int(r.get("rating"))
            review.review_title = r.get("review_title", "")
            review.review_text = r.get("review_body", "")
            review.timestamp = int(r.get("timestamp"))
            results.append(review)
        return results

    def get_user_profile(self, user_id: str, limit_items: int = 50) -> UserProfile:
        """
        Minimal user profile for User Profile Page.
        """
        df = self.reviews.loc[self.reviews["user_id"] == user_id]
        df = df.sort_values("timestamp", ascending=False)
        df = df.head(limit_items)
        items = []
        for _, r in df.iterrows():
            items.append(
                Review(
                    user_id=user_id,
                    item_id=r.get("item_id"),
                    review_title=r.get("review_title", ""),
                    review_text=r.get("review_body", ""),
                    rating=int(r.get("rating")),
                    timestamp=int(r.get("timestamp")),
                )
            )

        return UserProfile(user_id=user_id, items=items)

    def get_all_users(self) -> List[str]:
        """
        Returns a list of all available user IDs.
        """
        return list(self.uid2idx.keys())


recommender_service = RecommenderService(Path("model"))