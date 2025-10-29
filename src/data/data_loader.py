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

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Path & IO helpers
# -----------------------------
def _P(p: str) -> Path:
    path = Path(p)
    return path if path.exists() else Path.cwd() / p

def _dir_of(p: str) -> Path:
    q = _P(p)
    return q if q.is_dir() else q.parent

def _read_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Dataset
# -----------------------------
class InteractionDataset(Dataset):
    """
    Yield: (user_features, item_features, text_features, labels, ratings)
    Train: on-the-fly negative sampling with ratio=neg_ratio
    Val/Test: positives only
    """

    def __init__(
        self,
        df: pd.DataFrame,
        row_ids: List[int],
        uid2idx: Dict[str, int],
        iid2idx: Dict[str, int],
        item_text_emb: Dict[str, List[float]],
        mode: str = "train",
        neg_ratio: int = 0,
        seed: int = 42,
        emb_dim: int = 768,              # item embedding dim == review dim
        review_col: str = "review_emb",  # column in parquet
    ):
        self.df = df.reset_index(drop=True)
        max_idx = len(self.df) - 1
        self.rows = [rid for rid in row_ids if 0 <= rid <= max_idx]

        self.uid2idx = {str(k): int(v) for k, v in uid2idx.items()}
        self.iid2idx = {str(k): int(v) for k, v in iid2idx.items()}
        self.item_text_emb = {str(k): v for k, v in item_text_emb.items()}

        self.mode = mode
        self.neg_ratio = max(0, int(neg_ratio))
        self.rng = random.Random(seed)

        self.emb_dim = int(emb_dim)
        self.review_col = review_col
        self._zero = torch.zeros(self.emb_dim, dtype=torch.float32)

        # user positives set for rejection sampling
        self.user_pos = self.df.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(str))).to_dict()
        self.all_items = list(self.iid2idx.keys())

        # expand indices if training with negatives
        if self.mode == "train" and self.neg_ratio > 0:
            expanded = []
            for rid in self.rows:
                expanded.append((rid, 1))
                for _ in range(self.neg_ratio):
                    expanded.append((rid, 0))
            self.sample_index = expanded
        else:
            self.sample_index = [(rid, 1) for rid in self.rows]

    def __len__(self):
        return len(self.sample_index)

    def _sample_neg_item(self, uid: str) -> str:
        pos = self.user_pos.get(uid, set())
        while True:
            iid = self.rng.choice(self.all_items)
            if iid not in pos:
                return iid

    def _fuse_text(self, iid_str: str, review_vec) -> torch.Tensor:
        """
        Fuse item_emb (from JSON) and review_emb (from parquet) by element-wise sum
        to keep output dim == emb_dim.
        """
        # item embedding
        e_item = self.item_text_emb.get(iid_str)
        v_item = torch.tensor(e_item, dtype=torch.float32) if e_item is not None else self._zero

        # review embedding
        if review_vec is None:
            v_rev = self._zero
        else:
            v_rev = torch.tensor(review_vec, dtype=torch.float32)

        # ensure same length; if mismatch, fall back to item only
        if v_rev.numel() != v_item.numel():
            return v_item
        alpha = 0.6
        beta = 0.4
        return beta * v_item + alpha * v_rev 

    def _encode(self, uid: str, iid: str, rating: float, label: int, review_vec):
        uid_str = str(uid)
        iid_str = str(iid)

        user_idx = self.uid2idx.get(uid_str, 0)
        item_idx = self.iid2idx.get(iid_str, 0)

        user_feats = {"user_id": torch.tensor(user_idx, dtype=torch.long)}
        item_feats = {"item_id": torch.tensor(item_idx, dtype=torch.long)}  # keep only item_id

        text_feats = self._fuse_text(iid_str, review_vec)

        labels = torch.tensor(label, dtype=torch.long)
        ratings = torch.tensor(rating if label == 1 else 0.0, dtype=torch.float32)
        return user_feats, item_feats, text_feats, labels, ratings

    def __getitem__(self, idx: int):
        rid, is_pos = self.sample_index[idx]
        if rid >= len(self.df):
            rid = 0
        row = self.df.iloc[rid]
        uid = str(row["user_id"])
        rating = float(row.get("rating", 1.0))
        review_vec = row.get(self.review_col, None)

        if is_pos == 1:
            iid = str(row["item_id"])
            return self._encode(uid, iid, rating, 1, review_vec)
        else:
            iid = self._sample_neg_item(uid)
            return self._encode(uid, iid, 0.0, 0, None)


# -----------------------------
# Collate & Loader
# -----------------------------
def _collate(samples):
    user_b, item_b = {}, {}
    t_list, y_list, r_list = [], [], []

    u_keys = set().union(*(s[0].keys() for s in samples))
    i_keys = set().union(*(s[1].keys() for s in samples))

    for u, it, t, y, r in samples:
        for k in u_keys: user_b.setdefault(k, []).append(u[k])
        for k in i_keys: item_b.setdefault(k, []).append(it[k])
        t_list.append(t); y_list.append(y); r_list.append(r)

    for k in list(user_b.keys()): user_b[k] = torch.stack(user_b[k], 0)
    for k in list(item_b.keys()): item_b[k] = torch.stack(item_b[k], 0)
    text_features = torch.stack(t_list, 0)
    labels = torch.stack(y_list, 0).float()  # float for BCE/BCEWithLogits
    ratings = torch.stack(r_list, 0)

    return {
        "user_features": user_b,
        "item_features": item_b,
        "text_features": text_features,  # (B, 768)
        "labels": labels,
        "targets": labels,               # alias for eval code
        "ratings": ratings,
    }

def create_data_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate, num_workers=0)


# -----------------------------
# Processor
# -----------------------------
class DataProcessor:
    """
    Expect layout:
      data/processed/
        interactions_ready.parquet   # user_id, item_id, rating, review_emb(list[float])
        uid2idx.json / iid2idx.json  # optional
        splits.json                  # optional
        items_text_emb_*.json        # item embeddings (index-keyed: "0","1",...)
    """
    def __init__(self, data_cfg: Dict):
        self.cfg = data_cfg or {}
        self.neg_ratio = int(self.cfg.get("neg_ratio", 0))
        self.seed = int(self.cfg.get("seed", 42))
        self.emb_dim = int(self.cfg.get("emb_dim", 768))  # item emb dim; final text dim stays 768 after fusion
        self.data_path = self.cfg.get("data_path", "data/processed")
        self.text_embedding_path = self.cfg.get("text_embedding_path", None)

        self.user_feature_dims: Dict[str, int] = {}
        self.item_feature_dims: Dict[str, int] = {}
        self.eval_negatives: Optional[Dict] = None  # optional

    # ---- loaders ----
    def _load_core(self, base: Path) -> Tuple[pd.DataFrame, Optional[Dict[str, List[int]]]]:
        inter = base / "interactions_ready.parquet"
        splits_p = base / "splits.json"
        if not inter.exists():
            raise FileNotFoundError(f"Expected {inter}")
        df = pd.read_parquet(inter)

        # basic checks
        for c in ("user_id", "item_id", "rating", "review_emb"):
            if c not in df.columns:
                raise ValueError(f"`{c}` column is required in {inter.name}.")
        return df, (_read_json(splits_p) if splits_p.exists() else None)

    def _load_id_maps(self, base: Path, df: pd.DataFrame):
        uid_p = base / "uid2idx.json"
        iid_p = base / "iid2idx.json"
        if uid_p.exists() and iid_p.exists():
            uid2idx = _read_json(uid_p)
            iid2idx = _read_json(iid_p)
        else:
            users = sorted(df["user_id"].astype(str).unique())
            items = sorted(df["item_id"].astype(str).unique())
            uid2idx = {u: i for i, u in enumerate(users)}
            iid2idx = {v: i for i, v in enumerate(items)}
        uid2idx = {str(k): int(v) for k, v in uid2idx.items()}
        iid2idx = {str(k): int(v) for k, v in iid2idx.items()}
        return uid2idx, iid2idx

    def _load_embeddings(self, base: Path) -> Dict[str, List[float]]:
        if self.text_embedding_path:
            p = _P(self.text_embedding_path)
            if not p.exists():
                raise FileNotFoundError(f"text_embedding_path not found: {p}")
            return _read_json(p)
        # default: pick first items_text_emb_*.json
        cands = sorted(base.glob("items_text_emb_*.json"))
        if not cands:
            raise FileNotFoundError(f"No items_text_emb_*.json found under {base}")
        return _read_json(cands[0])

    def _build_splits_if_missing(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        df2 = df.copy()
        if "timestamp" in df2.columns:
            df2 = df2.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        else:
            df2 = df2.sort_values(["user_id"]).reset_index(drop=True)
        df2["_rid"] = np.arange(len(df2))
        sp = {"train": [], "valid": [], "test": []}
        for _, g in df2.groupby("user_id", sort=False):
            r = g["_rid"].tolist()
            if len(r) == 1:
                sp["test"].append(r[-1])
            elif len(r) == 2:
                sp["valid"].append(r[-2]); sp["test"].append(r[-1])
            else:
                sp["train"].extend(r[:-2]); sp["valid"].append(r[-2]); sp["test"].append(r[-1])
        return sp

    # ---- public ----
    def process_data(self, data_path: Optional[str] = None) -> Tuple[Dataset, Dataset, Dataset]:
        base = _dir_of(data_path or self.data_path)
        print(f"[DataProcessor] base_dir = {base}")

        # 1) interactions & splits
        df, splits = self._load_core(base)
        # normalize id types
        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        if "rating" not in df.columns:
            df["rating"] = 1.0

        # 2) id maps
        uid2idx, iid2idx = self._load_id_maps(base, df)

        # 3) item embeddings (index-keyed)
        emb = self._load_embeddings(base)
        emb = {str(k): v for k, v in emb.items()}
        try:
            first_vec = next(iter(emb.values()))
            if isinstance(first_vec, list): self.emb_dim = len(first_vec)
        except StopIteration:
            pass  # keep default

        # 4) splits
        if splits is None:
            splits = self._build_splits_if_missing(df)

        # 5) datasets
        train_ds = InteractionDataset(
            df, splits["train"], uid2idx, iid2idx, emb,
            mode="train", neg_ratio=self.neg_ratio, seed=self.seed,
            emb_dim=self.emb_dim, review_col="review_emb"
        )
        val_ds = InteractionDataset(
            df, splits["valid"], uid2idx, iid2idx, emb,
            mode="valid", neg_ratio=0, seed=self.seed,
            emb_dim=self.emb_dim, review_col="review_emb"
        )
        test_ds = InteractionDataset(
            df, splits["test"], uid2idx, iid2idx, emb,
            mode="test", neg_ratio=0, seed=self.seed,
            emb_dim=self.emb_dim, review_col="review_emb"
        )

        # 6) feature dims exposed for model init
        self.user_feature_dims = {"user_id": len(uid2idx)}
        self.item_feature_dims = {"item_id": len(iid2idx)}

        print(f"[DataProcessor] user_feature_dims: {self.user_feature_dims}")
        print(f"[DataProcessor] item_feature_dims : {self.item_feature_dims}")
        print(f"[DataProcessor] text_embedding_dim: {self.emb_dim}  # after fusion stays the same")
        return train_ds, val_ds, test_ds