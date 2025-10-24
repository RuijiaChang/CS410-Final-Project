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
from __future__ import annotations
import os, json, ast
from typing import Dict, List, Optional, Tuple, Callable, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------- utils ----------------

def _read_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

def _as_long(x) -> torch.LongTensor:
    return torch.as_tensor(x, dtype=torch.long)

def _as_float(x) -> torch.FloatTensor:
    return torch.as_tensor(x, dtype=torch.float32)

def parse_vec(cell) -> Optional[np.ndarray]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)): return None
    if isinstance(cell, (list, tuple, np.ndarray)):
        arr = np.asarray(cell, dtype=np.float32)
        return arr
    s = str(cell).strip()
    if not s: return None
    # JSON / Python list / CSV
    for parser in (lambda z: json.loads(z), lambda z: ast.literal_eval(z), lambda z: [float(x) for x in z.split(",")]):
        try:
            arr = np.asarray(parser(s), dtype=np.float32)
            return arr
        except Exception:
            continue
    return None

# -------------- categorical encoders --------------

class CatEncoder:
    def __init__(self, name: str, stoi: Dict[str, int], unk_idx: int = 0):
        self.name, self.stoi, self.unk_idx = name, stoi, unk_idx

    @classmethod
    def fit(cls, values: pd.Series, name: str, add_unk=True) -> "CatEncoder":
        uniq = pd.Series(values.astype(str).fillna("")).unique().tolist()
        stoi, idx = ({}, 0)
        if add_unk:
            stoi["<UNK>"] = 0; idx = 1
        for v in uniq:
            if v == "" and add_unk: continue
            stoi[str(v)] = idx; idx += 1
        return cls(name, stoi, 0 if add_unk else -1)

    def encode(self, s: pd.Series) -> np.ndarray:
        def f(v):
            v = "" if pd.isna(v) else str(v)
            return self.stoi.get(v, self.unk_idx if self.unk_idx >= 0 else 0)
        return s.astype(str).map(f).astype(np.int64).to_numpy()

    def to_json(self): return {"name": self.name, "stoi": self.stoi, "unk_idx": self.unk_idx}
    @classmethod
    def from_json(cls, d): return cls(d["name"], d["stoi"], d.get("unk_idx", 0))

# -------------- text encoder (optional) --------------

def build_text_encoder(model_name="sentence-transformers/all-mpnet-base-v2") -> Callable[[List[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    if dim != 768:
        print(f"[warn] text encoder dim {dim} != 768; will still proceed.")
    def encode(texts: List[str]) -> np.ndarray:
        if not texts: return np.zeros((0, dim), dtype=np.float32)
        emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(np.float32)
    return encode

# -------------- dataset --------------

class InteractionsDataset(Dataset):
    """Holds positive rows; negatives are created in collate_fn."""
    def __init__(self, df: pd.DataFrame, cfg: Dict,
                 user_enc: Dict[str, CatEncoder], item_enc: Dict[str, CatEncoder],
                 item_lookup: Dict[str, Dict[str, str]],
                 item_text_map: Dict[str, np.ndarray],
                 user_pos: Dict[str, set],
                 all_item_ids: np.ndarray,
                 item_numeric_map: Optional[Dict[str, np.ndarray]] = None,
                 user_profile_map: Optional[Dict[str, np.ndarray]] = None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.user_enc, self.item_enc = user_enc, item_enc
        self.item_lookup = item_lookup
        self.item_text_map = item_text_map
        self.item_numeric_map = item_numeric_map
        self.user_profile_map = user_profile_map
        self.user_cols = cfg["user_features"]
        self.item_cols = cfg["item_features"]
        self.neg_ratio = int(cfg.get("neg_ratio", 1))
        self.rng = np.random.default_rng(int(cfg.get("seed", 42)))

        # cache encoded positives
        self._user = self.df["user_id"].astype(str).to_numpy()
        self._item = self.df["item_id"].astype(str).to_numpy()
        self._rating = self.df.get("rating", pd.Series([1.0]*len(self.df))).astype(float).to_numpy()
        self._user_feats = {c: user_enc[c].encode(self.df[c]) for c in self.user_cols}
        self._item_feats = {c: item_enc[c].encode(self.df[c]) for c in self.item_cols}
        self.user_pos = user_pos
        self.all_items = all_item_ids

    def __len__(self): return len(self.df)
    def __getitem__(self, idx): return idx

def make_collate_fn(ds: InteractionsDataset,
                    text_dim=768,
                    num_item_numeric: int = 0,
                    has_user_profile: bool = False) -> Callable:

    def collate(indices: List[int]) -> Dict[str, torch.Tensor]:
        Bp = len(indices)                                  # positives
        total = Bp * (1 + ds.neg_ratio)

        # containers
        uf = {c: np.zeros(total, dtype=np.int64) for c in ds.user_cols}
        itf = {c: np.zeros(total, dtype=np.int64) for c in ds.item_cols}
        text = np.zeros((total, text_dim), dtype=np.float32)
        labels = np.zeros(total, dtype=np.int64)
        ratings = np.zeros(total, dtype=np.float32)
        if num_item_numeric > 0:
            item_numeric = np.zeros((total, num_item_numeric), dtype=np.float32)
        else:
            item_numeric = None
        if has_user_profile:
            user_prof = np.zeros((total, 768), dtype=np.float32)
        else:
            user_prof = None

        w = 0
        for idx in indices:
            uid = ds._user[idx]; iid = ds._item[idx]
            # positive
            for c in ds.user_cols: uf[c][w] = ds._user_feats[c][idx]
            for c in ds.item_cols: itf[c][w] = ds._item_feats[c][idx]
            emb = ds.item_text_map.get(iid);  text[w] = emb if emb is not None and emb.shape[0]==text_dim else 0.0
            if item_numeric is not None:
                item_numeric[w] = ds.item_numeric_map.get(iid, 0.0)
            if user_prof is not None:
                user_prof[w] = ds.user_profile_map.get(uid, 0.0)
            labels[w] = 1; ratings[w] = ds._rating[idx]; w += 1

            # negatives
            pos_set = ds.user_pos.get(uid, set())
            need, tries = ds.neg_ratio, 0
            while need > 0 and tries < need * 50:
                cand = str(ds.rng.choice(ds.all_items))
                tries += 1
                if cand in pos_set: continue
                # copy user feats
                for c in ds.user_cols: uf[c][w] = ds._user_feats[c][idx]
                # item feats by lookup
                row = ds.item_lookup.get(cand)
                if row is None: continue
                for c in ds.item_cols:
                    val = row.get(c, "")
                    itf[c][w] = ds.item_enc[c].stoi.get(str(val), ds.item_enc[c].unk_idx)
                emb = ds.item_text_map.get(cand); text[w] = emb if emb is not None and emb.shape[0]==text_dim else 0.0
                if item_numeric is not None:
                    item_numeric[w] = ds.item_numeric_map.get(cand, 0.0)
                if user_prof is not None:
                    user_prof[w] = ds.user_profile_map.get(uid, 0.0)
                labels[w] = 0; ratings[w] = 0.0; w += 1; need -= 1

        # trim
        uf = {k: v[:w] for k, v in uf.items()}
        itf = {k: v[:w] for k, v in itf.items()}
        batch = {
            "user_features": {k: _as_long(v) for k, v in uf.items()},
            "item_features": {k: _as_long(v) for k, v in itf.items()},
            "text_features": _as_float(text[:w]),
            "labels": _as_long(labels[:w]),
            "ratings": _as_float(ratings[:w]),
        }
        if item_numeric is not None:
            batch["item_numeric"] = _as_float(item_numeric[:w])
        if user_prof is not None:
            batch["user_profile_emb"] = _as_float(user_prof[:w])
        return batch
    return collate

# -------------- high-level factory --------------

def build_dataloaders(
    data_path: str,
    config: Dict,
    encoders_json: Optional[str] = None,
    text_encoder_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict]:
    """
    config:
      user_features: ["user_id", ...]
      item_features: ["item_id", "category", "brand", ...]
      text_feature_columns: ["title", "text"]     # OR:
      text_embedding_col: "item_text_emb"        # Jenny’s part (length = 768)
      item_numeric_features: ["price","discount_pct"]  (optional, Jenny’s part)
      numeric_scaler_path: "data/final/scalers.json"   (optional)
      user_profile_emb_col: "user_text_emb"            (optional, Jenny’s part; 768)
      batch_size: 1024
      neg_ratio: 1
      num_workers: 4
      shuffle: true
      split: {"train":0.85,"valid":0.05}  # optional; default = temporal LOO by 'timestamp'
      timestamp_col: "timestamp"
      seed: 42
    """
    df = _read_table(data_path).copy()
    assert "user_id" in df.columns and "item_id" in df.columns, "data must contain user_id & item_id"
    if "rating" not in df.columns: df["rating"] = 1.0
    ts_col = config.get("timestamp_col", "timestamp")
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)

    # ----- split -----
    split_cfg = config.get("split")
    if split_cfg is None:
        df = df.sort_values(["user_id", ts_col])
        rid = np.arange(len(df)); df["_rid"] = rid
        train_idx, valid_idx, test_idx = [], [], []
        for uid, g in df.groupby("user_id", sort=False):
            g = g.sort_values(ts_col)
            r = g["_rid"].to_numpy()
            if len(r)==1: test_idx.append(r[-1])
            elif len(r)==2: valid_idx.append(r[-2]); test_idx.append(r[-1])
            else: train_idx.extend(r[:-2]); valid_idx.append(r[-2]); test_idx.append(r[-1])
        df.drop(columns=["_rid"], inplace=True)
    else:
        idx = np.arange(len(df)); rng.shuffle(idx)
        n_tr = int(len(idx) * split_cfg.get("train", 0.85))
        n_va = int(len(idx) * split_cfg.get("valid", 0.05))
        train_idx = idx[:n_tr].tolist()
        valid_idx = idx[n_tr:n_tr+n_va].tolist()
        test_idx  = idx[n_tr+n_va:].tolist()

    # ----- fit/load encoders on TRAIN -----
    user_cols = config["user_features"]
    item_cols = config["item_features"]
    user_enc, item_enc = {}, {}

    if encoders_json and os.path.exists(encoders_json):
        blob = json.load(open(encoders_json, "r", encoding="utf-8"))
        user_enc = {k: CatEncoder.from_json(v) for k, v in blob["user"].items()}
        item_enc = {k: CatEncoder.from_json(v) for k, v in blob["item"].items()}
    else:
        dtr = df.iloc[train_idx]
        for c in user_cols: user_enc[c] = CatEncoder.fit(dtr[c], c)
        for c in item_cols: item_enc[c] = CatEncoder.fit(dtr[c], c)
        if encoders_json:
            json.dump({"user":{k:v.to_json() for k,v in user_enc.items()},
                       "item":{k:v.to_json() for k,v in item_enc.items()}},
                      open(encoders_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ----- item lookup -----
    item_keys = ["item_id"] + [c for c in item_cols if c!="item_id"]
    item_tbl = df[item_keys].drop_duplicates("item_id", keep="last")
    item_lookup = { str(r["item_id"]): {c: r.get(c, "") for c in item_cols} for _, r in item_tbl.iterrows() }
    all_item_ids = item_tbl["item_id"].astype(str).to_numpy()

    # ----- item text embeddings -----
    text_map: Dict[str, np.ndarray] = {}
    emb_col = config.get("text_embedding_col")
    text_cols = config.get("text_feature_columns", None)
    if emb_col and emb_col in df.columns:
        for _, r in item_tbl.iterrows():
            vec = parse_vec(r.get(emb_col))
            if vec is None: continue
            v = np.zeros(768, dtype=np.float32); v[:min(768, vec.shape[0])] = vec[:min(768, vec.shape[0])]
            text_map[str(r["item_id"])] = v
    else:
        if text_cols is None: text_cols = []
        if text_cols:
            if text_encoder_fn is None:
                text_encoder_fn = build_text_encoder("sentence-transformers/all-mpnet-base-v2")
            texts, ids = [], []
            for _, r in item_tbl.iterrows():
                parts = [str(r.get(c, "")) for c in text_cols if pd.notna(r.get(c, ""))]
                texts.append(" [SEP] ".join([p for p in parts if p]))
                ids.append(str(r["item_id"]))
            embs = text_encoder_fn(texts) if texts else np.zeros((0, 768), dtype=np.float32)
            for iid, e in zip(ids, embs):
                text_map[iid] = e.astype(np.float32)

    # ----- optional: numeric features (Jenny’s part) -----
    num_cols = config.get("item_numeric_features", []) or []
    scalers = {}
    sp = config.get("numeric_scaler_path")
    if num_cols and sp and os.path.exists(sp):
        scalers = json.load(open(sp, "r", encoding="utf-8"))

    def standardize(col, val):
        if col in scalers:
            m = float(scalers[col].get("mean", 0.0))
            s = max(float(scalers[col].get("std", 1.0)), 1e-6)
            return (float(val) - m) / s
        return float(val)

    item_numeric_map = {}
    if num_cols:
        for _, r in item_tbl.iterrows():
            iid = str(r["item_id"])
            vec = [standardize(c, r.get(c, 0.0)) for c in num_cols]
            item_numeric_map[iid] = np.asarray(vec, dtype=np.float32)

    # ----- optional: user profile embedding (Jenny’s part, 768) -----
    user_profile_map = {}
    up_col = config.get("user_profile_emb_col")
    if up_col and up_col in df.columns:
        # Expect the data_path already contains this column (or you can load/merge an external user_profile.parquet)
        for uid, sub in df[["user_id", up_col]].drop_duplicates("user_id").iterrows():
            v = parse_vec(sub[up_col])
            if v is not None:
                t = np.zeros(768, dtype=np.float32); t[:min(768, v.shape[0])] = v[:min(768, v.shape[0])]
                user_profile_map[str(sub["user_id"])] = t

    # ----- user -> positive items -----
    user_pos = { str(uid): set(g["item_id"].astype(str)) for uid, g in df.groupby("user_id", sort=False) }

    # ----- build datasets/loaders -----
    def _make_loader(indices: List[int], shuffle: bool) -> DataLoader:
        sub = df.iloc[indices].copy()
        ds = InteractionsDataset(
            sub, config, user_enc, item_enc, item_lookup,
            text_map, user_pos, all_item_ids,
            item_numeric_map=item_numeric_map if num_cols else None,
            user_profile_map=user_profile_map if up_col else None,
        )
        collate = make_collate_fn(
            ds,
            text_dim=768,
            num_item_numeric=len(num_cols),
            has_user_profile=bool(up_col)
        )
        return DataLoader(
            ds,
            batch_size=int(config.get("batch_size", 1024)),
            shuffle=shuffle,
            num_workers=int(config.get("num_workers", 4)),
            pin_memory=True,
            collate_fn=collate,
        )

    train_loader = _make_loader(train_idx, shuffle=bool(config.get("shuffle", True)))
    valid_loader = _make_loader(valid_idx, shuffle=False) if len(valid_idx) else None
    test_loader  = _make_loader(test_idx,  shuffle=False) if len(test_idx) else None

    artifacts = {
        "user_encoders": {k: v.to_json() for k, v in user_enc.items()},
        "item_encoders": {k: v.to_json() for k, v in item_enc.items()},
        "num_users": len(user_enc.get("user_id", CatEncoder("user_id", {})).stoi),
        "num_items": len(item_enc.get("item_id", CatEncoder("item_id", {})).stoi),
        "feature_info": {
            "user": config["user_features"],
            "item": config["item_features"],
            "item_numeric": num_cols,
            "has_user_profile": bool(up_col),
        }
    }
    return train_loader, valid_loader, test_loader, artifacts