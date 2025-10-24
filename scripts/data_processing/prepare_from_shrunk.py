"""
Prepare a loader-friendly CSV and item meta from all_categories_shrunk.parquet

Outputs (under --outdir):
  - loader_ready.csv                  # for your DataLoader (core user/item/text columns)
  - items_meta.parquet                # item-level metadata (Jenny’s part)
  - splits.json                       # per-user temporal leave-one-out split indices (for loader_ready.csv)
  - uid2idx.json / iid2idx.json       # ID→index mappings (for model/eval)
  - [optional] interactions_full.parquet     # full fields backup
  - [optional] negatives.json                 # fixed negatives for valid/test (for evaluation)
"""

import argparse, json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

# ----------- column candidates -----------
CAND_USER = ["user_id", "reviewerID", "uid", "reviewer_id"]
CAND_ITEM_P = ["parent_asin"]
CAND_ITEM_A = ["asin", "item_id", "iid", "product_id"]
CAND_RATE = ["rating", "overall", "stars"]
CAND_TIME = ["timestamp", "unixReviewTime", "time", "review_time"]

def pick_col(df: pd.DataFrame, cands) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def normalize_base(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize key columns: user_id / item_id (prefer parent_asin) / rating / timestamp; dedupe by keeping most recent per (user,item)."""
    u = pick_col(df, CAND_USER)
    i_p = pick_col(df, CAND_ITEM_P)
    i_a = pick_col(df, CAND_ITEM_A)
    r = pick_col(df, CAND_RATE)
    t = pick_col(df, CAND_TIME)
    if u is None or (i_p is None and i_a is None) or t is None:
        raise ValueError(f"Missing required cols. Have: {list(df.columns)[:20]} ...")

    out = pd.DataFrame()
    out["user_id"] = df[u].astype(str)
    out["item_id"] = (df[i_p] if i_p else df[i_a]).astype(str)
    out["rating"] = (pd.to_numeric(df[r], errors="coerce") if r else 1.0).fillna(1.0)
    ts = pd.to_numeric(df[t], errors="coerce")
    # handle ms vs s
    med = ts.dropna().median() if ts.notna().any() else np.nan
    if pd.notna(med) and med > 1e10:
        ts = (ts / 1000).round()
    out["timestamp"] = ts.astype("Int64")

    # bring back the remaining original columns (without overwriting normalized ones)
    for c in df.columns:
        if c not in out.columns:
            out[c] = df[c]

    # drop rows without timestamp
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    # keep most recent per (user,item)
    out = (
        out.sort_values(["user_id", "item_id", "timestamp"])
           .drop_duplicates(["user_id", "item_id"], keep="last")
           .reset_index(drop=True)
    )
    return out

def extract_category(x: Any) -> str:
    # x can be list/str/None
    try:
        if isinstance(x, list) and len(x) > 0:
            return str(x[0])
        if isinstance(x, str) and x.strip():
            return x.strip()
    except Exception:
        pass
    return "Unknown"

def extract_brand(row: Dict[str, Any]) -> str:
    # multi-source fallback: brand / Brand / details.brand / details.Brand / store
    for key in ["brand", "Brand"]:
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key]).strip()
    det = row.get("details", None)
    if isinstance(det, dict):
        for key in ["brand", "Brand"]:
            v = det.get(key)
            if v:
                return str(v).strip()
    if "store" in row and pd.notna(row["store"]) and str(row["store"]).strip():
        return str(row["store"]).strip()
    return "Unknown"

def leave_one_out_split(df: pd.DataFrame, ts_col: str = "timestamp"):
    """Per-user ascending by time: last=test, second-to-last=valid, the rest=train; returns index lists for the current row order."""
    df = df.sort_values(["user_id", ts_col]).reset_index(drop=True)
    rid = np.arange(len(df))
    df["_rid"] = rid
    splits = {"train": [], "valid": [], "test": []}
    for _, g in df.groupby("user_id", sort=False):
        g = g.sort_values(ts_col)
        r = g["_rid"].to_list()
        if len(r) == 1:
            splits["test"].append(r[-1])
        elif len(r) == 2:
            splits["valid"].append(r[-2])
            splits["test"].append(r[-1])
        else:
            splits["train"].extend(r[:-2])
            splits["valid"].append(r[-2])
            splits["test"].append(r[-1])
    df.drop(columns=["_rid"], inplace=True)
    return splits

def build_id_maps(df: pd.DataFrame):
    """Build uid2idx / iid2idx (string IDs to contiguous indices)."""
    users = sorted(df["user_id"].astype(str).unique())
    items = sorted(df["item_id"].astype(str).unique())
    uid2idx = {u: i for i, u in enumerate(users)}
    iid2idx = {v: i for i, v in enumerate(items)}
    return uid2idx, iid2idx

def build_eval_negatives(df: pd.DataFrame, splits, k=100, seed=42):
    """Generate fixed negatives for valid/test only (rejection sampling), for reproducible evaluation."""
    rng = np.random.default_rng(seed)
    items = df["item_id"].astype(str).unique()
    n = items.size
    user_pos = {str(u): set(g["item_id"].astype(str)) for u, g in df.groupby("user_id", sort=False)}
    res = {"valid": [], "test": []}
    def gen(name):
        out = []
        for rid in splits[name]:
            uid = str(df.iloc[rid]["user_id"])
            pos = user_pos.get(uid, set())
            negs, tries = [], 0
            while len(negs) < k and tries < k * 50:
                it = items[int(rng.integers(0, n))]
                if it not in pos:
                    negs.append(it)
                tries += 1
            out.append({"row_id": int(rid), "user_id": uid, "neg_items": negs})
        return out
    res["valid"] = gen("valid"); res["test"] = gen("test")
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True, help="data/shrunk/all_categories_shrunk.parquet")
    ap.add_argument("--outdir", default="data/final")
    ap.add_argument("--target_rows", type=int, default=0, help="Optional: downsample to a fixed number of rows (0 = no downsample)")
    ap.add_argument("--keep_all_fields", action="store_true", help="Also write interactions_full.parquet (keep all original fields)")
    # Evaluation negatives (valid/test)
    ap.add_argument("--write_negatives", action="store_true", help="Additionally write negatives.json (fixed negatives for valid/test)")
    ap.add_argument("--k_eval", type=int, default=100, help="Number of negatives per positive for evaluation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) read and normalize base columns
    raw = pd.read_parquet(args.input_parquet)
    df = normalize_base(raw)
    print(f"[1/6] normalized & dedup: {len(df):,} rows")

    # 2) Derive item-side business columns (aligning with DataLoader; Jenny’s part)
    df["category"] = df.get("categories", pd.Series([None]*len(df))).apply(extract_category)
    df["brand"] = df.apply(lambda r: extract_brand(r.to_dict()), axis=1)
    title = df.get("title", "").fillna("")
    text  = df.get("text", "").fillna("")
    df["item_text"] = (title.astype(str) + " [SEP] " + text.astype(str)).str.strip()

    # 3) Optional: global downsample
    if args.target_rows and len(df) > args.target_rows:
        df = df.sample(n=args.target_rows, random_state=args.seed).reset_index(drop=True)
        print(f"[2/6] downsampled to {len(df):,} rows")

    # 4) Write DataLoader input (clean CSV)
    keep_cols = ["user_id","item_id","rating","timestamp","category","brand","item_text"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan
    loader_csv = outdir / "loader_ready.csv"
    df[keep_cols].to_csv(loader_csv, index=False)
    print(f"[3/6] wrote loader_ready.csv → {loader_csv}")

    # 5) Split & mappings & (optional) negatives
    # 5.1 leave-one-out split (indices correspond to loader_ready.csv rows)
    splits = leave_one_out_split(df, ts_col="timestamp")
    with open(outdir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    print(f"[4/6] splits: train={len(splits['train']):,}, valid={len(splits['valid']):,}, test={len(splits['test']):,}")

    # 5.2 ID mappings (for model/evaluation)
    uid2idx, iid2idx = build_id_maps(df)
    with open(outdir / "uid2idx.json","w",encoding="utf-8") as f: json.dump(uid2idx,f,indent=2,ensure_ascii=False)
    with open(outdir / "iid2idx.json","w",encoding="utf-8") as f: json.dump(iid2idx,f,indent=2,ensure_ascii=False)

    # 5.3 Optional: fixed negatives for valid/test (reproducible evaluation)
    if args.write_negatives:
        negs = build_eval_negatives(df, splits, k=args.k_eval, seed=args.seed)
        with open(outdir / "negatives.json","w",encoding="utf-8") as f:
            json.dump(negs, f, indent=2, ensure_ascii=False)
        print(f"[5/6] negatives.json written (k_eval={args.k_eval})")

    # 6) items_meta.parquet (Jenny’s part); and optional full-field parquet
    #    — drop duplicate columns; convert list/dict columns to JSON strings to avoid parquet type issues
    meta_cols = ["title","text","images","category","brand","asin","parent_asin","store","details","subtitle"]
    present_meta = [c for c in meta_cols if c in df.columns]
    cols = ["item_id"] + list(dict.fromkeys(present_meta))
    items_meta = (df[cols].drop_duplicates("item_id", keep="last").reset_index(drop=True))
    for c in ["images","details","category"]:
        if c in items_meta.columns:
            items_meta[c] = items_meta[c].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
            )
    items_meta.to_parquet(outdir / "items_meta.parquet", index=False)

    if args.keep_all_fields:
        df.to_parquet(outdir / "interactions_full.parquet", index=False)

    print(f"[6/6] DONE → {outdir}")
    print("   - loader_ready.csv")
    print("   - splits.json")
    if args.write_negatives:
        print("   - negatives.json")
    print("   - uid2idx.json / iid2idx.json")
    print("   - items_meta.parquet")
    if args.keep_all_fields:
        print("   - interactions_full.parquet")

if __name__ == "__main__":
    main()
