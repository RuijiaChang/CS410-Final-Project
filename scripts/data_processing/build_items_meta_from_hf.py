#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_items_meta_from_hf.py
-------------------------------------------------------
Build a de-duplicated item-level metadata table from the
HuggingFace Amazon-Reviews-2023 "meta_categories" subsets.

No CLI args needed. Just run:  python build_items_meta_from_hf.py

Inputs (edit paths below if needed):
  - data/final/loader_ready.csv
  - data/final/iid2idx.json
HuggingFace:
  - McAuley-Lab/Amazon-Reviews-2023  (streaming, split="full")

Outputs:
  - data/item_data/items_meta.parquet  (one row per item_id)
  - data/item_data/items_meta_chunks/  (intermediate chunk files)
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Set, List

import pandas as pd
from datasets import get_dataset_config_names, load_dataset

# ====== Config (edit if needed) ======
DATASET_ID  = "McAuley-Lab/Amazon-Reviews-2023"
LOADER_PATH = "data/final/loader_ready.csv"
IID2IDX_PATH= "data/final/iid2idx.json"
CHUNK_DIR   = Path("data/item_data/items_meta_chunks")
FINAL_OUT   = Path("data/item_data/items_meta.parquet")
BATCH_SIZE  = 100_000      # rows per chunk
META_PREFIX = "raw_meta_"  # scan only meta subsets starting with this prefix


def load_unique_targets(loader_path: str, iid2idx_path: str) -> Set[str]:
    """Load unique item_ids from loader and intersect with iid2idx keys."""
    loader = pd.read_csv(loader_path, usecols=["item_id"])
    loader_ids = set(loader["item_id"].astype(str).dropna().unique())
    try:
        iid2idx = json.load(open(iid2idx_path, "r", encoding="utf-8"))
        idx_ids = set(map(str, iid2idx.keys()))
        targets = loader_ids & idx_ids
        print(f"[INFO] unique item_ids in loader: {len(loader_ids):,}")
        print(f"[INFO] items in iid2idx:          {len(idx_ids):,}")
        print(f"[INFO] target intersection:       {len(targets):,}")
        return targets
    except Exception:
        print("[WARN] Failed to read iid2idx.json. Using loader unique IDs only.")
        return loader_ids


def list_meta_subsets(dataset_id: str, prefix: str) -> List[str]:
    """List all raw_meta_* configs."""
    configs = get_dataset_config_names(dataset_id, trust_remote_code=True)
    subsets = [c for c in configs if c.startswith(prefix)]
    print(f"[INFO] meta subsets found: {len(subsets)}")
    return subsets


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Keep all fields; convert list/dict to JSON strings (parquet friendly)."""
    row = {}
    for k, v in rec.items():
        if isinstance(v, (list, dict)):
            row[k] = json.dumps(v, ensure_ascii=False)
        else:
            row[k] = v
    return row


def stream_one_subset(
    dataset_id: str,
    subset: str,
    targets: Set[str],
    seen: Set[str],
    batch_size: int,
    chunk_dir: Path,
    chunk_prefix: str = "meta",
) -> int:
    """Stream one meta subset; write matched unique items in chunks. Return newly written rows."""
    try:
        ds = load_dataset(
            dataset_id,
            subset,
            split="full",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[WARN] skip {subset}: {e}")
        return 0

    # Resume-friendly chunk index
    existing = sorted(chunk_dir.glob(f"{chunk_prefix}_*.parquet"))
    chunk_id = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 0

    buf: List[Dict[str, Any]] = []
    written = 0

    for rec in ds:
        asin = rec.get("parent_asin") or rec.get("asin")
        if asin is None:
            continue
        asin = str(asin)
        if asin not in targets or asin in seen:
            continue

        row = {"item_id": asin}
        row.update(normalize_record(rec))
        buf.append(row)
        seen.add(asin)

        if len(buf) >= batch_size:
            out_path = chunk_dir / f"{chunk_prefix}_{chunk_id:06d}.parquet"
            pd.DataFrame(buf).to_parquet(out_path, index=False)
            written += len(buf)
            buf.clear()
            chunk_id += 1
            print(f"[INFO] wrote chunk {out_path.name}; matched so far: {len(seen):,} / {len(targets):,}")

        # Early stop if all targets collected
        if len(seen) >= len(targets):
            break

    if buf:
        out_path = chunk_dir / f"{chunk_prefix}_{chunk_id:06d}.parquet"
        pd.DataFrame(buf).to_parquet(out_path, index=False)
        written += len(buf)
        print(f"[INFO] wrote chunk {out_path.name}; matched so far: {len(seen):,} / {len(targets):,}")

    return written


def merge_chunks(chunk_dir: Path, final_out: Path) -> None:
    """Concatenate chunks, drop duplicates by item_id, write final parquet."""
    chunks = sorted(chunk_dir.glob("meta_*.parquet"))
    if not chunks:
        raise SystemExit(f"[ERROR] no chunks found under {chunk_dir}")

    dfs = [pd.read_parquet(p) for p in chunks]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates("item_id").reset_index(drop=True)
    final_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(final_out, index=False)
    print(f"[DONE] saved {final_out} : {len(df)} rows, {len(df.columns)} columns")


def main():
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    targets = load_unique_targets(LOADER_PATH, IID2IDX_PATH)
    if not targets:
        raise SystemExit("[ERROR] no target item_ids to match.")

    subsets = list_meta_subsets(DATASET_ID, META_PREFIX)
    seen: Set[str] = set()
    total_written = 0

    for i, cfg in enumerate(subsets, 1):
        if len(seen) >= len(targets):
            print("[INFO] all targets collected; early stop.")
            break
        print(f"[INFO] ({i}/{len(subsets)}) processing subset: {cfg}")
        total_written += stream_one_subset(
            dataset_id=DATASET_ID,
            subset=cfg,
            targets=targets,
            seen=seen,
            batch_size=BATCH_SIZE,
            chunk_dir=CHUNK_DIR,
            chunk_prefix="meta",
        )
        print(f"[INFO] subset {cfg} done. matched: {len(seen):,} / {len(targets):,}")

    merge_chunks(CHUNK_DIR, FINAL_OUT)


if __name__ == "__main__":
    main()
