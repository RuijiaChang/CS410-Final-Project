# scripts/data_processing/encode_review_emb.py
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# ------- 配置区（按需修改） -------
PARQUET_IN  = Path(r"data/processed/interactions_mapped.parquet") 
PARQUET_OUT = Path(r"data/processed/interactions_mapped_with_review_emb.parquet") 
TEXT_COL    = "item_text"  
OUT_COL     = "review_emb"  
MODEL_NAME  = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE  = 256           
NORMALIZE   = True          
# --------------------------------

def main():
    assert PARQUET_IN.exists(), f"Not found: {PARQUET_IN}"
    print(f"[load] {PARQUET_IN}")
    df = pd.read_parquet(PARQUET_IN)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing '{TEXT_COL}' in parquet")

    texts = df[TEXT_COL].astype(str).fillna("").tolist()
    n = len(texts)
    print(f"[info] rows = {n}")

    # lazy import to speed up error messages
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] loading {MODEL_NAME} on {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)


    def preprocess(s: str) -> str:
        return s.strip()

    embs = np.zeros((n, model.get_sentence_embedding_dimension()), dtype=np.float32)

    for i in tqdm(range(0, n, BATCH_SIZE), desc="encode"):
        batch_txt = [preprocess(t) for t in texts[i:i+BATCH_SIZE]]
        vec = model.encode(
            batch_txt,
            batch_size=len(batch_txt),
            convert_to_numpy=True,
            normalize_embeddings=NORMALIZE,
            show_progress_bar=False
        ).astype(np.float32)
        embs[i:i+len(vec)] = vec

    df[OUT_COL] = [row.tolist() for row in embs]

    for col in ("category", "brand"):
        if col in df.columns:
            df = df.drop(columns=[col])

    print(f"[save] {PARQUET_OUT}")
    df.to_parquet(PARQUET_OUT, index=False)
    print(f"[done] {PARQUET_OUT} | shape={df.shape} | review_dim={embs.shape[1]}")

if __name__ == "__main__":
    main()
