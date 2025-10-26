# scripts/tools/rebuild_id_maps.py
from pathlib import Path
import pandas as pd, json

base = Path("data/processed")
df = pd.read_parquet(base/"interactions_mapped.parquet")
df["user_id"] = df["user_id"].astype(str)

users = sorted(df["user_id"].unique())
uid2idx = {u:i for i,u in enumerate(users)}

(json.dump(uid2idx, open(base/"uid2idx.json","w",encoding="utf-8"), indent=2, ensure_ascii=False))
print("rebuilt:", len(uid2idx), "users;")
