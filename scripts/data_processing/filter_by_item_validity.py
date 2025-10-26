import pandas as pd
import json

loader = pd.read_csv("data/interim/loader_ready.csv")
items = pd.read_parquet("data/interim/items_meta.parquet") 
iid2idx = json.load(open("data/interim/iid2idx.json", "r", encoding="utf-8"))

valid_items = set(items["item_id"].astype(str))

loader_filtered = loader[loader["item_id"].astype(str).isin(valid_items)]
loader_filtered.to_csv("data/processed/interaction.csv", index=False)

iid2idx_filtered = {k:v for k,v in iid2idx.items() if k in valid_items}
json.dump(iid2idx_filtered, open("data/processed/iid2idx_filtered.json", "w", encoding="utf-8"), indent=2)
