import sys
from pathlib import Path

# ============================================
# Robust root finder — will ALWAYS find project root
# ============================================

FILE = Path(__file__).resolve()
ROOT = FILE

while ROOT != ROOT.parent:
    if (ROOT / "src").exists():
        break
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))

# ============================================
# Import modules
# ============================================
import torch
import json
import numpy as np
import pandas as pd

from src.data.data_loader import DataProcessor
from src.models.two_tower_model import TwoTowerModel



# ============================================
# CONFIG
# ============================================
MODEL_PATH = "outputs/results/best_model.pth"
DATA_PATH = "data/processed"      # contains: items_meta.parquet, uid2idx.json, iid2idx.json
ITEM_META_PATH = "data/processed/items_meta.parquet"
OUT_DIR = "outputs/serving"


def load_model(model_path, user_dim, item_dim, text_dim):
    print("Loading TwoTowerModel:", model_path)

    model = TwoTowerModel(
        user_feature_dims={"user_id": user_dim},
        item_feature_dims={"item_id": item_dim},
        text_feature_dim=text_dim,
        embedding_dim=128,
        user_hidden_dims=[256, 128],
        item_hidden_dims=[256, 128],
        dropout_rate=0.2,
    )

    # FIX: allow loading numpy objects (PyTorch 2.6+)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    OUT = Path(OUT_DIR)
    OUT.mkdir(parents=True, exist_ok=True)

    print("===============================================")
    print("  Generating embeddings for backend/frontend")
    print("===============================================")

    # ----------------------------------------
    # 1. Load DataProcessor (loads uid2idx/iid2idx & item_text_emb)
    # ----------------------------------------
    processor = DataProcessor({"data_path": DATA_PATH})
    train_ds, val_ds, test_ds = processor.process_data()

    text_emb_dim = processor.emb_dim                # e.g., 768

    # load id maps
    uid2idx = json.load(open(f"{DATA_PATH}/uid2idx.json", "r"))
    iid2idx = json.load(open(f"{DATA_PATH}/iid2idx.json", "r"))

    # load text embeddings (item_text_emb_*.json)
    item_text_emb = processor._load_embeddings(Path(DATA_PATH))

    # ----------------------------------------
    # 2. Load item meta (your final meta file)
    # ----------------------------------------
    item_meta = pd.read_parquet(ITEM_META_PATH)
    item_meta["item_id"] = item_meta["item_id"].astype(str)

    print(f"Loaded items_meta.parquet with {len(item_meta)} items.")

    # ----------------------------------------
    # 3. Load trained model
    # ----------------------------------------
    model = load_model(
        MODEL_PATH,
        user_dim=len(uid2idx),
        item_dim=len(iid2idx),
        text_dim=text_emb_dim
    )

    # ----------------------------------------
    # 4. Generate item embeddings
    # ----------------------------------------
    print("🔵 Generating item embeddings…")

    item_emb_list = []
    missing_emb = 0

    for iid in item_meta["item_id"].tolist():
        if iid not in iid2idx:
            print("item_id not found in iid2idx:", iid)
            continue

        # id → index
        iid_idx = torch.tensor([iid2idx[iid]], dtype=torch.long)

        # text embedding (from your JSON)
        fused = item_text_emb.get(iid)
        if fused is None:
            missing_emb += 1
            text_vec = torch.zeros((1, text_emb_dim), dtype=torch.float32)
        else:
            text_vec = torch.tensor([fused], dtype=torch.float32)

        item_feature = {"item_id": iid_idx}

        with torch.no_grad():
            emb = model.get_item_embeddings(item_feature, text_vec).numpy()

        item_emb_list.append(emb[0])

    item_emb_matrix = np.vstack(item_emb_list)
    np.save(OUT / "item_emb.npy", item_emb_matrix)

    print(f"Saved item_emb.npy  ({item_emb_matrix.shape})")
    print(f"   Missing text embeddings: {missing_emb}")

    # ----------------------------------------
    # 5. Generate user embeddings
    # ----------------------------------------
    print("Generating user embeddings…")

    user_emb_list = []
    for uid, idx in uid2idx.items():
        uid_tensor = torch.tensor([idx], dtype=torch.long)
        user_feature = {"user_id": uid_tensor}

        with torch.no_grad():
            emb = model.get_user_embeddings(user_feature).numpy()

        user_emb_list.append(emb[0])

    user_emb_matrix = np.vstack(user_emb_list)
    np.save(OUT / "user_emb.npy", user_emb_matrix)

    print(f"Saved user_emb.npy  ({user_emb_matrix.shape})")

    # ----------------------------------------
    # 6. Save mapping + item meta
    # ----------------------------------------
    json.dump(uid2idx, open(OUT / "uid2idx.json", "w"))
    json.dump(iid2idx, open(OUT / "iid2idx.json", "w"))

    item_meta.to_parquet(OUT / "items_meta.parquet", index=False)

    print(f"Saved items_meta.parquet → {OUT / 'items_meta.parquet'}")

    print("\nDONE! Embeddings are ready for backend / frontend.")
    print("   Output folder:", OUT)


if __name__ == "__main__":
    main()
