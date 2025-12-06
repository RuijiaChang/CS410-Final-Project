import argparse
import os
import pandas as pd
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", type=str, default="data")
    parser.add_argument("--idx_filename", type=str, default="iid2idx_filtered.json")
    parser.add_argument("--metadata_filename", type=str, default="items_meta.parquet")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2") # intfloat/multilingual-e5-base
    parser.add_argument("--batch_size", type=int, default=64)
    '''
        ['item_id', 'main_category', 'title', 'average_rating', 'rating_number',
       'features', 'description', 'price', 'images', 'videos', 'store',
       'categories', 'details', 'parent_asin', 'bought_together', 'subtitle',
       'author']
    '''
    parser.add_argument("--attrs", nargs="+", default=["title", "main_category", "average_rating", "features", "price", "details"])

    args = parser.parse_args()
    
    return args

def construct_item_text(args, item_id, item_df):
    row = item_df.loc[item_df["item_id"] == item_id]
    if row.empty:
        # logger.warning(f"No row identified for item_id: {item_id}")
        return None
    # logger.info(f"Located row for item_id: {item_id}")
    
    row = row.iloc[0]
    item_text = ""
    for attr in args.attrs:
        if attr in row:
            value = str(row[attr]) if pd.notna(row[attr]) else ""
            item_text += f"{attr}: {value} [SEP]\n"

    return item_text.strip()

def main():
    args = get_args()

    # load item metadata
    item_df = pd.read_parquet(os.path.join(args.data_folder, args.metadata_filename))
    logger.info(f"Loaded item metadata from: {os.path.join(args.data_folder, args.metadata_filename)}, len: {len(item_df)}")

    # load item index
    with open(os.path.join(args.data_folder, args.idx_filename), "r") as f:
        items_idx = json.load(f)
    logger.info(f"Loaded items idx from: {os.path.join(args.data_folder, args.idx_filename)}, len: {len(items_idx)}")

    # load model
    model = SentenceTransformer(args.model_name)
    logger.info(f"Loaded model: {args.model_name}")
    
    # construct item text
    item_texts = []
    item_ids = []
    attr_suffix = "_".join(args.attrs)
    text_path = os.path.join(args.data_folder, "texts", f"item_texts_{attr_suffix}.csv")

    if os.path.exists(text_path):
        df_text = pd.read_csv(text_path)
        item_ids = df_text["item_id"].tolist()
        item_texts = df_text["item_text"].fillna("").tolist()
    else:
        for idx in tqdm(items_idx.keys()):
            text =  construct_item_text(args, idx, item_df)
            if not text:
                continue
            item_texts.append(text)
            item_ids.append(items_idx[str(idx)])
        pd.DataFrame({"item_id": item_ids, "item_text": item_texts}).to_csv(text_path, index=False)
        logger.info(f"Saved item_texts to {text_path}")
    logger.info(f"Constructed text for {len(item_texts)} items")

    # get embeddings
    item_emb_json = os.path.join(args.data_folder, "embeddings", f"items_text_emb_{attr_suffix}_{args.model_name.split("/")[1]}.json")
    logger.info("Encoding item texts ...")
    if os.path.exists(item_emb_json):
        with open(item_emb_json, "r") as f:
            embeddings_dict = json.load(f)
    else:
        embeddings = model.encode(
            item_texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        logger.info(f"Embeddings shape: {embeddings.shape}")

        assert len(item_ids) == len(embeddings)
        embeddings_dict = {item_id: emb.tolist() for item_id, emb in zip(item_ids, embeddings)}
        with open(item_emb_json, "w") as f:
            json.dump(embeddings_dict, f)
        logger.info(f"Saved embedding to {item_emb_json}")
    
    # get FAISS index
    embeddings = np.array([emb for emb in embeddings_dict.values()], dtype=np.float32)
    item_ids = np.array([k for k in embeddings_dict.keys()])

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    logger.info(f"Added {len(embeddings)} item embeddings to FAISS index")

    faiss_path = os.path.join(args.data_folder, "faiss", f"faiss_index_{attr_suffix}_{args.model_name.split('/')[1]}_{attr_suffix}.index")
    faiss.write_index(index, faiss_path)
    logger.info(f"Saved FAISS index to {faiss_path}")

if __name__ == "__main__":
    main()
