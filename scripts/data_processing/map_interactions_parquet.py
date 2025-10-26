import json
import argparse
import pandas as pd

def load_map(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return {str(k): int(v) for k, v in m.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter", required=True, help="path to interactions.parquet")
    ap.add_argument("--uid2idx", required=True, help="path to uid2idx.json")
    ap.add_argument("--iid2idx", required=True, help="path to iid2idx.json")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--user_col", default="user_id")
    ap.add_argument("--item_col", default="item_id")
    ap.add_argument("--strict", action="store_true", help="error if any id missing in map")
    ap.add_argument("--keep_raw", action="store_true", help="keep raw id columns as *_raw")
    args = ap.parse_args()

    print("[1] loading maps ...")
    uid_map = load_map(args.uid2idx)
    iid_map = load_map(args.iid2idx)
    print(f"  uid_map: {len(uid_map):,}  iid_map: {len(iid_map):,}")

    print("[2] reading interactions parquet ...")
    df = pd.read_parquet(args.inter)
    if args.user_col not in df.columns or args.item_col not in df.columns:
        raise KeyError(f"columns `{args.user_col}` or `{args.item_col}` not found. got: {df.columns.tolist()}")

    if args.keep_raw:
        df[f"{args.user_col}_raw"] = df[args.user_col].astype(str)
        df[f"{args.item_col}_raw"] = df[args.item_col].astype(str)

    print("[3] mapping IDs to int ...")
    df[args.user_col] = df[args.user_col].astype(str).map(uid_map)
    df[args.item_col] = df[args.item_col].astype(str).map(iid_map)

    miss_user = int(df[args.user_col].isna().sum())
    miss_item = int(df[args.item_col].isna().sum())
    print(f"  missing after map -> {args.user_col}: {miss_user}, {args.item_col}: {miss_item}")

    if args.strict and (miss_user > 0 or miss_item > 0):
        raise ValueError("Missing mappings found. Rerun without --strict to fill -1, or fix maps.")

    df[args.user_col] = df[args.user_col].fillna(-1).astype("int64")
    df[args.item_col] = df[args.item_col].fillna(-1).astype("int64")

    print("[4] writing parquet ...")
    df.to_parquet(args.out, index=False)
    print(f"done. shape={df.shape}, saved to: {args.out}")

if __name__ == "__main__":
    main()
